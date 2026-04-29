// HACK(eddyb) avoids rewriting all of the imports (see `lib.rs` and `build.rs`).
use crate::maybe_pqp_cg_ssa as rustc_codegen_ssa;

use crate::codegen_cx::{CodegenArgs, SpirvMetadata};
use crate::linker;
use crate::naga_transpile::should_transpile;
use crate::target::{SpirvTarget, SpirvTargetVariant};
use ar::{Archive, GnuBuilder, Header};
use rspirv::binary::Assemble;
use rspirv::dr::Module;
use rustc_attr_parsing::eval_config_entry;
use rustc_codegen_spirv_types::{CompileResult, ModuleResult};
use rustc_codegen_ssa::{CompiledModules, CrateInfo, NativeLib};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Diag;
use rustc_hir::attrs::NativeLibKind;
use rustc_metadata::{EncodedMetadata, fs::METADATA_FILENAME};
use rustc_middle::bug;
use rustc_middle::middle::dependency_format::Linkage;
use rustc_session::Session;
use rustc_session::config::{
    CrateType, DebugInfo, OptLevel, OutFileName, OutputFilenames, OutputType,
};
use rustc_session::output::{check_file_is_writeable, invalid_output_for_target, out_filename};
use rustc_span::Symbol;
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufWriter, Read};
use std::iter;
use std::path::{Path, PathBuf};

pub fn link(
    sess: &Session,
    compiled_modules: &CompiledModules,
    crate_info: &CrateInfo,
    metadata: &EncodedMetadata,
    outputs: &OutputFilenames,
    crate_name: &str,
) {
    let output_metadata = sess.opts.output_types.contains_key(&OutputType::Metadata);
    for &crate_type in sess.opts.crate_types.iter() {
        if (sess.opts.unstable_opts.no_codegen || !sess.opts.output_types.should_codegen())
            && !output_metadata
            && crate_type == CrateType::Executable
        {
            continue;
        }

        if invalid_output_for_target(sess, crate_type) {
            bug!(
                "invalid output type `{:?}` for target os `{}`",
                crate_type,
                sess.opts.target_triple
            );
        }

        for obj in compiled_modules
            .modules
            .iter()
            .filter_map(|m| m.object.as_ref())
        {
            check_file_is_writeable(obj, sess);
        }

        if outputs.outputs.should_codegen() {
            let out_filename = out_filename(sess, crate_type, outputs, Symbol::intern(crate_name));
            let out_filename_file_for_writing = out_filename.file_for_writing(
                outputs,
                OutputType::Exe,
                crate_name,
                sess.invocation_temp.as_deref(),
            );
            match crate_type {
                CrateType::Rlib => {
                    link_rlib(
                        sess,
                        compiled_modules,
                        crate_info,
                        metadata,
                        &out_filename_file_for_writing,
                    );
                }
                CrateType::Executable | CrateType::Cdylib | CrateType::Dylib => {
                    // HACK(eddyb) there's no way way to access `outputs.filestem`,
                    // so we pay the cost of building a whole `PathBuf` instead.
                    let disambiguated_crate_name_for_dumps = outputs
                        .with_extension("")
                        .file_name()
                        .unwrap()
                        .to_os_string();

                    link_exe(
                        sess,
                        crate_type,
                        &out_filename_file_for_writing,
                        compiled_modules,
                        crate_info,
                        outputs,
                        &disambiguated_crate_name_for_dumps,
                    );
                }
                other => {
                    sess.dcx()
                        .err(format!("CrateType {other:?} not supported yet"));
                }
            }
            match out_filename {
                OutFileName::Real(_) => {
                    // Already written to, above.
                }
                OutFileName::Stdout => {
                    // HACK(eddyb) wrote a file above, time to read it back out.
                    std::io::copy(
                        &mut std::io::BufReader::new(
                            std::fs::File::open(out_filename_file_for_writing).unwrap(),
                        ),
                        &mut std::io::stdout(),
                    )
                    .unwrap();
                }
            }
        }
    }
}

fn link_rlib(
    sess: &Session,
    compiled_modules: &CompiledModules,
    crate_info: &CrateInfo,
    metadata: &EncodedMetadata,
    out_filename: &Path,
) {
    let mut file_list = Vec::<&Path>::new();
    for obj in compiled_modules
        .modules
        .iter()
        .filter_map(|m| m.object.as_ref())
    {
        file_list.push(obj);
    }
    for lib in crate_info.used_libraries.iter() {
        if let NativeLibKind::Static {
            bundle: None | Some(true),
            ..
        } = lib.kind
        {
            sess.dcx().err(format!(
                "adding native library to rlib not supported yet: {}",
                lib.name
            ));
        }
    }

    create_archive(&file_list, metadata.stub_or_full(), out_filename);
}

fn link_exe(
    sess: &Session,
    crate_type: CrateType,
    out_filename: &Path,
    compiled_modules: &CompiledModules,
    crate_info: &CrateInfo,
    outputs: &OutputFilenames,
    disambiguated_crate_name_for_dumps: &OsStr,
) {
    let mut objects = Vec::new();
    let mut rlibs = Vec::new();
    for obj in compiled_modules
        .modules
        .iter()
        .filter_map(|m| m.object.as_ref())
    {
        objects.push(obj.clone());
    }

    link_local_crate_native_libs_and_dependent_crate_libs(&mut rlibs, sess, crate_type, crate_info);

    let cg_args = CodegenArgs::from_session(sess);

    // HACK(eddyb) this removes the `.json` in `.spv.json`, from `out_filename`.
    let out_path_spv = out_filename.with_extension("");

    let link_result = do_link(
        sess,
        &cg_args,
        &objects,
        &rlibs,
        outputs,
        disambiguated_crate_name_for_dumps,
    );
    let compile_result = match link_result {
        linker::LinkResult::SingleModule(module) => {
            let entry_points = entry_points(&module);
            post_link_single_module(sess, &cg_args, *module, &out_path_spv, None);
            CompileResult {
                entry_points,
                module: ModuleResult::SingleModule(out_path_spv),
            }
        }
        linker::LinkResult::MultipleModules {
            file_stem_to_entry_name_and_module,
        } => {
            let out_dir = out_path_spv.with_extension("spvs");
            if !out_dir.is_dir() {
                std::fs::create_dir_all(&out_dir).unwrap();
            }

            let entry_name_to_file_path: BTreeMap<_, _> = file_stem_to_entry_name_and_module
                .into_iter()
                .map(|(file_stem, (entry_name, module))| {
                    let mut out_file_name = file_stem;
                    out_file_name.push(".spv");
                    let out_file_path = out_dir.join(out_file_name);
                    post_link_single_module(
                        sess,
                        &cg_args,
                        module,
                        &out_file_path,
                        Some(disambiguated_crate_name_for_dumps),
                    );
                    (entry_name, out_file_path)
                })
                .collect();
            CompileResult {
                entry_points: entry_name_to_file_path.keys().cloned().collect(),
                module: ModuleResult::MultiModule(entry_name_to_file_path),
            }
        }
    };

    let file = File::create(out_filename).unwrap();
    // FIXME(eddyb) move this functionality into `rustc_codegen_spirv_types`.
    rustc_codegen_spirv_types::serde_json::to_writer(BufWriter::new(file), &compile_result)
        .unwrap();
}

fn entry_points(module: &rspirv::dr::Module) -> Vec<String> {
    module
        .entry_points
        .iter()
        .filter(|inst| inst.class.opcode == rspirv::spirv::Op::EntryPoint)
        .map(|inst| inst.operands[2].unwrap_literal_string().to_string())
        .collect()
}

/// For Kernel entry points using the `debug-printf` abort strategy, convert
/// `NonSemantic.DebugPrintf` instructions into `OpenCL.std printf` calls
/// (opcode 184) with byte-array format strings in `UniformConstant` storage.
/// The abort block's `OpReturn` terminator is left unchanged.
fn kernel_debug_printf_to_opencl(module: &mut Module) {
    use rspirv::dr::{Instruction, Operand};
    use rspirv::spirv::{Capability, ExecutionModel, Op, StorageClass, Word};
    use std::collections::{HashMap, HashSet};

    let has_kernel = module
        .entry_points
        .iter()
        .any(|ep| ep.operands[0].unwrap_execution_model() == ExecutionModel::Kernel);
    if !has_kernel {
        return;
    }

    let debug_printf_ext_id = module
        .ext_inst_imports
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::ExtInstImport
                && inst.operands[0].unwrap_literal_string() == "NonSemantic.DebugPrintf"
        })
        .and_then(|inst| inst.result_id);
    let Some(debug_printf_ext_id) = debug_printf_ext_id else {
        return;
    };

    let header = module.header.as_mut().unwrap();
    let mut next_id = || {
        let id = header.bound;
        header.bound += 1;
        id
    };

    // --- Basic types ---

    let u32_ty = module
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypeInt
                && inst.operands[0].unwrap_literal_bit32() == 32
                && inst.operands[1].unwrap_literal_bit32() == 0
        })
        .and_then(|inst| inst.result_id)
        .unwrap_or_else(|| {
            let id = next_id();
            module.types_global_values.push(Instruction::new(
                Op::TypeInt,
                None,
                Some(id),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ));
            id
        });

    let u8_ty = module
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypeInt
                && inst.operands[0].unwrap_literal_bit32() == 8
                && inst.operands[1].unwrap_literal_bit32() == 0
        })
        .and_then(|inst| inst.result_id)
        .unwrap_or_else(|| {
            let id = next_id();
            module.types_global_values.push(Instruction::new(
                Op::TypeInt,
                None,
                Some(id),
                vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
            ));
            id
        });

    let has_int8 = module.capabilities.iter().any(|inst| {
        inst.class.opcode == Op::Capability
            && inst.operands[0].unwrap_capability() == Capability::Int8
    });
    if !has_int8 {
        module.capabilities.push(Instruction::new(
            Op::Capability,
            None,
            None,
            vec![Operand::Capability(Capability::Int8)],
        ));
    }

    // --- OpenCL.std ext inst import ---

    let opencl_ext_id = module
        .ext_inst_imports
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::ExtInstImport
                && inst.operands[0].unwrap_literal_string() == "OpenCL.std"
        })
        .and_then(|inst| inst.result_id)
        .unwrap_or_else(|| {
            let id = next_id();
            module.ext_inst_imports.push(Instruction::new(
                Op::ExtInstImport,
                None,
                Some(id),
                vec![Operand::LiteralString("OpenCL.std".to_string())],
            ));
            id
        });

    // --- Collect Kernel function IDs ---

    let kernel_func_ids: Vec<Word> = module
        .entry_points
        .iter()
        .filter(|ep| ep.operands[0].unwrap_execution_model() == ExecutionModel::Kernel)
        .map(|ep| ep.operands[1].unwrap_id_ref())
        .collect();

    // --- Pre-collect format strings from DebugPrintf in Kernel functions ---

    let mut string_ids: HashSet<Word> = HashSet::new();
    for func in &module.functions {
        let func_id = func.def.as_ref().and_then(|d| d.result_id).unwrap_or(0);
        if !kernel_func_ids.contains(&func_id) {
            continue;
        }
        for block in &func.blocks {
            for inst in &block.instructions {
                if inst.class.opcode == Op::ExtInst
                    && inst.operands.len() > 2
                    && inst.operands[0].unwrap_id_ref() == debug_printf_ext_id
                {
                    string_ids.insert(inst.operands[2].unwrap_id_ref());
                }
            }
        }
    }

    // --- Create byte-array variables in UniformConstant for each format string ---

    let mut string_to_var: HashMap<Word, Word> = HashMap::new();
    let mut byte_const_cache: HashMap<u8, Word> = HashMap::new();

    for &string_id in &string_ids {
        let string_content = module
            .debug_string_source
            .iter()
            .find(|inst| inst.class.opcode == Op::String && inst.result_id == Some(string_id))
            .map(|inst| inst.operands[0].unwrap_literal_string().to_string())
            .unwrap();

        let bytes: Vec<u8> = string_content.bytes().chain(std::iter::once(0)).collect();
        let len = bytes.len();

        for &b in &bytes {
            byte_const_cache.entry(b).or_insert_with(|| {
                let id = next_id();
                module.types_global_values.push(Instruction::new(
                    Op::Constant,
                    Some(u8_ty),
                    Some(id),
                    vec![Operand::LiteralBit32(u32::from(b))],
                ));
                id
            });
        }

        let len_const_id = next_id();
        module.types_global_values.push(Instruction::new(
            Op::Constant,
            Some(u32_ty),
            Some(len_const_id),
            vec![Operand::LiteralBit32(len as u32)],
        ));

        let arr_ty = next_id();
        module.types_global_values.push(Instruction::new(
            Op::TypeArray,
            None,
            Some(arr_ty),
            vec![Operand::IdRef(u8_ty), Operand::IdRef(len_const_id)],
        ));

        let composite_id = next_id();
        let byte_operands: Vec<Operand> = bytes
            .iter()
            .map(|&b| Operand::IdRef(byte_const_cache[&b]))
            .collect();
        module.types_global_values.push(Instruction::new(
            Op::ConstantComposite,
            Some(arr_ty),
            Some(composite_id),
            byte_operands,
        ));

        let ptr_uc_arr = next_id();
        module.types_global_values.push(Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_uc_arr),
            vec![
                Operand::StorageClass(StorageClass::UniformConstant),
                Operand::IdRef(arr_ty),
            ],
        ));

        let var_id = next_id();
        module.types_global_values.push(Instruction::new(
            Op::Variable,
            Some(ptr_uc_arr),
            Some(var_id),
            vec![
                Operand::StorageClass(StorageClass::UniformConstant),
                Operand::IdRef(composite_id),
            ],
        ));

        string_to_var.insert(string_id, var_id);
    }

    // --- Convert DebugPrintf → OpenCL.std printf in Kernel functions ---

    for func in &mut module.functions {
        let func_id = func.def.as_ref().and_then(|d| d.result_id).unwrap_or(0);
        if !kernel_func_ids.contains(&func_id) {
            continue;
        }

        for block in &mut func.blocks {
            for inst in &mut block.instructions {
                if inst.class.opcode == Op::ExtInst
                    && !inst.operands.is_empty()
                    && inst.operands[0].unwrap_id_ref() == debug_printf_ext_id
                {
                    inst.operands[0] = Operand::IdRef(opencl_ext_id);
                    inst.operands[1] = Operand::LiteralExtInstInteger(184);
                    let string_id = inst.operands[2].unwrap_id_ref();
                    inst.operands[2] = Operand::IdRef(string_to_var[&string_id]);
                    inst.result_type = Some(u32_ty);
                    if inst.result_id.is_none() {
                        inst.result_id = Some(next_id());
                    }
                }
            }
        }
    }

    // --- Cleanup ---

    let consumed_string_ids: HashSet<Word> = string_to_var.keys().copied().collect();
    module.debug_string_source.retain(|inst| {
        inst.class.opcode != Op::String
            || !inst
                .result_id
                .is_some_and(|id| consumed_string_ids.contains(&id))
    });

    let still_referenced = module.functions.iter().any(|func| {
        func.blocks.iter().any(|block| {
            block.instructions.iter().any(|inst| {
                inst.class.opcode == Op::ExtInst
                    && !inst.operands.is_empty()
                    && inst.operands[0].unwrap_id_ref() == debug_printf_ext_id
            })
        })
    });
    if !still_referenced {
        module
            .ext_inst_imports
            .retain(|inst| inst.result_id != Some(debug_printf_ext_id));

        let has_non_semantic = module.ext_inst_imports.iter().any(|inst| {
            inst.class.opcode == Op::ExtInstImport
                && inst.operands[0]
                    .unwrap_literal_string()
                    .starts_with("NonSemantic.")
        });
        if !has_non_semantic {
            module.extensions.retain(|inst| {
                inst.class.opcode != Op::Extension
                    || inst.operands[0].unwrap_literal_string() != "SPV_KHR_non_semantic_info"
            });
        }
    }
}

fn post_link_single_module(
    sess: &Session,
    cg_args: &CodegenArgs,
    mut module: Module,
    out_filename: &Path,
    dump_prefix: Option<&OsStr>,
) {
    cg_args.do_disassemble(&module);
    kernel_debug_printf_to_opencl(&mut module);
    let spv_binary = module.assemble();

    if let Some(dir) = &cg_args.dump_post_link {
        // FIXME(eddyb) rename `filename` with `file_path` to make this less confusing.
        let out_filename_file_name = out_filename.file_name().unwrap();
        let dump_path = match dump_prefix {
            Some(prefix) => dir.join(prefix).with_extension(out_filename_file_name),
            None => dir.join(out_filename_file_name),
        };
        std::fs::write(dump_path, spirv_tools::binary::from_binary(&spv_binary)).unwrap();
    }

    let val_options = spirv_tools::val::ValidatorOptions {
        relax_struct_store: cg_args.relax_struct_store,
        relax_logical_pointer: cg_args.relax_logical_pointer,
        before_legalization: false,
        relax_block_layout: cg_args.relax_block_layout,
        uniform_buffer_standard_layout: cg_args.uniform_buffer_standard_layout,
        scalar_block_layout: cg_args.scalar_block_layout,
        skip_block_layout: cg_args.skip_block_layout,
        max_limits: vec![],
    };
    let opt_options = spirv_tools::opt::Options {
        validator_options: Some(val_options.clone()),
        max_id_bound: None,
        preserve_bindings: cg_args.linker_opts.preserve_bindings,
        preserve_spec_constants: false,
    };

    let spv_binary = if sess.opts.optimize != OptLevel::No
        || (sess.opts.debuginfo == DebugInfo::None && cg_args.spirv_metadata == SpirvMetadata::None)
    {
        if cg_args.run_spirv_opt {
            let _timer = sess.timer("link_spirv_opt");
            do_spirv_opt(sess, cg_args, spv_binary, out_filename, opt_options)
        } else {
            let reason = match (sess.opts.optimize, sess.opts.debuginfo == DebugInfo::None) {
                (OptLevel::No, true) => "debuginfo=None".to_string(),
                (optlevel, false) => format!("optlevel={optlevel:?}"),
                (optlevel, true) => format!("optlevel={optlevel:?}, debuginfo=None"),
            };
            sess.dcx().warn(format!(
                "`spirv-opt` should have ran ({reason}) but was disabled by `--no-spirv-opt`"
            ));
            spv_binary
        }
    } else {
        spv_binary
    };

    if cg_args.run_spirv_val {
        do_spirv_val(sess, &spv_binary, out_filename, val_options);
    }

    {
        let save_modules_timer = sess.timer("link_save_modules");
        if let Err(e) = std::fs::write(out_filename, spirv_tools::binary::from_binary(&spv_binary))
        {
            let mut err = sess
                .dcx()
                .struct_err("failed to serialize spirv-binary to disk");
            err.note(format!("module `{}`", out_filename.display()));
            err.note(format!("I/O error: {e:#}"));
            err.emit();
        }

        drop(save_modules_timer);
    }

    if let Ok(Some(transpile)) = should_transpile(sess) {
        transpile(sess, cg_args, &spv_binary, out_filename).ok();
    }
}

fn do_spirv_opt(
    sess: &Session,
    cg_args: &CodegenArgs,
    spv_binary: Vec<u32>,
    filename: &Path,
    options: spirv_tools::opt::Options,
) -> Vec<u32> {
    use spirv_tools::opt::{self, Optimizer};

    let target = SpirvTarget::parse_env(sess.target.options.env.desc()).unwrap();
    let target_env = target.to_spirv_tools();
    let mut optimizer = opt::create(Some(target_env));

    match sess.opts.optimize {
        OptLevel::No => {}
        OptLevel::Less | OptLevel::More | OptLevel::Aggressive => {
            optimizer.register_performance_passes();
        }
        OptLevel::Size | OptLevel::SizeMin => {
            optimizer.register_size_passes();
        }
    }

    if sess.opts.debuginfo == DebugInfo::None && cg_args.spirv_metadata == SpirvMetadata::None {
        optimizer
            .register_pass(opt::Passes::EliminateDeadConstant)
            .register_pass(opt::Passes::StripDebugInfo);
    }

    // NOTE(Kerilk) spirv-opt can crash (SIGSEGV) on some valid SPIR-V,
    // particularly with Kernel targets and certain dead branch patterns.
    // When using compiled tools (in-process FFI), a crash would kill the
    // entire compiler. Run performance/size passes in a forked child
    // process to isolate crashes, then apply safe passes (DCE, strip
    // debug) in-process on the result.
    #[cfg(unix)]
    if cfg!(feature = "use-compiled-tools") {
        return do_spirv_opt_forked(
            sess,
            cg_args,
            spv_binary,
            filename,
            options,
            Some(target_env),
            &optimizer,
        );
    }

    do_spirv_opt_inner(sess, &optimizer, spv_binary, filename, options)
}

fn do_spirv_opt_inner(
    sess: &Session,
    optimizer: &impl spirv_tools::opt::Optimizer,
    spv_binary: Vec<u32>,
    filename: &Path,
    options: spirv_tools::opt::Options,
) -> Vec<u32> {
    use spirv_tools::error;

    let result = optimizer.optimize(
        &spv_binary,
        &mut |msg: error::Message| {
            use error::MessageLevel as Level;

            let mut err = match msg.level {
                Level::Error | Level::Fatal | Level::InternalError => {
                    Diag::<()>::new(sess.dcx(), rustc_errors::Level::Error, msg.message)
                }
                Level::Warning => sess.dcx().struct_warn(msg.message),
                Level::Info | Level::Debug => sess.dcx().struct_note(msg.message),
            };

            err.note(format!("module `{}`", filename.display()));
            err.emit();
        },
        Some(options),
    );

    match result {
        Ok(spirv_tools::binary::Binary::OwnedU32(words)) => words,
        Ok(binary) => binary.as_words().to_vec(),
        Err(e) => {
            let mut err = sess.dcx().struct_warn(e.to_string());
            err.note("spirv-opt failed, leaving as unoptimized");
            err.note(format!("module `{}`", filename.display()));
            err.emit();
            spv_binary
        }
    }
}

/// Run spirv-opt with crash isolation for compiled tools.
///
/// Performance/size passes can crash (SIGSEGV) in spirv-tools on some
/// valid SPIR-V (e.g., Kernel targets with certain dead branch patterns).
/// The full optimizer runs in a forked child process. If it crashes, we
/// fall back to safe cleanup passes (DCE, strip debug) in-process —
/// these don't crash and produce a small enough binary for consumers.
#[cfg(unix)]
fn do_spirv_opt_forked(
    sess: &Session,
    cg_args: &CodegenArgs,
    spv_binary: Vec<u32>,
    filename: &Path,
    options: spirv_tools::opt::Options,
    target_env: Option<spirv_tools::TargetEnv>,
    optimizer: &impl spirv_tools::opt::Optimizer,
) -> Vec<u32> {
    use spirv_tools::opt::{self, Optimizer};

    let tmp_out = filename.with_extension("spirv-opt-out.spv");

    // Fork a child to run the full optimizer.
    let pid = unsafe { libc::fork() };
    if pid == -1 {
        sess.dcx()
            .warn("spirv-opt: fork() failed, running in-process");
        return do_spirv_opt_inner(sess, optimizer, spv_binary, filename, options);
    }

    if pid == 0 {
        // Child: run the full optimizer, write result, exit.
        let result = optimizer.optimize(
            &spv_binary,
            &mut |_msg: spirv_tools::error::Message| {},
            Some(options),
        );
        match result {
            Ok(binary) => {
                let _ = std::fs::write(
                    &tmp_out,
                    spirv_tools::binary::from_binary(binary.as_words()),
                );
                unsafe { libc::_exit(0) };
            }
            Err(_) => {
                unsafe { libc::_exit(1) };
            }
        }
    }

    // Parent: wait for the child.
    let mut status: libc::c_int = 0;
    unsafe { libc::waitpid(pid, &mut status, 0) };

    if libc::WIFEXITED(status)
        && libc::WEXITSTATUS(status) == 0
        && let Ok(bytes) = std::fs::read(&tmp_out)
    {
        let _ = std::fs::remove_file(&tmp_out);
        if let Ok(words) = spirv_tools::binary::to_binary(bytes.as_slice()) {
            return words.to_vec();
        }
    }

    // Child crashed or failed — run safe cleanup passes in-process.
    let _ = std::fs::remove_file(&tmp_out);
    sess.dcx()
        .warn("spirv-opt performance passes crashed, falling back to cleanup-only");

    let mut safe_opt = opt::create(target_env);
    // Register the key optimization passes individually, skipping
    // DeadBranchElim which crashes on some Kernel SPIR-V patterns.
    safe_opt
        .register_pass(opt::Passes::InlineExhaustive)
        .register_pass(opt::Passes::ConditionalConstantPropagation)
        .register_pass(opt::Passes::AggressiveDCE)
        .register_pass(opt::Passes::EliminateDeadFunctions)
        .register_pass(opt::Passes::EliminateDeadMembers)
        .register_pass(opt::Passes::EliminateDeadConstant)
        .register_pass(opt::Passes::DeadVariableElimination)
        .register_pass(opt::Passes::CFGCleanup)
        .register_pass(opt::Passes::BlockMerge);
    if sess.opts.debuginfo == DebugInfo::None && cg_args.spirv_metadata == SpirvMetadata::None {
        safe_opt.register_pass(opt::Passes::StripDebugInfo);
    }

    do_spirv_opt_inner(sess, &safe_opt, spv_binary, filename, Default::default())
}

fn do_spirv_val(
    sess: &Session,
    spv_binary: &[u32],
    filename: &Path,
    options: spirv_tools::val::ValidatorOptions,
) {
    use spirv_tools::val::{self, Validator};

    let target = SpirvTarget::parse_env(sess.target.options.env.desc()).unwrap();
    let validator = val::create(Some(target.to_spirv_tools()));

    if let Err(e) = validator.validate(spv_binary, Some(options)) {
        let mut err = sess.dcx().struct_err(e.to_string());
        err.note("spirv-val failed");
        err.note(format!("module `{}`", filename.display()));
        err.emit();
    }
}

fn link_local_crate_native_libs_and_dependent_crate_libs(
    rlibs: &mut Vec<PathBuf>,
    sess: &Session,
    crate_type: CrateType,
    crate_info: &CrateInfo,
) {
    if sess.opts.unstable_opts.link_native_libraries {
        add_local_native_libraries(sess, crate_info);
    }
    add_upstream_rust_crates(sess, rlibs, crate_info, crate_type);
    if sess.opts.unstable_opts.link_native_libraries {
        add_upstream_native_libraries(sess, crate_info, crate_type);
    }
}

fn add_local_native_libraries(sess: &Session, crate_info: &CrateInfo) {
    let relevant_libs = crate_info
        .used_libraries
        .iter()
        .filter(|l| relevant_lib(sess, l));
    assert_eq!(relevant_libs.count(), 0);
}

fn add_upstream_rust_crates(
    sess: &Session,
    rlibs: &mut Vec<PathBuf>,
    crate_info: &CrateInfo,
    crate_type: CrateType,
) {
    let data = crate_info
        .dependency_formats
        .get(&crate_type)
        .expect("failed to find crate type in dependency format list");
    for &cnum in &crate_info.used_crates {
        let src = &crate_info.used_crate_source[&cnum];
        match data[cnum] {
            Linkage::NotLinked | Linkage::IncludedFromDylib => {}
            Linkage::Static => rlibs.push(src.rlib.as_ref().unwrap().clone()),
            //Linkage::Dynamic => rlibs.push(src.dylib.as_ref().unwrap().0.clone()),
            Linkage::Dynamic => {
                sess.dcx().err("TODO: Linkage::Dynamic not supported yet");
            }
        }
    }
}

fn add_upstream_native_libraries(sess: &Session, crate_info: &CrateInfo, crate_type: CrateType) {
    let data = crate_info
        .dependency_formats
        .get(&crate_type)
        .expect("failed to find crate type in dependency format list");

    for &cnum in &crate_info.used_crates {
        for lib in crate_info.native_libraries[&cnum].iter() {
            if !relevant_lib(sess, lib) {
                continue;
            }
            match lib.kind {
                NativeLibKind::Static {
                    bundle: Some(false),
                    ..
                } if data[cnum] != Linkage::Static => {}

                NativeLibKind::Static {
                    bundle: None | Some(true),
                    ..
                } => {}

                _ => sess.dcx().fatal(format!(
                    "`NativeLibKind::{:?}` (name={:?}) not supported yet",
                    lib.kind, lib.name
                )),
            }
        }
    }
}

// FIXME(eddyb) upstream has code like this already, maybe we can reuse most of it?
// (see `compiler/rustc_codegen_ssa/src/back/link.rs`)
fn relevant_lib(sess: &Session, lib: &NativeLib) -> bool {
    match lib.cfg {
        Some(ref cfg) => eval_config_entry(sess, cfg).as_bool(),
        None => true,
    }
}

fn create_archive(files: &[&Path], metadata: &[u8], out_filename: &Path) {
    let files_with_names = files.iter().map(|file| {
        (
            file,
            file.file_name()
                .unwrap()
                .to_str()
                .expect("archive file names should be valid ASCII/UTF-8"),
        )
    });
    let out_file = File::create(out_filename).unwrap();
    let mut builder = GnuBuilder::new(
        out_file,
        iter::once(METADATA_FILENAME)
            .chain(files_with_names.clone().map(|(_, name)| name))
            .map(|name| name.as_bytes().to_vec())
            .collect(),
    );
    builder
        .append(
            &Header::new(METADATA_FILENAME.as_bytes().to_vec(), metadata.len() as u64),
            metadata,
        )
        .unwrap();

    let mut filenames = FxHashSet::default();
    filenames.insert(METADATA_FILENAME);
    for (file, name) in files_with_names {
        assert!(
            filenames.insert(name),
            "Duplicate filename in archive: {:?}",
            file.file_name().unwrap()
        );

        // NOTE(eddyb) we can't use `append_path` or `append_file`, as they
        // record too much metadata by default (mtime/UID/GID, at least),
        // which is determintal to reproducible build artifacts, but also
        // can misbehave in environments with high UIDs/GIDs (see #889).
        let file = File::open(file).unwrap();
        let header = Header::new(name.as_bytes().to_vec(), file.metadata().unwrap().len());
        // NOTE(eddyb) either `fs::File`, or the result of `fs::read`, could fit
        // here, but `fs::File` has specialized file->file copying on some OSes.
        builder.append(&header, file).unwrap();
    }
    builder.into_inner().unwrap();
}

// HACK(eddyb) hiding the actual implementation to avoid `rspirv::dr::Loader`
// being hardcoded (as future work may need to customize it for various reasons).
pub fn with_rspirv_loader<E>(
    f: impl FnOnce(&mut dyn rspirv::binary::Consumer) -> Result<(), E>,
) -> Result<rspirv::dr::Module, E> {
    let mut loader = rspirv::dr::Loader::new();
    f(&mut loader)?;
    Ok(loader.module())
}

/// This is the actual guts of linking: the rest of the link-related functions are just digging through rustc's
/// shenanigans to collect all the object files we need to link.
fn do_link(
    sess: &Session,
    cg_args: &CodegenArgs,
    objects: &[PathBuf],
    rlibs: &[PathBuf],
    outputs: &OutputFilenames,
    disambiguated_crate_name_for_dumps: &OsStr,
) -> linker::LinkResult {
    let load_modules_timer = sess.timer("link_load_modules");

    let mut modules = Vec::new();
    let mut add_module = |file_name: &OsStr, bytes: &[u8]| {
        let module =
            with_rspirv_loader(|loader| rspirv::binary::parse_bytes(bytes, loader)).unwrap();
        if let Some(dir) = &cg_args.dump_pre_link {
            // FIXME(eddyb) is it a good idea to re-`assemble` the `rspirv::dr`
            // module, or should this just save the original bytes?
            std::fs::write(
                dir.join(file_name).with_extension("spv"),
                spirv_tools::binary::from_binary(&module.assemble()),
            )
            .unwrap();
        }
        modules.push(module);
    };

    // `objects` are the plain obj files we need to link - usually produced by the final crate.
    for obj in objects {
        add_module(obj.file_name().unwrap(), &std::fs::read(obj).unwrap());
    }

    // `rlibs` are archive files we've created in `create_archive`, usually produced by crates that are being
    // referenced. We need to unpack them and add the modules inside.
    for rlib in rlibs {
        let mut archive = Archive::new(File::open(rlib).unwrap());
        while let Some(entry) = archive.next_entry() {
            let mut entry = entry.unwrap();
            if entry.header().identifier() != METADATA_FILENAME.as_bytes() {
                // std::fs::read adds 1 to the size, so do the same here - see comment:
                // https://github.com/rust-lang/rust/blob/72868e017bdade60603a25889e253f556305f996/library/std/src/fs.rs#L200-L202
                let mut bytes = Vec::with_capacity(entry.header().size() as usize + 1);
                entry.read_to_end(&mut bytes).unwrap();

                let file_name = std::str::from_utf8(entry.header().identifier()).unwrap();
                add_module(OsStr::new(file_name), &bytes);
            }
        }
    }

    drop(load_modules_timer);

    // Do the link...
    let link_result = linker::link(
        sess,
        modules,
        &cg_args.linker_opts,
        outputs,
        disambiguated_crate_name_for_dumps,
    );

    if let Ok(v) = link_result {
        v
    } else {
        sess.dcx().abort_if_errors();
        bug!("Linker errored, but no error reported");
    }
}
