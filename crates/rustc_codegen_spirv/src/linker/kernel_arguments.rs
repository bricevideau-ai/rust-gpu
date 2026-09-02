//! Convert `Kernel` entry stubs from `void(void)` + global `OpVariable` pattern
//! to proper function parameters, as required by the `OpenCL` SPIR-V environment.
//!
//! After inlining and specialization, `Kernel` entry points look like:
//!
//! ```text
//! %entry = OpFunction %void None %fn_void_void
//! %label = OpLabel
//! %val   = OpLoad %T %global_var
//! ...
//! OpReturn
//! OpFunctionEnd
//! ```
//!
//! `OpenCL` consumers (e.g., pocl) expect:
//!
//! ```text
//! %entry = OpFunction %void None %fn_type_with_params
//! %param = OpFunctionParameter %T
//! %label = OpLabel
//! ... (uses %param instead of OpLoad %global_var)
//! OpReturn
//! OpFunctionEnd
//! ```

use crate::custom_decorations::{CustomDecoration, KernelParamPositionDecoration};
use rspirv::dr::{Function, Instruction, Module, Operand};
use rspirv::spirv::{AddressingModel, BuiltIn, Decoration, ExecutionModel, Op, StorageClass};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};

/// Convert `Kernel` entry points from global `OpVariable` to `OpFunctionParameter`,
/// and fix `BuiltIn` variable types for `Physical64` addressing.
pub fn convert_kernel_arguments(module: &mut Module) {
    fix_builtin_types(module);

    let entries = collect_kernel_entries(module);
    if entries.is_empty() {
        // Strip any leftover position decorations even if no Kernel entries
        // exist (defensive — currently codegen only emits these for Kernel).
        KernelParamPositionDecoration::remove_all(module);
        return;
    }

    let index = ModuleIndex::build(module);
    let plans: Vec<Plan> = entries
        .iter()
        .filter_map(|e| Plan::build(module, &index, e))
        .collect();

    let mut fn_ptr_type_cache: FxHashMap<u32, u32> = FxHashMap::default();
    // Each plan gets its own callee-rewrite map. Previously all plans
    // wrote into a single shared map and we panicked on conflict; now we
    // resolve conflicts after all plans by cloning the conflicting
    // callees, so each plan's requirements survive intact.
    let mut per_plan_rewrites: Vec<CalleeRewriteMap> = std::iter::repeat_with(FxHashMap::default)
        .take(plans.len())
        .collect();
    for (plan, rewrites) in plans.iter().zip(per_plan_rewrites.iter_mut()) {
        apply_plan(module, plan, &mut fn_ptr_type_cache, rewrites);
    }
    resolve_and_apply_callee_rewrites(module, &index, &plans, per_plan_rewrites);

    // Position decorations served their purpose; strip them so they don't
    // survive into the final SPIR-V (they piggyback on `UserTypeGOOGLE`,
    // which OpenCL consumers wouldn't understand).
    KernelParamPositionDecoration::remove_all(module);
}

/// Per-callee, per-arg-index, the new `*Function Image` pointer type that the
/// arg should accept (collected once per plan).
type CalleeRewriteMap = FxHashMap<u32, FxHashMap<usize, u32>>;

/// One `OpEntryPoint` instruction's relevant fields.
struct KernelEntry {
    func_id: u32,
    interface_ids: Vec<u32>,
}

fn collect_kernel_entries(module: &Module) -> Vec<KernelEntry> {
    module
        .entry_points
        .iter()
        .filter(|ep| ep.operands[0].unwrap_execution_model() == ExecutionModel::Kernel)
        .map(|ep| KernelEntry {
            func_id: ep.operands[1].unwrap_id_ref(),
            interface_ids: ep.operands[3..]
                .iter()
                .map(|op| op.unwrap_id_ref())
                .collect(),
        })
        .collect()
}

/// Module-wide lookup tables built once and shared across all plans.
struct ModuleIndex {
    /// Function ID → index into `module.functions`. Lets us use
    /// `&mut module.functions[idx]` instead of repeated `.iter().find()`.
    func_idx: FxHashMap<u32, usize>,
    /// Per-`OpVariable` info.
    var_info: FxHashMap<u32, VarInfo>,
    /// Per-`OpVariable` source-parameter sort key keyed by entry function.
    /// Map: `(entry_func_id, var_id)` → position. Codegen emits one
    /// `KernelParamPositionDecoration` per kernel-arg `OpVariable`; the
    /// linker reads these to recover Rust source parameter order. Without
    /// this, codegen's emission order in `types_global_values` is unreliable
    /// (multi-kernel modules + slice decomposition + global dedup-by-type
    /// can interleave variables arbitrarily).
    kernel_param_position: FxHashMap<(u32, u32), u32>,
    /// Pointer-type → pointee-type.
    ptr_to_pointee: FxHashMap<u32, u32>,
    /// IDs of all `OpTypeImage` types.
    image_type_ids: FxHashSet<u32>,
    /// Variables decorated with `BuiltIn` (must stay as global `Input` vars).
    builtin_ids: FxHashSet<u32>,
}

struct VarInfo {
    ptr_type: u32,
    storage_class: StorageClass,
    /// Variables with an initializer are program-scope statics, not kernel
    /// arguments — leave them alone.
    has_initializer: bool,
}

impl ModuleIndex {
    fn build(module: &Module) -> Self {
        let func_idx = module
            .functions
            .iter()
            .enumerate()
            .filter_map(|(i, f)| f.def_id().map(|id| (id, i)))
            .collect();

        let mut var_info = FxHashMap::default();
        let mut ptr_to_pointee = FxHashMap::default();
        let mut image_type_ids = FxHashSet::default();

        for inst in &module.types_global_values {
            match inst.class.opcode {
                Op::Variable => {
                    let id = inst.result_id.unwrap();
                    var_info.insert(
                        id,
                        VarInfo {
                            ptr_type: inst.result_type.unwrap(),
                            storage_class: inst.operands[0].unwrap_storage_class(),
                            has_initializer: inst.operands.len() > 1,
                        },
                    );
                }
                Op::TypePointer => {
                    ptr_to_pointee
                        .insert(inst.result_id.unwrap(), inst.operands[1].unwrap_id_ref());
                }
                Op::TypeImage => {
                    image_type_ids.insert(inst.result_id.unwrap());
                }
                _ => {}
            }
        }

        let builtin_ids = module
            .annotations
            .iter()
            .filter(|inst| {
                inst.class.opcode == Op::Decorate
                    && inst.operands[1].unwrap_decoration() == Decoration::BuiltIn
            })
            .map(|inst| inst.operands[0].unwrap_id_ref())
            .collect();

        // Read codegen-emitted KernelParamPositionDecoration into a map.
        let kernel_param_position: FxHashMap<(u32, u32), u32> =
            KernelParamPositionDecoration::decode_all(module)
                .map(|(var_id, decoded)| {
                    let d = decoded.decode();
                    ((d.entry_func, var_id), d.position)
                })
                .collect();

        Self {
            func_idx,
            var_info,
            kernel_param_position,
            ptr_to_pointee,
            image_type_ids,
            builtin_ids,
        }
    }
}

/// What a kernel argument needs done to its references in the body.
#[derive(Clone, Copy)]
enum ParamKind {
    /// `*CrossWorkgroup T` — passed as-is; `OpLoad` through it stays an
    /// `OpLoad` (operand[0] swapped to the new param ID).
    CrossWorkgroupPtr,
    /// `OpTypeImage` value — needs a Function-storage local `OpVariable` so
    /// pointer-based references in the body keep working.
    ImageByValue { pointee_type: u32 },
    /// Scalar value (e.g. `u64` slice length) — `OpLoad` of the original
    /// global becomes `OpCopyObject %param`.
    ScalarByValue,
}

struct PlanParam {
    /// Original module-scope `OpVariable` ID (to be removed at cleanup).
    var_id: u32,
    /// New `OpFunctionParameter` ID.
    param_id: u32,
    /// `OpTypeFunction` operand for this param (pointer for `CrossWorkgroup`,
    /// value otherwise).
    param_type: u32,
    kind: ParamKind,
}

struct Plan {
    func_id: u32,
    func_idx: usize,
    /// `BuiltIn` variables (e.g. `GlobalInvocationId`) — stay as `Input` vars
    /// in the entry-point interface.
    builtin_interface: Vec<u32>,
    params: Vec<PlanParam>,
    new_fn_type_id: u32,
}

impl Plan {
    fn build(module: &mut Module, index: &ModuleIndex, entry: &KernelEntry) -> Option<Self> {
        let func_idx = *index.func_idx.get(&entry.func_id)?;

        // Split the interface list: builtins stay, everything else becomes a
        // parameter candidate.
        let mut builtin_interface = Vec::new();
        let mut interface_param_ids = Vec::new();
        for &id in &entry.interface_ids {
            if index.builtin_ids.contains(&id) {
                builtin_interface.push(id);
            } else {
                interface_param_ids.push(id);
            }
        }

        // SPIR-V <= 1.3 only lists Input/Output in the interface, so scan the
        // body for CrossWorkgroup and UniformConstant globals (kernel args,
        // not statics).
        let mut body_var_ids: Vec<u32> = Vec::new();
        for block in &module.functions[func_idx].blocks {
            for inst in &block.instructions {
                for op in inst.operands.iter().filter_map(|o| o.id_ref_any()) {
                    let Some(vi) = index.var_info.get(&op) else {
                        continue;
                    };
                    let is_kernel_arg = matches!(
                        vi.storage_class,
                        StorageClass::CrossWorkgroup | StorageClass::UniformConstant
                    ) && !vi.has_initializer;
                    if is_kernel_arg
                        && !interface_param_ids.contains(&op)
                        && !body_var_ids.contains(&op)
                    {
                        body_var_ids.push(op);
                    }
                }
            }
        }

        // Sort by the codegen-emitted source-parameter position to recover
        // Rust source order. Variables without a position decoration sort
        // last (defensive — currently codegen tags all kernel-arg globals).
        let mut all_var_ids: Vec<u32> = interface_param_ids
            .into_iter()
            .chain(body_var_ids)
            .collect();
        all_var_ids.sort_by_key(|id| {
            index
                .kernel_param_position
                .get(&(entry.func_id, *id))
                .copied()
                .unwrap_or(u32::MAX)
        });

        if all_var_ids.is_empty() {
            return None;
        }

        let params: Vec<PlanParam> = all_var_ids
            .into_iter()
            .map(|var_id| {
                let vi = &index.var_info[&var_id];
                let (param_type, kind) = if vi.storage_class == StorageClass::CrossWorkgroup {
                    (vi.ptr_type, ParamKind::CrossWorkgroupPtr)
                } else {
                    let pointee = index.ptr_to_pointee[&vi.ptr_type];
                    let kind = if index.image_type_ids.contains(&pointee) {
                        ParamKind::ImageByValue {
                            pointee_type: pointee,
                        }
                    } else {
                        ParamKind::ScalarByValue
                    };
                    (pointee, kind)
                };
                PlanParam {
                    var_id,
                    param_id: next_id(&mut module.header),
                    param_type,
                    kind,
                }
            })
            .collect();

        Some(Plan {
            func_id: entry.func_id,
            func_idx,
            builtin_interface,
            params,
            new_fn_type_id: next_id(&mut module.header),
        })
    }
}

/// One Function-storage local `OpVariable` backing an `ImageByValue` parameter.
struct ImageLocal {
    /// The Function-storage `OpVariable` ID.
    local_id: u32,
    /// `*Function Image` pointer type — needed when the local is passed to
    /// a callee, whose parameter type must change accordingly.
    fn_ptr_type: u32,
}

fn apply_plan(
    module: &mut Module,
    plan: &Plan,
    fn_ptr_type_cache: &mut FxHashMap<u32, u32>,
    plan_callee_rewrites: &mut CalleeRewriteMap,
) {
    emit_new_function_type(module, plan);
    swap_function_signature(module, plan);

    let image_locals = setup_image_locals(module, plan, fn_ptr_type_cache);
    rewrite_function_body(&mut module.functions[plan.func_idx], plan, &image_locals);
    fix_image_load_qualifier(module, plan, &image_locals);
    collect_called_fn_rewrites(module, plan, &image_locals, plan_callee_rewrites);

    cleanup_globals(module, plan);
    update_entry_point_interface(module, plan);
}

/// Codegen can emit `OpBitcast %ptr_function_image_default_qualifier %src`
/// where `%src` is a Function-local image variable created by
/// `setup_image_locals` with the per-parameter access qualifier. The bitcast
/// converts to `abi.rs`'s default-qualifier image pointer type, and the
/// downstream `OpLoad` then produces a value with the *wrong* qualifier —
/// which `OpenCL` consumers (e.g. pocl) follow when mangling `write_image*`
/// and then fail to link.
///
/// Walk the body and for each `OpLoad %X %src` where `%src` is the result of
/// such a bitcast off one of our locals, redirect the load to read from the
/// local directly and update its result type to the local's image type. The
/// bitcast then becomes dead (unused) and harmless.
fn fix_image_load_qualifier(
    module: &mut Module,
    plan: &Plan,
    image_locals: &FxHashMap<u32, ImageLocal>,
) {
    if image_locals.is_empty() {
        return;
    }
    // local_id → underlying image type (the pointee of fn_ptr_type)
    let local_to_image: FxHashMap<u32, u32> = image_locals
        .values()
        .filter_map(|l| {
            let pointee = module.types_global_values.iter().find_map(|inst| {
                if inst.class.opcode == Op::TypePointer && inst.result_id == Some(l.fn_ptr_type) {
                    Some(inst.operands[1].unwrap_id_ref())
                } else {
                    None
                }
            })?;
            Some((l.local_id, pointee))
        })
        .collect();

    let func = &mut module.functions[plan.func_idx];
    // Pass 1: collect bitcasts whose source is one of our locals.
    // Map: bitcast_result_id → (source_local_id, source_image_type).
    let mut bitcast_redirect: FxHashMap<u32, (u32, u32)> = FxHashMap::default();
    for block in &func.blocks {
        for inst in &block.instructions {
            if inst.class.opcode == Op::Bitcast
                && let Some(result_id) = inst.result_id
                && let Some(src_id) = inst.operands.first().and_then(|o| o.id_ref_any())
                && let Some(&src_image) = local_to_image.get(&src_id)
            {
                bitcast_redirect.insert(result_id, (src_id, src_image));
            }
        }
    }
    if bitcast_redirect.is_empty() {
        return;
    }
    // Pass 2: rewrite OpLoad through the bitcast to load directly from the
    // local with the local's image type.
    for block in &mut func.blocks {
        for inst in &mut block.instructions {
            if inst.class.opcode == Op::Load
                && let Some(src_id) = inst.operands.first().and_then(|o| o.id_ref_any())
                && let Some(&(local_id, src_image)) = bitcast_redirect.get(&src_id)
            {
                inst.operands[0] = Operand::IdRef(local_id);
                inst.result_type = Some(src_image);
            }
        }
    }
}

/// Emit a new `OpTypeFunction` matching the plan's parameter types.
fn emit_new_function_type(module: &mut Module, plan: &Plan) {
    let void_type = module.functions[plan.func_idx]
        .def
        .as_ref()
        .unwrap()
        .result_type
        .unwrap();
    module.types_global_values.push(Instruction::new(
        Op::TypeFunction,
        None,
        Some(plan.new_fn_type_id),
        std::iter::once(Operand::IdRef(void_type))
            .chain(plan.params.iter().map(|p| Operand::IdRef(p.param_type)))
            .collect(),
    ));
}

/// Replace the function's type in its def + populate parameters.
fn swap_function_signature(module: &mut Module, plan: &Plan) {
    let func = &mut module.functions[plan.func_idx];
    func.def.as_mut().unwrap().operands[1] = Operand::IdRef(plan.new_fn_type_id);
    func.parameters = plan
        .params
        .iter()
        .map(|p| {
            Instruction::new(
                Op::FunctionParameter,
                Some(p.param_type),
                Some(p.param_id),
                vec![],
            )
        })
        .collect();
}

/// For each `ImageByValue` parameter, emit a Function-local `OpVariable` at
/// the start of the entry block and `OpStore` the parameter into it. Returns
/// a map from the original global var ID to the local var info.
fn setup_image_locals(
    module: &mut Module,
    plan: &Plan,
    fn_ptr_type_cache: &mut FxHashMap<u32, u32>,
) -> FxHashMap<u32, ImageLocal> {
    let mut locals: FxHashMap<u32, ImageLocal> = FxHashMap::default();
    let mut new_var_insts: Vec<Instruction> = Vec::new();
    let mut new_store_insts: Vec<Instruction> = Vec::new();

    for p in &plan.params {
        let ParamKind::ImageByValue { pointee_type } = p.kind else {
            continue;
        };
        let fn_ptr_type = *fn_ptr_type_cache
            .entry(pointee_type)
            .or_insert_with(|| emit_function_ptr_type(module, pointee_type));
        let local_id = next_id(&mut module.header);
        locals.insert(
            p.var_id,
            ImageLocal {
                local_id,
                fn_ptr_type,
            },
        );
        new_var_insts.push(Instruction::new(
            Op::Variable,
            Some(fn_ptr_type),
            Some(local_id),
            vec![Operand::StorageClass(StorageClass::Function)],
        ));
        new_store_insts.push(Instruction::new(
            Op::Store,
            None,
            None,
            vec![Operand::IdRef(local_id), Operand::IdRef(p.param_id)],
        ));
    }

    if new_var_insts.is_empty() {
        return locals;
    }

    // SPIR-V requires all OpVariables in a function to live at the top of
    // the first block, before any other (non-debug) instructions.
    let block = &mut module.functions[plan.func_idx].blocks[0];
    for (i, inst) in new_var_insts.into_iter().enumerate() {
        block.instructions.insert(i, inst);
    }
    let store_pos = block
        .instructions
        .iter()
        .position(|inst| !matches!(inst.class.opcode, Op::Variable | Op::Line | Op::NoLine))
        .unwrap_or(block.instructions.len());
    for (i, inst) in new_store_insts.into_iter().enumerate() {
        block.instructions.insert(store_pos + i, inst);
    }
    locals
}

fn emit_function_ptr_type(module: &mut Module, pointee_type: u32) -> u32 {
    let id = next_id(&mut module.header);
    module.types_global_values.push(Instruction::new(
        Op::TypePointer,
        None,
        Some(id),
        vec![
            Operand::StorageClass(StorageClass::Function),
            Operand::IdRef(pointee_type),
        ],
    ));
    id
}

/// Rewrite all references to the original global `OpVariables` in the body:
/// - `CrossWorkgroup` pointer var → `OpFunctionParameter` (still a pointer).
/// - Image-by-value var → Function-local `OpVariable`.
/// - Scalar var: only valid as `OpLoad` source; converted to `OpCopyObject %param`.
fn rewrite_function_body(
    func: &mut Function,
    plan: &Plan,
    image_locals: &FxHashMap<u32, ImageLocal>,
) {
    // Build the rewrite map for the generic operand walk. Scalar params are
    // intentionally absent — they're handled by the OpLoad special case
    // below (their param_id is a value, not a pointer, so any non-Load
    // reference would be invalid IR anyway).
    let mut id_rewrite: FxHashMap<u32, u32> = FxHashMap::default();
    let mut scalar_var_to_param: FxHashMap<u32, u32> = FxHashMap::default();
    for p in &plan.params {
        match p.kind {
            ParamKind::CrossWorkgroupPtr => {
                id_rewrite.insert(p.var_id, p.param_id);
            }
            ParamKind::ImageByValue { .. } => {
                id_rewrite.insert(p.var_id, image_locals[&p.var_id].local_id);
            }
            ParamKind::ScalarByValue => {
                scalar_var_to_param.insert(p.var_id, p.param_id);
            }
        }
    }

    for block in &mut func.blocks {
        for inst in &mut block.instructions {
            if inst.class.opcode == Op::Load
                && let Some(src) = inst.operands[0].id_ref_any()
                && let Some(&param_id) = scalar_var_to_param.get(&src)
            {
                *inst = Instruction::new(
                    Op::CopyObject,
                    inst.result_type,
                    inst.result_id,
                    vec![Operand::IdRef(param_id)],
                );
                continue;
            }
            for op in &mut inst.operands {
                if let Operand::IdRef(id) = op
                    && let Some(&new_id) = id_rewrite.get(id)
                {
                    *op = Operand::IdRef(new_id);
                }
            }
        }
    }
}

/// When a Function-storage image local is passed to a callee, the callee's
/// parameter type changes from `*UniformConstant Image` to `*Function Image`.
/// Record the required rewrites for this plan; conflicts across plans are
/// resolved later by cloning the callee (see `resolve_and_apply_callee_rewrites`).
fn collect_called_fn_rewrites(
    module: &Module,
    plan: &Plan,
    image_locals: &FxHashMap<u32, ImageLocal>,
    out: &mut CalleeRewriteMap,
) {
    if image_locals.is_empty() {
        return;
    }
    let local_to_fn_ptr: FxHashMap<u32, u32> = image_locals
        .values()
        .map(|info| (info.local_id, info.fn_ptr_type))
        .collect();

    for block in &module.functions[plan.func_idx].blocks {
        for inst in &block.instructions {
            if inst.class.opcode != Op::FunctionCall {
                continue;
            }
            let called_fn_id = inst.operands[0].unwrap_id_ref();
            for (arg_idx, op) in inst.operands[1..].iter().enumerate() {
                if let Operand::IdRef(arg_id) = op
                    && let Some(&new_ptr) = local_to_fn_ptr.get(arg_id)
                {
                    out.entry(called_fn_id)
                        .or_default()
                        .insert(arg_idx, new_ptr);
                }
            }
        }
    }
}

/// Resolve callee-rewrite requirements across all plans, cloning callees
/// when two plans need the same helper with different signatures.
///
/// The first plan to claim a given callee uses the original function;
/// each subsequent distinct signature gets a fresh function clone with
/// renumbered IDs, and that plan's `OpFunctionCall %original` is rewritten
/// to `%clone`.
///
/// In the (currently common) case where every plan agrees on the callee
/// signature, no clones are made and the result is bit-identical to the
/// pre-cloning behaviour: a single new `OpTypeFunction` per callee, applied
/// in place.
///
/// # On the cloning path's reachability
///
/// The cloning branch is currently unreachable from rust-gpu codegen: both
/// `&Image` and `&mut Image` resolve to consistent SPIR-V access qualifiers
/// across all kernels in a module (`ImageReadWrite` is a per-module
/// capability, and Rust's reference mutability is fixed per call site). So
/// in practice every plan agrees on each callee's signature today.
///
/// The cloning logic exists to keep the linker robust against future
/// codegen changes that *could* produce divergent qualifiers — e.g. a
/// per-kernel `#[spirv(...)]` attribute that overrides the default, or
/// ABI changes that allow image references to coerce across qualifiers.
/// The previous implementation panicked on divergence; this one quietly
/// produces correct SPIR-V.
fn resolve_and_apply_callee_rewrites(
    module: &mut Module,
    index: &ModuleIndex,
    plans: &[Plan],
    per_plan: Vec<CalleeRewriteMap>,
) {
    // (callee_id, sorted-(arg_idx, new_ptr_type) signature) → effective callee
    type Sig = Vec<(usize, u32)>;
    let to_sig = |m: &FxHashMap<usize, u32>| -> Sig {
        let mut v: Sig = m.iter().map(|(&k, &v)| (k, v)).collect();
        v.sort_unstable();
        v
    };

    let mut sig_to_id: FxHashMap<(u32, Sig), u32> = FxHashMap::default();
    let mut callees_with_first_sig: FxHashSet<u32> = FxHashSet::default();
    // (plan_idx, original_callee_id) → effective callee_id (only when remapped).
    let mut plan_callee_remap: FxHashMap<(usize, u32), u32> = FxHashMap::default();
    // effective callee_id → signature to apply.
    let mut callees_to_apply: FxHashMap<u32, FxHashMap<usize, u32>> = FxHashMap::default();
    // Pairs of (original_callee_id, new_clone_id) — clones to materialise.
    let mut clones_needed: Vec<(u32, u32)> = Vec::new();

    for (plan_i, per_callee) in per_plan.iter().enumerate() {
        for (&callee_id, sig_map) in per_callee {
            let sig = to_sig(sig_map);
            let key = (callee_id, sig);
            let effective_id = if let Some(&existing) = sig_to_id.get(&key) {
                existing
            } else if callees_with_first_sig.insert(callee_id) {
                // First signature for this callee — use the original ID.
                callee_id
            } else {
                // Conflicting signature — allocate a clone.
                let new_id = next_id(&mut module.header);
                clones_needed.push((callee_id, new_id));
                new_id
            };
            sig_to_id.insert(key, effective_id);
            callees_to_apply
                .entry(effective_id)
                .or_default()
                .extend(sig_map.iter().map(|(&k, &v)| (k, v)));
            if effective_id != callee_id {
                plan_callee_remap.insert((plan_i, callee_id), effective_id);
            }
        }
    }

    // Materialise clones (no-op when no conflicts were detected).
    let first_clone_idx = module.functions.len();
    for &(original_callee_id, new_callee_id) in &clones_needed {
        let original_idx = *index.func_idx.get(&original_callee_id).unwrap();
        // Clone the function out first to release the borrow on `module`,
        // then renumber IDs using just the header.
        let original = module.functions[original_idx].clone();
        let clone = clone_function_with_fresh_ids(original, new_callee_id, &mut module.header);
        module.functions.push(clone);
    }

    // Build a `func_idx` map that includes the clones, so per-callee apply
    // and the apply step below can resolve cloned IDs.
    let mut func_idx_full = index.func_idx.clone();
    for (i, &(_, new_id)) in clones_needed.iter().enumerate() {
        func_idx_full.insert(new_id, first_clone_idx + i);
    }

    apply_callee_rewrites(module, &func_idx_full, callees_to_apply);

    // Rewrite `OpFunctionCall %original → %clone` in any plan that's been
    // remapped. Only touched when there were actual conflicts.
    if !plan_callee_remap.is_empty() {
        for (plan_i, plan) in plans.iter().enumerate() {
            let plan_remap: FxHashMap<u32, u32> = plan_callee_remap
                .iter()
                .filter_map(|(&(pi, orig), &new)| (pi == plan_i).then_some((orig, new)))
                .collect();
            if plan_remap.is_empty() {
                continue;
            }
            for block in &mut module.functions[plan.func_idx].blocks {
                for inst in &mut block.instructions {
                    if inst.class.opcode == Op::FunctionCall
                        && let Some(callee_op) = inst.operands.first_mut()
                        && let Operand::IdRef(callee_id) = callee_op
                        && let Some(&new) = plan_remap.get(callee_id)
                    {
                        *callee_op = Operand::IdRef(new);
                    }
                }
            }
        }
    }
}

/// Apply each (effective callee → new param-type) rewrite by minting a new
/// `OpTypeFunction` for that callee and updating its def + parameter types.
fn apply_callee_rewrites(
    module: &mut Module,
    func_idx: &FxHashMap<u32, usize>,
    rewrites: FxHashMap<u32, FxHashMap<usize, u32>>,
) {
    for (called_fn_id, param_changes) in rewrites {
        let &called_idx = func_idx.get(&called_fn_id).unwrap();
        let old_fn_type_id =
            module.functions[called_idx].def.as_ref().unwrap().operands[1].unwrap_id_ref();

        let mut new_operands = module
            .types_global_values
            .iter()
            .find(|inst| {
                inst.class.opcode == Op::TypeFunction && inst.result_id == Some(old_fn_type_id)
            })
            .unwrap()
            .operands
            .clone();
        for (&param_idx, &new_type) in &param_changes {
            new_operands[param_idx + 1] = Operand::IdRef(new_type);
        }
        let new_fn_type_id = next_id(&mut module.header);
        module.types_global_values.push(Instruction::new(
            Op::TypeFunction,
            None,
            Some(new_fn_type_id),
            new_operands,
        ));

        let called_fn = &mut module.functions[called_idx];
        called_fn.def.as_mut().unwrap().operands[1] = Operand::IdRef(new_fn_type_id);
        for (&param_idx, &new_type) in &param_changes {
            if let Some(param) = called_fn.parameters.get_mut(param_idx) {
                param.result_type = Some(new_type);
            }
        }
    }
}

/// Deep-clone `original` with all internal result IDs renumbered. The new
/// function's def takes `new_def_id`; everything else (parameters, block
/// labels, body instructions) gets a freshly-allocated ID. References to
/// IDs internal to the function are rewritten through the same map;
/// references to external IDs (other functions, types, globals, imported
/// extensions) are left alone.
fn clone_function_with_fresh_ids(
    original: Function,
    new_def_id: u32,
    header: &mut Option<rspirv::dr::ModuleHeader>,
) -> Function {
    let mut clone = original;
    let mut id_remap: FxHashMap<u32, u32> = FxHashMap::default();

    // Function def gets the pre-allocated ID.
    if let Some(def) = clone.def.as_mut()
        && let Some(old) = def.result_id
    {
        id_remap.insert(old, new_def_id);
        def.result_id = Some(new_def_id);
    }

    // Parameters, block labels, and body instructions get fresh IDs.
    for param in &mut clone.parameters {
        if let Some(old) = param.result_id {
            let new = next_id(header);
            id_remap.insert(old, new);
            param.result_id = Some(new);
        }
    }
    for block in &mut clone.blocks {
        if let Some(label) = block.label.as_mut()
            && let Some(old) = label.result_id
        {
            let new = next_id(header);
            id_remap.insert(old, new);
            label.result_id = Some(new);
        }
        for inst in &mut block.instructions {
            if let Some(old) = inst.result_id {
                let new = next_id(header);
                id_remap.insert(old, new);
                inst.result_id = Some(new);
            }
        }
    }

    // Rewrite all IdRef operands using the remap. External IDs (not in the
    // map) pass through unchanged.
    let rewrite = |inst: &mut Instruction, remap: &FxHashMap<u32, u32>| {
        for op in &mut inst.operands {
            if let Operand::IdRef(id) = op
                && let Some(&new) = remap.get(id)
            {
                *op = Operand::IdRef(new);
            }
        }
    };
    if let Some(def) = clone.def.as_mut() {
        rewrite(def, &id_remap);
    }
    for param in &mut clone.parameters {
        rewrite(param, &id_remap);
    }
    for block in &mut clone.blocks {
        if let Some(label) = block.label.as_mut() {
            rewrite(label, &id_remap);
        }
        for inst in &mut block.instructions {
            rewrite(inst, &id_remap);
        }
    }

    clone
}

/// Drop the original global `OpVariables` (now replaced by parameters) and
/// any decorations targeting them. `OpName` decorations are *transferred*
/// from the original `OpVariable` to the new `OpFunctionParameter` (and
/// likewise for `OpMemberName`), so reflection APIs like `clGetKernelArgInfo`
/// can recover the source-level argument names.
///
/// For `&[T]` slices, codegen emits two `OpVariable`s named `<binding>` and
/// `<binding>.len` (e.g. `data` + `data.len`); both names carry through to
/// the corresponding pointer and length kernel parameters.
fn cleanup_globals(module: &mut Module, plan: &Plan) {
    let var_to_param: FxHashMap<u32, u32> =
        plan.params.iter().map(|p| (p.var_id, p.param_id)).collect();
    let was_removed = |id: u32| var_to_param.contains_key(&id);
    module
        .types_global_values
        .retain(|inst| inst.result_id.is_none_or(|id| !was_removed(id)));
    module.annotations.retain(|inst| {
        inst.operands
            .first()
            .and_then(|op| op.id_ref_any())
            .is_none_or(|id| !was_removed(id))
    });
    // Rewrite the target ID of `OpName` (and `OpMemberName`) entries that
    // referred to one of the removed `OpVariable`s, so reflection sees the
    // name on the `OpFunctionParameter` that replaced it.
    for inst in &mut module.debug_names {
        if let Some(target) = inst.operands.first().and_then(|op| op.id_ref_any())
            && let Some(&new_id) = var_to_param.get(&target)
        {
            inst.operands[0] = Operand::IdRef(new_id);
        }
    }
}

/// Truncate the entry-point interface to the `BuiltIn` variables that must
/// remain (e.g. `GlobalInvocationId`).
fn update_entry_point_interface(module: &mut Module, plan: &Plan) {
    let ep = module
        .entry_points
        .iter_mut()
        .find(|ep| ep.operands[1].unwrap_id_ref() == plan.func_id)
        .unwrap();
    ep.operands.truncate(3);
    for &bid in &plan.builtin_interface {
        ep.operands.push(Operand::IdRef(bid));
    }
}

fn next_id(header: &mut Option<rspirv::dr::ModuleHeader>) -> u32 {
    let header = header.as_mut().unwrap();
    let id = header.bound;
    header.bound += 1;
    id
}

/// Fix `BuiltIn` variable types for `Kernel` entry points on `Physical64` addressing.
///
/// The `OpenCL` SPIR-V environment requires `GlobalInvocationId` and similar
/// builtins to use `v3ulong` (vec3 of u64) on `Physical64`, not `v3uint`.
/// Also adds the `Constant` decoration required by some implementations.
fn fix_builtin_types(module: &mut Module) {
    // Only applies to Physical64 addressing.
    let is_physical64 = module
        .memory_model
        .as_ref()
        .is_some_and(|mm| mm.operands[0].unwrap_addressing_model() == AddressingModel::Physical64);
    if !is_physical64 {
        return;
    }

    // Check for Kernel entry points.
    let has_kernel = module
        .entry_points
        .iter()
        .any(|ep| ep.operands[0].unwrap_execution_model() == ExecutionModel::Kernel);
    if !has_kernel {
        return;
    }

    // Find BuiltIn-decorated variables that need type conversion.
    let builtin_var_ids: FxHashSet<u32> = module
        .annotations
        .iter()
        .filter(|inst| {
            inst.class.opcode == Op::Decorate
                && inst.operands[1].unwrap_decoration() == Decoration::BuiltIn
                && matches!(
                    inst.operands[2].unwrap_built_in(),
                    BuiltIn::GlobalInvocationId
                        | BuiltIn::LocalInvocationId
                        | BuiltIn::WorkgroupId
                        | BuiltIn::NumWorkgroups
                        | BuiltIn::GlobalSize
                        | BuiltIn::EnqueuedWorkgroupSize
                        | BuiltIn::GlobalOffset
                )
        })
        .map(|inst| inst.operands[0].unwrap_id_ref())
        .collect();

    if builtin_var_ids.is_empty() {
        return;
    }

    // Add Constant decoration for builtin variables (required by some impls).
    for &var_id in &builtin_var_ids {
        let has_constant = module.annotations.iter().any(|inst| {
            inst.class.opcode == Op::Decorate
                && inst.operands[0].unwrap_id_ref() == var_id
                && inst.operands[1].unwrap_decoration() == Decoration::Constant
        });
        if !has_constant {
            module.annotations.push(Instruction::new(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(var_id),
                    Operand::Decoration(Decoration::Constant),
                ],
            ));
        }
    }

    // Find the u32 and u64 type IDs, and v3uint type ID.
    let mut u32_type: Option<u32> = None;
    let mut u64_type: Option<u32> = None;
    let mut v3uint_type: Option<u32> = None;
    let mut ptr_input_v3uint: Option<u32> = None;

    for inst in &module.types_global_values {
        match inst.class.opcode {
            Op::TypeInt => {
                let width = inst.operands[0].unwrap_literal_bit32();
                if width == 32 {
                    u32_type = inst.result_id;
                } else if width == 64 {
                    u64_type = inst.result_id;
                }
            }
            Op::TypeVector => {
                let elem = inst.operands[0].unwrap_id_ref();
                let count = inst.operands[1].unwrap_literal_bit32();
                if count == 3 && Some(elem) == u32_type {
                    v3uint_type = inst.result_id;
                }
            }
            Op::TypePointer => {
                let sc = inst.operands[0].unwrap_storage_class();
                let pointee = inst.operands[1].unwrap_id_ref();
                if sc == StorageClass::Input && Some(pointee) == v3uint_type {
                    ptr_input_v3uint = inst.result_id;
                }
            }
            _ => {}
        }
    }

    let (Some(_u32_ty), Some(u64_ty), Some(v3uint_ty)) = (u32_type, u64_type, v3uint_type) else {
        return;
    };

    // Create v3ulong and *Input v3ulong types. Insert them before the first
    // OpVariable so they're defined before use (SPIR-V requires this).
    let v3ulong_ty = next_id(&mut module.header);
    let ptr_input_v3ulong_ty = next_id(&mut module.header);

    let insert_pos = module
        .types_global_values
        .iter()
        .position(|inst| inst.class.opcode == Op::Variable)
        .unwrap_or(module.types_global_values.len());

    module.types_global_values.insert(
        insert_pos,
        Instruction::new(
            Op::TypeVector,
            None,
            Some(v3ulong_ty),
            vec![Operand::IdRef(u64_ty), Operand::LiteralBit32(3)],
        ),
    );
    module.types_global_values.insert(
        insert_pos + 1,
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_input_v3ulong_ty),
            vec![
                Operand::StorageClass(StorageClass::Input),
                Operand::IdRef(v3ulong_ty),
            ],
        ),
    );

    // Change the BuiltIn OpVariable types from *Input v3uint to *Input v3ulong.
    for inst in &mut module.types_global_values {
        if inst.class.opcode == Op::Variable
            && inst
                .result_id
                .is_some_and(|id| builtin_var_ids.contains(&id))
            && inst.result_type == ptr_input_v3uint
        {
            inst.result_type = Some(ptr_input_v3ulong_ty);
        }
    }

    // In all functions, fix OpLoad of builtin variables:
    // 1. Change result type from v3uint to v3ulong (to match the variable type)
    // 2. Insert OpUConvert v3ulong → v3uint right after, keeping the original
    //    result ID so all downstream code continues to work with v3uint.
    for func in &mut module.functions {
        for block in &mut func.blocks {
            let mut insertions: Vec<(usize, Instruction)> = Vec::new();

            for (i, inst) in block.instructions.iter_mut().enumerate() {
                if inst.class.opcode == Op::Load
                    && inst.result_type == Some(v3uint_ty)
                    && builtin_var_ids.contains(&inst.operands[0].unwrap_id_ref())
                {
                    let original_id = inst.result_id.unwrap();
                    let new_v3ulong_id = next_id(&mut module.header);

                    // Load produces v3ulong with a new ID.
                    inst.result_type = Some(v3ulong_ty);
                    inst.result_id = Some(new_v3ulong_id);

                    // Convert v3ulong → v3uint with the original ID.
                    insertions.push((
                        i + 1,
                        Instruction::new(
                            Op::UConvert,
                            Some(v3uint_ty),
                            Some(original_id),
                            vec![Operand::IdRef(new_v3ulong_id)],
                        ),
                    ));
                }
            }

            for (idx, inst) in insertions.into_iter().rev() {
                block.instructions.insert(idx, inst);
            }
        }
    }
}
