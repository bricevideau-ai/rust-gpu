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

    for plan in &plans {
        apply_plan(module, plan);
    }

    // Position decorations served their purpose; strip them so they don't
    // survive into the final SPIR-V (they piggyback on `UserTypeGOOGLE`,
    // which OpenCL consumers wouldn't understand).
    KernelParamPositionDecoration::remove_all(module);
}

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
        // body for CrossWorkgroup globals (kernel args).
        let mut body_var_ids: Vec<u32> = Vec::new();
        for block in &module.functions[func_idx].blocks {
            for inst in &block.instructions {
                for op in inst.operands.iter().filter_map(|o| o.id_ref_any()) {
                    let Some(vi) = index.var_info.get(&op) else {
                        continue;
                    };
                    if vi.storage_class == StorageClass::CrossWorkgroup
                        && !vi.has_initializer
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
                    (index.ptr_to_pointee[&vi.ptr_type], ParamKind::ScalarByValue)
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

fn apply_plan(module: &mut Module, plan: &Plan) {
    emit_new_function_type(module, plan);
    swap_function_signature(module, plan);
    rewrite_function_body(&mut module.functions[plan.func_idx], plan);
    cleanup_globals(module, plan);
    update_entry_point_interface(module, plan);
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

/// Rewrite all references to the original global `OpVariables` in the body:
/// - `CrossWorkgroup` pointer var → `OpFunctionParameter` (still a pointer).
/// - Scalar var: only valid as `OpLoad` source; converted to `OpCopyObject %param`.
fn rewrite_function_body(func: &mut Function, plan: &Plan) {
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
