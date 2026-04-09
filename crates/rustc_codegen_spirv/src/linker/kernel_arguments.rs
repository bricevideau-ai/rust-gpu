//! Convert Kernel entry stubs from `void(void)` + global `OpVariable` pattern
//! to proper function parameters, as required by the `OpenCL` SPIR-V environment.
//!
//! After inlining and specialization, Kernel entry points look like:
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
//! ```text
//! %entry = OpFunction %void None %fn_type_with_params
//! %param = OpFunctionParameter %T
//! %label = OpLabel
//! ... (uses %param instead of OpLoad %global_var)
//! OpReturn
//! OpFunctionEnd
//! ```

use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::{AddressingModel, BuiltIn, Decoration, ExecutionModel, Op, StorageClass};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};

/// Convert Kernel entry points from global-`OpVariable` to `OpFunctionParameter`,
/// and fix `BuiltIn` variable types for Physical64 addressing.
pub fn convert_kernel_arguments(module: &mut Module) {
    fix_builtin_types(module);
    // Find all Kernel entry points and their interface variables.
    let kernel_entries: Vec<(u32, Vec<u32>)> = module
        .entry_points
        .iter()
        .filter(|ep| ep.operands[0].unwrap_execution_model() == ExecutionModel::Kernel)
        .map(|ep| {
            let func_id = ep.operands[1].unwrap_id_ref();
            let interface_ids: Vec<u32> = ep.operands[3..]
                .iter()
                .map(|op| op.unwrap_id_ref())
                .collect();
            (func_id, interface_ids)
        })
        .collect();

    if kernel_entries.is_empty() {
        return;
    }

    // Pre-compute the set of builtin-decorated variable IDs.
    let builtin_ids: FxHashSet<u32> = module
        .annotations
        .iter()
        .filter(|inst| {
            inst.class.opcode == Op::Decorate
                && inst.operands[1].unwrap_decoration() == Decoration::BuiltIn
        })
        .map(|inst| inst.operands[0].unwrap_id_ref())
        .collect();

    // Build maps from global OpVariable IDs to their info.
    let mut var_info: FxHashMap<u32, VarInfo> = FxHashMap::default();
    for inst in &module.types_global_values {
        if inst.class.opcode == Op::Variable {
            let result_id = inst.result_id.unwrap();
            let ptr_type = inst.result_type.unwrap();
            let storage_class = inst.operands[0].unwrap_storage_class();
            // Variables with initializers are program-scope globals (statics),
            // not kernel arguments — don't convert them to function parameters.
            let has_initializer = inst.operands.len() > 1;
            var_info.insert(
                result_id,
                VarInfo {
                    ptr_type,
                    storage_class,
                    has_initializer,
                },
            );
        }
    }

    // Build map from pointer type to pointee type.
    let mut ptr_to_pointee: FxHashMap<u32, u32> = FxHashMap::default();
    for inst in &module.types_global_values {
        if inst.class.opcode == Op::TypePointer {
            let result_id = inst.result_id.unwrap();
            let pointee = inst.operands[1].unwrap_id_ref();
            ptr_to_pointee.insert(result_id, pointee);
        }
    }

    // For each Kernel entry point, collect the transformation plan.
    struct EntryPlan {
        func_id: u32,
        builtin_interface: Vec<u32>,
        param_var_ids: Vec<u32>,
        param_types: Vec<u32>,
        param_id_map: FxHashMap<u32, u32>,
        fn_params: Vec<Instruction>,
        new_fn_type_id: u32,
    }

    let mut plans: Vec<EntryPlan> = Vec::new();

    for (func_id, interface_ids) in &kernel_entries {
        let func = module
            .functions
            .iter()
            .find(|f| f.def_id() == Some(*func_id))
            .unwrap();

        // Classify interface variables.
        let entry_builtin_ids: Vec<u32> = interface_ids
            .iter()
            .copied()
            .filter(|id| builtin_ids.contains(id))
            .collect();
        let interface_param_ids: Vec<u32> = interface_ids
            .iter()
            .copied()
            .filter(|id| !builtin_ids.contains(id))
            .collect();

        // Find CrossWorkgroup variables referenced in the function body but
        // not in the interface list (SPIR-V <= 1.3 only lists Input/Output).
        let mut body_var_ids: Vec<u32> = Vec::new();
        for block in &func.blocks {
            for inst in &block.instructions {
                for op in inst.operands.iter().filter_map(|o| o.id_ref_any()) {
                    if let Some(vi) = var_info.get(&op)
                        && vi.storage_class == StorageClass::CrossWorkgroup
                        && !vi.has_initializer
                        && !interface_param_ids.contains(&op)
                        && !body_var_ids.contains(&op)
                    {
                        body_var_ids.push(op);
                    }
                }
            }
        }

        let all_param_var_ids: Vec<u32> = interface_param_ids
            .iter()
            .chain(body_var_ids.iter())
            .copied()
            .collect();

        if all_param_var_ids.is_empty() {
            continue;
        }

        // Determine parameter types.
        let param_types: Vec<u32> = all_param_var_ids
            .iter()
            .map(|var_id| {
                let vi = &var_info[var_id];
                if vi.storage_class == StorageClass::CrossWorkgroup {
                    vi.ptr_type // Pointer type (e.g., *CrossWorkgroup u32)
                } else {
                    ptr_to_pointee[&vi.ptr_type] // Value type (e.g., u64)
                }
            })
            .collect();

        // Allocate IDs for the new function type and parameters.
        let new_fn_type_id = next_id(&mut module.header);
        let mut param_id_map: FxHashMap<u32, u32> = FxHashMap::default();
        let mut fn_params = Vec::new();
        for (var_id, param_type) in all_param_var_ids.iter().zip(param_types.iter()) {
            let param_id = next_id(&mut module.header);
            param_id_map.insert(*var_id, param_id);
            fn_params.push(Instruction::new(
                Op::FunctionParameter,
                Some(*param_type),
                Some(param_id),
                vec![],
            ));
        }

        plans.push(EntryPlan {
            func_id: *func_id,
            builtin_interface: entry_builtin_ids,
            param_var_ids: all_param_var_ids,
            param_types,
            param_id_map,
            fn_params,
            new_fn_type_id,
        });
    }

    // Apply all plans.
    for plan in &plans {
        // Add the new function type to the types section.
        let func = module
            .functions
            .iter()
            .find(|f| f.def_id() == Some(plan.func_id))
            .unwrap();
        let void_type = func.def.as_ref().unwrap().result_type.unwrap();

        module.types_global_values.push(Instruction::new(
            Op::TypeFunction,
            None,
            Some(plan.new_fn_type_id),
            std::iter::once(Operand::IdRef(void_type))
                .chain(plan.param_types.iter().map(|t| Operand::IdRef(*t)))
                .collect(),
        ));

        // Update the function: change type and add parameters.
        let func = module
            .functions
            .iter_mut()
            .find(|f| f.def_id() == Some(plan.func_id))
            .unwrap();

        func.def.as_mut().unwrap().operands[1] = Operand::IdRef(plan.new_fn_type_id);
        func.parameters = plan.fn_params.clone();

        // Replace references to global OpVariables in the function body.
        for block in &mut func.blocks {
            for inst in &mut block.instructions {
                if inst.class.opcode == Op::Load {
                    let loaded_from = inst.operands[0].unwrap_id_ref();
                    if let Some(&param_id) = plan.param_id_map.get(&loaded_from) {
                        let vi = &var_info[&loaded_from];
                        if vi.storage_class == StorageClass::CrossWorkgroup {
                            // Param is the pointer — keep OpLoad, change source.
                            inst.operands[0] = Operand::IdRef(param_id);
                        } else {
                            // Param is the value — replace OpLoad with OpCopyObject.
                            *inst = Instruction::new(
                                Op::CopyObject,
                                inst.result_type,
                                inst.result_id,
                                vec![Operand::IdRef(param_id)],
                            );
                        }
                    }
                }
                // Also replace any direct references to the variable (e.g., in
                // OpInBoundsPtrAccessChain or OpStore).
                for op in &mut inst.operands {
                    if let Operand::IdRef(id) = op
                        && let Some(&param_id) = plan.param_id_map.get(id)
                    {
                        *op = Operand::IdRef(param_id);
                    }
                }
            }
        }

        // Remove the converted global OpVariables and their decorations.
        module.types_global_values.retain(|inst| {
            inst.result_id
                .is_none_or(|id| !plan.param_var_ids.contains(&id))
        });
        module.annotations.retain(|inst| {
            inst.operands
                .first()
                .and_then(|op| op.id_ref_any())
                .is_none_or(|id| !plan.param_var_ids.contains(&id))
        });
        module.debug_names.retain(|inst| {
            inst.operands
                .first()
                .and_then(|op| op.id_ref_any())
                .is_none_or(|id| !plan.param_var_ids.contains(&id))
        });

        // Update the entry point interface to only include builtins.
        // Builtins (Input OpVariables with BuiltIn decorations) must remain
        // in the interface for the runtime to populate them per work item.
        let ep = module
            .entry_points
            .iter_mut()
            .find(|ep| ep.operands[1].unwrap_id_ref() == plan.func_id)
            .unwrap();
        ep.operands.truncate(3); // Keep execution model, func ID, name
        for bid in &plan.builtin_interface {
            ep.operands.push(Operand::IdRef(*bid));
        }
    }
}

struct VarInfo {
    ptr_type: u32,
    storage_class: StorageClass,
    has_initializer: bool,
}

fn next_id(header: &mut Option<rspirv::dr::ModuleHeader>) -> u32 {
    let header = header.as_mut().unwrap();
    let id = header.bound;
    header.bound += 1;
    id
}

/// Fix `BuiltIn` variable types for Kernel entry points on Physical64 addressing.
///
/// The `OpenCL` SPIR-V environment requires `GlobalInvocationId` and similar
/// builtins to use `v3ulong` (vec3 of u64) on Physical64, not `v3uint`.
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

    // Skip the Constant decoration for Kernel targets: it is not valid on
    // Input storage class variables per the SPIR-V spec (only UniformConstant)
    // and causes Intel NEO (IGC) to fail with CL_OUT_OF_HOST_MEMORY.
    if !has_kernel {
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
