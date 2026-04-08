//! Kernel-mode group operations using the `Groups` capability.
//!
//! These use `OpGroupAll`, `OpGroupAny`, `OpGroupBroadcast`, `OpGroupIAdd`,
//! etc. which are valid for the `Kernel` execution model (`OpenCL` SPIR-V).
//!
//! For Vulkan/Shader targets, use the `subgroup_*` functions instead,
//! which use the `GroupNonUniform*` capabilities.

#[cfg(target_arch = "spirv")]
use core::arch::asm;

#[cfg(target_arch = "spirv")]
const SUBGROUP: u32 = crate::memory::Scope::Subgroup as u32;

#[cfg(target_arch = "spirv")]
const WORKGROUP: u32 = crate::memory::Scope::Workgroup as u32;

/// Evaluates a predicate for all invocations in the group. Returns `true`
/// if `predicate` is `true` for **all** invocations.
///
/// Requires Capability `Groups`.
#[spirv_std_macros::gpu_only]
#[doc(alias = "OpGroupAll")]
#[inline]
pub fn group_all(predicate: bool) -> bool {
    let mut result = false;
    unsafe {
        asm! {
            "%bool = OpTypeBool",
            "%u32 = OpTypeInt 32 0",
            "%scope = OpConstant %u32 {scope}",
            "%predicate = OpLoad _ {predicate}",
            "%result = OpGroupAll %bool %scope %predicate",
            "OpStore {result} %result",
            scope = const SUBGROUP,
            predicate = in(reg) &predicate,
            result = in(reg) &mut result,
        }
    }
    result
}

/// Evaluates a predicate for all invocations in the group. Returns `true`
/// if `predicate` is `true` for **any** invocation.
///
/// Requires Capability `Groups`.
#[spirv_std_macros::gpu_only]
#[doc(alias = "OpGroupAny")]
#[inline]
pub fn group_any(predicate: bool) -> bool {
    let mut result = false;
    unsafe {
        asm! {
            "%bool = OpTypeBool",
            "%u32 = OpTypeInt 32 0",
            "%scope = OpConstant %u32 {scope}",
            "%predicate = OpLoad _ {predicate}",
            "%result = OpGroupAny %bool %scope %predicate",
            "OpStore {result} %result",
            scope = const SUBGROUP,
            predicate = in(reg) &predicate,
            result = in(reg) &mut result,
        }
    }
    result
}

/// Broadcasts `value` from the invocation with `local_id` to all invocations
/// in the group.
///
/// `local_id` must be the same value for all invocations in the group.
///
/// Requires Capability `Groups`.
#[spirv_std_macros::gpu_only]
#[doc(alias = "OpGroupBroadcast")]
#[inline]
pub fn group_broadcast_u32(value: u32, local_id: u32) -> u32 {
    let mut result: u32 = 0;
    unsafe {
        asm! {
            "%u32 = OpTypeInt 32 0",
            "%scope = OpConstant %u32 {scope}",
            "%value = OpLoad _ {value}",
            "%id = OpLoad _ {id}",
            "%result = OpGroupBroadcast %u32 %scope %value %id",
            "OpStore {result} %result",
            scope = const SUBGROUP,
            value = in(reg) &value,
            id = in(reg) &local_id,
            result = in(reg) &mut result,
        }
    }
    result
}

// Helper macro to generate group arithmetic operations with Reduce,
// InclusiveScan, and ExclusiveScan variants — matching the pattern
// used by the Vulkan subgroup_* functions.
macro_rules! group_op {
    ($scope:ident, $ty:ty, $spirv_ty:literal, $asm_op:literal,
     $reduce:ident, $inclusive:ident, $exclusive:ident,
     $doc:literal) => {
        #[doc = concat!($doc, "\n\nReturns the reduction across all invocations in the group.\n\nRequires Capability `Groups`.")]
        #[spirv_std_macros::gpu_only]
        #[doc(alias = $asm_op)]
        #[inline]
        pub fn $reduce(value: $ty) -> $ty {
            let mut result: $ty = Default::default();
            unsafe {
                asm! {
                    concat!("%ty = ", $spirv_ty),
                    "%u32 = OpTypeInt 32 0",
                    "%scope = OpConstant %u32 {scope}",
                    "%value = OpLoad _ {value}",
                    concat!("%result = ", $asm_op, " %ty %scope Reduce %value"),
                    "OpStore {result} %result",
                    scope = const $scope,
                    value = in(reg) &value,
                    result = in(reg) &mut result,
                }
            }
            result
        }

        #[doc = concat!($doc, "\n\nReturns the inclusive scan (prefix sum including this invocation).\n\nRequires Capability `Groups`.")]
        #[spirv_std_macros::gpu_only]
        #[doc(alias = $asm_op)]
        #[inline]
        pub fn $inclusive(value: $ty) -> $ty {
            let mut result: $ty = Default::default();
            unsafe {
                asm! {
                    concat!("%ty = ", $spirv_ty),
                    "%u32 = OpTypeInt 32 0",
                    "%scope = OpConstant %u32 {scope}",
                    "%value = OpLoad _ {value}",
                    concat!("%result = ", $asm_op, " %ty %scope InclusiveScan %value"),
                    "OpStore {result} %result",
                    scope = const $scope,
                    value = in(reg) &value,
                    result = in(reg) &mut result,
                }
            }
            result
        }

        #[doc = concat!($doc, "\n\nReturns the exclusive scan (prefix sum excluding this invocation).\n\nRequires Capability `Groups`.")]
        #[spirv_std_macros::gpu_only]
        #[doc(alias = $asm_op)]
        #[inline]
        pub fn $exclusive(value: $ty) -> $ty {
            let mut result: $ty = Default::default();
            unsafe {
                asm! {
                    concat!("%ty = ", $spirv_ty),
                    "%u32 = OpTypeInt 32 0",
                    "%scope = OpConstant %u32 {scope}",
                    "%value = OpLoad _ {value}",
                    concat!("%result = ", $asm_op, " %ty %scope ExclusiveScan %value"),
                    "OpStore {result} %result",
                    scope = const $scope,
                    value = in(reg) &value,
                    result = in(reg) &mut result,
                }
            }
            result
        }
    };
}

group_op!(
    SUBGROUP,
    u32,
    "OpTypeInt 32 0",
    "OpGroupIAdd",
    group_i_add,
    group_inclusive_i_add,
    group_exclusive_i_add,
    "Integer add group operation."
);

group_op!(
    SUBGROUP,
    f32,
    "OpTypeFloat 32",
    "OpGroupFAdd",
    group_f_add,
    group_inclusive_f_add,
    group_exclusive_f_add,
    "Float add group operation."
);

group_op!(
    SUBGROUP,
    u32,
    "OpTypeInt 32 0",
    "OpGroupUMin",
    group_u_min,
    group_inclusive_u_min,
    group_exclusive_u_min,
    "Unsigned integer minimum group operation."
);

group_op!(
    SUBGROUP,
    u32,
    "OpTypeInt 32 0",
    "OpGroupUMax",
    group_u_max,
    group_inclusive_u_max,
    group_exclusive_u_max,
    "Unsigned integer maximum group operation."
);

// NOTE: Kernel capability requires signedness=0 for all OpTypeInt.
// Sign is encoded in the operation (OpGroupSMin vs OpGroupUMin), not the type.
group_op!(
    SUBGROUP,
    i32,
    "OpTypeInt 32 0",
    "OpGroupSMin",
    group_s_min,
    group_inclusive_s_min,
    group_exclusive_s_min,
    "Signed integer minimum group operation."
);

group_op!(
    SUBGROUP,
    i32,
    "OpTypeInt 32 0",
    "OpGroupSMax",
    group_s_max,
    group_inclusive_s_max,
    group_exclusive_s_max,
    "Signed integer maximum group operation."
);

group_op!(
    SUBGROUP,
    f32,
    "OpTypeFloat 32",
    "OpGroupFMin",
    group_f_min,
    group_inclusive_f_min,
    group_exclusive_f_min,
    "Float minimum group operation."
);

group_op!(
    SUBGROUP,
    f32,
    "OpTypeFloat 32",
    "OpGroupFMax",
    group_f_max,
    group_inclusive_f_max,
    group_exclusive_f_max,
    "Float maximum group operation."
);

// Double-precision (f64) variants.

group_op!(
    SUBGROUP,
    f64,
    "OpTypeFloat 64",
    "OpGroupFAdd",
    group_f64_add,
    group_inclusive_f64_add,
    group_exclusive_f64_add,
    "Double-precision float add group operation."
);

group_op!(
    SUBGROUP,
    f64,
    "OpTypeFloat 64",
    "OpGroupFMin",
    group_f64_min,
    group_inclusive_f64_min,
    group_exclusive_f64_min,
    "Double-precision float minimum group operation."
);

group_op!(
    SUBGROUP,
    f64,
    "OpTypeFloat 64",
    "OpGroupFMax",
    group_f64_max,
    group_inclusive_f64_max,
    group_exclusive_f64_max,
    "Double-precision float maximum group operation."
);

// ─── Workgroup-scoped variants (OpenCL `work_group_*`) ───────────────────
//
// Same `OpGroup*` operations, but with `Workgroup` scope so the reduction /
// scan / broadcast spans the whole work-group instead of a single subgroup.
// These map to the OpenCL C `work_group_reduce_*`, `work_group_scan_*`,
// and `work_group_broadcast` family.
//
// All invocations in the work-group must execute the operation in uniform
// control flow; otherwise behavior is undefined (per OpenCL SPIR-V env).

/// Evaluates a predicate for all invocations in the work-group. Returns
/// `true` if `predicate` is `true` for **all** invocations.
///
/// Maps to `OpenCL` `work_group_all`. Requires Capability `Groups`.
#[spirv_std_macros::gpu_only]
#[doc(alias = "OpGroupAll")]
#[inline]
pub fn work_group_all(predicate: bool) -> bool {
    let mut result = false;
    unsafe {
        asm! {
            "%bool = OpTypeBool",
            "%u32 = OpTypeInt 32 0",
            "%scope = OpConstant %u32 {scope}",
            "%predicate = OpLoad _ {predicate}",
            "%result = OpGroupAll %bool %scope %predicate",
            "OpStore {result} %result",
            scope = const WORKGROUP,
            predicate = in(reg) &predicate,
            result = in(reg) &mut result,
        }
    }
    result
}

/// Evaluates a predicate for all invocations in the work-group. Returns
/// `true` if `predicate` is `true` for **any** invocation.
///
/// Maps to `OpenCL` `work_group_any`. Requires Capability `Groups`.
#[spirv_std_macros::gpu_only]
#[doc(alias = "OpGroupAny")]
#[inline]
pub fn work_group_any(predicate: bool) -> bool {
    let mut result = false;
    unsafe {
        asm! {
            "%bool = OpTypeBool",
            "%u32 = OpTypeInt 32 0",
            "%scope = OpConstant %u32 {scope}",
            "%predicate = OpLoad _ {predicate}",
            "%result = OpGroupAny %bool %scope %predicate",
            "OpStore {result} %result",
            scope = const WORKGROUP,
            predicate = in(reg) &predicate,
            result = in(reg) &mut result,
        }
    }
    result
}

/// Broadcasts `value` from the invocation with `local_id` to all invocations
/// in the work-group.
///
/// `local_id` must be the same value for all invocations in the work-group.
/// The `OpenCL` SPIR-V environment requires the `LocalId` to be a `size_t` for
/// `Workgroup` scope, so this takes `usize` (which is `ulong` on Physical64);
/// passing a narrower type yields a mangled symbol that `pocl`'s kernel
/// library doesn't expose.
///
/// Maps to `OpenCL` `work_group_broadcast`. Requires Capability `Groups`.
#[spirv_std_macros::gpu_only]
#[doc(alias = "OpGroupBroadcast")]
#[inline]
pub fn work_group_broadcast_u32(value: u32, local_id: usize) -> u32 {
    let mut result: u32 = 0;
    unsafe {
        asm! {
            "%u32 = OpTypeInt 32 0",
            "%scope = OpConstant %u32 {scope}",
            "%value = OpLoad _ {value}",
            "%id = OpLoad _ {id}",
            "%result = OpGroupBroadcast %u32 %scope %value %id",
            "OpStore {result} %result",
            scope = const WORKGROUP,
            value = in(reg) &value,
            id = in(reg) &local_id,
            result = in(reg) &mut result,
        }
    }
    result
}

group_op!(
    WORKGROUP,
    u32,
    "OpTypeInt 32 0",
    "OpGroupIAdd",
    work_group_i_add,
    work_group_inclusive_i_add,
    work_group_exclusive_i_add,
    "Integer add work-group operation."
);

group_op!(
    WORKGROUP,
    f32,
    "OpTypeFloat 32",
    "OpGroupFAdd",
    work_group_f_add,
    work_group_inclusive_f_add,
    work_group_exclusive_f_add,
    "Float add work-group operation."
);

group_op!(
    WORKGROUP,
    u32,
    "OpTypeInt 32 0",
    "OpGroupUMin",
    work_group_u_min,
    work_group_inclusive_u_min,
    work_group_exclusive_u_min,
    "Unsigned integer minimum work-group operation."
);

group_op!(
    WORKGROUP,
    u32,
    "OpTypeInt 32 0",
    "OpGroupUMax",
    work_group_u_max,
    work_group_inclusive_u_max,
    work_group_exclusive_u_max,
    "Unsigned integer maximum work-group operation."
);

group_op!(
    WORKGROUP,
    i32,
    "OpTypeInt 32 0",
    "OpGroupSMin",
    work_group_s_min,
    work_group_inclusive_s_min,
    work_group_exclusive_s_min,
    "Signed integer minimum work-group operation."
);

group_op!(
    WORKGROUP,
    i32,
    "OpTypeInt 32 0",
    "OpGroupSMax",
    work_group_s_max,
    work_group_inclusive_s_max,
    work_group_exclusive_s_max,
    "Signed integer maximum work-group operation."
);

group_op!(
    WORKGROUP,
    f32,
    "OpTypeFloat 32",
    "OpGroupFMin",
    work_group_f_min,
    work_group_inclusive_f_min,
    work_group_exclusive_f_min,
    "Float minimum work-group operation."
);

group_op!(
    WORKGROUP,
    f32,
    "OpTypeFloat 32",
    "OpGroupFMax",
    work_group_f_max,
    work_group_inclusive_f_max,
    work_group_exclusive_f_max,
    "Float maximum work-group operation."
);

// Double-precision (f64) variants.

group_op!(
    WORKGROUP,
    f64,
    "OpTypeFloat 64",
    "OpGroupFAdd",
    work_group_f64_add,
    work_group_inclusive_f64_add,
    work_group_exclusive_f64_add,
    "Double-precision float add work-group operation."
);

group_op!(
    WORKGROUP,
    f64,
    "OpTypeFloat 64",
    "OpGroupFMin",
    work_group_f64_min,
    work_group_inclusive_f64_min,
    work_group_exclusive_f64_min,
    "Double-precision float minimum work-group operation."
);

group_op!(
    WORKGROUP,
    f64,
    "OpTypeFloat 64",
    "OpGroupFMax",
    work_group_f64_max,
    work_group_inclusive_f64_max,
    work_group_exclusive_f64_max,
    "Double-precision float maximum work-group operation."
);
