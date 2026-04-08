// build-pass
// only-opencl2.0
// compile-flags: -C target-feature=+Groups

// Test Kernel-mode subgroup operations using the Groups capability
// and OpGroup* instructions (valid for Kernel execution model).
//
// The Groups capability uses OpGroupAll, OpGroupAny, OpGroupIAdd, etc.
// which are distinct from Vulkan's GroupNonUniform* capabilities.

use spirv_std::spirv;

// OpGroupAll — true if predicate is true for all invocations in the subgroup.
#[spirv(kernel(threads(32)))]
pub fn test_group_all(#[spirv(cross_workgroup)] out: &mut u32) {
    let result: u32;
    unsafe {
        core::arch::asm!(
            "%bool = OpTypeBool",
            "%uint = OpTypeInt 32 0",
            "%subgroup = OpConstant %uint 3",
            "%true = OpConstantTrue %bool",
            "%one = OpConstant %uint 1",
            "%zero = OpConstant %uint 0",
            "%pred = OpGroupAll %bool %subgroup %true",
            "{result} = OpSelect %uint %pred %one %zero",
            result = out(reg) result,
        );
    }
    *out = result;
}

// OpGroupAny — true if predicate is true for any invocation in the subgroup.
#[spirv(kernel(threads(32)))]
pub fn test_group_any(#[spirv(cross_workgroup)] out: &mut u32) {
    let result: u32;
    unsafe {
        core::arch::asm!(
            "%bool = OpTypeBool",
            "%uint = OpTypeInt 32 0",
            "%subgroup = OpConstant %uint 3",
            "%true = OpConstantTrue %bool",
            "%one = OpConstant %uint 1",
            "%zero = OpConstant %uint 0",
            "%pred = OpGroupAny %bool %subgroup %true",
            "{result} = OpSelect %uint %pred %one %zero",
            result = out(reg) result,
        );
    }
    *out = result;
}

// OpGroupBroadcast — broadcast value from local_id to all invocations.
#[spirv(kernel(threads(32)))]
pub fn test_group_broadcast(#[spirv(cross_workgroup)] out: &mut u32, value: u32) {
    let result: u32;
    unsafe {
        core::arch::asm!(
            "%uint = OpTypeInt 32 0",
            "%subgroup = OpConstant %uint 3",
            "%zero = OpConstant %uint 0",
            "{result} = OpGroupBroadcast %uint %subgroup {value} %zero",
            value = in(reg) value,
            result = out(reg) result,
        );
    }
    *out = result;
}

// OpGroupIAdd — integer reduction across the subgroup.
#[spirv(kernel(threads(32)))]
pub fn test_group_iadd_reduce(#[spirv(cross_workgroup)] out: &mut u32, value: u32) {
    let result: u32;
    unsafe {
        core::arch::asm!(
            "%uint = OpTypeInt 32 0",
            "%subgroup = OpConstant %uint 3",
            "{result} = OpGroupIAdd %uint %subgroup Reduce {value}",
            value = in(reg) value,
            result = out(reg) result,
        );
    }
    *out = result;
}

// Subgroup builtins.
#[spirv(kernel(threads(32)))]
pub fn test_subgroup_builtins(
    #[spirv(cross_workgroup)] out: &mut u32,
    #[spirv(subgroup_id)] subgroup_id: u32,
    #[spirv(subgroup_local_invocation_id)] local_id: u32,
) {
    *out = subgroup_id * 1000 + local_id;
}
