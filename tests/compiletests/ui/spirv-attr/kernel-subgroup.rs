// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-opencl1.2
// compile-flags: -C target-feature=+Groups

// Test Kernel-mode subgroup operations using the Groups capability
// and the spirv_std::arch::group_* API.

use spirv_std::arch;
use spirv_std::spirv;

#[spirv(kernel(threads(32)))]
pub fn test_group_all(#[spirv(cross_workgroup)] out: &mut u32) {
    *out = arch::group_all(true) as u32;
}

#[spirv(kernel(threads(32)))]
pub fn test_group_any(#[spirv(cross_workgroup)] out: &mut u32) {
    *out = arch::group_any(true) as u32;
}

#[spirv(kernel(threads(32)))]
pub fn test_group_broadcast(#[spirv(cross_workgroup)] out: &mut u32, value: u32) {
    *out = arch::group_broadcast_u32(value, 0);
}

#[spirv(kernel(threads(32)))]
pub fn test_group_iadd_reduce(#[spirv(cross_workgroup)] out: &mut u32, value: u32) {
    *out = arch::group_i_add(value);
}

#[spirv(kernel(threads(32)))]
pub fn test_group_iadd_inclusive(#[spirv(cross_workgroup)] out: &mut u32, value: u32) {
    *out = arch::group_inclusive_i_add(value);
}

#[spirv(kernel(threads(32)))]
pub fn test_group_iadd_exclusive(#[spirv(cross_workgroup)] out: &mut u32, value: u32) {
    *out = arch::group_exclusive_i_add(value);
}

#[spirv(kernel(threads(32)))]
pub fn test_group_fadd_reduce(#[spirv(cross_workgroup)] out: &mut f32, value: f32) {
    *out = arch::group_f_add(value);
}

#[spirv(kernel(threads(32)))]
pub fn test_group_umin(#[spirv(cross_workgroup)] out: &mut u32, value: u32) {
    *out = arch::group_u_min(value);
}

#[spirv(kernel(threads(32)))]
pub fn test_group_fmax(#[spirv(cross_workgroup)] out: &mut f32, value: f32) {
    *out = arch::group_f_max(value);
}

#[spirv(kernel(threads(32)))]
pub fn test_subgroup_builtins(
    #[spirv(cross_workgroup)] out: &mut u32,
    #[spirv(subgroup_id)] subgroup_id: u32,
    #[spirv(subgroup_local_invocation_id)] local_id: u32,
) {
    *out = subgroup_id * 1000 + local_id;
}
