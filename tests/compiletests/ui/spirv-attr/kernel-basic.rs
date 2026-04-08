// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::glam::*;
use spirv_std::spirv;

/// Minimal kernel with no parameters.
#[spirv(kernel)]
pub fn kernel_noop() {}

/// Kernel with a built-in global_invocation_id.
#[spirv(kernel)]
pub fn kernel_with_global_id(#[spirv(global_invocation_id)] _id: USizeVec3) {}

/// Kernel with multiple compute builtins.
#[spirv(kernel)]
pub fn kernel_with_builtins(
    #[spirv(global_invocation_id)] _global_id: USizeVec3,
    #[spirv(local_invocation_id)] _local_id: USizeVec3,
    #[spirv(workgroup_id)] _wg_id: USizeVec3,
    #[spirv(num_workgroups)] _num_wgs: USizeVec3,
    #[spirv(local_invocation_index)] _local_idx: u32,
) {
}

/// Kernel with a mutable buffer parameter (CrossWorkgroup).
#[spirv(kernel)]
pub fn kernel_with_buffer(
    #[spirv(global_invocation_id)] _id: USizeVec3,
    #[spirv(cross_workgroup)] buf: &mut u32,
) {
    *buf *= 2;
}

/// Kernel with an immutable CrossWorkgroup reference.
#[spirv(kernel)]
pub fn kernel_with_readonly_buffer(
    #[spirv(cross_workgroup)] input: &u32,
    #[spirv(cross_workgroup)] output: &mut u32,
) {
    *output = *input;
}

/// Kernel with a scalar by-value parameter (not a pointer).
#[spirv(kernel)]
pub fn kernel_with_scalar(#[spirv(cross_workgroup)] buf: &mut u32, factor: u32) {
    *buf *= factor;
}

/// Kernel with optional threads() specification.
#[spirv(kernel(threads(64)))]
pub fn kernel_with_threads(#[spirv(global_invocation_id)] _id: USizeVec3) {}

/// Kernel with a mutable slice parameter (CrossWorkgroup).
/// Slices decompose into (data pointer, length) kernel arguments.
#[spirv(kernel)]
pub fn kernel_with_slice(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(cross_workgroup)] data: &mut [u32],
) {
    let index = id.x as usize;
    data[index] = data[index] * 2;
}

/// Kernel with an immutable slice parameter.
#[spirv(kernel)]
pub fn kernel_with_readonly_slice(
    #[spirv(cross_workgroup)] input: &[u32],
    #[spirv(cross_workgroup)] output: &mut u32,
) {
    *output = input[0];
}

/// Kernel with workgroup (shared/local) memory.
#[spirv(kernel(threads(64)))]
pub fn kernel_with_workgroup(
    #[spirv(global_invocation_id)] _id: USizeVec3,
    #[spirv(workgroup)] _shared: &mut [u32; 64],
) {
}
