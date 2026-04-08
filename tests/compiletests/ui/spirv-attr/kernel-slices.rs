// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::spirv;

// Safe slice indexing.

#[spirv(kernel)]
pub fn test_safe_read(
    #[spirv(cross_workgroup)] data: &[u32],
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = data[0];
}

#[spirv(kernel)]
pub fn test_safe_write(#[spirv(cross_workgroup)] data: &mut [u32]) {
    data[0] = 42;
}

#[spirv(kernel)]
pub fn test_safe_read_write(#[spirv(cross_workgroup)] data: &mut [u32]) {
    data[0] = data[0] * 2;
}

// Slice length.

#[spirv(kernel)]
pub fn test_len(#[spirv(cross_workgroup)] data: &[u32], #[spirv(cross_workgroup)] out: &mut u32) {
    *out = data.len() as u32;
}

// Multiple slice parameters.

#[spirv(kernel)]
pub fn test_copy_slice(
    #[spirv(cross_workgroup)] src: &[u32],
    #[spirv(cross_workgroup)] dst: &mut [u32],
) {
    dst[0] = src[0];
}

// Dynamic indexing with builtin.

#[spirv(kernel)]
pub fn test_dynamic_index(
    #[spirv(global_invocation_id)] id: spirv_std::glam::UVec3,
    #[spirv(cross_workgroup)] data: &mut [u32],
) {
    let index = id.x as usize;
    data[index] = data[index] + 1;
}

// Slice with scalar parameter.

#[spirv(kernel)]
pub fn test_fill(#[spirv(cross_workgroup)] data: &mut [u32], value: u32) {
    data[0] = value;
}
