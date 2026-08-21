// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// u32 -> {u64, i32, f32} width and type conversions in kernel context.

use spirv_std::spirv;

#[spirv(kernel)]
pub fn as_u64(#[spirv(cross_workgroup)] a: &u32, #[spirv(cross_workgroup)] out: &mut u64) {
    *out = *a as u64;
}

#[spirv(kernel)]
pub fn as_i32(#[spirv(cross_workgroup)] a: &u32, #[spirv(cross_workgroup)] out: &mut i32) {
    *out = *a as i32;
}

#[spirv(kernel)]
pub fn as_f32(#[spirv(cross_workgroup)] a: &u32, #[spirv(cross_workgroup)] out: &mut f32) {
    *out = *a as f32;
}
