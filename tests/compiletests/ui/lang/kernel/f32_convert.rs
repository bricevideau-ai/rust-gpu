// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// f32 -> {u32, i32} and u32 -> f32 conversions in kernel context.

use spirv_std::spirv;

#[spirv(kernel)]
pub fn as_u32(#[spirv(cross_workgroup)] a: &f32, #[spirv(cross_workgroup)] out: &mut u32) {
    *out = *a as u32;
}

#[spirv(kernel)]
pub fn as_i32(#[spirv(cross_workgroup)] a: &f32, #[spirv(cross_workgroup)] out: &mut i32) {
    *out = *a as i32;
}

#[spirv(kernel)]
pub fn from_u32(#[spirv(cross_workgroup)] a: &u32, #[spirv(cross_workgroup)] out: &mut f32) {
    *out = *a as f32;
}
