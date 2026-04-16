// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// i32 -> {u32, f32} conversions in kernel context.

use spirv_std::spirv;

#[spirv(kernel)]
pub fn as_u32(#[spirv(cross_workgroup)] a: &i32, #[spirv(cross_workgroup)] out: &mut u32) {
    *out = *a as u32;
}

#[spirv(kernel)]
pub fn as_f32(#[spirv(cross_workgroup)] a: &i32, #[spirv(cross_workgroup)] out: &mut f32) {
    *out = *a as f32;
}
