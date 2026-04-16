// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// u64 -> {u32, i64} narrowing/sign-bit conversions in kernel context.

use spirv_std::spirv;

#[spirv(kernel)]
pub fn as_u32(#[spirv(cross_workgroup)] a: &u64, #[spirv(cross_workgroup)] out: &mut u32) {
    *out = *a as u32;
}

#[spirv(kernel)]
pub fn as_i64(#[spirv(cross_workgroup)] a: &u64, #[spirv(cross_workgroup)] out: &mut i64) {
    *out = *a as i64;
}
