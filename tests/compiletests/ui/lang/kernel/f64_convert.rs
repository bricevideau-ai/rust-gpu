// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// compile-flags: -C target-feature=+Float64

// f64 conversions in kernel context: to/from f32, signed and unsigned
// 32-/64-bit integers.

use spirv_std::spirv;

#[spirv(kernel)]
pub fn as_f32(#[spirv(cross_workgroup)] a: &f64, #[spirv(cross_workgroup)] out: &mut f32) {
    *out = *a as f32;
}

#[spirv(kernel)]
pub fn as_u64(#[spirv(cross_workgroup)] a: &f64, #[spirv(cross_workgroup)] out: &mut u64) {
    *out = *a as u64;
}

#[spirv(kernel)]
pub fn from_f32(#[spirv(cross_workgroup)] a: &f32, #[spirv(cross_workgroup)] out: &mut f64) {
    *out = *a as f64;
}

#[spirv(kernel)]
pub fn from_i32(#[spirv(cross_workgroup)] a: &i32, #[spirv(cross_workgroup)] out: &mut f64) {
    *out = *a as f64;
}

#[spirv(kernel)]
pub fn from_u64(#[spirv(cross_workgroup)] a: &u64, #[spirv(cross_workgroup)] out: &mut f64) {
    *out = *a as f64;
}
