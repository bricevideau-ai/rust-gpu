// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// compile-flags: -C target-feature=+Float64

// Test double-precision (f64) operations in kernel context.
// Critical for HPC workloads.

use spirv_std::spirv;

// Basic arithmetic.

#[spirv(kernel)]
pub fn test_f64_add(
    #[spirv(cross_workgroup)] a: &f64,
    #[spirv(cross_workgroup)] b: &f64,
    #[spirv(cross_workgroup)] out: &mut f64,
) {
    *out = *a + *b;
}

#[spirv(kernel)]
pub fn test_f64_sub(
    #[spirv(cross_workgroup)] a: &f64,
    #[spirv(cross_workgroup)] b: &f64,
    #[spirv(cross_workgroup)] out: &mut f64,
) {
    *out = *a - *b;
}

#[spirv(kernel)]
pub fn test_f64_mul(
    #[spirv(cross_workgroup)] a: &f64,
    #[spirv(cross_workgroup)] out: &mut f64,
    factor: f64,
) {
    *out = *a * factor;
}

#[spirv(kernel)]
pub fn test_f64_div(
    #[spirv(cross_workgroup)] a: &f64,
    #[spirv(cross_workgroup)] b: &f64,
    #[spirv(cross_workgroup)] out: &mut f64,
) {
    *out = *a / *b;
}

#[spirv(kernel)]
pub fn test_f64_neg(#[spirv(cross_workgroup)] a: &f64, #[spirv(cross_workgroup)] out: &mut f64) {
    *out = -*a;
}

// FMA pattern.

#[spirv(kernel)]
pub fn test_f64_fma(
    #[spirv(cross_workgroup)] a: &f64,
    #[spirv(cross_workgroup)] b: &f64,
    #[spirv(cross_workgroup)] c: &f64,
    #[spirv(cross_workgroup)] out: &mut f64,
) {
    *out = *a * *b + *c;
}

// Comparisons.

#[spirv(kernel)]
pub fn test_f64_min_max(
    #[spirv(cross_workgroup)] a: &f64,
    #[spirv(cross_workgroup)] b: &f64,
    #[spirv(cross_workgroup)] out_min: &mut f64,
    #[spirv(cross_workgroup)] out_max: &mut f64,
) {
    *out_min = if *a < *b { *a } else { *b };
    *out_max = if *a > *b { *a } else { *b };
}

// Conversions.

#[spirv(kernel)]
pub fn test_f64_from_f32(
    #[spirv(cross_workgroup)] a: &f32,
    #[spirv(cross_workgroup)] out: &mut f64,
) {
    *out = *a as f64;
}

#[spirv(kernel)]
pub fn test_f32_from_f64(
    #[spirv(cross_workgroup)] a: &f64,
    #[spirv(cross_workgroup)] out: &mut f32,
) {
    *out = *a as f32;
}

#[spirv(kernel)]
pub fn test_f64_from_i32(
    #[spirv(cross_workgroup)] a: &i32,
    #[spirv(cross_workgroup)] out: &mut f64,
) {
    *out = *a as f64;
}

#[spirv(kernel)]
pub fn test_f64_from_u64(
    #[spirv(cross_workgroup)] a: &u64,
    #[spirv(cross_workgroup)] out: &mut f64,
) {
    *out = *a as f64;
}

#[spirv(kernel)]
pub fn test_u64_from_f64(
    #[spirv(cross_workgroup)] a: &f64,
    #[spirv(cross_workgroup)] out: &mut u64,
) {
    *out = *a as u64;
}

// Slice operations with f64.

#[spirv(kernel)]
pub fn test_f64_slice_access(#[spirv(cross_workgroup)] data: &mut [f64], index: u32) {
    let i = index as usize;
    data[i] = data[i] * 2.0;
}

// Mixed precision: f64 accumulator with f32 inputs.

#[spirv(kernel)]
pub fn test_mixed_precision(
    #[spirv(cross_workgroup)] input: &[f32],
    #[spirv(cross_workgroup)] out: &mut f64,
) {
    let mut acc: f64 = 0.0;
    acc += input[0] as f64;
    acc += input[1] as f64;
    acc += input[2] as f64;
    acc += input[3] as f64;
    *out = acc;
}

// DAXPY: Double-precision A*X Plus Y — classic HPC kernel.

#[spirv(kernel)]
pub fn test_daxpy(
    #[spirv(global_invocation_id)] id: spirv_std::glam::U64Vec3,
    #[spirv(cross_workgroup)] x: &[f64],
    #[spirv(cross_workgroup)] y: &mut [f64],
    alpha: f64,
) {
    let i = id.x as usize;
    y[i] = alpha * x[i] + y[i];
}
