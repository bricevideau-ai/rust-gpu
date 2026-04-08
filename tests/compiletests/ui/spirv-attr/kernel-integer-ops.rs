// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::spirv;

// Arithmetic operations.

#[spirv(kernel)]
pub fn test_add(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] b: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = *a + *b;
}

#[spirv(kernel)]
pub fn test_sub(#[spirv(cross_workgroup)] a: &u32, #[spirv(cross_workgroup)] out: &mut u32) {
    *out = *a - 1;
}

#[spirv(kernel)]
pub fn test_mul(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
    factor: u32,
) {
    *out = *a * factor;
}

#[spirv(kernel)]
pub fn test_div_mod(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] out_div: &mut u32,
    #[spirv(cross_workgroup)] out_mod: &mut u32,
    divisor: u32,
) {
    *out_div = *a / divisor;
    *out_mod = *a % divisor;
}

// Bitwise operations.

#[spirv(kernel)]
pub fn test_bitwise(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] b: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = (*a & *b) | (*a ^ *b);
}

#[spirv(kernel)]
pub fn test_shifts(
    #[spirv(cross_workgroup)] val: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
    shift: u32,
) {
    *out = (*val << shift) | (*val >> shift);
}

#[spirv(kernel)]
pub fn test_not(#[spirv(cross_workgroup)] val: &u32, #[spirv(cross_workgroup)] out: &mut u32) {
    *out = !*val;
}

// Comparison operations.

#[spirv(kernel)]
pub fn test_min_max(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] b: &u32,
    #[spirv(cross_workgroup)] out_min: &mut u32,
    #[spirv(cross_workgroup)] out_max: &mut u32,
) {
    *out_min = if *a < *b { *a } else { *b };
    *out_max = if *a > *b { *a } else { *b };
}

// Signed integer operations.

#[spirv(kernel)]
pub fn test_signed_arith(
    #[spirv(cross_workgroup)] a: &i32,
    #[spirv(cross_workgroup)] out: &mut i32,
) {
    *out = -*a;
}

#[spirv(kernel)]
pub fn test_signed_compare(
    #[spirv(cross_workgroup)] a: &i32,
    #[spirv(cross_workgroup)] b: &i32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = if *a < *b { 1 } else { 0 };
}

// Saturating and wrapping operations.

#[spirv(kernel)]
pub fn test_wrapping_add(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = a.wrapping_add(1);
}

#[spirv(kernel)]
pub fn test_wrapping_mul(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
    factor: u32,
) {
    *out = a.wrapping_mul(factor);
}

// Type conversions.

#[spirv(kernel)]
pub fn test_u32_to_u64(#[spirv(cross_workgroup)] a: &u32, #[spirv(cross_workgroup)] out: &mut u64) {
    *out = *a as u64;
}

#[spirv(kernel)]
pub fn test_u64_to_u32(#[spirv(cross_workgroup)] a: &u64, #[spirv(cross_workgroup)] out: &mut u32) {
    *out = *a as u32;
}

#[spirv(kernel)]
pub fn test_i32_to_u32(#[spirv(cross_workgroup)] a: &i32, #[spirv(cross_workgroup)] out: &mut u32) {
    *out = *a as u32;
}
