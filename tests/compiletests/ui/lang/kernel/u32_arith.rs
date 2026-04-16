// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// u32 arithmetic, bitwise, shift, comparison, and wrapping operations
// in kernel context.

use spirv_std::spirv;

#[spirv(kernel)]
pub fn add(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] b: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = *a + *b;
}

#[spirv(kernel)]
pub fn sub(#[spirv(cross_workgroup)] a: &u32, #[spirv(cross_workgroup)] out: &mut u32) {
    *out = *a - 1;
}

#[spirv(kernel)]
pub fn mul(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
    factor: u32,
) {
    *out = *a * factor;
}

#[spirv(kernel)]
pub fn div(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
    divisor: u32,
) {
    *out = *a / divisor;
}

#[spirv(kernel)]
pub fn rem(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
    divisor: u32,
) {
    *out = *a % divisor;
}

#[spirv(kernel)]
pub fn and(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] b: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = *a & *b;
}

#[spirv(kernel)]
pub fn or(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] b: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = *a | *b;
}

#[spirv(kernel)]
pub fn xor(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] b: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = *a ^ *b;
}

#[spirv(kernel)]
pub fn not(#[spirv(cross_workgroup)] val: &u32, #[spirv(cross_workgroup)] out: &mut u32) {
    *out = !*val;
}

#[spirv(kernel)]
pub fn shl(
    #[spirv(cross_workgroup)] val: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
    shift: u32,
) {
    *out = *val << shift;
}

#[spirv(kernel)]
pub fn shr(
    #[spirv(cross_workgroup)] val: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
    shift: u32,
) {
    *out = *val >> shift;
}

#[spirv(kernel)]
pub fn min_max(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] b: &u32,
    #[spirv(cross_workgroup)] out_min: &mut u32,
    #[spirv(cross_workgroup)] out_max: &mut u32,
) {
    *out_min = if *a < *b { *a } else { *b };
    *out_max = if *a > *b { *a } else { *b };
}

#[spirv(kernel)]
pub fn wrapping_add(#[spirv(cross_workgroup)] a: &u32, #[spirv(cross_workgroup)] out: &mut u32) {
    *out = a.wrapping_add(1);
}

#[spirv(kernel)]
pub fn wrapping_mul(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
    factor: u32,
) {
    *out = a.wrapping_mul(factor);
}
