// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// u64 arithmetic, bitwise, and shift operations in kernel context.
// Exercises 64-bit integer codegen — Int64 is mandatory in the OpenCL
// SPIR-V environment, so no extra cap directive is needed.

use spirv_std::spirv;

#[spirv(kernel)]
pub fn add(
    #[spirv(cross_workgroup)] a: &u64,
    #[spirv(cross_workgroup)] b: &u64,
    #[spirv(cross_workgroup)] out: &mut u64,
) {
    *out = *a + *b;
}

#[spirv(kernel)]
pub fn sub(#[spirv(cross_workgroup)] a: &u64, #[spirv(cross_workgroup)] out: &mut u64) {
    *out = *a - 1;
}

#[spirv(kernel)]
pub fn mul(
    #[spirv(cross_workgroup)] a: &u64,
    #[spirv(cross_workgroup)] out: &mut u64,
    factor: u64,
) {
    *out = *a * factor;
}

#[spirv(kernel)]
pub fn div(
    #[spirv(cross_workgroup)] a: &u64,
    #[spirv(cross_workgroup)] out: &mut u64,
    divisor: u64,
) {
    *out = *a / divisor;
}

#[spirv(kernel)]
pub fn rem(
    #[spirv(cross_workgroup)] a: &u64,
    #[spirv(cross_workgroup)] out: &mut u64,
    divisor: u64,
) {
    *out = *a % divisor;
}

#[spirv(kernel)]
pub fn and(
    #[spirv(cross_workgroup)] a: &u64,
    #[spirv(cross_workgroup)] b: &u64,
    #[spirv(cross_workgroup)] out: &mut u64,
) {
    *out = *a & *b;
}

#[spirv(kernel)]
pub fn or(
    #[spirv(cross_workgroup)] a: &u64,
    #[spirv(cross_workgroup)] b: &u64,
    #[spirv(cross_workgroup)] out: &mut u64,
) {
    *out = *a | *b;
}

#[spirv(kernel)]
pub fn xor(
    #[spirv(cross_workgroup)] a: &u64,
    #[spirv(cross_workgroup)] b: &u64,
    #[spirv(cross_workgroup)] out: &mut u64,
) {
    *out = *a ^ *b;
}

#[spirv(kernel)]
pub fn shl(
    #[spirv(cross_workgroup)] val: &u64,
    #[spirv(cross_workgroup)] out: &mut u64,
    shift: u32,
) {
    *out = *val << shift;
}

#[spirv(kernel)]
pub fn shr(
    #[spirv(cross_workgroup)] val: &u64,
    #[spirv(cross_workgroup)] out: &mut u64,
    shift: u32,
) {
    *out = *val >> shift;
}

#[spirv(kernel)]
pub fn min_max(
    #[spirv(cross_workgroup)] a: &u64,
    #[spirv(cross_workgroup)] b: &u64,
    #[spirv(cross_workgroup)] out_min: &mut u64,
    #[spirv(cross_workgroup)] out_max: &mut u64,
) {
    *out_min = if *a < *b { *a } else { *b };
    *out_max = if *a > *b { *a } else { *b };
}
