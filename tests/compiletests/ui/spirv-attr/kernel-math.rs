// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::spirv;

// Float arithmetic.

#[spirv(kernel)]
pub fn test_f32_add(
    #[spirv(cross_workgroup)] a: &f32,
    #[spirv(cross_workgroup)] b: &f32,
    #[spirv(cross_workgroup)] out: &mut f32,
) {
    *out = *a + *b;
}

#[spirv(kernel)]
pub fn test_f32_mul(
    #[spirv(cross_workgroup)] a: &f32,
    #[spirv(cross_workgroup)] out: &mut f32,
    factor: f32,
) {
    *out = *a * factor;
}

#[spirv(kernel)]
pub fn test_f32_neg(#[spirv(cross_workgroup)] a: &f32, #[spirv(cross_workgroup)] out: &mut f32) {
    *out = -*a;
}

// Float conversions.

#[spirv(kernel)]
pub fn test_f32_to_u32(#[spirv(cross_workgroup)] a: &f32, #[spirv(cross_workgroup)] out: &mut u32) {
    *out = *a as u32;
}

#[spirv(kernel)]
pub fn test_u32_to_f32(#[spirv(cross_workgroup)] a: &u32, #[spirv(cross_workgroup)] out: &mut f32) {
    *out = *a as f32;
}

#[spirv(kernel)]
pub fn test_f32_to_i32(#[spirv(cross_workgroup)] a: &f32, #[spirv(cross_workgroup)] out: &mut i32) {
    *out = *a as i32;
}

// Float comparisons.

#[spirv(kernel)]
pub fn test_f32_min_max(
    #[spirv(cross_workgroup)] a: &f32,
    #[spirv(cross_workgroup)] b: &f32,
    #[spirv(cross_workgroup)] out_min: &mut f32,
    #[spirv(cross_workgroup)] out_max: &mut f32,
) {
    *out_min = if *a < *b { *a } else { *b };
    *out_max = if *a > *b { *a } else { *b };
}

// Mixed int/float operations.

#[spirv(kernel)]
pub fn test_fma_manual(
    #[spirv(cross_workgroup)] a: &f32,
    #[spirv(cross_workgroup)] b: &f32,
    #[spirv(cross_workgroup)] c: &f32,
    #[spirv(cross_workgroup)] out: &mut f32,
) {
    *out = *a * *b + *c;
}

// Boolean operations.

#[spirv(kernel)]
pub fn test_bool_logic(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] b: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    let x = *a != 0;
    let y = *b != 0;
    *out = if x && y {
        3
    } else if x || y {
        2
    } else if !x {
        1
    } else {
        0
    };
}
