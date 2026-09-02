// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// f32 arithmetic and comparison in kernel context.

use spirv_std::spirv;

#[spirv(kernel)]
pub fn add(
    #[spirv(cross_workgroup)] a: &f32,
    #[spirv(cross_workgroup)] b: &f32,
    #[spirv(cross_workgroup)] out: &mut f32,
) {
    *out = *a + *b;
}

#[spirv(kernel)]
pub fn mul(
    #[spirv(cross_workgroup)] a: &f32,
    #[spirv(cross_workgroup)] out: &mut f32,
    factor: f32,
) {
    *out = *a * factor;
}

#[spirv(kernel)]
pub fn neg(#[spirv(cross_workgroup)] a: &f32, #[spirv(cross_workgroup)] out: &mut f32) {
    *out = -*a;
}

#[spirv(kernel)]
pub fn min_max(
    #[spirv(cross_workgroup)] a: &f32,
    #[spirv(cross_workgroup)] b: &f32,
    #[spirv(cross_workgroup)] out_min: &mut f32,
    #[spirv(cross_workgroup)] out_max: &mut f32,
) {
    *out_min = if *a < *b { *a } else { *b };
    *out_max = if *a > *b { *a } else { *b };
}

#[spirv(kernel)]
pub fn fma(
    #[spirv(cross_workgroup)] a: &f32,
    #[spirv(cross_workgroup)] b: &f32,
    #[spirv(cross_workgroup)] c: &f32,
    #[spirv(cross_workgroup)] out: &mut f32,
) {
    *out = *a * *b + *c;
}
