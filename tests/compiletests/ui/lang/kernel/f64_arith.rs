// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// compile-flags: -C target-feature=+Float64

// f64 arithmetic and comparison in kernel context. Float64 is opt-in
// per the OpenCL SPIR-V environment.

use spirv_std::spirv;

#[spirv(kernel)]
pub fn add(
    #[spirv(cross_workgroup)] a: &f64,
    #[spirv(cross_workgroup)] b: &f64,
    #[spirv(cross_workgroup)] out: &mut f64,
) {
    *out = *a + *b;
}

#[spirv(kernel)]
pub fn sub(
    #[spirv(cross_workgroup)] a: &f64,
    #[spirv(cross_workgroup)] b: &f64,
    #[spirv(cross_workgroup)] out: &mut f64,
) {
    *out = *a - *b;
}

#[spirv(kernel)]
pub fn mul(
    #[spirv(cross_workgroup)] a: &f64,
    #[spirv(cross_workgroup)] out: &mut f64,
    factor: f64,
) {
    *out = *a * factor;
}

#[spirv(kernel)]
pub fn div(
    #[spirv(cross_workgroup)] a: &f64,
    #[spirv(cross_workgroup)] b: &f64,
    #[spirv(cross_workgroup)] out: &mut f64,
) {
    *out = *a / *b;
}

#[spirv(kernel)]
pub fn neg(#[spirv(cross_workgroup)] a: &f64, #[spirv(cross_workgroup)] out: &mut f64) {
    *out = -*a;
}

#[spirv(kernel)]
pub fn min_max(
    #[spirv(cross_workgroup)] a: &f64,
    #[spirv(cross_workgroup)] b: &f64,
    #[spirv(cross_workgroup)] out_min: &mut f64,
    #[spirv(cross_workgroup)] out_max: &mut f64,
) {
    *out_min = if *a < *b { *a } else { *b };
    *out_max = if *a > *b { *a } else { *b };
}

#[spirv(kernel)]
pub fn fma(
    #[spirv(cross_workgroup)] a: &f64,
    #[spirv(cross_workgroup)] b: &f64,
    #[spirv(cross_workgroup)] c: &f64,
    #[spirv(cross_workgroup)] out: &mut f64,
) {
    *out = *a * *b + *c;
}
