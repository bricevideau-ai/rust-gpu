// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

use spirv_std::arch::opencl_std;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn test_erf(#[spirv(cross_workgroup)] x: &f32, #[spirv(cross_workgroup)] out: &mut f32) {
    *out = opencl_std::erf(*x);
}

#[spirv(kernel)]
pub fn test_erfc(#[spirv(cross_workgroup)] x: &f32, #[spirv(cross_workgroup)] out: &mut f32) {
    *out = opencl_std::erfc(*x);
}

#[spirv(kernel)]
pub fn test_fdim(
    #[spirv(cross_workgroup)] x: &f32,
    #[spirv(cross_workgroup)] y: &f32,
    #[spirv(cross_workgroup)] out: &mut f32,
) {
    *out = opencl_std::fdim(*x, *y);
}

#[spirv(kernel)]
pub fn test_hypot(
    #[spirv(cross_workgroup)] x: &f32,
    #[spirv(cross_workgroup)] y: &f32,
    #[spirv(cross_workgroup)] out: &mut f32,
) {
    *out = opencl_std::hypot(*x, *y);
}

#[spirv(kernel)]
pub fn test_ilogb(#[spirv(cross_workgroup)] x: &f32, #[spirv(cross_workgroup)] out: &mut i32) {
    *out = opencl_std::ilogb(*x);
}

#[spirv(kernel)]
pub fn test_lgamma(#[spirv(cross_workgroup)] x: &f32, #[spirv(cross_workgroup)] out: &mut f32) {
    *out = opencl_std::lgamma(*x);
}

#[spirv(kernel)]
pub fn test_lgamma_r(
    #[spirv(cross_workgroup)] x: &f32,
    #[spirv(cross_workgroup)] out_val: &mut f32,
    #[spirv(cross_workgroup)] out_sign: &mut i32,
) {
    let (val, sign) = opencl_std::lgamma_r(*x);
    *out_val = val;
    *out_sign = sign;
}

#[spirv(kernel)]
pub fn test_tgamma(#[spirv(cross_workgroup)] x: &f32, #[spirv(cross_workgroup)] out: &mut f32) {
    *out = opencl_std::tgamma(*x);
}

#[spirv(kernel)]
pub fn test_nextafter(
    #[spirv(cross_workgroup)] x: &f32,
    #[spirv(cross_workgroup)] y: &f32,
    #[spirv(cross_workgroup)] out: &mut f32,
) {
    *out = opencl_std::nextafter(*x, *y);
}

#[spirv(kernel)]
pub fn test_remainder(
    #[spirv(cross_workgroup)] x: &f32,
    #[spirv(cross_workgroup)] y: &f32,
    #[spirv(cross_workgroup)] out: &mut f32,
) {
    *out = opencl_std::remainder(*x, *y);
}

#[spirv(kernel)]
pub fn test_remquo(
    #[spirv(cross_workgroup)] x: &f32,
    #[spirv(cross_workgroup)] y: &f32,
    #[spirv(cross_workgroup)] out_rem: &mut f32,
    #[spirv(cross_workgroup)] out_quo: &mut i32,
) {
    let (rem, quo) = opencl_std::remquo(*x, *y);
    *out_rem = rem;
    *out_quo = quo;
}

#[spirv(kernel)]
pub fn test_ldexp(
    #[spirv(cross_workgroup)] x: &f32,
    #[spirv(cross_workgroup)] n: &i32,
    #[spirv(cross_workgroup)] out: &mut f32,
) {
    *out = opencl_std::ldexp(*x, *n);
}

#[spirv(kernel)]
pub fn test_s_add_sat(
    #[spirv(cross_workgroup)] a: &i32,
    #[spirv(cross_workgroup)] b: &i32,
    #[spirv(cross_workgroup)] out: &mut i32,
) {
    *out = opencl_std::s_add_sat(*a, *b);
}

#[spirv(kernel)]
pub fn test_u_add_sat(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] b: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = opencl_std::u_add_sat(*a, *b);
}

#[spirv(kernel)]
pub fn test_s_sub_sat(
    #[spirv(cross_workgroup)] a: &i32,
    #[spirv(cross_workgroup)] b: &i32,
    #[spirv(cross_workgroup)] out: &mut i32,
) {
    *out = opencl_std::s_sub_sat(*a, *b);
}

#[spirv(kernel)]
pub fn test_u_sub_sat(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] b: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = opencl_std::u_sub_sat(*a, *b);
}
