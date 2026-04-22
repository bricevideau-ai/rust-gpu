// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// Multi-output ops — each returns a tuple. fract / modf / frexp / sincos.

use spirv_std::arch::opencl_std as ocl;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] x: &f32,
    #[spirv(cross_workgroup)] out_frac: &mut f32,
    #[spirv(cross_workgroup)] out_int: &mut f32,
    #[spirv(cross_workgroup)] out_mantissa: &mut f32,
    #[spirv(cross_workgroup)] out_exp: &mut i32,
    #[spirv(cross_workgroup)] out_sin: &mut f32,
    #[spirv(cross_workgroup)] out_cos: &mut f32,
) {
    let (frac, ipart) = ocl::fract(*x);
    *out_frac = frac;
    *out_int = ipart;

    let (_modf_frac, _modf_int) = ocl::modf(*x);

    let (mantissa, exp) = ocl::frexp(*x);
    *out_mantissa = mantissa;
    *out_exp = exp;

    let (s, c) = ocl::sincos(*x);
    *out_sin = s;
    *out_cos = c;
}
