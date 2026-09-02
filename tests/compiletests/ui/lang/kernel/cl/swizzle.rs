// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// `cl::s!` swizzle macro — letter form (xyzw, widths 1-4) and
// OpenCL `sN` form (hex digits, all widths up to 16). Single-index
// returns the scalar; multi-index returns the matching `cl::*N` type.

use spirv_std::cl::{Float2, Float3, Float4, Float8, Float16, s};
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] v4: &Float4,
    #[spirv(cross_workgroup)] v8: &Float8,
    #[spirv(cross_workgroup)] v16: &Float16,
    #[spirv(cross_workgroup)] out_scalar: &mut f32,
    #[spirv(cross_workgroup)] out2: &mut Float2,
    #[spirv(cross_workgroup)] out3: &mut Float3,
    #[spirv(cross_workgroup)] out4: &mut Float4,
    #[spirv(cross_workgroup)] out8: &mut Float8,
    #[spirv(cross_workgroup)] out16: &mut Float16,
) {
    // Letter form, widths 1-4.
    *out_scalar = s!(*v4, x);
    *out2 = s!(*v4, xy);
    *out3 = s!(*v4, zyx);
    *out4 = s!(*v4, wzyx);

    // sN form, all widths.
    *out2 = s!(*v8, s07);
    *out3 = s!(*v16, sFc8);
    *out4 = s!(*v8, s0123);
    *out8 = s!(*v16, sFEDCBA98);
    *out16 = s!(*v16, sFEDCBA9876543210);

    // Underscores allowed and ignored.
    *out4 = s!(*v16, s0_4_8_C);
}
