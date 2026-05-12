// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// `cl::s!(v, lo|hi|even|odd)` — named-group swizzles. Returns scalar
// for source width 2; half-width vector for widths 4/8/16.

use spirv_std::cl::{Float2, Float4, Float8, Float16, UInt8, s};
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] v2: &Float2,
    #[spirv(cross_workgroup)] v4: &Float4,
    #[spirv(cross_workgroup)] v8: &Float8,
    #[spirv(cross_workgroup)] v16: &Float16,
    #[spirv(cross_workgroup)] u8v: &UInt8,
    #[spirv(cross_workgroup)] out_scalar: &mut f32,
    #[spirv(cross_workgroup)] out2: &mut Float2,
    #[spirv(cross_workgroup)] out4: &mut Float4,
    #[spirv(cross_workgroup)] out8: &mut Float8,
    #[spirv(cross_workgroup)] u4v: &mut spirv_std::cl::UInt4,
) {
    // Width 2 — scalar.
    *out_scalar = s!(*v2, lo);
    *out_scalar = s!(*v2, hi);
    *out_scalar = s!(*v2, even);
    *out_scalar = s!(*v2, odd);

    // Width 4 → Float2.
    *out2 = s!(*v4, lo);
    *out2 = s!(*v4, hi);
    *out2 = s!(*v4, even);
    *out2 = s!(*v4, odd);

    // Width 8 → Float4. Also chains naturally via repeated halving.
    *out4 = s!(*v8, lo);
    *out4 = s!(*v8, hi);
    *out2 = s!(s!(*v8, lo), even);

    // Width 16 → Float8.
    *out8 = s!(*v16, even);
    *out8 = s!(*v16, hi);

    // Integer family works the same.
    *u4v = s!(*u8v, lo);
}
