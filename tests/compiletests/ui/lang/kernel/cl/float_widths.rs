// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// Exercises every width of `spirv_std::cl::FloatN` to confirm widths
// 8 and 16 codegen — they require the SPIR-V `Vector16` capability and
// the kernel-only count widening in `abi.rs`.

use spirv_std::cl::{Float2, Float3, Float4, Float8, Float16};
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] a2: &Float2,
    #[spirv(cross_workgroup)] a3: &Float3,
    #[spirv(cross_workgroup)] a4: &Float4,
    #[spirv(cross_workgroup)] a8: &Float8,
    #[spirv(cross_workgroup)] a16: &Float16,
    #[spirv(cross_workgroup)] out2: &mut Float2,
    #[spirv(cross_workgroup)] out3: &mut Float3,
    #[spirv(cross_workgroup)] out4: &mut Float4,
    #[spirv(cross_workgroup)] out8: &mut Float8,
    #[spirv(cross_workgroup)] out16: &mut Float16,
) {
    *out2 = *a2 + Float2::splat(1.0);
    *out3 = *a3 - Float3::splat(1.0);
    *out4 = *a4 * Float4::splat(2.0);
    *out8 = *a8 / Float8::splat(2.0);
    *out16 = *a16 + *a16;
}
