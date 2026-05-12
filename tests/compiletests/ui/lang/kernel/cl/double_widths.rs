// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// compile-flags: -C target-feature=+Float64

// `spirv_std::cl::DoubleN` for every supported width — `Float64` is
// opt-in per the OpenCL SPIR-V environment; `Vector16` is auto-enabled
// on Kernel targets.

use spirv_std::cl::{Double2, Double3, Double4, Double8, Double16};
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] a2: &Double2,
    #[spirv(cross_workgroup)] a3: &Double3,
    #[spirv(cross_workgroup)] a4: &Double4,
    #[spirv(cross_workgroup)] a8: &Double8,
    #[spirv(cross_workgroup)] a16: &Double16,
    #[spirv(cross_workgroup)] out2: &mut Double2,
    #[spirv(cross_workgroup)] out3: &mut Double3,
    #[spirv(cross_workgroup)] out4: &mut Double4,
    #[spirv(cross_workgroup)] out8: &mut Double8,
    #[spirv(cross_workgroup)] out16: &mut Double16,
) {
    *out2 = *a2 + Double2::splat(1.0);
    *out3 = *a3 - Double3::splat(1.0);
    *out4 = *a4 * Double4::splat(2.0);
    *out8 = *a8 / Double8::splat(2.0);
    *out16 = *a16 + *a16;
}
