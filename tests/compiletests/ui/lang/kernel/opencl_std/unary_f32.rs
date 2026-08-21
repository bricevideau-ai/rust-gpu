// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

use spirv_std::arch::opencl_std as ocl;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] a: &f32, #[spirv(cross_workgroup)] out: &mut f32) {
    *out = ocl::sqrt(*a)
        + ocl::rsqrt(*a)
        + ocl::sin(*a)
        + ocl::cos(*a)
        + ocl::tan(*a)
        + ocl::asin(*a)
        + ocl::acos(*a)
        + ocl::atan(*a)
        + ocl::sinh(*a)
        + ocl::cosh(*a)
        + ocl::tanh(*a)
        + ocl::exp(*a)
        + ocl::exp2(*a)
        + ocl::log(*a)
        + ocl::log2(*a)
        + ocl::cbrt(*a)
        + ocl::ceil(*a)
        + ocl::floor(*a)
        + ocl::round(*a)
        + ocl::trunc(*a)
        + ocl::fabs(*a)
        + ocl::sign(*a);
}
