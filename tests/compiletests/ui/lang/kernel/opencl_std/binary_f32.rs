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
pub fn main(
    #[spirv(cross_workgroup)] a: &f32,
    #[spirv(cross_workgroup)] b: &f32,
    #[spirv(cross_workgroup)] out: &mut f32,
) {
    *out = ocl::pow(*a, *b)
        + ocl::atan2(*a, *b)
        + ocl::fmin(*a, *b)
        + ocl::fmax(*a, *b)
        + ocl::fmod(*a, *b)
        + ocl::hypot(*a, *b)
        + ocl::copysign(*a, *b);
}
