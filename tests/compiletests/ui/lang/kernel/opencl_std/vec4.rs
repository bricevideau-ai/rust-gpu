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
use spirv_std::glam::Vec4;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] a: &Vec4, #[spirv(cross_workgroup)] out: &mut Vec4) {
    *out = ocl::native_sqrt(*a) + ocl::fabs(*a) + ocl::sign(*a);
}
