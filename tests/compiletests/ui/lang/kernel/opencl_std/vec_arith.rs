// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// `opencl_std::add`/`sub`/`mul`/`div` — core SPIR-V `OpFAdd`/`OpFSub`/
// `OpFMul`/`OpFDiv` exposed as functions for callers who want
// guaranteed-vector codegen instead of relying on the auto-vectoriser
// to refuse glam's per-component scalarisation. Works on scalar or
// vector float operands.

use spirv_std::arch::opencl_std as ocl;
use spirv_std::glam::{Vec3, Vec4};
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] a3: &Vec3,
    #[spirv(cross_workgroup)] b3: &Vec3,
    #[spirv(cross_workgroup)] a4: &Vec4,
    #[spirv(cross_workgroup)] b4: &Vec4,
    #[spirv(cross_workgroup)] s: &f32,
    #[spirv(cross_workgroup)] out_v3: &mut Vec3,
    #[spirv(cross_workgroup)] out_v4: &mut Vec4,
    #[spirv(cross_workgroup)] out_s: &mut f32,
) {
    *out_v3 = ocl::div(ocl::mul(ocl::add(*a3, *b3), ocl::sub(*a3, *b3)), *a3);
    *out_v4 = ocl::add(ocl::sub(*a4, *b4), ocl::mul(*a4, *b4));
    *out_s = ocl::add(*s, ocl::div(*s, *s));
}
