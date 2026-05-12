#![cfg_attr(target_arch = "spirv", no_std)]

use abi_vector_layout_opencl_cpu::layout::eval_cl_layouts;
use spirv_std::glam::UVec3;
use spirv_std::spirv;

#[spirv(kernel(threads(1)))]
pub fn main_kernel(
    #[spirv(cross_workgroup)] output: &mut [u32],
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    eval_cl_layouts(gid.x, output);
}
