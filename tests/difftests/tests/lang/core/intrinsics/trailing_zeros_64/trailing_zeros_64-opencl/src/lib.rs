#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::glam::UVec3;
use spirv_std::spirv;

#[spirv(kernel(threads(1)))]
pub fn main_kernel(
    #[spirv(cross_workgroup)] input: &[u64],
    #[spirv(cross_workgroup)] output: &mut [u32],
    #[spirv(global_invocation_id)] global_id: UVec3,
) {
    let tid = global_id.x as usize;

    if tid < input.len() && tid < output.len() {
        output[tid] = input[tid].trailing_zeros();
    }
}
