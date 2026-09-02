#![cfg_attr(target_arch = "spirv", no_std)]

use bitwise_ops_opencl_shared::compute;
use spirv_std::glam::UVec3;
use spirv_std::spirv;

#[spirv(kernel(threads(64)))]
pub fn main_kernel(
    #[spirv(cross_workgroup)] input_a: &[u32],
    #[spirv(cross_workgroup)] input_b: &[u32],
    #[spirv(cross_workgroup)] output: &mut [u32],
    #[spirv(global_invocation_id)] global_id: UVec3,
) {
    let tid = global_id.x as usize;
    if tid < input_a.len() && tid < input_b.len() && tid < output.len() {
        output[tid] = compute(tid, input_a[tid], input_b[tid]);
    }
}

#[cfg(target_arch = "spirv")]
fn main() {}
