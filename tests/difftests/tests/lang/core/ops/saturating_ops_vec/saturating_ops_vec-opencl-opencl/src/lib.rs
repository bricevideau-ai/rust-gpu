#![cfg_attr(target_arch = "spirv", no_std)]

use saturating_ops_vec_shared::{FUNCS, opencl_std_compute};
use spirv_std::glam::UVec3;
use spirv_std::spirv;

#[spirv(kernel(threads(64)))]
pub fn main_kernel(
    #[spirv(cross_workgroup)] input_a: &[i32],
    #[spirv(cross_workgroup)] input_b: &[i32],
    #[spirv(cross_workgroup)] output: &mut [u32],
    #[spirv(global_invocation_id)] global_id: UVec3,
) {
    let tid = global_id.x as usize;
    let threads = input_a.len() / 4;

    if tid >= threads {
        return;
    }

    let base = tid * 4;
    let a = [
        input_a[base],
        input_a[base + 1],
        input_a[base + 2],
        input_a[base + 3],
    ];
    let b = [
        input_b[base],
        input_b[base + 1],
        input_b[base + 2],
        input_b[base + 3],
    ];
    let out_base = tid * FUNCS;
    opencl_std_compute(a, b, &mut output[out_base..out_base + FUNCS]);
}
