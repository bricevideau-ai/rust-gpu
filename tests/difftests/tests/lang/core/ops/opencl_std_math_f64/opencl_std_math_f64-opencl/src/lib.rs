#![cfg_attr(target_arch = "spirv", no_std)]

use opencl_std_math_f64_shared::{FUNCS, compute};
use spirv_std::glam::UVec3;
use spirv_std::spirv;

#[spirv(kernel(threads(64)))]
pub fn main_kernel(
    #[spirv(cross_workgroup)] input_a: &[f64],
    #[spirv(cross_workgroup)] input_b: &[f64],
    #[spirv(cross_workgroup)] input_c: &[f64],
    #[spirv(cross_workgroup)] output: &mut [f64],
    #[spirv(global_invocation_id)] global_id: UVec3,
) {
    let tid = global_id.x as usize;
    if tid >= input_a.len() {
        return;
    }
    let base = tid * FUNCS;
    compute(
        input_a[tid],
        input_b[tid],
        input_c[tid],
        &mut output[base..base + FUNCS],
    );
}
