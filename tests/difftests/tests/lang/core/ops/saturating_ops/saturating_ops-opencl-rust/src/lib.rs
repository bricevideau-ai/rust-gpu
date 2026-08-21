#![cfg_attr(target_arch = "spirv", no_std)]

use saturating_ops_shared::{FUNCS, rust_compute};
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
    if tid >= input_a.len() {
        return;
    }
    let base = tid * FUNCS;
    rust_compute(input_a[tid], input_b[tid], &mut output[base..base + FUNCS]);
}
