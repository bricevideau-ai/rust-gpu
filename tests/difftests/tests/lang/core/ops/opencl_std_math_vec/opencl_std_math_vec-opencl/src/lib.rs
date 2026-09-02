#![cfg_attr(target_arch = "spirv", no_std)]

use opencl_std_math_vec_shared::{FUNCS, compute};
use spirv_std::cl::Float4;
use spirv_std::glam::UVec3;
use spirv_std::spirv;

#[spirv(kernel(threads(64)))]
pub fn main_kernel(
    #[spirv(cross_workgroup)] input_a: &[f32],
    #[spirv(cross_workgroup)] input_b: &[f32],
    #[spirv(cross_workgroup)] input_c: &[f32],
    #[spirv(cross_workgroup)] output: &mut [f32],
    #[spirv(global_invocation_id)] global_id: UVec3,
) {
    let tid = global_id.x as usize;
    let threads = input_a.len() / 4;

    if tid >= threads {
        return;
    }

    let base = tid * 4;
    let va = Float4::new(
        input_a[base],
        input_a[base + 1],
        input_a[base + 2],
        input_a[base + 3],
    );
    let vb = Float4::new(
        input_b[base],
        input_b[base + 1],
        input_b[base + 2],
        input_b[base + 3],
    );
    let vc = Float4::new(
        input_c[base],
        input_c[base + 1],
        input_c[base + 2],
        input_c[base + 3],
    );

    let out_base = tid * 4 * FUNCS;
    compute(va, vb, vc, &mut output[out_base..out_base + 4 * FUNCS]);
}
