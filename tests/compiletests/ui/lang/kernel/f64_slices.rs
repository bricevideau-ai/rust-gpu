// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6
// compile-flags: -C target-feature=+Float64

// f64 kernels exercising slice support (depends on commit 5's slice +
// OpPtrAccessChain). Three HPC-style patterns: simple slice access,
// mixed-precision accumulation (f32 -> f64), and DAXPY.

use spirv_std::glam::U64Vec3;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn slice_access(#[spirv(cross_workgroup)] data: &mut [f64], index: u32) {
    let i = index as usize;
    data[i] = data[i] * 2.0;
}

#[spirv(kernel)]
pub fn mixed_precision(
    #[spirv(cross_workgroup)] input: &[f32],
    #[spirv(cross_workgroup)] out: &mut f64,
) {
    let mut acc: f64 = 0.0;
    acc += input[0] as f64;
    acc += input[1] as f64;
    acc += input[2] as f64;
    acc += input[3] as f64;
    *out = acc;
}

// DAXPY: Double-precision A*X Plus Y — classic HPC kernel.
#[spirv(kernel)]
pub fn daxpy(
    #[spirv(global_invocation_id)] id: U64Vec3,
    #[spirv(cross_workgroup)] x: &[f64],
    #[spirv(cross_workgroup)] y: &mut [f64],
    alpha: f64,
) {
    let i = id.x as usize;
    y[i] = alpha * x[i] + y[i];
}
