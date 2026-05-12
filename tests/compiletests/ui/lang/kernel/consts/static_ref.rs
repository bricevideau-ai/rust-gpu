// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// `&'static u32` becomes a real global; on Kernel targets it lives in
// UniformConstant storage. (`const SCALAR: u32 = ...` const-folds to a
// literal and doesn't exercise this path.)

use spirv_std::spirv;

#[inline(never)]
fn scalar_load(r: &'static u32) -> u32 {
    *r
}

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] out: &mut u32) {
    *out = scalar_load(&123);
}
