// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

use spirv_std::spirv;

fn array_bounds_check(x: [u32; 4], i: usize) -> u32 {
    x[i]
}

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] out: &mut u32) {
    *out = array_bounds_check([10, 20, 30, 40], 2);
}
