// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::spirv;

fn sum_array(a: [u32; 4]) -> u32 {
    a[0] + a[1] + a[2] + a[3]
}

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] out: &mut u32) {
    *out = sum_array([10, 20, 30, 40]);
}
