// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] buf: &mut u32, n: u32) {
    let mut sum = 0u32;
    for i in 0..n {
        for j in 0..n {
            sum += i * n + j;
        }
    }
    *buf = sum;
}
