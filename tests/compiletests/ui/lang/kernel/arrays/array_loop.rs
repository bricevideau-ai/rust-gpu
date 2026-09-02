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

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] out: &mut u32) {
    let array = [1u32, 2, 3, 4, 5];
    let mut sum = 0u32;
    for i in 0..5 {
        sum += array[i];
    }
    *out = sum;
}
