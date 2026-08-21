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
pub fn main(#[spirv(cross_workgroup)] buf: &mut u32, val: u32) {
    if val > 100 {
        *buf = 3;
    } else if val > 10 {
        *buf = 2;
    } else if val > 0 {
        *buf = 1;
    } else {
        *buf = 0;
    }
}
