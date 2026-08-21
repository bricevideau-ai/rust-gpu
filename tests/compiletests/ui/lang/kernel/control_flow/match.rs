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
    *buf = match val {
        0 => 10,
        1 => 20,
        2 => 30,
        _ => 0,
    };
}
