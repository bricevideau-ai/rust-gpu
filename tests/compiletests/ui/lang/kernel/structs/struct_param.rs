// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::spirv;

#[derive(Clone, Copy)]
struct Pair {
    x: u32,
    y: u32,
}

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] out: &mut u32, x: u32, y: u32) {
    let p = Pair { x, y };
    *out = p.x + p.y;
}
