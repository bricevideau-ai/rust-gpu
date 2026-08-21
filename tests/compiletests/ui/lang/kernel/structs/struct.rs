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

#[derive(Clone, Copy)]
struct Pair {
    x: u32,
    y: u32,
}

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] out_x: &mut u32, #[spirv(cross_workgroup)] out_y: &mut u32) {
    let p = Pair { x: 10, y: 20 };
    *out_x = p.x;
    *out_y = p.y;
}
