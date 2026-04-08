// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// Pass a struct by value through a function call.

use spirv_std::spirv;

#[derive(Clone, Copy)]
struct Pair {
    x: u32,
    y: u32,
}

fn swap(p: Pair) -> Pair {
    Pair { x: p.y, y: p.x }
}

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] out_x: &mut u32, #[spirv(cross_workgroup)] out_y: &mut u32) {
    let p = swap(Pair { x: 42, y: 99 });
    *out_x = p.x;
    *out_y = p.y;
}
