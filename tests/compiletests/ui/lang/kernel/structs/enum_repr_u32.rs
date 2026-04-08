// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// `#[repr(u32)]` to avoid u8 discriminant requiring Int8.

use spirv_std::spirv;

#[derive(Clone, Copy)]
#[repr(u32)]
enum Color {
    Red = 0,
    Green = 1,
    Blue = 2,
}

fn color_value(c: Color) -> u32 {
    match c {
        Color::Red => 0xFF0000,
        Color::Green => 0x00FF00,
        Color::Blue => 0x0000FF,
    }
}

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] out: &mut u32, which: u32) {
    let c = match which {
        0 => Color::Red,
        1 => Color::Green,
        _ => Color::Blue,
    };
    *out = color_value(c);
}
