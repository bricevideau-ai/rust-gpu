// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::spirv;

// Basic struct operations.

#[derive(Clone, Copy)]
struct Pair {
    x: u32,
    y: u32,
}

#[spirv(kernel)]
pub fn test_struct(
    #[spirv(cross_workgroup)] out_x: &mut u32,
    #[spirv(cross_workgroup)] out_y: &mut u32,
) {
    let p = Pair { x: 10, y: 20 };
    *out_x = p.x;
    *out_y = p.y;
}

#[spirv(kernel)]
pub fn test_struct_param(#[spirv(cross_workgroup)] out: &mut u32, x: u32, y: u32) {
    let p = Pair { x, y };
    *out = p.x + p.y;
}

// Nested structs.

#[derive(Clone, Copy)]
struct Nested {
    inner: Pair,
    z: u32,
}

#[spirv(kernel)]
pub fn test_nested_struct(#[spirv(cross_workgroup)] out: &mut u32) {
    let n = Nested {
        inner: Pair { x: 1, y: 2 },
        z: 3,
    };
    *out = n.inner.x + n.inner.y + n.z;
}

// Tuple operations.

#[spirv(kernel)]
pub fn test_tuple(#[spirv(cross_workgroup)] out: &mut u32) {
    let t: (u32, u32, u32) = (10, 20, 30);
    *out = t.0 + t.1 + t.2;
}

// Option<T> — this tests the MemberDecorate Offset skip for Kernel.

fn maybe_double(x: u32) -> Option<u32> {
    if x == 0 { None } else { Some(x * 2) }
}

#[spirv(kernel)]
pub fn test_option(#[spirv(cross_workgroup)] input: &u32, #[spirv(cross_workgroup)] out: &mut u32) {
    *out = maybe_double(*input).unwrap_or(u32::MAX);
}

#[spirv(kernel)]
pub fn test_option_map(
    #[spirv(cross_workgroup)] input: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = maybe_double(*input).map(|v| v + 1).unwrap_or(0);
}

// Enum operations (use #[repr(u32)] to avoid u8 discriminant requiring Int8).

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
pub fn test_enum(#[spirv(cross_workgroup)] out: &mut u32, which: u32) {
    let c = match which {
        0 => Color::Red,
        1 => Color::Green,
        _ => Color::Blue,
    };
    *out = color_value(c);
}

// Function calls with structs.

fn swap(p: Pair) -> Pair {
    Pair { x: p.y, y: p.x }
}

#[spirv(kernel)]
pub fn test_struct_fn(
    #[spirv(cross_workgroup)] out_x: &mut u32,
    #[spirv(cross_workgroup)] out_y: &mut u32,
) {
    let p = swap(Pair { x: 42, y: 99 });
    *out_x = p.x;
    *out_y = p.y;
}
