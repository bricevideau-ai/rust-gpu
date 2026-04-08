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

#[derive(Clone, Copy)]
struct Nested {
    inner: Pair,
    z: u32,
}

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] out: &mut u32) {
    let n = Nested {
        inner: Pair { x: 1, y: 2 },
        z: 3,
    };
    *out = n.inner.x + n.inner.y + n.z;
}
