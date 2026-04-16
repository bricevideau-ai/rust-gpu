// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::spirv;

fn apply<F: FnOnce(u32) -> u32>(f: F, x: u32) -> u32 {
    f(x)
}

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] out: &mut u32, x: u32, y: u32) {
    let sum = x + y;
    *out = apply(|v| v + sum, 10);
}
