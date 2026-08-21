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

fn apply<F: FnOnce(u32) -> u32>(f: F, x: u32) -> u32 {
    f(x)
}

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] out: &mut u32, x: u32) {
    *out = apply(|v| v * 2, x);
}
