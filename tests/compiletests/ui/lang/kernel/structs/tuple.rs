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
pub fn main(#[spirv(cross_workgroup)] out: &mut u32) {
    let t: (u32, u32, u32) = (10, 20, 30);
    *out = t.0 + t.1 + t.2;
}
