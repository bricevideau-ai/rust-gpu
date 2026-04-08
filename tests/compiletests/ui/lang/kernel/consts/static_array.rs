// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::spirv;

static LOOKUP: [u32; 4] = [10, 20, 30, 40];

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] out: &mut u32, index: u32) {
    *out = LOOKUP[index as usize];
}
