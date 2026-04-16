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
pub fn main(#[spirv(cross_workgroup)] buf: &mut u32, limit: u32) {
    let mut i = 0u32;
    loop {
        if i >= limit {
            break;
        }
        i += 1;
    }
    *buf = i;
}
