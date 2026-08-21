// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// `static mut` becomes a CrossWorkgroup program-scope global on Kernel.

use spirv_std::spirv;

static mut COUNTER: u32 = 0;

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] out: &mut u32) {
    unsafe {
        COUNTER += 1;
        *out = COUNTER;
    }
}
