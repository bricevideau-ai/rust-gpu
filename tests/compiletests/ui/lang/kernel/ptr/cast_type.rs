// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// Type-changing pointer cast (`*const u32` -> `*const u8`). Rejected by
// Vulkan's Logical addressing as a zombie value; valid in Physical
// addressing via OpBitcast.

use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] a: &u32, #[spirv(cross_workgroup)] out: &mut u32) {
    let p = a as *const u32 as *const u8;
    *out = unsafe { *p } as u32;
}
