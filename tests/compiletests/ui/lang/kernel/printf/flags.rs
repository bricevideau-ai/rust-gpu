// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(value: i32, uval: u32) {
    spirv_std::printf!("%+d\n", value);
    spirv_std::printf!("%-10u\n", uval);
    spirv_std::printf!("%#x\n", uval);
    spirv_std::printf!("%08x\n", uval);
    spirv_std::printf!("% d\n", value);
}
