// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(sval: i8, uval: u8) {
    spirv_std::printf!("%hhd\n", sval);
    spirv_std::printf!("%hhu\n", uval);
    spirv_std::printf!("%hhx\n", uval);
}
