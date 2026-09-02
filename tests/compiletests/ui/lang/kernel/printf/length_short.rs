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
pub fn main(sval: i16, uval: u16) {
    spirv_std::printf!("%hd\n", sval);
    spirv_std::printf!("%hu\n", uval);
    spirv_std::printf!("%hx\n", uval);
}
