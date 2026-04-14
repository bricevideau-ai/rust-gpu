// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(sval: i64, uval: u64) {
    spirv_std::printf!("%ld\n", sval);
    spirv_std::printf!("%li\n", sval);
    spirv_std::printf!("%lu\n", uval);
    spirv_std::printf!("%lx\n", uval);
    spirv_std::printf!("%lX\n", uval);
    spirv_std::printf!("%lo\n", uval);
}
