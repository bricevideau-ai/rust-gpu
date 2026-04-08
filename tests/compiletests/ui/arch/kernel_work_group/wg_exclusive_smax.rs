// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-opencl1.2
// ignore-spv1.5
// ignore-spv1.6
// compile-flags: -C target-feature=+Groups

use spirv_std::arch;
use spirv_std::spirv;

#[spirv(kernel(threads(32)))]
pub fn main(#[spirv(cross_workgroup)] out: &mut i32, value: i32) {
    *out = arch::work_group_exclusive_s_max(value);
}
