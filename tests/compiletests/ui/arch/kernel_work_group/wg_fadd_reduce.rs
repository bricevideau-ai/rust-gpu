// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-opencl1.2
// compile-flags: -C target-feature=+Groups

use spirv_std::arch;
use spirv_std::spirv;

#[spirv(kernel(threads(32)))]
pub fn main(#[spirv(cross_workgroup)] out: &mut f32, value: f32) {
    *out = arch::work_group_f_add(value);
}
