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

use spirv_std::spirv;

#[spirv(kernel(threads(32)))]
pub fn main(
    #[spirv(cross_workgroup)] out: &mut u32,
    #[spirv(subgroup_id)] subgroup_id: u32,
    #[spirv(subgroup_local_invocation_id)] local_id: u32,
) {
    *out = subgroup_id * 1000 + local_id;
}
