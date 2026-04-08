// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// Mutable slice param: decomposes into (data ptr, length) kernel args.

use spirv_std::glam::UVec3;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(#[spirv(global_invocation_id)] id: UVec3, #[spirv(cross_workgroup)] data: &mut [u32]) {
    let i = id.x as usize;
    data[i] = data[i] * 2;
}
