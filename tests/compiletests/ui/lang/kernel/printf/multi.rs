// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::glam::USizeVec3;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(#[spirv(global_invocation_id)] id: USizeVec3, #[spirv(cross_workgroup)] data: &[u32]) {
    let i = id.x as u32;
    let val = data[id.x];
    spirv_std::printf!("id=%u value=%u\n", i, val);
}
