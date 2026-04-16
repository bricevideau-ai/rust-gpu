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
pub fn main(#[spirv(local_invocation_id)] _id: USizeVec3) {}
