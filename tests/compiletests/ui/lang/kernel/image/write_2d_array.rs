// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// 2D image array — coord is `IVec3` (x, y, layer). Dim=2D
// carries no per-Dim capability requirement, so no extra
// auto-declare is needed beyond `ImageBasic`. Locks in that the
// arrayed bit doesn't accidentally pull in a wrong cap.

#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(clippy::missing_safety_doc)]

use glam::*;
use spirv_std::{Image, glam, spirv};

#[spirv(kernel)]
pub fn main(
    #[spirv(global_invocation_id)] id: USizeVec3,
    #[spirv(image_access = "write_only")] image: &mut Image!(2D, arrayed=true, type=f32, sampled=false),
    #[spirv(cross_workgroup)] input: &[Vec4],
) {
    let coord = IVec3::new(id.x as i32, id.y as i32, id.z as i32);
    unsafe {
        image.write(coord, input[id.x]);
    }
}
