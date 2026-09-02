// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(clippy::missing_safety_doc)]

// Read from one image, write to another: distinct OpTypeImage per access mode
// (ReadOnly vs WriteOnly) — supported by OpenCL 1.2.

use glam::*;
use spirv_std::{Image, glam, spirv};

#[spirv(kernel)]
pub fn main(
    #[spirv(global_invocation_id)] id: USizeVec3,
    src: &Image!(2D, type=f32, sampled=false),
    #[spirv(image_access = "write_only")] dst: &mut Image!(2D, type=f32, sampled=false),
) {
    let coord = IVec2::new(id.x as i32, 0);
    let texel: Vec4 = src.read(coord);
    unsafe {
        dst.write(coord, texel);
    }
}
