// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

#![cfg_attr(target_arch = "spirv", no_std)]

use glam::*;
use spirv_std::{Image, glam, spirv};

#[spirv(kernel)]
pub fn main(
    #[spirv(global_invocation_id)] id: USizeVec3,
    image: &Image!(2D, type=f32, sampled=false),
    #[spirv(cross_workgroup)] output: &mut [Vec4],
) {
    let coord = IVec2::new(id.x as i32, 0);
    let texel: Vec4 = image.read(coord);
    output[id.x] = texel;
}
