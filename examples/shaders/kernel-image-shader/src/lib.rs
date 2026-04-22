#![cfg_attr(target_arch = "spirv", no_std)]

use glam::{USizeVec3, UVec4};
use spirv_std::{Image, glam, spirv};

#[spirv(kernel)]
pub fn read_image_test(
    #[spirv(global_invocation_id)] id: USizeVec3,
    image: &Image!(2D, type=u32, sampled=false),
    #[spirv(cross_workgroup)] output: &mut [u32],
    width: u32,
) {
    let px = id.x as u32;
    let py = id.y as u32;
    let coord = glam::IVec2::new(px as i32, py as i32);
    let pixel: UVec4 = image.read(coord);
    output[(py * width + px) as usize] =
        pixel.x | (pixel.y << 8) | (pixel.z << 16) | (pixel.w << 24);
}
