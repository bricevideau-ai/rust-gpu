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

use glam::{USizeVec3, Vec2, Vec4};
use spirv_std::{Image, const_sampler, glam, spirv};

#[spirv(kernel)]
pub fn main(
    #[spirv(global_invocation_id)] _id: USizeVec3,
    src: &Image!(2D, type=f32, sampled=true),
    #[spirv(cross_workgroup)] out: &mut [Vec4],
) {
    let nearest = const_sampler!(addr = ClampToEdge, normalized = false, filter = Nearest);
    let linear = const_sampler!(addr = Repeat, normalized = true, filter = Linear);
    let a: Vec4 = src.sample_by_lod(nearest, Vec2::new(0.0, 0.0), 0.0);
    let b: Vec4 = src.sample_by_lod(linear, Vec2::new(0.25, 0.75), 0.0);
    out[0] = a + b;
}
