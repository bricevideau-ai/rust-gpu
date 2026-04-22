// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// compile-flags: -C target-feature=+LiteralSampler

#![cfg_attr(target_arch = "spirv", no_std)]

use glam::{USizeVec3, Vec2, Vec4};
use spirv_std::{Image, Sampler, glam, spirv};

#[spirv(kernel)]
pub fn main(
    #[spirv(global_invocation_id)] _id: USizeVec3,
    img: &Image!(2D, type=f32, sampled=true),
    sampler: &Sampler,
    #[spirv(cross_workgroup)] out: &mut [Vec4],
) {
    out[0] = img.sample_by_lod(*sampler, Vec2::new(0.5, 0.5), 0.0);
}
