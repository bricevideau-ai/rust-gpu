// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6
// compile-flags: -C target-feature=+LiteralSampler

#![cfg_attr(target_arch = "spirv", no_std)]

use glam::{Vec2, Vec4};
use spirv_std::{Image, Sampler, glam, spirv};

#[spirv(kernel)]
pub fn main(
    img: &Image!(2D, type=f32, sampled=true),
    sampler_a: &Sampler,
    sampler_b: &Sampler,
    #[spirv(cross_workgroup)] out: &mut [Vec4],
) {
    let a = img.sample_by_lod(*sampler_a, Vec2::new(0.0, 0.0), 0.0);
    let b = img.sample_by_lod(*sampler_b, Vec2::new(1.0, 1.0), 0.0);
    out[0] = a + b;
}
