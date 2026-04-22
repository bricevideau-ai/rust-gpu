// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

#![cfg_attr(target_arch = "spirv", no_std)]

// Constant sampler via `OpConstantSampler`. The `LiteralSampler`
// capability is auto-added by the `const_sampler!` macro.

use glam::{IVec2, USizeVec3, Vec2, Vec4};
use spirv_std::{Image, const_sampler, glam, spirv};

#[spirv(kernel)]
pub fn main(
    #[spirv(global_invocation_id)] _id: USizeVec3,
    src: &Image!(2D, type=f32, sampled=true),
    #[spirv(image_access = "write_only")] dst: &mut Image!(2D, type=f32, sampled=false),
) {
    let sampler = const_sampler!(addr = ClampToEdge, normalized = true, filter = Linear);
    let color: Vec4 = src.sample_by_lod(sampler, Vec2::new(0.5, 0.5), 0.0);
    unsafe { dst.write(IVec2::new(0, 0), color) };
}
