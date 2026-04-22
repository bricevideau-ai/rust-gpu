#![cfg_attr(target_arch = "spirv", no_std)]
#![cfg_attr(target_arch = "spirv", feature(asm_experimental_arch))]

use glam::{USizeVec3, Vec2, Vec4};
use spirv_std::{Image, Sampler, const_sampler, glam, spirv};

/// Read a 2D float sampled image with `OpImageSampleExplicitLod` (the
/// only sampling form valid in Kernel mode — no implicit-LOD without
/// fragment derivatives) and write the texel to a 2D storage image.
///
/// Coordinates are normalised in `[0, 1]`, so when the destination is
/// larger than the source the sampler does the upscale according to its
/// configured filter and addressing mode.
#[spirv(kernel)]
pub fn upscale_2d(
    #[spirv(global_invocation_id)] id: USizeVec3,
    src: &Image!(2D, type=f32, sampled=true),
    sampler: &Sampler,
    dst: &mut Image!(2D, type=f32, sampled=false),
    dst_width: u32,
    dst_height: u32,
) {
    let px = id.x as u32;
    let py = id.y as u32;
    if px >= dst_width || py >= dst_height {
        return;
    }
    // Sample the centre of each destination pixel.
    let u = (px as f32 + 0.5) / dst_width as f32;
    let v = (py as f32 + 0.5) / dst_height as f32;
    let texel: Vec4 = src.sample_by_lod(*sampler, Vec2::new(u, v), 0.0);
    let coord = glam::IVec2::new(px as i32, py as i32);
    unsafe {
        dst.write(coord, texel);
    }
}

/// Same upscale, but using `OpConstantSampler` (no kernel-arg sampler).
/// The sampler is baked into the SPIR-V module via `const_sampler!`,
/// which auto-adds the `LiteralSampler` capability.
#[spirv(kernel)]
pub fn upscale_2d_const_sampler(
    #[spirv(global_invocation_id)] id: USizeVec3,
    src: &Image!(2D, type=f32, sampled=true),
    dst: &mut Image!(2D, type=f32, sampled=false),
    dst_width: u32,
    dst_height: u32,
) {
    let px = id.x as u32;
    let py = id.y as u32;
    if px >= dst_width || py >= dst_height {
        return;
    }
    let sampler = const_sampler!(addr = ClampToEdge, normalized = true, filter = Linear);
    let u = (px as f32 + 0.5) / dst_width as f32;
    let v = (py as f32 + 0.5) / dst_height as f32;
    let texel: Vec4 = src.sample_by_lod(sampler, Vec2::new(u, v), 0.0);
    let coord = glam::IVec2::new(px as i32, py as i32);
    unsafe {
        dst.write(coord, texel);
    }
}
