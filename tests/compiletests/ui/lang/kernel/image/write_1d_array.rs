// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// 1D image array — coord is `IVec2` (x, layer). Exercises the
// `(Dim::OneD, Arrayed::True)` ImageCoordinate impl in
// spirv-std/src/image/params.rs (which was a gap before
// 2026-06-02), plus the Image1D capability auto-declare (needed
// for any Dim=1D OpTypeImage including arrayed).

#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(clippy::missing_safety_doc)]

use glam::*;
use spirv_std::{Image, glam, spirv};

#[spirv(kernel)]
pub fn main(
    #[spirv(global_invocation_id)] id: USizeVec3,
    #[spirv(image_access = "write_only")] image: &mut Image!(1D, arrayed=true, type=f32, sampled=false),
    #[spirv(cross_workgroup)] input: &[Vec4],
) {
    let coord = IVec2::new(id.x as i32, id.y as i32);
    unsafe {
        image.write(coord, input[id.x]);
    }
}
