// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// Confirms `Dim=3D` is reachable on Kernel — the SPIR-V core spec
// attaches no per-`Dim` capability requirement to 3D (only 1D,
// Buffer, Cube, Rect, SubpassData carry one). `ImageBasic` alone
// is enough. The 2D companion lives in `write.rs`, the 1D
// companion in `write_1d.rs`.

#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(clippy::missing_safety_doc)]

use glam::*;
use spirv_std::{Image, glam, spirv};

#[spirv(kernel)]
pub fn main(
    #[spirv(global_invocation_id)] id: USizeVec3,
    #[spirv(image_access = "write_only")] image: &mut Image!(3D, type=f32, sampled=false),
    #[spirv(cross_workgroup)] input: &[Vec4],
) {
    let coord = IVec3::new(id.x as i32, 0, 0);
    unsafe {
        image.write(coord, input[id.x]);
    }
}
