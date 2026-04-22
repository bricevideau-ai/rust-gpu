// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// Confirms that the codegen auto-declares `OpCapability Image1D` when
// a Kernel image param uses `Dim=1D`. Without that auto-declare,
// `spirv-val` rejects the module: "Operand 3 of TypeImage requires
// Sampled1D". This file would `build-pass` only if the cap is
// declared. The 2D companion lives in `write.rs`.

#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(clippy::missing_safety_doc)]

use glam::*;
use spirv_std::{Image, glam, spirv};

#[spirv(kernel)]
pub fn main(
    #[spirv(global_invocation_id)] id: USizeVec3,
    #[spirv(image_access = "write_only")] image: &mut Image!(1D, type=f32, sampled=false),
    #[spirv(cross_workgroup)] input: &[Vec4],
) {
    unsafe {
        image.write(id.x as i32, input[id.x]);
    }
}
