#![cfg_attr(target_arch = "spirv", no_std)]
#![cfg_attr(target_arch = "spirv", feature(asm_experimental_arch))]
// HACK(eddyb) can't easily see warnings otherwise from `spirv-builder` builds.
#![deny(warnings)]

use glam::USizeVec3;
use spirv_std::{glam, spirv};

/// Test printf with f32 and f64 values.
#[spirv(kernel)]
pub fn printf_fp64_test(
    #[spirv(global_invocation_id)] id: USizeVec3,
    #[spirv(cross_workgroup)] floats: &[f32],
    #[spirv(cross_workgroup)] doubles: &[f64],
) {
    let i = id.x;
    spirv_std::printf!("item %u: f32=%f f64=%f\n", i as u32, floats[i], doubles[i]);
}
