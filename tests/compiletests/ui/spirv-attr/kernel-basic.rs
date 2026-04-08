// build-pass
// only-opencl1.2

use spirv_std::glam::*;
use spirv_std::spirv;

/// Minimal kernel with no parameters.
#[spirv(kernel)]
pub fn kernel_noop() {}

/// Kernel with a built-in global_invocation_id.
#[spirv(kernel)]
pub fn kernel_with_global_id(#[spirv(global_invocation_id)] _id: USizeVec3) {}
