// build-fail
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// Compile-time validation in `cl::s!` — bad letters, bad digits,
// invalid result widths.

use spirv_std::cl::{Float4, s};
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] v: &Float4) {
    // Letter form rejects non-{x,y,z,w} components.
    let _ = s!(*v, xyq);
    // sN form rejects non-hex digits.
    let _ = s!(*v, sgg);
    // Result width 5 is not a valid OpenCL/SPIR-V vector width.
    let _ = s!(*v, xyzwx);
    // Result width 0 (empty after the `s` prefix) is not valid.
    let _ = s!(*v, s);
}
