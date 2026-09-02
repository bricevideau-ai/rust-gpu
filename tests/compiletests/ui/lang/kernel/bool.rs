// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// Boolean short-circuit logic in kernel context.

use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] b: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    let x = *a != 0;
    let y = *b != 0;
    *out = if x && y {
        3
    } else if x || y {
        2
    } else if !x {
        1
    } else {
        0
    };
}
