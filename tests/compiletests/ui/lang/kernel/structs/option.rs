// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// `Option<T>` exercises the MemberDecorate Offset skip for Kernel.

use spirv_std::spirv;

fn maybe_double(x: u32) -> Option<u32> {
    if x == 0 { None } else { Some(x * 2) }
}

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] input: &u32, #[spirv(cross_workgroup)] out: &mut u32) {
    *out = maybe_double(*input).unwrap_or(u32::MAX);
}
