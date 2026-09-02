// build-fail
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// Kernel-target equivalent of `glam/invalid_vector_type.rs` — verifies
// that `#[rust_gpu::vector::v1]` rejects widths outside {2, 3, 4, 8, 16}
// even on Kernel targets where `Vector16` is enabled.

use spirv_std::spirv;

#[rust_gpu::vector::v1]
pub struct OneField {
    _x: f32,
}

#[rust_gpu::vector::v1]
pub struct FiveFields {
    _x: f32,
    _y: f32,
    _z: f32,
    _w: f32,
    _v: f32,
}

#[rust_gpu::vector::v1]
pub struct SevenFields {
    _a: f32,
    _b: f32,
    _c: f32,
    _d: f32,
    _e: f32,
    _f: f32,
    _g: f32,
}

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] _: &OneField,
    #[spirv(cross_workgroup)] _: &FiveFields,
    #[spirv(cross_workgroup)] _: &SevenFields,
) {
}
