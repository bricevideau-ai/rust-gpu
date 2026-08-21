// Probe: can a GENERIC #[rust_gpu::vector] struct monomorphize into valid
// OpTypeVector at several element types? Decides whether spirv_std::cl can
// grow a generic vector layer (claspr instantiate vector-family question).
// NOTE: unlike the concrete cl::* types, a generic struct cannot carry the
// element-dependent #[repr(align(N))] — host-side buffer ABI would need
// separate treatment even if device codegen works.

// build-pass
// compile-flags: -C target-feature=+Float64
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

use spirv_std::spirv;

#[derive(Copy, Clone)]
#[repr(C)]
#[cfg_attr(target_arch = "spirv", rust_gpu::vector::v1)]
pub struct GVec2<T> {
    pub x: T,
    pub y: T,
}

#[spirv(kernel)]
pub fn gv_f32(#[spirv(cross_workgroup)] data: &mut [f32]) {
    let v = GVec2 {
        x: data[0],
        y: data[1],
    };
    let w = GVec2 { x: v.y, y: v.x };
    data[0] = w.x + w.y;
}

#[spirv(kernel)]
pub fn gv_f64(#[spirv(cross_workgroup)] data: &mut [f64]) {
    let v = GVec2 {
        x: data[0],
        y: data[1],
    };
    let w = GVec2 { x: v.y, y: v.x };
    data[0] = w.x + w.y;
}

#[spirv(kernel)]
pub fn gv_u32(#[spirv(cross_workgroup)] data: &mut [u32]) {
    let v = GVec2 {
        x: data[0],
        y: data[1],
    };
    let w = GVec2 { x: v.y, y: v.x };
    data[0] = w.x ^ w.y;
}
