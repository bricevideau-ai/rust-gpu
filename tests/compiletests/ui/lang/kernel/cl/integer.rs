// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// `spirv_std::cl` integer vectors — covers signed (`OpSDiv`) and
// unsigned (`OpUDiv`) divide-codegen and exercises Int8/16/64
// scalar-element widths plus Vector16 component widths.

use spirv_std::cl::{
    Char4, Int2, Int4, Int8, Int16, Long4, Short4, UChar4, UInt2, UInt4, UInt8, UInt16, ULong4,
    UShort4,
};
use spirv_std::spirv;

#[spirv(kernel)]
pub fn signed(
    #[spirv(cross_workgroup)] a8: &Char4,
    #[spirv(cross_workgroup)] a16: &Short4,
    #[spirv(cross_workgroup)] a32: &Int4,
    #[spirv(cross_workgroup)] a64: &Long4,
    #[spirv(cross_workgroup)] wide: &Int16,
    #[spirv(cross_workgroup)] narrow: &Int2,
    #[spirv(cross_workgroup)] mid: &Int8,
    #[spirv(cross_workgroup)] out8: &mut Char4,
    #[spirv(cross_workgroup)] out16: &mut Short4,
    #[spirv(cross_workgroup)] out32: &mut Int4,
    #[spirv(cross_workgroup)] out64: &mut Long4,
    #[spirv(cross_workgroup)] outwide: &mut Int16,
    #[spirv(cross_workgroup)] outnarrow: &mut Int2,
    #[spirv(cross_workgroup)] outmid: &mut Int8,
) {
    *out8 = *a8 + Char4::splat(1);
    *out16 = *a16 - Short4::splat(1);
    *out32 = (*a32 * Int4::splat(2)) / Int4::splat(3);
    *out64 = *a64 + Long4::splat(7);
    *outwide = *wide / Int16::splat(2);
    *outnarrow = *narrow + *narrow;
    *outmid = *mid * Int8::splat(2);
}

#[spirv(kernel)]
pub fn unsigned(
    #[spirv(cross_workgroup)] a8: &UChar4,
    #[spirv(cross_workgroup)] a16: &UShort4,
    #[spirv(cross_workgroup)] a32: &UInt4,
    #[spirv(cross_workgroup)] a64: &ULong4,
    #[spirv(cross_workgroup)] wide: &UInt16,
    #[spirv(cross_workgroup)] narrow: &UInt2,
    #[spirv(cross_workgroup)] mid: &UInt8,
    #[spirv(cross_workgroup)] out8: &mut UChar4,
    #[spirv(cross_workgroup)] out16: &mut UShort4,
    #[spirv(cross_workgroup)] out32: &mut UInt4,
    #[spirv(cross_workgroup)] out64: &mut ULong4,
    #[spirv(cross_workgroup)] outwide: &mut UInt16,
    #[spirv(cross_workgroup)] outnarrow: &mut UInt2,
    #[spirv(cross_workgroup)] outmid: &mut UInt8,
) {
    *out8 = *a8 + UChar4::splat(1);
    *out16 = *a16 - UShort4::splat(1);
    *out32 = (*a32 * UInt4::splat(2)) / UInt4::splat(3);
    *out64 = *a64 + ULong4::splat(7);
    *outwide = *wide / UInt16::splat(2);
    *outnarrow = *narrow + *narrow;
    *outmid = *mid * UInt8::splat(2);
}
