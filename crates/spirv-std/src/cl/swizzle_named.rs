//! Named-group swizzle traits — `lo`, `hi`, `even`, `odd` —
//! per the `OpenCL` spec.
//!
//! For a width-N source vector, each named group returns:
//!
//! | Source `N` | `lo` / `hi` / `even` / `odd` |
//! |------------|-------------------------------|
//! | 2          | scalar (1 component)          |
//! | 4          | width-2 vector                |
//! | 8          | width-4 vector                |
//! | 16         | width-8 vector                |
//!
//! Width-3 vectors are intentionally not supported — the `OpenCL` spec
//! defines named-group swizzles on `vec3` as "operate as if `vec4` with
//! component 3 undefined", which is too easy to misuse. Use the
//! explicit `s!(v3, sN)` form instead if you need a slice of a width-3.
//!
//! Each trait is implemented for every concrete `cl::*` type at
//! supported widths via a per-source/per-target asm template that
//! emits a single `OpVectorShuffle` (or `OpCompositeExtract` for the
//! width-2 case which yields a scalar).

use crate::cl::{
    Char2, Char4, Char8, Char16, Double2, Double4, Double8, Double16, Float2, Float4, Float8,
    Float16, Int2, Int4, Int8, Int16, Long2, Long4, Long8, Long16, Short2, Short4, Short8, Short16,
    UChar2, UChar4, UChar8, UChar16, UInt2, UInt4, UInt8, UInt16, ULong2, ULong4, ULong8, ULong16,
    UShort2, UShort4, UShort8, UShort16,
};

/// Returns the lower half of an `OpenCL` vector.
pub trait SwizzleLo {
    /// The result type — scalar for source width 2, half-width vector otherwise.
    type Output;
    /// Returns components `0..N/2` (or component `0` for width 2).
    fn lo(self) -> Self::Output;
}

/// Returns the upper half of an `OpenCL` vector.
pub trait SwizzleHi {
    /// The result type — scalar for source width 2, half-width vector otherwise.
    type Output;
    /// Returns components `N/2..N` (or component `1` for width 2).
    fn hi(self) -> Self::Output;
}

/// Returns the even-indexed components of an `OpenCL` vector.
pub trait SwizzleEven {
    /// The result type — scalar for source width 2, half-width vector otherwise.
    type Output;
    /// Returns components `0, 2, 4, ...` (or component `0` for width 2).
    fn even(self) -> Self::Output;
}

/// Returns the odd-indexed components of an `OpenCL` vector.
pub trait SwizzleOdd {
    /// The result type — scalar for source width 2, half-width vector otherwise.
    type Output;
    /// Returns components `1, 3, 5, ...` (or component `1` for width 2).
    fn odd(self) -> Self::Output;
}

// ----- Macro helpers ---------------------------------------------------------

/// Implements one named swizzle that returns a vector (width >= 2) via
/// `OpVectorShuffle` on the SPIR-V side and `from_array` on the host.
macro_rules! impl_named_vec {
    ($src:ty, $out:ty, $trait:ident, $method:ident, [$($idx:literal),+ $(,)?]) => {
        impl $trait for $src {
            type Output = $out;

            #[cfg(target_arch = "spirv")]
            #[inline]
            fn $method(self) -> Self::Output {
                let mut result = <$out as ::core::default::Default>::default();
                unsafe {
                    ::core::arch::asm!(
                        "%src = OpLoad _ {src}",
                        ::core::concat!(
                            "%dst = OpVectorShuffle typeof*{dst} %src %src ",
                            $( ::core::stringify!($idx), " ", )+
                        ),
                        "OpStore {dst} %dst",
                        src = in(reg) &self,
                        dst = in(reg) &mut result,
                    );
                }
                result
            }

            #[cfg(not(target_arch = "spirv"))]
            #[inline]
            fn $method(self) -> Self::Output {
                let arr = self.to_array();
                <$out>::from_array([$(arr[$idx]),+])
            }
        }
    };
}

/// Implements one named swizzle that returns a scalar (source width 2) via
/// `OpCompositeExtract` on the SPIR-V side.
macro_rules! impl_named_scalar {
    ($src:ty, $scalar:ty, $trait:ident, $method:ident, $idx:literal) => {
        impl $trait for $src {
            type Output = $scalar;

            #[cfg(target_arch = "spirv")]
            #[inline]
            fn $method(self) -> Self::Output {
                let mut result: $scalar = <$scalar as ::core::default::Default>::default();
                unsafe {
                    ::core::arch::asm!(
                        "%src = OpLoad _ {src}",
                        ::core::concat!(
                            "%dst = OpCompositeExtract typeof*{dst} %src ",
                            ::core::stringify!($idx),
                        ),
                        "OpStore {dst} %dst",
                        src = in(reg) &self,
                        dst = in(reg) &mut result,
                    );
                }
                result
            }

            #[cfg(not(target_arch = "spirv"))]
            #[inline]
            fn $method(self) -> Self::Output {
                self.to_array()[$idx]
            }
        }
    };
}

/// Implements all four named swizzles for one source type at one supported
/// width. Width 2 → scalar, widths 4/8/16 → half-width vector.
macro_rules! impl_named {
    // Width 2 → scalar.
    ($src:ty, scalar = $scalar:ty, w2) => {
        impl_named_scalar!($src, $scalar, SwizzleLo, lo, 0);
        impl_named_scalar!($src, $scalar, SwizzleHi, hi, 1);
        impl_named_scalar!($src, $scalar, SwizzleEven, even, 0);
        impl_named_scalar!($src, $scalar, SwizzleOdd, odd, 1);
    };
    // Width 4 → width-2 vector.
    ($src:ty, half = $half:ty, w4) => {
        impl_named_vec!($src, $half, SwizzleLo, lo, [0, 1]);
        impl_named_vec!($src, $half, SwizzleHi, hi, [2, 3]);
        impl_named_vec!($src, $half, SwizzleEven, even, [0, 2]);
        impl_named_vec!($src, $half, SwizzleOdd, odd, [1, 3]);
    };
    // Width 8 → width-4 vector.
    ($src:ty, half = $half:ty, w8) => {
        impl_named_vec!($src, $half, SwizzleLo, lo, [0, 1, 2, 3]);
        impl_named_vec!($src, $half, SwizzleHi, hi, [4, 5, 6, 7]);
        impl_named_vec!($src, $half, SwizzleEven, even, [0, 2, 4, 6]);
        impl_named_vec!($src, $half, SwizzleOdd, odd, [1, 3, 5, 7]);
    };
    // Width 16 → width-8 vector.
    ($src:ty, half = $half:ty, w16) => {
        impl_named_vec!($src, $half, SwizzleLo, lo, [0, 1, 2, 3, 4, 5, 6, 7]);
        impl_named_vec!($src, $half, SwizzleHi, hi, [8, 9, 10, 11, 12, 13, 14, 15]);
        impl_named_vec!($src, $half, SwizzleEven, even, [0, 2, 4, 6, 8, 10, 12, 14]);
        impl_named_vec!($src, $half, SwizzleOdd, odd, [1, 3, 5, 7, 9, 11, 13, 15]);
    };
}

/// Implements all four named swizzles for an entire scalar family.
macro_rules! impl_named_family {
    ($scalar:ty, $w2:ty, $w4:ty, $w8:ty, $w16:ty) => {
        impl_named!($w2, scalar = $scalar, w2);
        impl_named!($w4, half = $w2, w4);
        impl_named!($w8, half = $w4, w8);
        impl_named!($w16, half = $w8, w16);
    };
}

impl_named_family!(f32, Float2, Float4, Float8, Float16);
impl_named_family!(f64, Double2, Double4, Double8, Double16);
impl_named_family!(i8, Char2, Char4, Char8, Char16);
impl_named_family!(u8, UChar2, UChar4, UChar8, UChar16);
impl_named_family!(i16, Short2, Short4, Short8, Short16);
impl_named_family!(u16, UShort2, UShort4, UShort8, UShort16);
impl_named_family!(i32, Int2, Int4, Int8, Int16);
impl_named_family!(u32, UInt2, UInt4, UInt8, UInt16);
impl_named_family!(i64, Long2, Long4, Long8, Long16);
impl_named_family!(u64, ULong2, ULong4, ULong8, ULong16);
