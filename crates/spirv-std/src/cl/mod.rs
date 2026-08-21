//! Native `OpenCL` vector types: `Float2/3/4/8/16`, `Double2/3/4/8/16`,
//! and `Char`/`UChar`/`Short`/`UShort`/`Int`/`UInt`/`Long`/`ULong` at the
//! same widths.
//!
//! Distinct from [`crate::glam`] — these exist to give kernel authors guaranteed
//! `OpTypeVector` codegen at every supported width (including 8 and 16, which
//! `glam` does not provide), and to expose the `OpenCL`-spec swizzle syntax via
//! the `s!` macro.
//!
//! ```ignore
//! use spirv_std::cl::Float4;
//!
//! let v = Float4::new(1.0, 2.0, 3.0, 4.0);
//! let w = v + Float4::splat(0.5);
//! ```
//!
//! Only available when targeting the `OpenCL` Kernel execution model. The
//! `Vector16` capability (required for widths 8/16) is auto-enabled for
//! Kernel targets; `Float64` (required for the `Double*` types) is opt-in
//! via `-C target-feature=+Float64`.

pub(crate) mod macros;

mod double;
mod float;
mod integer;
mod swizzle_named;

/// `OpenCL` swizzle macro: `s!(v, xyzw)` or `s!(v, sFEDC)`.
///
/// See [`spirv_std_macros::cl_swizzle`] for the full grammar. Re-exported
/// here so user code can write `spirv_std::cl::s!`.
pub use crate::macros::cl_swizzle as s;
pub use swizzle_named::{SwizzleEven, SwizzleHi, SwizzleLo, SwizzleOdd};

pub use double::{Double2, Double3, Double4, Double8, Double16};
pub use float::{Float2, Float3, Float4, Float8, Float16};
pub use integer::{
    Char2, Char3, Char4, Char8, Char16, Int2, Int3, Int4, Int8, Int16, Long2, Long3, Long4, Long8,
    Long16, Short2, Short3, Short4, Short8, Short16, UChar2, UChar3, UChar4, UChar8, UChar16,
    UInt2, UInt3, UInt4, UInt8, UInt16, ULong2, ULong3, ULong4, ULong8, ULong16, UShort2, UShort3,
    UShort4, UShort8, UShort16,
};
