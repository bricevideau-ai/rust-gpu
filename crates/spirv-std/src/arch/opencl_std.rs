//! Math intrinsics from the `OpenCL.std` extended instruction set.
//!
//! All functions in this module emit `OpExtInst %opencl_std <op> ...`,
//! where `%opencl_std` is the result of `OpExtInstImport "OpenCL.std"`.
//! These instructions are valid only in `Kernel` SPIR-V (i.e. `OpenCL`
//! targets); behaviour on Vulkan/Shader targets is undefined.
//!
//! For Vulkan/shader targets, use the equivalents in `crate::arch::*` or
//! `crate::float`, which call the `GLSL.std.450` set instead.
//!
//! # Naming conventions
//!
//! Functions match the `OpenCL` C names from the SPIR-V `OpenCL.std`
//! extended instruction set spec. Where signedness matters for integers,
//! the `s_` and `u_` prefixes follow the `OpenCL.std` naming
//! (`s_min`, `u_min`, `s_clamp`, `u_clamp`, …).
//!
//! All ops accept both scalar and `glam`-vector arguments; on a vector
//! the underlying `OpExtInst` is applied componentwise. The bounds are:
//!
//! - Float ops: [`FloatOrFloatVector`] — `f32`, `f64`, `Vec2`, `Vec3`,
//!   `Vec3A`, `Vec4`, `DVec2`, `DVec3`, `DVec4`
//! - Signed integer ops (`s_*`): [`SignedIntegerOrSignedVector`] —
//!   `i8`/`i16`/`i32`/`i64`, `IVec2`/`IVec3`/`IVec4`
//! - Unsigned integer ops (`u_*`): [`UnsignedIntegerOrUnsignedVector`]
//!   — `u8`/`u16`/`u32`/`u64`, `UVec2`/`UVec3`/`UVec4`
//! - Sign-agnostic integer ops (`popcount`, `clz`, `ctz`):
//!   [`IntegerOrIntegerVector`] — any of the above integer types
//!
//! # `native_*` ops
//!
//! `native_sqrt`, `native_sin`, `native_cos`, `native_exp`, `native_log`
//! are implementation-defined-precision faster variants of the IEEE
//! versions. ULP error is implementation-defined and typically larger
//! than the corresponding non-`native_` op. Use them when you need speed
//! and tolerate the precision loss; otherwise prefer the non-prefixed
//! versions.
//!
//! # `mad` vs `fma`
//!
//! [`mad`] is allowed to use unconstrained intermediate precision (the
//! GPU may fuse it differently than `fma` or evaluate it as separate
//! `mul` then `add`). For IEEE-754-deterministic fused multiply-add,
//! use [`fma`].
//!
//! # Required capability
//!
//! No extra capability beyond `Kernel`. The `OpExtInstImport` for
//! `"OpenCL.std"` is emitted inline in each call; the linker's
//! `remove_duplicate_ext_inst_imports` pass collapses them to a single
//! module-level import.

// Host arms below pass `|x| num_traits::Float::method(x)` to per-component
// macros for readability and consistency across the entire op table — the
// closure form is uniform whether the body is one method call or a small
// expression. Allow the redundant-closure lint at module scope rather than
// per-call to keep the table dense.
#![allow(clippy::redundant_closure)]

#[cfg(target_arch = "spirv")]
use core::arch::asm;

use crate::{Float, Integer, ScalarOrVector, SignedInteger, UnsignedInteger};

/// Per-component access for the host arms of the `OpenCL.std` intrinsics.
///
/// On `target_arch = "spirv"` the GPU paths emit `OpExtInst` / `OpDot` /
/// etc. directly and never see individual components, so this trait does
/// not exist on the SPIR-V side — every type that flows through
/// [`FloatOrFloatVector`] must impl this *only* for the host build.
///
/// The trait is exposed as `pub` because it appears in the host-side bound
/// of [`FloatOrFloatVector`], but it's not part of the user-facing surface:
/// users write `ocl::dot(a, b)`, not `a.zip_componentwise(...)`.
#[cfg(not(target_arch = "spirv"))]
pub trait Componentwise: Copy {
    /// The per-component scalar type. Named `Component` (not `Scalar`) to
    /// avoid associated-type ambiguity when this trait travels with
    /// [`ScalarOrVector`] (which already exposes its own `Scalar`).
    type Component: Copy;
    /// Apply `f` to each component independently and collect into a fresh
    /// value of the same type.
    fn map_componentwise(self, f: impl FnMut(Self::Component) -> Self::Component) -> Self;
    /// Zip with another value of the same type, applying `f` componentwise.
    fn zip_componentwise(
        self,
        other: Self,
        f: impl FnMut(Self::Component, Self::Component) -> Self::Component,
    ) -> Self;
    /// Reduce: visit each component, threading an accumulator. Used for
    /// reductions like `dot` / `length_squared`.
    fn fold_componentwise<R: Copy>(self, init: R, f: impl FnMut(R, Self::Component) -> R) -> R;

    /// Three-way zip — apply `f` to `(self_i, b_i, c_i)` componentwise
    /// and collect into a fresh value. Default implementation walks `b`
    /// and `c` once each via `fold_componentwise` to materialise their
    /// components into stack arrays that the user closure can index.
    /// Concrete impls don't need to override this unless they want a
    /// faster path.
    fn zip3_componentwise(
        self,
        b: Self,
        c: Self,
        mut f: impl FnMut(Self::Component, Self::Component, Self::Component) -> Self::Component,
    ) -> Self {
        // Use a `Cell`-backed cursor through `b` and `c` so the
        // `map_componentwise(self, ...)` walker can pull one component
        // from each of `b` and `c` per iteration without needing to
        // materialise the full arrays through a const-N trait method.
        use core::cell::RefCell;
        let b_iter = RefCell::new(IterState::<Self::Component, Self>::new(b));
        let c_iter = RefCell::new(IterState::<Self::Component, Self>::new(c));
        self.map_componentwise(|a_i| {
            let b_i = b_iter.borrow_mut().next();
            let c_i = c_iter.borrow_mut().next();
            f(a_i, b_i, c_i)
        })
    }
}

/// Tiny iterator-ish helper that walks the components of a `Componentwise`
/// value lazily by repeatedly applying `fold_componentwise`. Used by the
/// default `zip3_componentwise` impl to thread `b` and `c` cursors.
#[cfg(not(target_arch = "spirv"))]
struct IterState<C: Copy, V: Componentwise<Component = C>> {
    src: V,
    pos: usize,
}

#[cfg(not(target_arch = "spirv"))]
impl<C: Copy, V: Componentwise<Component = C>> IterState<C, V> {
    fn new(src: V) -> Self {
        Self { src, pos: 0 }
    }
    /// Returns the `pos`-th component of `src` and advances the cursor.
    /// Implemented via `fold_componentwise` so we don't need an explicit
    /// `to_array`/`get` on the trait. O(N) per call (used at most N times).
    fn next(&mut self) -> C {
        let target = self.pos;
        self.pos += 1;
        let mut found: Option<C> = None;
        let mut idx = 0usize;
        self.src.fold_componentwise((), |(), x| {
            if idx == target {
                found = Some(x);
            }
            idx += 1;
        });
        found.expect("zip3_componentwise: index out of range")
    }
}

#[cfg(not(target_arch = "spirv"))]
mod componentwise_impls {
    use super::Componentwise;
    use crate::glam;

    macro_rules! componentwise_scalar {
        ($($t:ty),+ $(,)?) => {
            $(
                impl Componentwise for $t {
                    type Component = $t;
                    #[inline]
                    fn map_componentwise(
                        self,
                        mut f: impl FnMut(Self::Component) -> Self::Component,
                    ) -> Self {
                        f(self)
                    }
                    #[inline]
                    fn zip_componentwise(
                        self,
                        other: Self,
                        mut f: impl FnMut(Self::Component, Self::Component) -> Self::Component,
                    ) -> Self {
                        f(self, other)
                    }
                    #[inline]
                    fn fold_componentwise<R: Copy>(
                        self,
                        init: R,
                        mut f: impl FnMut(R, Self::Component) -> R,
                    ) -> R {
                        f(init, self)
                    }
                }
            )+
        };
    }
    componentwise_scalar!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);

    /// Generates a `Componentwise` impl for a glam vector that has
    /// inherent `to_array(self) -> [T; N]` and `from_array([T; N]) -> Self`.
    macro_rules! componentwise_glam_vec {
        ($vec:ty, $scalar:ty) => {
            impl Componentwise for $vec {
                type Component = $scalar;
                #[inline]
                fn map_componentwise(
                    self,
                    mut f: impl FnMut(Self::Component) -> Self::Component,
                ) -> Self {
                    let mut a = self.to_array();
                    for x in &mut a {
                        *x = f(*x);
                    }
                    Self::from_array(a)
                }
                #[inline]
                fn zip_componentwise(
                    self,
                    other: Self,
                    mut f: impl FnMut(Self::Component, Self::Component) -> Self::Component,
                ) -> Self {
                    let mut a = self.to_array();
                    let b = other.to_array();
                    for (x, y) in a.iter_mut().zip(b.iter()) {
                        *x = f(*x, *y);
                    }
                    Self::from_array(a)
                }
                #[inline]
                fn fold_componentwise<R: Copy>(
                    self,
                    init: R,
                    mut f: impl FnMut(R, Self::Component) -> R,
                ) -> R {
                    let a = self.to_array();
                    let mut acc = init;
                    for x in a {
                        acc = f(acc, x);
                    }
                    acc
                }
            }
        };
    }

    componentwise_glam_vec!(glam::Vec2, f32);
    componentwise_glam_vec!(glam::Vec3, f32);
    componentwise_glam_vec!(glam::Vec3A, f32);
    componentwise_glam_vec!(glam::Vec4, f32);
    componentwise_glam_vec!(glam::DVec2, f64);
    componentwise_glam_vec!(glam::DVec3, f64);
    componentwise_glam_vec!(glam::DVec4, f64);
    componentwise_glam_vec!(glam::IVec2, i32);
    componentwise_glam_vec!(glam::IVec3, i32);
    componentwise_glam_vec!(glam::IVec4, i32);
    componentwise_glam_vec!(glam::UVec2, u32);
    componentwise_glam_vec!(glam::UVec3, u32);
    componentwise_glam_vec!(glam::UVec4, u32);
}

// On host, every type implementing this trait must also implement
// `Componentwise` so the host arms of `dot`/`length`/etc. can iterate
// per-component. On SPIR-V the supertrait is dropped — the GPU paths
// emit `OpDot` etc. directly and never touch components individually.

/// A float scalar (`f32`, `f64`) or a vector of floats (`glam::Vec2`,
/// `glam::Vec3`, `glam::Vec3A`, `glam::Vec4`, `glam::DVec2`, `glam::DVec3`,
/// `glam::DVec4`) — the argument type for `OpenCL.std` extended instructions
/// that are polymorphic over `genFloat` in the `OpenCL` SPIR-V spec.
///
/// On vector arguments the underlying `OpExtInst` is applied componentwise,
/// matching `OpenCL` C semantics.
#[cfg(target_arch = "spirv")]
pub trait FloatOrFloatVector: ScalarOrVector + Copy
where
    Self::Scalar: Float,
{
}

/// A float scalar (`f32`, `f64`) or a vector of floats (`glam::Vec2`,
/// `glam::Vec3`, `glam::Vec3A`, `glam::Vec4`, `glam::DVec2`, `glam::DVec3`,
/// `glam::DVec4`) — the argument type for `OpenCL.std` extended instructions
/// that are polymorphic over `genFloat` in the `OpenCL` SPIR-V spec.
///
/// On vector arguments the underlying `OpExtInst` is applied componentwise,
/// matching `OpenCL` C semantics.
#[cfg(not(target_arch = "spirv"))]
pub trait FloatOrFloatVector:
    ScalarOrVector + Copy + Componentwise<Component = <Self as ScalarOrVector>::Scalar>
where
    <Self as ScalarOrVector>::Scalar: Float,
{
}

#[cfg(target_arch = "spirv")]
impl<T> FloatOrFloatVector for T
where
    T: ScalarOrVector + Copy,
    T::Scalar: Float,
{
}

#[cfg(not(target_arch = "spirv"))]
impl<T> FloatOrFloatVector for T
where
    T: ScalarOrVector + Copy + Componentwise<Component = <T as ScalarOrVector>::Scalar>,
    <T as ScalarOrVector>::Scalar: Float,
{
}

/// Any integer scalar or integer vector — argument type for `OpenCL.std`
/// integer instructions polymorphic over `genIType`/`genUType` whose
/// signedness doesn't matter (`popcount`, `clz`, `ctz`).
#[cfg(target_arch = "spirv")]
pub trait IntegerOrIntegerVector: ScalarOrVector + Copy
where
    Self::Scalar: Integer,
{
}

/// Any integer scalar or integer vector — argument type for `OpenCL.std`
/// integer instructions polymorphic over `genIType`/`genUType` whose
/// signedness doesn't matter (`popcount`, `clz`, `ctz`).
#[cfg(not(target_arch = "spirv"))]
pub trait IntegerOrIntegerVector:
    ScalarOrVector + Copy + Componentwise<Component = <Self as ScalarOrVector>::Scalar>
where
    <Self as ScalarOrVector>::Scalar: Integer,
{
}

#[cfg(target_arch = "spirv")]
impl<T> IntegerOrIntegerVector for T
where
    T: ScalarOrVector + Copy,
    T::Scalar: Integer,
{
}

#[cfg(not(target_arch = "spirv"))]
impl<T> IntegerOrIntegerVector for T
where
    T: ScalarOrVector + Copy + Componentwise<Component = <T as ScalarOrVector>::Scalar>,
    <T as ScalarOrVector>::Scalar: Integer,
{
}

/// A signed-integer scalar (`i8`/`i16`/`i32`/`i64`) or a signed-integer
/// vector (`glam::IVec2`/`IVec3`/`IVec4`) — argument type for `OpenCL.std`
/// integer instructions polymorphic over `genIType` (the `s_` prefixed
/// ops: `s_abs`, `s_min`, `s_max`, `s_clamp`).
#[cfg(target_arch = "spirv")]
pub trait SignedIntegerOrSignedVector: ScalarOrVector + Copy
where
    Self::Scalar: SignedInteger,
{
}

/// A signed-integer scalar (`i8`/`i16`/`i32`/`i64`) or a signed-integer
/// vector (`glam::IVec2`/`IVec3`/`IVec4`) — argument type for `OpenCL.std`
/// integer instructions polymorphic over `genIType` (the `s_` prefixed
/// ops: `s_abs`, `s_min`, `s_max`, `s_clamp`).
#[cfg(not(target_arch = "spirv"))]
pub trait SignedIntegerOrSignedVector:
    ScalarOrVector + Copy + Componentwise<Component = <Self as ScalarOrVector>::Scalar>
where
    <Self as ScalarOrVector>::Scalar: SignedInteger,
{
}

#[cfg(target_arch = "spirv")]
impl<T> SignedIntegerOrSignedVector for T
where
    T: ScalarOrVector + Copy,
    T::Scalar: SignedInteger,
{
}

#[cfg(not(target_arch = "spirv"))]
impl<T> SignedIntegerOrSignedVector for T
where
    T: ScalarOrVector + Copy + Componentwise<Component = <T as ScalarOrVector>::Scalar>,
    <T as ScalarOrVector>::Scalar: SignedInteger,
{
}

/// An unsigned-integer scalar (`u8`/`u16`/`u32`/`u64`) or an
/// unsigned-integer vector (`glam::UVec2`/`UVec3`/`UVec4`) — argument
/// type for `OpenCL.std` integer instructions polymorphic over
/// `genUType` (the `u_` prefixed ops: `u_min`, `u_max`, `u_clamp`).
#[cfg(target_arch = "spirv")]
pub trait UnsignedIntegerOrUnsignedVector: ScalarOrVector + Copy
where
    Self::Scalar: UnsignedInteger,
{
}

/// An unsigned-integer scalar (`u8`/`u16`/`u32`/`u64`) or an
/// unsigned-integer vector (`glam::UVec2`/`UVec3`/`UVec4`) — argument
/// type for `OpenCL.std` integer instructions polymorphic over
/// `genUType` (the `u_` prefixed ops: `u_min`, `u_max`, `u_clamp`).
#[cfg(not(target_arch = "spirv"))]
pub trait UnsignedIntegerOrUnsignedVector:
    ScalarOrVector + Copy + Componentwise<Component = <Self as ScalarOrVector>::Scalar>
where
    <Self as ScalarOrVector>::Scalar: UnsignedInteger,
{
}

#[cfg(target_arch = "spirv")]
impl<T> UnsignedIntegerOrUnsignedVector for T
where
    T: ScalarOrVector + Copy,
    T::Scalar: UnsignedInteger,
{
}

#[cfg(not(target_arch = "spirv"))]
impl<T> UnsignedIntegerOrUnsignedVector for T
where
    T: ScalarOrVector + Copy + Componentwise<Component = <T as ScalarOrVector>::Scalar>,
    <T as ScalarOrVector>::Scalar: UnsignedInteger,
{
}

#[cfg(target_arch = "spirv")]
unsafe fn opencl_unary<T: Default + Copy, const OP: u32>(x: T) -> T {
    let mut result = T::default();
    unsafe {
        asm! {
            "%opencl = OpExtInstImport \"OpenCL.std\"",
            "%x = OpLoad _ {x}",
            "%result = OpExtInst typeof*{result} %opencl {op} %x",
            "OpStore {result} %result",
            x = in(reg) &x,
            result = in(reg) &mut result,
            op = const OP,
        }
    }
    result
}

#[cfg(target_arch = "spirv")]
unsafe fn opencl_binary<T: Default + Copy, const OP: u32>(a: T, b: T) -> T {
    let mut result = T::default();
    unsafe {
        asm! {
            "%opencl = OpExtInstImport \"OpenCL.std\"",
            "%a = OpLoad _ {a}",
            "%b = OpLoad _ {b}",
            "%result = OpExtInst typeof*{result} %opencl {op} %a %b",
            "OpStore {result} %result",
            a = in(reg) &a,
            b = in(reg) &b,
            result = in(reg) &mut result,
            op = const OP,
        }
    }
    result
}

#[cfg(target_arch = "spirv")]
unsafe fn opencl_ternary<T: Default + Copy, const OP: u32>(a: T, b: T, c: T) -> T {
    let mut result = T::default();
    unsafe {
        asm! {
            "%opencl = OpExtInstImport \"OpenCL.std\"",
            "%a = OpLoad _ {a}",
            "%b = OpLoad _ {b}",
            "%c = OpLoad _ {c}",
            "%result = OpExtInst typeof*{result} %opencl {op} %a %b %c",
            "OpStore {result} %result",
            a = in(reg) &a,
            b = in(reg) &b,
            c = in(reg) &c,
            result = in(reg) &mut result,
            op = const OP,
        }
    }
    result
}

/// Same as `opencl_unary` but returns the per-component scalar type
/// (used by `length` / `fast_length`, where `length(Vec3) -> f32`).
#[cfg(target_arch = "spirv")]
unsafe fn opencl_unary_to_scalar<V: ScalarOrVector + Copy, const OP: u32>(x: V) -> V::Scalar
where
    V::Scalar: Default + Copy,
{
    let mut result = V::Scalar::default();
    unsafe {
        asm! {
            "%opencl = OpExtInstImport \"OpenCL.std\"",
            "%x = OpLoad _ {x}",
            "%result = OpExtInst typeof*{result} %opencl {op} %x",
            "OpStore {result} %result",
            x = in(reg) &x,
            result = in(reg) &mut result,
            op = const OP,
        }
    }
    result
}

/// Same as `opencl_binary` but returns the per-component scalar type
/// (used by `distance` / `fast_distance`).
#[cfg(target_arch = "spirv")]
unsafe fn opencl_binary_to_scalar<V: ScalarOrVector + Copy, const OP: u32>(a: V, b: V) -> V::Scalar
where
    V::Scalar: Default + Copy,
{
    let mut result = V::Scalar::default();
    unsafe {
        asm! {
            "%opencl = OpExtInstImport \"OpenCL.std\"",
            "%a = OpLoad _ {a}",
            "%b = OpLoad _ {b}",
            "%result = OpExtInst typeof*{result} %opencl {op} %a %b",
            "OpStore {result} %result",
            a = in(reg) &a,
            b = in(reg) &b,
            result = in(reg) &mut result,
            op = const OP,
        }
    }
    result
}

/// Helper for `OpenCL.std` ops that produce two outputs: a return value
/// (`F`) and a value (`P`) written through a Function-storage pointer.
/// Used by `fract`, `modf`, `frexp`, `sincos`. Returns both as a tuple.
///
/// The out-pointer's backing slot is allocated internally so callers
/// see a clean `(F, P)` Rust API instead of having to thread an `&mut`
/// argument through.
#[cfg(target_arch = "spirv")]
unsafe fn opencl_with_ptr_out<F, P, const OP: u32>(value: F) -> (F, P)
where
    F: Default + Copy,
    P: Default + Copy,
{
    let mut result = F::default();
    let mut out = P::default();
    unsafe {
        asm! {
            "%opencl = OpExtInstImport \"OpenCL.std\"",
            "%v = OpLoad _ {value}",
            "%result = OpExtInst typeof*{result} %opencl {op} %v {out}",
            "OpStore {result} %result",
            value = in(reg) &value,
            out = in(reg) &mut out,
            result = in(reg) &mut result,
            op = const OP,
        }
    }
    (result, out)
}

// ── Float, unary ──────────────────────────────────────────────────────

macro_rules! float_unary {
    // Two forms: with an explicit per-scalar host fallback (expressed as a
    // function on one `F::Scalar`), and without — the latter still resolves
    // via `gpu_only` to an `unimplemented!()` stub and is tracked for follow-up.
    //
    // The `host_scalar = …` arm takes a path-style function (e.g. `f.sqrt()`
    // expressed as the closure `|f| f.sqrt()`); the closure is monomorphised
    // separately per scalar type at each public call site, so `Float`-trait
    // methods that exist on both `f32` and `f64` work uniformly.
    ($(#[$attr:meta])* $name:ident, $opcode:expr, host_scalar = $host:expr $(,)?) => {
        $(#[$attr])*
        #[inline]
        pub fn $name<F: FloatOrFloatVector>(x: F) -> F
        where
            F::Scalar: Float,
        {
            #[cfg(target_arch = "spirv")]
            { unsafe { opencl_unary::<F, $opcode>(x) } }
            #[cfg(not(target_arch = "spirv"))]
            {
                // `Componentwise` is reachable via the host-only supertrait
                // of `FloatOrFloatVector`, so no explicit `use` is needed —
                // the method is found through that bound.
                x.map_componentwise($host)
            }
        }
    };
    ($(#[$attr:meta])* $name:ident, $opcode:expr) => {
        $(#[$attr])*
        #[spirv_std_macros::gpu_only]
        #[inline]
        pub fn $name<F: FloatOrFloatVector>(x: F) -> F
        where
            F::Scalar: Float,
        {
            unsafe { opencl_unary::<F, $opcode>(x) }
        }
    };
}

// All unary host arms route through `num_traits::Float` so the same closure
// monomorphises correctly for both `f32` and `f64` per call site. The `nf`
// alias keeps the table readable.
float_unary!(
    /// Inverse cosine (`acos(x)`).
    acos, 0,
    host_scalar = |x| num_traits::Float::acos(x),
);
float_unary!(
    /// Inverse hyperbolic cosine (`acosh(x)`).
    acosh, 1,
    host_scalar = |x| num_traits::Float::acosh(x),
);
float_unary!(
    /// Inverse sine (`asin(x)`).
    asin, 3,
    host_scalar = |x| num_traits::Float::asin(x),
);
float_unary!(
    /// Inverse hyperbolic sine (`asinh(x)`).
    asinh, 4,
    host_scalar = |x| num_traits::Float::asinh(x),
);
float_unary!(
    /// Inverse tangent (`atan(x)`).
    atan, 6,
    host_scalar = |x| num_traits::Float::atan(x),
);
float_unary!(
    /// Inverse hyperbolic tangent (`atanh(x)`).
    atanh, 8,
    host_scalar = |x| num_traits::Float::atanh(x),
);
float_unary!(
    /// Cube root (`cbrt(x)`).
    cbrt, 11,
    host_scalar = |x| num_traits::Float::cbrt(x),
);
float_unary!(
    /// Round up to nearest integer (`ceil(x)`).
    ceil, 12,
    host_scalar = |x| num_traits::Float::ceil(x),
);
float_unary!(
    /// Cosine (`cos(x)`).
    cos, 14,
    host_scalar = |x| num_traits::Float::cos(x),
);
float_unary!(
    /// Hyperbolic cosine (`cosh(x)`).
    cosh, 15,
    host_scalar = |x| num_traits::Float::cosh(x),
);
float_unary!(
    /// Natural exponent (`e^x`).
    exp, 19,
    host_scalar = |x| num_traits::Float::exp(x),
);
float_unary!(
    /// Base-2 exponent (`2^x`).
    exp2, 20,
    host_scalar = |x| num_traits::Float::exp2(x),
);
float_unary!(
    /// Base-10 exponent (`10^x`).
    exp10, 21,
    host_scalar = |x| num_traits::Float::powf(num_traits::cast(10.0).unwrap(), x),
);
float_unary!(
    /// Absolute value (`|x|`).
    fabs, 23,
    host_scalar = |x| num_traits::Float::abs(x),
);
float_unary!(
    /// Round down to nearest integer (`floor(x)`).
    floor, 25,
    host_scalar = |x| num_traits::Float::floor(x),
);
float_unary!(
    /// Natural logarithm (`ln(x)`).
    log, 37,
    host_scalar = |x| num_traits::Float::ln(x),
);
float_unary!(
    /// Base-2 logarithm.
    log2, 38,
    host_scalar = |x| num_traits::Float::log2(x),
);
float_unary!(
    /// Base-10 logarithm.
    log10, 39,
    host_scalar = |x| num_traits::Float::log10(x),
);
float_unary!(
    /// Round to nearest integer, ties away from zero.
    round, 55,
    host_scalar = |x| num_traits::Float::round(x),
);
float_unary!(
    /// Reciprocal square root (`1/sqrt(x)`).
    rsqrt, 56,
    host_scalar = |x| num_traits::Float::recip(num_traits::Float::sqrt(x)),
);
float_unary!(
    /// Sine (`sin(x)`).
    sin, 57,
    host_scalar = |x| num_traits::Float::sin(x),
);
float_unary!(
    /// Hyperbolic sine.
    sinh, 59,
    host_scalar = |x| num_traits::Float::sinh(x),
);
float_unary!(
    /// Square root.
    sqrt, 61,
    host_scalar = |x| num_traits::Float::sqrt(x),
);
float_unary!(
    /// Tangent.
    tan, 62,
    host_scalar = |x| num_traits::Float::tan(x),
);
float_unary!(
    /// Hyperbolic tangent.
    tanh, 63,
    host_scalar = |x| num_traits::Float::tanh(x),
);
float_unary!(
    /// Truncate toward zero.
    trunc, 66,
    host_scalar = |x| num_traits::Float::trunc(x),
);
float_unary!(
    /// Sign of `x`: `-1`, `0`, or `+1`.
    sign, 103,
    host_scalar = |x| num_traits::Float::signum(x),
);

// `native_*` — implementation-defined-precision, faster than IEEE.
// On host we just use the IEEE versions; there's no faster CPU primitive.

float_unary!(
    /// Faster, lower-precision cosine. ULP error is implementation-defined.
    native_cos, 81,
    host_scalar = |x| num_traits::Float::cos(x),
);
float_unary!(
    /// Faster, lower-precision sine. ULP error is implementation-defined.
    native_sin, 92,
    host_scalar = |x| num_traits::Float::sin(x),
);
float_unary!(
    /// Faster, lower-precision square root. ULP error is implementation-defined.
    native_sqrt, 93,
    host_scalar = |x| num_traits::Float::sqrt(x),
);
float_unary!(
    /// Faster, lower-precision natural exponent. ULP error is implementation-defined.
    native_exp, 83,
    host_scalar = |x| num_traits::Float::exp(x),
);
float_unary!(
    /// Faster, lower-precision natural logarithm. ULP error is implementation-defined.
    native_log, 86,
    host_scalar = |x| num_traits::Float::ln(x),
);

// ── Float, binary ─────────────────────────────────────────────────────

macro_rules! float_binary {
    ($(#[$attr:meta])* $name:ident, $opcode:expr, host_scalar = $host:expr $(,)?) => {
        $(#[$attr])*
        #[inline]
        pub fn $name<F: FloatOrFloatVector>(a: F, b: F) -> F
        where
            F::Scalar: Float,
        {
            #[cfg(target_arch = "spirv")]
            { unsafe { opencl_binary::<F, $opcode>(a, b) } }
            #[cfg(not(target_arch = "spirv"))]
            { a.zip_componentwise(b, $host) }
        }
    };
    ($(#[$attr:meta])* $name:ident, $opcode:expr) => {
        $(#[$attr])*
        #[spirv_std_macros::gpu_only]
        #[inline]
        pub fn $name<F: FloatOrFloatVector>(a: F, b: F) -> F
        where
            F::Scalar: Float,
        {
            unsafe { opencl_binary::<F, $opcode>(a, b) }
        }
    };
}

float_binary!(
    /// Two-argument arctangent (`atan2(y, x)`), correctly handling quadrants.
    atan2, 7,
    host_scalar = |a, b| num_traits::Float::atan2(a, b),
);
float_binary!(
    /// Magnitude of `a` with the sign of `b`.
    copysign, 13,
    host_scalar = |a, b| num_traits::Float::copysign(a, b),
);
float_binary!(
    /// Maximum of two floats. Follows IEEE-754 `maxNum` for NaN handling.
    fmax, 27,
    host_scalar = |a, b| num_traits::Float::max(a, b),
);
float_binary!(
    /// Minimum of two floats. Follows IEEE-754 `minNum` for NaN handling.
    fmin, 28,
    host_scalar = |a, b| num_traits::Float::min(a, b),
);
float_binary!(
    /// Floating-point modulo (sign matches the dividend `a`).
    fmod, 29,
    host_scalar = |a, b| a - num_traits::Float::trunc(a / b) * b,
);
float_binary!(
    /// Square root of `a*a + b*b` without overflow / underflow for large/small inputs.
    hypot, 32,
    host_scalar = |a, b| num_traits::Float::hypot(a, b),
);
float_binary!(
    /// `a` raised to the power `b`.
    pow, 48,
    host_scalar = |a, b| num_traits::Float::powf(a, b),
);

// ── Float, ternary ────────────────────────────────────────────────────

macro_rules! float_ternary {
    ($(#[$attr:meta])* $name:ident, $opcode:expr, host_scalar = $host:expr $(,)?) => {
        $(#[$attr])*
        #[inline]
        pub fn $name<F: FloatOrFloatVector>(a: F, b: F, c: F) -> F
        where
            F::Scalar: Float,
        {
            #[cfg(target_arch = "spirv")]
            { unsafe { opencl_ternary::<F, $opcode>(a, b, c) } }
            #[cfg(not(target_arch = "spirv"))]
            { a.zip3_componentwise(b, c, $host) }
        }
    };
    ($(#[$attr:meta])* $name:ident, $opcode:expr) => {
        $(#[$attr])*
        #[spirv_std_macros::gpu_only]
        #[inline]
        pub fn $name<F: FloatOrFloatVector>(a: F, b: F, c: F) -> F
        where
            F::Scalar: Float,
        {
            unsafe { opencl_ternary::<F, $opcode>(a, b, c) }
        }
    };
}

float_ternary!(
    /// Fused multiply-add: `a * b + c`, computed with a single rounding (IEEE-754).
    fma, 26,
    host_scalar = |a, b, c| num_traits::Float::mul_add(a, b, c),
);
float_ternary!(
    /// Multiply-add `a * b + c` with implementation-defined intermediate precision.
    /// For IEEE-754 determinism, use [`fma`] instead.
    mad, 42,
    host_scalar = |a, b, c| a * b + c,
);
float_ternary!(
    /// Clamp `x` to the closed interval `[min, max]`. Argument order: `(x, min, max)`.
    clamp, 95,
    host_scalar = |x, lo, hi| num_traits::Float::min(num_traits::Float::max(x, lo), hi),
);
float_ternary!(
    /// Linear interpolation: `a + (b - a) * t`. Argument order: `(a, b, t)`.
    mix, 99,
    host_scalar = |a, b, t| a + (b - a) * t,
);
float_ternary!(
    /// Smooth Hermite interpolation between `0` and `1` for `x` in `[edge0, edge1]`.
    /// Argument order: `(edge0, edge1, x)`.
    smoothstep, 102,
    host_scalar = |edge0, edge1, x| smoothstep_scalar(edge0, edge1, x),
);

/// Host scalar implementation of `smoothstep`. Pinning the scalar type
/// here lets `num_traits::cast` infer the target without `let`-side
/// annotations in the macro-expansion site.
#[cfg(not(target_arch = "spirv"))]
#[inline]
fn smoothstep_scalar<F: Float>(edge0: F, edge1: F, x: F) -> F {
    let zero = F::zero();
    let one = F::one();
    let two = num_traits::cast::<f64, F>(2.0).unwrap();
    let three = num_traits::cast::<f64, F>(3.0).unwrap();
    let t = num_traits::Float::min(
        num_traits::Float::max((x - edge0) / (edge1 - edge0), zero),
        one,
    );
    t * t * (three - two * t)
}

// ── Integer, unary ────────────────────────────────────────────────────

/// Absolute value of a signed integer (or componentwise on a signed vector).
#[inline]
pub fn s_abs<I: SignedIntegerOrSignedVector>(x: I) -> I
where
    I::Scalar: SignedInteger,
{
    #[cfg(target_arch = "spirv")]
    {
        unsafe { opencl_unary::<I, 141>(x) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        x.map_componentwise(|c| {
            // `num_traits::Signed::abs` works for every signed primitive.
            num_traits::Signed::abs(&c)
        })
    }
}

/// Number of set bits (popcount), componentwise on vectors.
#[inline]
pub fn popcount<I: IntegerOrIntegerVector>(x: I) -> I
where
    I::Scalar: Integer,
{
    #[cfg(target_arch = "spirv")]
    {
        unsafe { opencl_unary::<I, 166>(x) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        // `count_ones()` returns u32; convert back to the input scalar
        // via num_traits::cast to fit any Integer impl.
        x.map_componentwise(|c| num_traits::cast(num_traits::PrimInt::count_ones(c)).unwrap())
    }
}

/// Count leading zero bits, componentwise on vectors.
#[inline]
pub fn clz<I: IntegerOrIntegerVector>(x: I) -> I
where
    I::Scalar: Integer,
{
    #[cfg(target_arch = "spirv")]
    {
        unsafe { opencl_unary::<I, 151>(x) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        x.map_componentwise(|c| num_traits::cast(num_traits::PrimInt::leading_zeros(c)).unwrap())
    }
}

/// Count trailing zero bits, componentwise on vectors.
#[inline]
pub fn ctz<I: IntegerOrIntegerVector>(x: I) -> I
where
    I::Scalar: Integer,
{
    #[cfg(target_arch = "spirv")]
    {
        unsafe { opencl_unary::<I, 152>(x) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        x.map_componentwise(|c| num_traits::cast(num_traits::PrimInt::trailing_zeros(c)).unwrap())
    }
}

// ── Integer, binary ───────────────────────────────────────────────────

/// Minimum of two signed integers (or componentwise on signed vectors).
#[inline]
pub fn s_min<I: SignedIntegerOrSignedVector>(a: I, b: I) -> I
where
    I::Scalar: SignedInteger,
{
    #[cfg(target_arch = "spirv")]
    {
        unsafe { opencl_binary::<I, 158>(a, b) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        a.zip_componentwise(b, core::cmp::min)
    }
}

/// Maximum of two signed integers (or componentwise on signed vectors).
#[inline]
pub fn s_max<I: SignedIntegerOrSignedVector>(a: I, b: I) -> I
where
    I::Scalar: SignedInteger,
{
    #[cfg(target_arch = "spirv")]
    {
        unsafe { opencl_binary::<I, 156>(a, b) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        a.zip_componentwise(b, core::cmp::max)
    }
}

/// Minimum of two unsigned integers (or componentwise on unsigned vectors).
#[inline]
pub fn u_min<I: UnsignedIntegerOrUnsignedVector>(a: I, b: I) -> I
where
    I::Scalar: UnsignedInteger,
{
    #[cfg(target_arch = "spirv")]
    {
        unsafe { opencl_binary::<I, 159>(a, b) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        a.zip_componentwise(b, core::cmp::min)
    }
}

/// Maximum of two unsigned integers (or componentwise on unsigned vectors).
#[inline]
pub fn u_max<I: UnsignedIntegerOrUnsignedVector>(a: I, b: I) -> I
where
    I::Scalar: UnsignedInteger,
{
    #[cfg(target_arch = "spirv")]
    {
        unsafe { opencl_binary::<I, 157>(a, b) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        a.zip_componentwise(b, core::cmp::max)
    }
}

// ── Integer, ternary ──────────────────────────────────────────────────

/// Clamp a signed integer `x` to `[min, max]` (or componentwise on
/// signed vectors). Argument order: `(x, min, max)`.
#[inline]
pub fn s_clamp<I: SignedIntegerOrSignedVector>(x: I, min: I, max: I) -> I
where
    I::Scalar: SignedInteger,
{
    #[cfg(target_arch = "spirv")]
    {
        unsafe { opencl_ternary::<I, 149>(x, min, max) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        x.zip3_componentwise(min, max, |x, lo, hi| {
            core::cmp::min(core::cmp::max(x, lo), hi)
        })
    }
}

/// Clamp an unsigned integer `x` to `[min, max]` (or componentwise on
/// unsigned vectors). Argument order: `(x, min, max)`.
#[inline]
pub fn u_clamp<I: UnsignedIntegerOrUnsignedVector>(x: I, min: I, max: I) -> I
where
    I::Scalar: UnsignedInteger,
{
    #[cfg(target_arch = "spirv")]
    {
        unsafe { opencl_ternary::<I, 150>(x, min, max) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        x.zip3_componentwise(min, max, |x, lo, hi| {
            core::cmp::min(core::cmp::max(x, lo), hi)
        })
    }
}

// ── Geometric ─────────────────────────────────────────────────────────
//
// `length`/`distance`/`fast_length`/`fast_distance` return the per-
// component scalar type of the input (e.g. `length(Vec3) -> f32`).
// `normalize`/`fast_normalize`/`cross` return the input vector type.
//
// Per the `OpenCL.std` spec, `cross` is restricted to vec3/vec4. The
// `FloatOrFloatVector` bound is wider; passing other types compiles
// but produces SPIR-V that `spirv-val` rejects. (Defining a tighter
// trait would buy nothing — the constraint only matters for `cross`.)

/// Vector length (Euclidean norm). For a vector `v`, returns
/// `sqrt(dot(v, v))`. Also accepts a scalar (returns its absolute value).
#[inline]
pub fn length<V: FloatOrFloatVector>(v: V) -> V::Scalar
where
    V::Scalar: Float,
{
    #[cfg(target_arch = "spirv")]
    {
        unsafe { opencl_unary_to_scalar::<V, 106>(v) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        num_traits::Float::sqrt(dot(v, v))
    }
}

/// Distance between two vectors: `length(a - b)`.
#[inline]
pub fn distance<V: FloatOrFloatVector>(a: V, b: V) -> V::Scalar
where
    V::Scalar: Float,
{
    #[cfg(target_arch = "spirv")]
    {
        unsafe { opencl_binary_to_scalar::<V, 105>(a, b) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        length(a.zip_componentwise(b, |x, y| x - y))
    }
}

/// Returns `v` scaled to unit length: `v / length(v)`.
#[inline]
pub fn normalize<V: FloatOrFloatVector>(v: V) -> V
where
    V::Scalar: Float,
{
    #[cfg(target_arch = "spirv")]
    {
        unsafe { opencl_unary::<V, 107>(v) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        let len = length(v);
        v.map_componentwise(|x| x / len)
    }
}

/// Cross product of two 3- or 4-component float vectors.
///
/// Per the `OpenCL.std` spec, only `Vec3`/`Vec3A`/`Vec4` (and `DVec3`/
/// `DVec4`) are valid; passing other `FloatOrFloatVector` types
/// produces SPIR-V that `spirv-val` rejects. The host fallback panics
/// for inputs of other widths so callers see a fast failure rather than
/// silently wrong values.
#[inline]
pub fn cross<V: FloatOrFloatVector>(a: V, b: V) -> V
where
    V::Scalar: Float,
{
    #[cfg(target_arch = "spirv")]
    {
        unsafe { opencl_binary::<V, 104>(a, b) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        // Materialise `a` and `b` into per-component arrays via
        // fold_componentwise so we can index them positionally. The
        // length must be 3 or 4 — anything else is a misuse.
        let mut a_buf = [V::Scalar::default(); 4];
        let mut b_buf = [V::Scalar::default(); 4];
        let mut n = 0usize;
        a.fold_componentwise((), |(), x| {
            if n < a_buf.len() {
                a_buf[n] = x;
            }
            n += 1;
        });
        let len = n;
        let mut m = 0usize;
        b.fold_componentwise((), |(), y| {
            if m < b_buf.len() {
                b_buf[m] = y;
            }
            m += 1;
        });
        assert!(
            len == 3 || len == 4,
            "ocl::cross host fallback: only width-3 or width-4 vectors are valid",
        );
        // For width-4 the OpenCL spec returns the cross of the .xyz lanes
        // with the w lane zeroed out. Same here.
        let cx = a_buf[1] * b_buf[2] - a_buf[2] * b_buf[1];
        let cy = a_buf[2] * b_buf[0] - a_buf[0] * b_buf[2];
        let cz = a_buf[0] * b_buf[1] - a_buf[1] * b_buf[0];
        let mut idx = 0usize;
        a.map_componentwise(|_| {
            let v = match idx {
                0 => cx,
                1 => cy,
                2 => cz,
                _ => V::Scalar::default(),
            };
            idx += 1;
            v
        })
    }
}

/// Faster, lower-precision `length`. ULP error is implementation-defined.
/// On host we just delegate to the IEEE version.
#[inline]
pub fn fast_length<V: FloatOrFloatVector>(v: V) -> V::Scalar
where
    V::Scalar: Float,
{
    #[cfg(target_arch = "spirv")]
    {
        unsafe { opencl_unary_to_scalar::<V, 109>(v) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        length(v)
    }
}

/// Faster, lower-precision `distance`. ULP error is implementation-defined.
/// On host we just delegate to the IEEE version.
#[inline]
pub fn fast_distance<V: FloatOrFloatVector>(a: V, b: V) -> V::Scalar
where
    V::Scalar: Float,
{
    #[cfg(target_arch = "spirv")]
    {
        unsafe { opencl_binary_to_scalar::<V, 108>(a, b) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        distance(a, b)
    }
}

/// Faster, lower-precision `normalize`. ULP error is implementation-defined.
/// On host we just delegate to the IEEE version.
#[inline]
pub fn fast_normalize<V: FloatOrFloatVector>(v: V) -> V
where
    V::Scalar: Float,
{
    #[cfg(target_arch = "spirv")]
    {
        unsafe { opencl_unary::<V, 110>(v) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        normalize(v)
    }
}

/// Vector dot product: `Σᵢ aᵢ · bᵢ`.
///
/// Lowers to the core SPIR-V `OpDot` instruction (opcode 148), not an
/// `OpenCL.std` extended instruction — `OpDot` is part of core SPIR-V
/// and the `OpenCL` SPIR-V environment spec lists it without any extra
/// capability requirement, so it's always available on Kernel modules.
/// Exposed here next to the `OpenCL.std` geometric ops for
/// discoverability — the geometric family
/// (`length`/`distance`/`normalize`/etc.) is conceptually built on
/// `dot`, so callers expect to find them together.
///
/// Operands must both be float vectors of the same length (2/3/4);
/// scalar arguments are not accepted by `OpDot`. Use `a * b` for the
/// scalar case.
#[inline]
pub fn dot<V: FloatOrFloatVector>(a: V, b: V) -> V::Scalar
where
    V::Scalar: Float,
{
    #[cfg(target_arch = "spirv")]
    {
        let mut result = V::Scalar::default();
        unsafe {
            asm! {
                "%a = OpLoad _ {a}",
                "%b = OpLoad _ {b}",
                "%result = OpDot typeof*{result} %a %b",
                "OpStore {result} %result",
                a = in(reg) &a,
                b = in(reg) &b,
                result = in(reg) &mut result,
            }
        }
        result
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        a.zip_componentwise(b, |x, y| x * y)
            .fold_componentwise(V::Scalar::default(), |s, x| s + x)
    }
}

// ── Vector arithmetic ────────────────────────────────────────────────
//
// Component-wise binary float ops on vectors (or scalars). Lower to
// the core SPIR-V `OpFAdd`/`OpFSub`/`OpFMul`/`OpFDiv` instructions
// directly — these are core SPIR-V (no capability) and work on both
// scalar and vector operands. Exposed here so that callers who want
// guaranteed-vector codegen can reach for them; using `a + b` etc. on
// `glam` types currently relies on LLVM auto-vectorisation to refuse
// the per-component scalarisation, which is brittle for anything
// beyond the simplest expressions.

/// Component-wise float addition: `a + b`. Lowers to a single
/// `OpFAdd %T %a %b`.
///
/// `a + b` on `glam` types may scalarise to per-component
/// `OpCompositeExtract` + scalar `OpFAdd` + `OpCompositeConstruct`
/// depending on the surrounding expression; this function guarantees
/// the vector instruction.
#[inline]
pub fn add<V: FloatOrFloatVector>(a: V, b: V) -> V
where
    V::Scalar: Float,
{
    #[cfg(target_arch = "spirv")]
    {
        let mut result = V::default();
        unsafe {
            asm! {
                "%a = OpLoad _ {a}",
                "%b = OpLoad _ {b}",
                "%result = OpFAdd typeof*{result} %a %b",
                "OpStore {result} %result",
                a = in(reg) &a,
                b = in(reg) &b,
                result = in(reg) &mut result,
            }
        }
        result
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        a.zip_componentwise(b, |x, y| x + y)
    }
}

/// Component-wise float subtraction: `a - b`. Lowers to a single
/// `OpFSub %T %a %b`. See [`add`] for the rationale on exposing it.
#[inline]
pub fn sub<V: FloatOrFloatVector>(a: V, b: V) -> V
where
    V::Scalar: Float,
{
    #[cfg(target_arch = "spirv")]
    {
        let mut result = V::default();
        unsafe {
            asm! {
                "%a = OpLoad _ {a}",
                "%b = OpLoad _ {b}",
                "%result = OpFSub typeof*{result} %a %b",
                "OpStore {result} %result",
                a = in(reg) &a,
                b = in(reg) &b,
                result = in(reg) &mut result,
            }
        }
        result
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        a.zip_componentwise(b, |x, y| x - y)
    }
}

/// Component-wise float multiplication: `a * b`. Lowers to a single
/// `OpFMul %T %a %b`. See [`add`] for the rationale on exposing it.
///
/// For vector × scalar, lift the scalar to a vector first (e.g.
/// `Vec3::splat(s)`); this function does not perform that broadcast.
#[inline]
pub fn mul<V: FloatOrFloatVector>(a: V, b: V) -> V
where
    V::Scalar: Float,
{
    #[cfg(target_arch = "spirv")]
    {
        let mut result = V::default();
        unsafe {
            asm! {
                "%a = OpLoad _ {a}",
                "%b = OpLoad _ {b}",
                "%result = OpFMul typeof*{result} %a %b",
                "OpStore {result} %result",
                a = in(reg) &a,
                b = in(reg) &b,
                result = in(reg) &mut result,
            }
        }
        result
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        a.zip_componentwise(b, |x, y| x * y)
    }
}

/// Component-wise float division: `a / b`. Lowers to a single
/// `OpFDiv %T %a %b`. See [`add`] for the rationale on exposing it.
#[inline]
pub fn div<V: FloatOrFloatVector>(a: V, b: V) -> V
where
    V::Scalar: Float,
{
    #[cfg(target_arch = "spirv")]
    {
        let mut result = V::default();
        unsafe {
            asm! {
                "%a = OpLoad _ {a}",
                "%b = OpLoad _ {b}",
                "%result = OpFDiv typeof*{result} %a %b",
                "OpStore {result} %result",
                a = in(reg) &a,
                b = in(reg) &b,
                result = in(reg) &mut result,
            }
        }
        result
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        a.zip_componentwise(b, |x, y| x / y)
    }
}

// ── Multi-output ops ──────────────────────────────────────────────────
//
// These map to `OpenCL.std` ops that produce two outputs (the function's
// return value plus one written through a pointer). Exposed here as
// tuple-returning functions; the helper allocates the out-pointer's
// backing slot internally so callers don't need to thread an `&mut`
// argument through.
//
// Scalar-only for now — vector forms are mechanically the same but
// would multiply test surface; deferred.

/// Splits `value` into its fractional part (returned) and the integer
/// part `floor(value)` (second tuple element). Fractional part is in
/// `[0.0, 1.0)`.
#[inline]
pub fn fract<F: Float + Default>(value: F) -> (F, F) {
    #[cfg(target_arch = "spirv")]
    {
        unsafe { opencl_with_ptr_out::<F, F, 30>(value) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        let i = num_traits::Float::floor(value);
        (value - i, i)
    }
}

/// Decomposes `value` into `(fractional, integer)` parts (sign-preserving).
/// The integer part is `trunc(value)`.
#[inline]
pub fn modf<F: Float + Default>(value: F) -> (F, F) {
    #[cfg(target_arch = "spirv")]
    {
        unsafe { opencl_with_ptr_out::<F, F, 45>(value) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        let i = num_traits::Float::trunc(value);
        (value - i, i)
    }
}

/// Decomposes `value` into `(mantissa, exponent)` such that
/// `value = mantissa * 2^exponent` and `|mantissa| ∈ [0.5, 1.0)`.
#[inline]
pub fn frexp<F: Float + Default>(value: F) -> (F, i32) {
    #[cfg(target_arch = "spirv")]
    {
        unsafe { opencl_with_ptr_out::<F, i32, 31>(value) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        // OpenCL spec form: value = mantissa * 2^exp where
        // |mantissa| ∈ [0.5, 1.0). Compute exp from `log2(|value|)`,
        // then divide out the power of two.
        let abs = num_traits::Float::abs(value);
        if abs == F::zero() {
            return (F::zero(), 0);
        }
        let exp = num_traits::ToPrimitive::to_i32(&num_traits::Float::floor(
            num_traits::Float::log2(abs),
        ))
        .unwrap_or(0)
            + 1;
        let two = num_traits::cast::<f64, F>(2.0).unwrap();
        let scale = num_traits::Float::powi(two, -exp);
        (value * scale, exp)
    }
}

/// Computes `(sin(value), cos(value))` in one call.
#[inline]
pub fn sincos<F: Float + Default>(value: F) -> (F, F) {
    #[cfg(target_arch = "spirv")]
    {
        unsafe { opencl_with_ptr_out::<F, F, 58>(value) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        (num_traits::Float::sin(value), num_traits::Float::cos(value))
    }
}

// Host-side smoke tests for the host arms of `opencl_std`. Compiled
// out on SPIR-V via `cfg(test)`; exercise that the host bodies type-
// check and produce plausible numeric results across both glam and
// `cl::*` inputs at multiple op categories (unary/binary/ternary,
// vector geometric, integer, multi-output).
#[cfg(all(test, not(target_arch = "spirv")))]
mod tests {
    use super::*;
    use crate::cl::Double3;
    use crate::glam::{DVec3, IVec3, Vec3};

    const EPS: f64 = 1e-12;

    #[test]
    fn dot_glam_and_cl_match() {
        let a_glam = DVec3::new(1.0, 2.0, 3.0);
        let b_glam = DVec3::new(4.0, -5.0, 6.0);
        let a_cl = Double3::from_array([1.0, 2.0, 3.0]);
        let b_cl = Double3::from_array([4.0, -5.0, 6.0]);
        let expected = 4.0 - 10.0 + 18.0;
        assert_eq!(dot(a_glam, b_glam), expected);
        assert_eq!(dot(a_cl, b_cl), expected);
    }

    #[test]
    fn unary_scalar_host() {
        assert!((rsqrt(4.0_f64) - 0.5).abs() < EPS);
        assert!((sqrt(9.0_f64) - 3.0).abs() < EPS);
        assert!((cos(0.0_f64) - 1.0).abs() < EPS);
        assert!((sin(0.0_f64)).abs() < EPS);
        assert!((exp(0.0_f64) - 1.0).abs() < EPS);
        assert!((log(num_traits::Float::exp(1.0_f64)) - 1.0).abs() < EPS);
        assert_eq!(fabs(-3.5_f64), 3.5);
        assert_eq!(floor(2.7_f64), 2.0);
        assert_eq!(ceil(2.1_f64), 3.0);
    }

    #[test]
    fn unary_vector_host() {
        let v = DVec3::new(4.0, 16.0, 0.25);
        let r = rsqrt(v);
        assert!((r.x - 0.5).abs() < EPS);
        assert!((r.y - 0.25).abs() < EPS);
        assert!((r.z - 2.0).abs() < EPS);

        // The same operation should also work on the matching `cl::*` type
        // — `cl::Double3` is an `OpenCL` `double3` (32-byte aligned, 24
        // bytes of payload + trailing pad) whose `Componentwise` impl
        // is provided in this commit.
        let v_cl = Double3::from_array([4.0, 16.0, 0.25]);
        let r_cl = rsqrt(v_cl).to_array();
        assert!((r_cl[0] - 0.5).abs() < EPS);
        assert!((r_cl[1] - 0.25).abs() < EPS);
        assert!((r_cl[2] - 2.0).abs() < EPS);
    }

    #[test]
    fn binary_host() {
        assert!((atan2(1.0_f64, 0.0) - core::f64::consts::FRAC_PI_2).abs() < EPS);
        assert_eq!(fmax(2.0_f64, 5.0), 5.0);
        assert_eq!(fmin(2.0_f64, 5.0), 2.0);
        assert!((pow(2.0_f64, 10.0) - 1024.0).abs() < EPS);
        assert!((hypot(3.0_f64, 4.0) - 5.0).abs() < EPS);
    }

    #[test]
    fn ternary_host() {
        assert!((fma(2.0_f64, 3.0, 1.0) - 7.0).abs() < EPS);
        assert!((mad(2.0_f64, 3.0, 1.0) - 7.0).abs() < EPS);
        assert_eq!(clamp(5.0_f64, 0.0, 1.0), 1.0);
        assert_eq!(clamp(-1.0_f64, 0.0, 1.0), 0.0);
        assert!((mix(0.0_f64, 10.0, 0.25) - 2.5).abs() < EPS);
        assert!((smoothstep(0.0_f64, 1.0, 0.5) - 0.5).abs() < EPS);
        // Ternary on a vector type should walk all three through the
        // default `zip3_componentwise` impl.
        let a = DVec3::new(2.0, 3.0, 4.0);
        let b = DVec3::new(3.0, 4.0, 5.0);
        let c = DVec3::new(1.0, 1.0, 1.0);
        let r = fma(a, b, c);
        assert_eq!(r, DVec3::new(7.0, 13.0, 21.0));
    }

    #[test]
    fn integer_host() {
        assert_eq!(s_abs(-5_i32), 5);
        assert_eq!(s_min(2_i32, -3), -3);
        assert_eq!(s_max(2_i32, -3), 2);
        assert_eq!(u_min(2_u32, 7), 2);
        assert_eq!(u_max(2_u32, 7), 7);
        assert_eq!(s_clamp(15_i32, -10, 10), 10);
        assert_eq!(s_clamp(-15_i32, -10, 10), -10);
        assert_eq!(u_clamp(15_u32, 0, 10), 10);
        assert_eq!(popcount(0b1010_1010_u32), 4);
        assert_eq!(clz(1_u32), 31);
        assert_eq!(ctz(0b1000_u32), 3);
        // Vector forms.
        assert_eq!(s_abs(IVec3::new(-1, 2, -3)), IVec3::new(1, 2, 3));
    }

    #[test]
    fn geometric_host() {
        let a = DVec3::new(1.0, 2.0, 3.0);
        let b = DVec3::new(4.0, -5.0, 6.0);
        assert!((length(a) - num_traits::Float::sqrt(14.0_f64)).abs() < EPS);
        assert!((distance(a, b) - num_traits::Float::sqrt(9.0 + 49.0 + 9.0_f64)).abs() < EPS);
        let n = normalize(a);
        let n_len = length(n);
        assert!((n_len - 1.0).abs() < EPS);
        // cross of x̂ and ŷ should be ẑ.
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        let xy = cross(x, y);
        assert_eq!(xy, Vec3::new(0.0, 0.0, 1.0));
        // Same for `cl::Double3`.
        let xc = Double3::from_array([1.0, 0.0, 0.0]);
        let yc = Double3::from_array([0.0, 1.0, 0.0]);
        let xyc = cross(xc, yc).to_array();
        assert_eq!(xyc, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn vector_arithmetic_host() {
        let a = DVec3::new(1.0, 2.0, 3.0);
        let b = DVec3::new(4.0, 5.0, 6.0);
        assert_eq!(add(a, b), DVec3::new(5.0, 7.0, 9.0));
        assert_eq!(sub(a, b), DVec3::new(-3.0, -3.0, -3.0));
        assert_eq!(mul(a, b), DVec3::new(4.0, 10.0, 18.0));
        assert_eq!(div(b, a), DVec3::new(4.0, 2.5, 2.0));
    }

    #[test]
    fn multi_output_host() {
        let (frac, int) = fract(2.75_f64);
        assert!((frac - 0.75).abs() < EPS);
        assert_eq!(int, 2.0);

        let (m, e) = modf(-2.75_f64);
        assert!((m - -0.75).abs() < EPS);
        assert_eq!(e, -2.0);

        let (s, c) = sincos(0.0_f64);
        assert_eq!(s, 0.0);
        assert_eq!(c, 1.0);

        let (mantissa, exp) = frexp(8.0_f64);
        // 8.0 = 0.5 * 2^4
        assert!((mantissa - 0.5).abs() < EPS);
        assert_eq!(exp, 4);
    }

    #[test]
    fn cl_method_forms_match_free_functions() {
        // Glam-style method forms on `cl::*` types must produce
        // identical results to the matching `ocl::*` free functions —
        // they're the same call, just different syntax.
        use crate::cl::{Double3, Float3, Int3, UInt3};

        let a = Float3::new(1.0, 2.0, 3.0);
        let b = Float3::new(4.0, -5.0, 6.0);
        assert_eq!(a.dot(b), dot(a, b));
        assert_eq!(a.length(), length(a));
        assert_eq!(a.length_squared(), dot(a, a));
        assert_eq!(a.distance(b), distance(a, b));
        // `normalize` uses sqrt — go via approximate compare.
        let n_method = a.normalize().to_array();
        let n_free = normalize(a).to_array();
        for i in 0..3 {
            assert!((n_method[i] - n_free[i]).abs() < 1e-6);
        }

        // Componentwise unary.
        assert_eq!(a.abs(), fabs(a));
        assert_eq!(a.floor(), floor(a));
        assert_eq!(a.sqrt(), sqrt(a));

        // Binary.
        assert_eq!(a.min(b), fmin(a, b));
        assert_eq!(a.max(b), fmax(a, b));
        assert_eq!(a.powf(2.0), pow(a, Float3::splat(2.0)));

        // Ternary.
        let lo = Float3::splat(0.0);
        let hi = Float3::splat(2.0);
        assert_eq!(a.clamp(lo, hi), clamp(a, lo, hi));
        assert_eq!(a.lerp(b, 0.25), mix(a, b, Float3::splat(0.25)));
        assert_eq!(a.mix(b, 0.25), a.lerp(b, 0.25));
        assert_eq!(a.mul_add(b, lo), fma(a, b, lo));

        // Cross — only widths 3/4.
        let x = Float3::new(1.0, 0.0, 0.0);
        let y = Float3::new(0.0, 1.0, 0.0);
        assert_eq!(x.cross(y), cross(x, y));

        // Recip = 1/self via OpFDiv (NOT native_recip).
        let eps_f32 = 1e-6_f32;
        let r = a.recip().to_array();
        assert!((r[0] - 1.0).abs() < eps_f32);
        assert!((r[1] - 0.5).abs() < eps_f32);
        assert!((r[2] - 1.0_f32 / 3.0).abs() < eps_f32);

        // Double3 sanity.
        let da = Double3::new(3.0, 4.0, 0.0);
        assert!((da.length() - 5.0).abs() < EPS);

        // Integer methods: signed.
        let i = Int3::new(-3, 5, 7);
        let j = Int3::new(2, 2, 2);
        assert_eq!(i.abs(), s_abs(i));
        assert_eq!(i.min(j), s_min(i, j));
        assert_eq!(i.max(j), s_max(i, j));
        let lo = Int3::splat(0);
        let hi = Int3::splat(10);
        assert_eq!(i.clamp(lo, hi), s_clamp(i, lo, hi));

        // Integer methods: unsigned.
        let u = UInt3::new(3, 5, 7);
        let v = UInt3::new(2, 8, 4);
        assert_eq!(u.min(v), u_min(u, v));
        assert_eq!(u.max(v), u_max(u, v));
        let lo = UInt3::splat(0);
        let hi = UInt3::splat(10);
        assert_eq!(u.clamp(lo, hi), u_clamp(u, lo, hi));
        assert_eq!(u.count_ones(), popcount(u));
    }
}
