//! Shared macros for declaring `cl` vector types.

/// Declares a homogeneous-float vector struct (e.g. `Float4`, `Double8`)
/// with the `OpenCL` `floatN`/`doubleN` ABI: `#[repr(C, align($align))]`,
/// `#[rust_gpu::vector::v1]`, named `s0..sN-1` fields, plus
/// `new`/`splat`/`from_array`/`to_array` and the four arithmetic
/// operators emitting native vector `OpFAdd`/`OpFSub`/`OpFMul`/`OpFDiv`.
///
/// `count`, `align`, and `size` must match the `OpenCL` C type's layout —
/// the macro emits a `const _: ()` size/align assertion.
macro_rules! decl_float_vector {
    (
        $name:ident, scalar = $scalar:ty,
        count = $count:literal, align = $align:literal, size = $size:literal,
        fields = [$($field:ident: $field_idx:tt),+],
    ) => {
        #[doc = concat!("`OpenCL` ", stringify!($scalar), " vector with ", stringify!($count), " components.")]
        #[derive(Copy, Clone, Default, PartialEq, Debug)]
        #[repr(C, align($align))]
        #[cfg_attr(target_arch = "spirv", rust_gpu::vector::v1)]
        pub struct $name {
            $(
                #[doc = concat!("Component `", stringify!($field), "`.")]
                pub $field: $scalar,
            )+
        }

        impl $name {
            /// Constructs a vector from its components.
            #[inline]
            #[must_use]
            #[allow(clippy::too_many_arguments)]
            pub const fn new($($field: $scalar),+) -> Self {
                Self { $($field),+ }
            }

            /// Returns a vector with all components set to `v`.
            #[inline]
            #[must_use]
            pub const fn splat(v: $scalar) -> Self {
                Self { $($field: v),+ }
            }

            /// Constructs a vector from an array.
            #[inline]
            #[must_use]
            pub const fn from_array(a: [$scalar; $count]) -> Self {
                Self { $($field: a[$field_idx]),+ }
            }

            /// Returns the components as an array.
            #[inline]
            #[must_use]
            pub const fn to_array(self) -> [$scalar; $count] {
                [$(self.$field),+]
            }
        }

        impl From<[$scalar; $count]> for $name {
            #[inline]
            fn from(a: [$scalar; $count]) -> Self { Self::from_array(a) }
        }

        impl From<$name> for [$scalar; $count] {
            #[inline]
            fn from(v: $name) -> Self { v.to_array() }
        }

        $crate::cl::macros::decl_float_vector!(@op $name, $scalar, Add, add, "OpFAdd", [$($field),+]);
        $crate::cl::macros::decl_float_vector!(@op $name, $scalar, Sub, sub, "OpFSub", [$($field),+]);
        $crate::cl::macros::decl_float_vector!(@op $name, $scalar, Mul, mul, "OpFMul", [$($field),+]);
        $crate::cl::macros::decl_float_vector!(@op $name, $scalar, Div, div, "OpFDiv", [$($field),+]);

        // `Mul<Scalar>` and `Mul<Self> for Scalar` use `OpVectorTimesScalar`,
        // SPIR-V's purpose-built op for the splat-multiply pattern.
        $crate::cl::macros::decl_float_vector!(@vts $name, $scalar, [$($field),+]);

        // Scalar +/-// go through splat-then-vector-op; the linker peephole
        // already collapses the resulting `OpFAdd v (OpCompositeConstruct s..s)`
        // pattern when the scalar is a constant.
        $crate::cl::macros::decl_float_vector!(@scalar_op $name, $scalar, Add, add, "OpFAdd", [$($field),+]);
        $crate::cl::macros::decl_float_vector!(@scalar_op $name, $scalar, Sub, sub, "OpFSub", [$($field),+]);
        $crate::cl::macros::decl_float_vector!(@scalar_op $name, $scalar, Div, div, "OpFDiv", [$($field),+]);

        // Compound-assignment forms — `+=`/`-=`/`*=`/`/=` for vector-vector
        // and vector-scalar. Each one delegates to the corresponding non-
        // assign impl, so codegen is identical to `x = x op y`.
        $crate::cl::macros::decl_assign_op!($name, AddAssign, add_assign, Add, add);
        $crate::cl::macros::decl_assign_op!($name, SubAssign, sub_assign, Sub, sub);
        $crate::cl::macros::decl_assign_op!($name, MulAssign, mul_assign, Mul, mul);
        $crate::cl::macros::decl_assign_op!($name, DivAssign, div_assign, Div, div);
        $crate::cl::macros::decl_assign_op!($name, $scalar, AddAssign, add_assign, Add, add);
        $crate::cl::macros::decl_assign_op!($name, $scalar, SubAssign, sub_assign, Sub, sub);
        $crate::cl::macros::decl_assign_op!($name, $scalar, MulAssign, mul_assign, Mul, mul);
        $crate::cl::macros::decl_assign_op!($name, $scalar, DivAssign, div_assign, Div, div);

        impl $crate::sealed::Sealed for $name {}
        impl $crate::ScalarComposite for $name {
            #[inline]
            fn transform<F: $crate::ScalarOrVectorTransform>(self, f: &mut F) -> Self {
                f.transform_vector(self)
            }
        }
        unsafe impl $crate::ScalarOrVector for $name {
            type Scalar = $scalar;
            const N: core::num::NonZeroUsize = core::num::NonZeroUsize::new($count).unwrap();
        }
        unsafe impl $crate::Vector<$scalar, $count> for $name {}

        // Host-only `Componentwise` impl so this type flows through the
        // host arms of `opencl_std::dot`/`length`/etc. uniformly with glam.
        #[cfg(not(target_arch = "spirv"))]
        impl $crate::arch::opencl_std::Componentwise for $name {
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

        const _: () = {
            assert!(core::mem::size_of::<$name>() == $size);
            assert!(core::mem::align_of::<$name>() == $align);
        };
    };

    (@op $name:ident, $scalar:ty, $trait:ident, $method:ident, $opname:literal, [$($field:ident),+]) => {
        impl core::ops::$trait for $name {
            type Output = Self;

            #[cfg(target_arch = "spirv")]
            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                let mut result = Self::default();
                unsafe {
                    core::arch::asm!(
                        "%a = OpLoad _ {a}",
                        "%b = OpLoad _ {b}",
                        concat!("%result = ", $opname, " typeof*{result} %a %b"),
                        "OpStore {result} %result",
                        a = in(reg) &self,
                        b = in(reg) &rhs,
                        result = in(reg) &mut result,
                    );
                }
                result
            }

            #[cfg(not(target_arch = "spirv"))]
            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                Self { $($field: core::ops::$trait::$method(self.$field, rhs.$field)),+ }
            }
        }
    };

    // `OpVectorTimesScalar` — both `vector * scalar` and `scalar * vector`.
    (@vts $name:ident, $scalar:ty, [$($field:ident),+]) => {
        impl core::ops::Mul<$scalar> for $name {
            type Output = Self;

            #[cfg(target_arch = "spirv")]
            #[inline]
            fn mul(self, rhs: $scalar) -> Self::Output {
                let mut result = Self::default();
                unsafe {
                    core::arch::asm!(
                        "%v = OpLoad _ {v}",
                        "%s = OpLoad _ {s}",
                        "%result = OpVectorTimesScalar typeof*{result} %v %s",
                        "OpStore {result} %result",
                        v = in(reg) &self,
                        s = in(reg) &rhs,
                        result = in(reg) &mut result,
                    );
                }
                result
            }

            #[cfg(not(target_arch = "spirv"))]
            #[inline]
            fn mul(self, rhs: $scalar) -> Self::Output {
                Self { $($field: self.$field * rhs),+ }
            }
        }

        impl core::ops::Mul<$name> for $scalar {
            type Output = $name;

            #[inline]
            fn mul(self, rhs: $name) -> Self::Output {
                rhs * self
            }
        }
    };

    // Splat-then-vector-op for the non-Mul scalar arithmetic paths.
    // Generates both `Vector op Scalar` and `Scalar op Vector` directions.
    (@scalar_op $name:ident, $scalar:ty, $trait:ident, $method:ident, $opname:literal, [$($field:ident),+]) => {
        impl core::ops::$trait<$scalar> for $name {
            type Output = Self;

            #[cfg(target_arch = "spirv")]
            #[inline]
            fn $method(self, rhs: $scalar) -> Self::Output {
                core::ops::$trait::$method(self, $name::splat(rhs))
            }

            #[cfg(not(target_arch = "spirv"))]
            #[inline]
            fn $method(self, rhs: $scalar) -> Self::Output {
                Self { $($field: core::ops::$trait::$method(self.$field, rhs)),+ }
            }
        }

        impl core::ops::$trait<$name> for $scalar {
            type Output = $name;

            #[cfg(target_arch = "spirv")]
            #[inline]
            fn $method(self, rhs: $name) -> Self::Output {
                core::ops::$trait::$method($name::splat(self), rhs)
            }

            #[cfg(not(target_arch = "spirv"))]
            #[inline]
            fn $method(self, rhs: $name) -> Self::Output {
                $name { $($field: core::ops::$trait::$method(self, rhs.$field)),+ }
            }
        }
    };
}

pub(crate) use decl_float_vector;

/// Declares a homogeneous-integer vector struct (e.g. `Int4`, `UChar16`).
/// Same shape as [`decl_float_vector!`] but emits the integer op opcodes:
/// `OpIAdd`/`OpISub`/`OpIMul` are signedness-agnostic; division is
/// `OpSDiv` for signed scalars and `OpUDiv` for unsigned.
macro_rules! decl_integer_vector {
    (
        $name:ident, scalar = $scalar:ty, signed = $signed:tt,
        count = $count:literal, align = $align:literal, size = $size:literal,
        fields = [$($field:ident: $field_idx:tt),+],
    ) => {
        #[doc = concat!("`OpenCL` ", stringify!($scalar), " vector with ", stringify!($count), " components.")]
        #[derive(Copy, Clone, Default, PartialEq, Eq, Debug)]
        #[repr(C, align($align))]
        #[cfg_attr(target_arch = "spirv", rust_gpu::vector::v1)]
        pub struct $name {
            $(
                #[doc = concat!("Component `", stringify!($field), "`.")]
                pub $field: $scalar,
            )+
        }

        impl $name {
            /// Constructs a vector from its components.
            #[inline]
            #[must_use]
            #[allow(clippy::too_many_arguments)]
            pub const fn new($($field: $scalar),+) -> Self {
                Self { $($field),+ }
            }

            /// Returns a vector with all components set to `v`.
            #[inline]
            #[must_use]
            pub const fn splat(v: $scalar) -> Self {
                Self { $($field: v),+ }
            }

            /// Constructs a vector from an array.
            #[inline]
            #[must_use]
            pub const fn from_array(a: [$scalar; $count]) -> Self {
                Self { $($field: a[$field_idx]),+ }
            }

            /// Returns the components as an array.
            #[inline]
            #[must_use]
            pub const fn to_array(self) -> [$scalar; $count] {
                [$(self.$field),+]
            }
        }

        impl From<[$scalar; $count]> for $name {
            #[inline]
            fn from(a: [$scalar; $count]) -> Self { Self::from_array(a) }
        }

        impl From<$name> for [$scalar; $count] {
            #[inline]
            fn from(v: $name) -> Self { v.to_array() }
        }

        $crate::cl::macros::decl_integer_vector!(@op $name, $scalar, Add, add, "OpIAdd", [$($field),+]);
        $crate::cl::macros::decl_integer_vector!(@op $name, $scalar, Sub, sub, "OpISub", [$($field),+]);
        $crate::cl::macros::decl_integer_vector!(@op $name, $scalar, Mul, mul, "OpIMul", [$($field),+]);
        $crate::cl::macros::decl_integer_vector!(@div $name, $scalar, $signed, [$($field),+]);

        // Scalar arithmetic — splat-then-vector-op for all four. SPIR-V has
        // no integer equivalent of `OpVectorTimesScalar`.
        $crate::cl::macros::decl_integer_vector!(@scalar_op $name, $scalar, Add, add, [$($field),+]);
        $crate::cl::macros::decl_integer_vector!(@scalar_op $name, $scalar, Sub, sub, [$($field),+]);
        $crate::cl::macros::decl_integer_vector!(@scalar_op $name, $scalar, Mul, mul, [$($field),+]);
        $crate::cl::macros::decl_integer_vector!(@scalar_op $name, $scalar, Div, div, [$($field),+]);

        // Compound-assignment forms.
        $crate::cl::macros::decl_assign_op!($name, AddAssign, add_assign, Add, add);
        $crate::cl::macros::decl_assign_op!($name, SubAssign, sub_assign, Sub, sub);
        $crate::cl::macros::decl_assign_op!($name, MulAssign, mul_assign, Mul, mul);
        $crate::cl::macros::decl_assign_op!($name, DivAssign, div_assign, Div, div);
        $crate::cl::macros::decl_assign_op!($name, $scalar, AddAssign, add_assign, Add, add);
        $crate::cl::macros::decl_assign_op!($name, $scalar, SubAssign, sub_assign, Sub, sub);
        $crate::cl::macros::decl_assign_op!($name, $scalar, MulAssign, mul_assign, Mul, mul);
        $crate::cl::macros::decl_assign_op!($name, $scalar, DivAssign, div_assign, Div, div);

        impl $crate::sealed::Sealed for $name {}
        impl $crate::ScalarComposite for $name {
            #[inline]
            fn transform<F: $crate::ScalarOrVectorTransform>(self, f: &mut F) -> Self {
                f.transform_vector(self)
            }
        }
        unsafe impl $crate::ScalarOrVector for $name {
            type Scalar = $scalar;
            const N: core::num::NonZeroUsize = core::num::NonZeroUsize::new($count).unwrap();
        }
        unsafe impl $crate::Vector<$scalar, $count> for $name {}

        // Host-only `Componentwise` impl so this type flows through the
        // host arms of `opencl_std::dot`/`length`/etc. uniformly with glam.
        #[cfg(not(target_arch = "spirv"))]
        impl $crate::arch::opencl_std::Componentwise for $name {
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

        const _: () = {
            assert!(core::mem::size_of::<$name>() == $size);
            assert!(core::mem::align_of::<$name>() == $align);
        };
    };

    (@op $name:ident, $scalar:ty, $trait:ident, $method:ident, $opname:literal, [$($field:ident),+]) => {
        impl core::ops::$trait for $name {
            type Output = Self;

            #[cfg(target_arch = "spirv")]
            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                let mut result = Self::default();
                unsafe {
                    core::arch::asm!(
                        "%a = OpLoad _ {a}",
                        "%b = OpLoad _ {b}",
                        concat!("%result = ", $opname, " typeof*{result} %a %b"),
                        "OpStore {result} %result",
                        a = in(reg) &self,
                        b = in(reg) &rhs,
                        result = in(reg) &mut result,
                    );
                }
                result
            }

            #[cfg(not(target_arch = "spirv"))]
            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                Self { $($field: core::ops::$trait::$method(self.$field, rhs.$field)),+ }
            }
        }
    };

    (@div $name:ident, $scalar:ty, signed,   [$($field:ident),+]) => {
        $crate::cl::macros::decl_integer_vector!(@op $name, $scalar, Div, div, "OpSDiv", [$($field),+]);
    };
    (@div $name:ident, $scalar:ty, unsigned, [$($field:ident),+]) => {
        $crate::cl::macros::decl_integer_vector!(@op $name, $scalar, Div, div, "OpUDiv", [$($field),+]);
    };

    // Splat-then-vector-op for both directions of scalar arithmetic.
    (@scalar_op $name:ident, $scalar:ty, $trait:ident, $method:ident, [$($field:ident),+]) => {
        impl core::ops::$trait<$scalar> for $name {
            type Output = Self;

            #[cfg(target_arch = "spirv")]
            #[inline]
            fn $method(self, rhs: $scalar) -> Self::Output {
                core::ops::$trait::$method(self, $name::splat(rhs))
            }

            #[cfg(not(target_arch = "spirv"))]
            #[inline]
            fn $method(self, rhs: $scalar) -> Self::Output {
                Self { $($field: core::ops::$trait::$method(self.$field, rhs)),+ }
            }
        }

        impl core::ops::$trait<$name> for $scalar {
            type Output = $name;

            #[cfg(target_arch = "spirv")]
            #[inline]
            fn $method(self, rhs: $name) -> Self::Output {
                core::ops::$trait::$method($name::splat(self), rhs)
            }

            #[cfg(not(target_arch = "spirv"))]
            #[inline]
            fn $method(self, rhs: $name) -> Self::Output {
                $name { $($field: core::ops::$trait::$method(self, rhs.$field)),+ }
            }
        }
    };
}

pub(crate) use decl_integer_vector;

/// Declares one compound-assignment trait impl by delegating to the
/// matching non-assign op. Two arms — vector-vector and vector-scalar —
/// distinguished by arity: three idents = vector-vector, four = vector-scalar.
macro_rules! decl_assign_op {
    // Vector op= Vector  (delegates to `op` impl for `(Self, Self) -> Self`)
    ($name:ident, $assign_trait:ident, $assign_method:ident, $op_trait:ident, $op_method:ident) => {
        impl core::ops::$assign_trait for $name {
            #[inline]
            fn $assign_method(&mut self, rhs: Self) {
                *self = core::ops::$op_trait::$op_method(*self, rhs);
            }
        }
    };
    // Vector op= Scalar  (delegates to `op<Scalar>` impl for `(Self, Scalar) -> Self`)
    ($name:ident, $scalar:ty, $assign_trait:ident, $assign_method:ident, $op_trait:ident, $op_method:ident) => {
        impl core::ops::$assign_trait<$scalar> for $name {
            #[inline]
            fn $assign_method(&mut self, rhs: $scalar) {
                *self = core::ops::$op_trait::$op_method(*self, rhs);
            }
        }
    };
}

pub(crate) use decl_assign_op;

/// `ZERO` / `ONE` constants — works for any width and scalar type
/// where `splat` exists. Generated once per concrete `cl::*` type.
macro_rules! decl_common_constants {
    ($name:ident, $scalar:ty) => {
        impl $name {
            /// All components set to `0`.
            pub const ZERO: Self = Self::splat(0 as $scalar);
            /// All components set to `1`.
            pub const ONE: Self = Self::splat(1 as $scalar);
        }
    };
}

pub(crate) use decl_common_constants;

/// `xyzw` accessor methods plus axis-direction unit constants
/// (`X`, `Y`, `Z`, `W`) for narrow widths (2, 3, 4). The `OpenCL` C
/// spec lists `xyzw` as aliases for `s0..s3` on widths up to 4 only.
macro_rules! decl_xyzw_w2 {
    ($name:ident, $scalar:ty) => {
        impl $name {
            /// First component (`s0`).
            #[inline]
            #[must_use]
            pub const fn x(self) -> $scalar {
                self.s0
            }
            /// Second component (`s1`).
            #[inline]
            #[must_use]
            pub const fn y(self) -> $scalar {
                self.s1
            }
            /// Unit vector along the `x` axis.
            pub const X: Self = Self::new(1 as $scalar, 0 as $scalar);
            /// Unit vector along the `y` axis.
            pub const Y: Self = Self::new(0 as $scalar, 1 as $scalar);
        }
    };
}

macro_rules! decl_xyzw_w3 {
    ($name:ident, $scalar:ty) => {
        impl $name {
            /// First component (`s0`).
            #[inline]
            #[must_use]
            pub const fn x(self) -> $scalar {
                self.s0
            }
            /// Second component (`s1`).
            #[inline]
            #[must_use]
            pub const fn y(self) -> $scalar {
                self.s1
            }
            /// Third component (`s2`).
            #[inline]
            #[must_use]
            pub const fn z(self) -> $scalar {
                self.s2
            }
            /// Unit vector along the `x` axis.
            pub const X: Self = Self::new(1 as $scalar, 0 as $scalar, 0 as $scalar);
            /// Unit vector along the `y` axis.
            pub const Y: Self = Self::new(0 as $scalar, 1 as $scalar, 0 as $scalar);
            /// Unit vector along the `z` axis.
            pub const Z: Self = Self::new(0 as $scalar, 0 as $scalar, 1 as $scalar);
        }
    };
}

macro_rules! decl_xyzw_w4 {
    ($name:ident, $scalar:ty) => {
        impl $name {
            /// First component (`s0`).
            #[inline]
            #[must_use]
            pub const fn x(self) -> $scalar {
                self.s0
            }
            /// Second component (`s1`).
            #[inline]
            #[must_use]
            pub const fn y(self) -> $scalar {
                self.s1
            }
            /// Third component (`s2`).
            #[inline]
            #[must_use]
            pub const fn z(self) -> $scalar {
                self.s2
            }
            /// Fourth component (`s3`).
            #[inline]
            #[must_use]
            pub const fn w(self) -> $scalar {
                self.s3
            }
            /// Unit vector along the `x` axis.
            pub const X: Self = Self::new(1 as $scalar, 0 as $scalar, 0 as $scalar, 0 as $scalar);
            /// Unit vector along the `y` axis.
            pub const Y: Self = Self::new(0 as $scalar, 1 as $scalar, 0 as $scalar, 0 as $scalar);
            /// Unit vector along the `z` axis.
            pub const Z: Self = Self::new(0 as $scalar, 0 as $scalar, 1 as $scalar, 0 as $scalar);
            /// Unit vector along the `w` axis.
            pub const W: Self = Self::new(0 as $scalar, 0 as $scalar, 0 as $scalar, 1 as $scalar);
        }
    };
}

pub(crate) use decl_xyzw_w2;
pub(crate) use decl_xyzw_w3;
pub(crate) use decl_xyzw_w4;

/// `extend(scalar) -> wider type` — `Float2.extend(z) -> Float3`,
/// `Float3.extend(w) -> Float4`. The standard graphics idiom for going
/// from rgb → rgba or xy → xyz.
macro_rules! decl_extend_w2_to_w3 {
    ($src:ident, $dst:ident, $scalar:ty) => {
        impl $src {
            #[doc = concat!("Extends `self` to a [`", stringify!($dst), "`] by appending `s2`.")]
            #[inline]
            #[must_use]
            pub const fn extend(self, s2: $scalar) -> $dst {
                $dst::new(self.s0, self.s1, s2)
            }
        }
    };
}

macro_rules! decl_extend_w3_to_w4 {
    ($src:ident, $dst:ident, $scalar:ty) => {
        impl $src {
            #[doc = concat!("Extends `self` to a [`", stringify!($dst), "`] by appending `s3`.")]
            #[inline]
            #[must_use]
            pub const fn extend(self, s3: $scalar) -> $dst {
                $dst::new(self.s0, self.s1, self.s2, s3)
            }
        }
    };
}

pub(crate) use decl_extend_w2_to_w3;
pub(crate) use decl_extend_w3_to_w4;

/// Per-component conversion between two `cl::*` types with the same
/// width but different scalar types. On SPIR-V this lowers to a single
/// `OpConvertFToU`/`OpConvertSToF`/`OpFConvert`/etc. (the chosen opcode
/// is passed in as a string literal). On host we walk the components
/// via `as` casts — Rust's primitive `as` covers every integer/float
/// pair the spec allows.
macro_rules! decl_componentwise_convert {
    ($src:ident, $dst:ident, $method:ident, $opname:literal, [$($field:ident),+]) => {
        impl $src {
            #[doc = concat!("Componentwise conversion to [`", stringify!($dst), "`].")]
            ///
            #[doc = concat!("Lowers to a single `", $opname, "` on SPIR-V; on host each component")]
            /// is cast individually with Rust's `as` operator.
            #[inline]
            #[must_use]
            pub fn $method(self) -> $dst {
                #[cfg(target_arch = "spirv")]
                {
                    let mut result = <$dst as ::core::default::Default>::default();
                    unsafe {
                        ::core::arch::asm!(
                            "%a = OpLoad _ {a}",
                            ::core::concat!("%result = ", $opname, " typeof*{result} %a"),
                            "OpStore {result} %result",
                            a = in(reg) &self,
                            result = in(reg) &mut result,
                        );
                    }
                    result
                }
                #[cfg(not(target_arch = "spirv"))]
                {
                    $dst::new($(self.$field as _),+)
                }
            }
        }
    };
}

pub(crate) use decl_componentwise_convert;

/// Glam-style ergonomic methods on a `cl::Float*` / `Double*` type that
/// wrap the matching free functions in [`crate::arch::opencl_std`].
/// Generated once per concrete `cl::*` float type.
///
/// Each method is a one-line forward to `ocl::*` — same SPIR-V codegen,
/// same host-side fallback. The free-function form remains public for
/// the cases that don't fit a method (multi-output `fract`/`modf`/
/// `frexp`/`sincos`, the precision-trade-off `native_*` ops, generic
/// `<V: FloatOrFloatVector>` user code).
///
/// `recip()` is intentionally `1.0 / self` (lowers to per-component
/// `OpFDiv`, IEEE-correct), **not** `ocl::native_recip`. The `native_*`
/// ops give implementation-defined precision and must be opted into
/// explicitly via `ocl::native_*`.
macro_rules! decl_float_vector_methods {
    ($name:ident, $scalar:ty) => {
        impl $name {
            // ── Geometric ─────────────────────────────────────────
            /// Vector dot product `Σᵢ self[i] * other[i]`. Lowers to
            /// `OpDot`. See [`crate::arch::opencl_std::dot`].
            #[inline]
            #[must_use]
            pub fn dot(self, other: Self) -> $scalar {
                $crate::arch::opencl_std::dot(self, other)
            }
            /// Vector length (Euclidean norm), `sqrt(self·self)`.
            #[inline]
            #[must_use]
            pub fn length(self) -> $scalar {
                $crate::arch::opencl_std::length(self)
            }
            /// `dot(self, self)` — saves a `sqrt` over `.length()` when
            /// only the squared magnitude is needed (length comparisons,
            /// gradient norms). No direct `OpenCL.std` equivalent.
            #[inline]
            #[must_use]
            pub fn length_squared(self) -> $scalar {
                $crate::arch::opencl_std::dot(self, self)
            }
            /// Distance to `other`: `length(self - other)`.
            #[inline]
            #[must_use]
            pub fn distance(self, other: Self) -> $scalar {
                $crate::arch::opencl_std::distance(self, other)
            }
            /// `self / length(self)` — returns a unit vector in the
            /// same direction.
            #[inline]
            #[must_use]
            pub fn normalize(self) -> Self {
                $crate::arch::opencl_std::normalize(self)
            }

            // ── Per-component unary ───────────────────────────────
            /// Componentwise `|self|`.
            #[inline]
            #[must_use]
            pub fn abs(self) -> Self {
                $crate::arch::opencl_std::fabs(self)
            }
            /// Componentwise `floor(self)`.
            #[inline]
            #[must_use]
            pub fn floor(self) -> Self {
                $crate::arch::opencl_std::floor(self)
            }
            /// Componentwise `ceil(self)`.
            #[inline]
            #[must_use]
            pub fn ceil(self) -> Self {
                $crate::arch::opencl_std::ceil(self)
            }
            /// Componentwise `round(self)` (ties away from zero).
            #[inline]
            #[must_use]
            pub fn round(self) -> Self {
                $crate::arch::opencl_std::round(self)
            }
            /// Componentwise `trunc(self)` (toward zero).
            #[inline]
            #[must_use]
            pub fn trunc(self) -> Self {
                $crate::arch::opencl_std::trunc(self)
            }
            /// Componentwise `1 / self`. IEEE-correct via per-component
            /// `OpFDiv`. For the lower-precision native form use
            /// `ocl::native_recip` directly.
            #[inline]
            #[must_use]
            pub fn recip(self) -> Self {
                Self::splat(1 as $scalar) / self
            }
            /// Componentwise `sqrt(self)`.
            #[inline]
            #[must_use]
            pub fn sqrt(self) -> Self {
                $crate::arch::opencl_std::sqrt(self)
            }
            /// Componentwise `1 / sqrt(self)`.
            #[inline]
            #[must_use]
            pub fn rsqrt(self) -> Self {
                $crate::arch::opencl_std::rsqrt(self)
            }
            /// Componentwise sign: `-1`, `0`, or `+1`.
            #[inline]
            #[must_use]
            pub fn signum(self) -> Self {
                $crate::arch::opencl_std::sign(self)
            }
            /// Componentwise `exp(self)`.
            #[inline]
            #[must_use]
            pub fn exp(self) -> Self {
                $crate::arch::opencl_std::exp(self)
            }
            /// Componentwise `2^self`.
            #[inline]
            #[must_use]
            pub fn exp2(self) -> Self {
                $crate::arch::opencl_std::exp2(self)
            }
            /// Componentwise `ln(self)`.
            #[inline]
            #[must_use]
            pub fn ln(self) -> Self {
                $crate::arch::opencl_std::log(self)
            }
            /// Componentwise `log2(self)`.
            #[inline]
            #[must_use]
            pub fn log2(self) -> Self {
                $crate::arch::opencl_std::log2(self)
            }
            /// Componentwise `log10(self)`.
            #[inline]
            #[must_use]
            pub fn log10(self) -> Self {
                $crate::arch::opencl_std::log10(self)
            }
            /// Componentwise `sin(self)`.
            #[inline]
            #[must_use]
            pub fn sin(self) -> Self {
                $crate::arch::opencl_std::sin(self)
            }
            /// Componentwise `cos(self)`.
            #[inline]
            #[must_use]
            pub fn cos(self) -> Self {
                $crate::arch::opencl_std::cos(self)
            }
            /// Componentwise `tan(self)`.
            #[inline]
            #[must_use]
            pub fn tan(self) -> Self {
                $crate::arch::opencl_std::tan(self)
            }
            /// Componentwise `asin(self)`.
            #[inline]
            #[must_use]
            pub fn asin(self) -> Self {
                $crate::arch::opencl_std::asin(self)
            }
            /// Componentwise `acos(self)`.
            #[inline]
            #[must_use]
            pub fn acos(self) -> Self {
                $crate::arch::opencl_std::acos(self)
            }
            /// Componentwise `atan(self)`.
            #[inline]
            #[must_use]
            pub fn atan(self) -> Self {
                $crate::arch::opencl_std::atan(self)
            }
            /// Componentwise hyperbolic sin/cos/tan/etc.
            #[inline]
            #[must_use]
            pub fn sinh(self) -> Self {
                $crate::arch::opencl_std::sinh(self)
            }
            /// Componentwise hyperbolic cosine.
            #[inline]
            #[must_use]
            pub fn cosh(self) -> Self {
                $crate::arch::opencl_std::cosh(self)
            }
            /// Componentwise hyperbolic tangent.
            #[inline]
            #[must_use]
            pub fn tanh(self) -> Self {
                $crate::arch::opencl_std::tanh(self)
            }

            // ── Per-component binary ──────────────────────────────
            /// Componentwise `pow(self, n)` — every component raised to `n`.
            #[inline]
            #[must_use]
            pub fn powf(self, n: $scalar) -> Self {
                $crate::arch::opencl_std::pow(self, Self::splat(n))
            }
            /// Componentwise `atan2(self, other)`.
            #[inline]
            #[must_use]
            pub fn atan2(self, other: Self) -> Self {
                $crate::arch::opencl_std::atan2(self, other)
            }
            /// Componentwise magnitude of `self` with the sign of `sign`.
            #[inline]
            #[must_use]
            pub fn copysign(self, sign: Self) -> Self {
                $crate::arch::opencl_std::copysign(self, sign)
            }
            /// Componentwise minimum.
            #[inline]
            #[must_use]
            pub fn min(self, other: Self) -> Self {
                $crate::arch::opencl_std::fmin(self, other)
            }
            /// Componentwise maximum.
            #[inline]
            #[must_use]
            pub fn max(self, other: Self) -> Self {
                $crate::arch::opencl_std::fmax(self, other)
            }
            /// Componentwise floating-point modulo (sign matches `self`).
            #[inline]
            #[must_use]
            pub fn rem_euclid(self, other: Self) -> Self {
                // `OpenCL.std::fmod` returns the dividend-signed remainder,
                // which matches `f32::rem_euclid` for positive divisors.
                // Glam-style name; users mixing positive/negative divisors
                // should reach for `ocl::fmod` directly.
                $crate::arch::opencl_std::fmod(self, other)
            }

            // ── Per-component ternary ─────────────────────────────
            /// Componentwise clamp to `[min, max]`.
            #[inline]
            #[must_use]
            pub fn clamp(self, min: Self, max: Self) -> Self {
                $crate::arch::opencl_std::clamp(self, min, max)
            }
            /// Componentwise linear interpolation: `self + (other - self) * t`.
            /// Aliased as [`Self::mix`] for callers using `OpenCL`/GLSL terminology.
            #[inline]
            #[must_use]
            pub fn lerp(self, other: Self, t: $scalar) -> Self {
                $crate::arch::opencl_std::mix(self, other, Self::splat(t))
            }
            /// Componentwise linear interpolation. Alias for [`Self::lerp`] —
            /// kept for symmetry with the `OpenCL` C / GLSL spelling.
            #[inline]
            #[must_use]
            pub fn mix(self, other: Self, t: $scalar) -> Self {
                self.lerp(other, t)
            }
            /// Smooth Hermite interpolation between `0` and `1` for `self`
            /// in `[edge0, edge1]`.
            #[inline]
            #[must_use]
            pub fn smoothstep(self, edge0: Self, edge1: Self) -> Self {
                $crate::arch::opencl_std::smoothstep(edge0, edge1, self)
            }
            /// Fused multiply-add: `self * a + b`, IEEE single-rounding.
            #[inline]
            #[must_use]
            pub fn mul_add(self, a: Self, b: Self) -> Self {
                $crate::arch::opencl_std::fma(self, a, b)
            }
        }
    };
}

pub(crate) use decl_float_vector_methods;

/// Methods specific to vector widths 3 and 4 — `cross` is only well-
/// defined there per the `OpenCL.std` spec.
macro_rules! decl_float_vector_methods_cross {
    ($name:ident) => {
        impl $name {
            /// Cross product. Defined for widths 3 and 4 per the
            /// `OpenCL.std` spec.
            #[inline]
            #[must_use]
            pub fn cross(self, other: Self) -> Self {
                $crate::arch::opencl_std::cross(self, other)
            }
        }
    };
}

pub(crate) use decl_float_vector_methods_cross;

/// Glam-style methods on integer vector types. Routes `min`/`max`/`clamp`
/// to the signed or unsigned `OpenCL.std` opcode based on the `$signed`
/// tag (same dispatch the existing `@div` arm uses).
macro_rules! decl_integer_vector_methods {
    ($name:ident, $scalar:ty, signed) => {
        impl $name {
            /// Componentwise `|self|`.
            #[inline]
            #[must_use]
            pub fn abs(self) -> Self {
                $crate::arch::opencl_std::s_abs(self)
            }
            /// Componentwise minimum (signed).
            #[inline]
            #[must_use]
            pub fn min(self, other: Self) -> Self {
                $crate::arch::opencl_std::s_min(self, other)
            }
            /// Componentwise maximum (signed).
            #[inline]
            #[must_use]
            pub fn max(self, other: Self) -> Self {
                $crate::arch::opencl_std::s_max(self, other)
            }
            /// Componentwise clamp to `[min, max]` (signed).
            #[inline]
            #[must_use]
            pub fn clamp(self, min: Self, max: Self) -> Self {
                $crate::arch::opencl_std::s_clamp(self, min, max)
            }
            /// Number of set bits, componentwise.
            #[inline]
            #[must_use]
            pub fn count_ones(self) -> Self {
                $crate::arch::opencl_std::popcount(self)
            }
            /// Count leading zero bits, componentwise.
            #[inline]
            #[must_use]
            pub fn leading_zeros(self) -> Self {
                $crate::arch::opencl_std::clz(self)
            }
            /// Count trailing zero bits, componentwise.
            #[inline]
            #[must_use]
            pub fn trailing_zeros(self) -> Self {
                $crate::arch::opencl_std::ctz(self)
            }
        }
    };
    ($name:ident, $scalar:ty, unsigned) => {
        impl $name {
            /// Componentwise minimum (unsigned).
            #[inline]
            #[must_use]
            pub fn min(self, other: Self) -> Self {
                $crate::arch::opencl_std::u_min(self, other)
            }
            /// Componentwise maximum (unsigned).
            #[inline]
            #[must_use]
            pub fn max(self, other: Self) -> Self {
                $crate::arch::opencl_std::u_max(self, other)
            }
            /// Componentwise clamp to `[min, max]` (unsigned).
            #[inline]
            #[must_use]
            pub fn clamp(self, min: Self, max: Self) -> Self {
                $crate::arch::opencl_std::u_clamp(self, min, max)
            }
            /// Number of set bits, componentwise.
            #[inline]
            #[must_use]
            pub fn count_ones(self) -> Self {
                $crate::arch::opencl_std::popcount(self)
            }
            /// Count leading zero bits, componentwise.
            #[inline]
            #[must_use]
            pub fn leading_zeros(self) -> Self {
                $crate::arch::opencl_std::clz(self)
            }
            /// Count trailing zero bits, componentwise.
            #[inline]
            #[must_use]
            pub fn trailing_zeros(self) -> Self {
                $crate::arch::opencl_std::ctz(self)
            }
        }
    };
}

pub(crate) use decl_integer_vector_methods;
