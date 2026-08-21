//! support functions for debug printf

use crate::{Scalar, Vector};

#[doc(hidden)]
pub fn assert_is_type<T>(ty: T) -> T {
    ty
}

#[doc(hidden)]
pub fn assert_is_vector<TY: Scalar, V: Vector<TY, SIZE>, const SIZE: usize>(vec: V) -> V {
    vec
}

/// Marker trait for types accepted by `%f` in `OpenCL` printf.
/// Accepts both `f32` and `f64` since `OpenCL` uses `%f` for all floats.
#[doc(hidden)]
pub trait PrintfFloat {}
impl PrintfFloat for f32 {}
impl PrintfFloat for f64 {}

#[doc(hidden)]
pub fn assert_is_float<T: PrintfFloat>(ty: T) -> T {
    ty
}

/// Marker trait for pointer types accepted by `%p` in `OpenCL` printf.
#[doc(hidden)]
pub trait PrintfPointer {}
impl<T: ?Sized> PrintfPointer for *const T {}
impl<T: ?Sized> PrintfPointer for *mut T {}

#[doc(hidden)]
pub fn assert_is_pointer<T: PrintfPointer>(ty: T) -> T {
    ty
}
