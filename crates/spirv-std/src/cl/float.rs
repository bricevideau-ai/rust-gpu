//! Single-precision `OpenCL` float vector types: `Float2`, `Float3`, `Float4`,
//! `Float8`, `Float16`.
//!
//! Layouts match `OpenCL` C: `float3` is sized and aligned as `float4` (16 bytes),
//! and the wider types follow the `power-of-two * scalar size` rule.

use crate::cl::integer::{Int2, Int3, Int4, UInt2, UInt3, UInt4};
use crate::cl::macros::{
    decl_common_constants, decl_componentwise_convert, decl_extend_w2_to_w3, decl_extend_w3_to_w4,
    decl_float_vector, decl_float_vector_methods, decl_float_vector_methods_cross, decl_xyzw_w2,
    decl_xyzw_w3, decl_xyzw_w4,
};

// OpenCL C alignment / size table for `floatN`:
//   N=2 → align 8,  size 8
//   N=3 → align 16, size 16  (padded to N=4)
//   N=4 → align 16, size 16
//   N=8 → align 32, size 32
//   N=16→ align 64, size 64
decl_float_vector!(Float2,  scalar = f32, count = 2,  align = 8,  size = 8,
    fields = [s0: 0, s1: 1],);
decl_float_vector!(Float3,  scalar = f32, count = 3,  align = 16, size = 16,
    fields = [s0: 0, s1: 1, s2: 2],);
decl_float_vector!(Float4,  scalar = f32, count = 4,  align = 16, size = 16,
    fields = [s0: 0, s1: 1, s2: 2, s3: 3],);
decl_float_vector!(Float8,  scalar = f32, count = 8,  align = 32, size = 32,
    fields = [s0: 0, s1: 1, s2: 2, s3: 3, s4: 4, s5: 5, s6: 6, s7: 7],);
decl_float_vector!(Float16, scalar = f32, count = 16, align = 64, size = 64,
    fields = [s0: 0, s1: 1, s2: 2, s3: 3, s4: 4, s5: 5, s6: 6, s7: 7,
              s8: 8, s9: 9, sa: 10, sb: 11, sc: 12, sd: 13, se: 14, sf: 15],);

// `ZERO` / `ONE` constants for every width.
decl_common_constants!(Float2, f32);
decl_common_constants!(Float3, f32);
decl_common_constants!(Float4, f32);
decl_common_constants!(Float8, f32);
decl_common_constants!(Float16, f32);

// `xyzw` accessors + axis constants for narrow widths only — the
// `OpenCL` C spec restricts these to widths 2/3/4.
decl_xyzw_w2!(Float2, f32);
decl_xyzw_w3!(Float3, f32);
decl_xyzw_w4!(Float4, f32);

// `extend` ladder — `Float2 → Float3 → Float4`.
decl_extend_w2_to_w3!(Float2, Float3, f32);
decl_extend_w3_to_w4!(Float3, Float4, f32);

// Float → integer conversions for narrow widths (the most common
// pixel/index sites). Lower to a single `OpConvertFToU` /
// `OpConvertFToS` per call, matching what `glam::Vec3::as_uvec3()`
// does. Truncates toward zero (the SPIR-V spec semantics).
decl_componentwise_convert!(Float2, UInt2, as_uint2, "OpConvertFToU", [s0, s1]);
decl_componentwise_convert!(Float3, UInt3, as_uint3, "OpConvertFToU", [s0, s1, s2]);
decl_componentwise_convert!(Float4, UInt4, as_uint4, "OpConvertFToU", [s0, s1, s2, s3]);
decl_componentwise_convert!(Float2, Int2, as_int2, "OpConvertFToS", [s0, s1]);
decl_componentwise_convert!(Float3, Int3, as_int3, "OpConvertFToS", [s0, s1, s2]);
decl_componentwise_convert!(Float4, Int4, as_int4, "OpConvertFToS", [s0, s1, s2, s3]);

// Glam-style ergonomic methods: `v.dot(w)` / `v.length()` / etc.
// Each method is a paper-thin wrapper around the matching
// `spirv_std::arch::opencl_std::*` free function — same SPIR-V codegen,
// nicer call site. Free functions remain `pub` for cases that don't fit
// methods (multi-output ops, `native_*` precision tradeoffs, generic
// kernel code over `FloatOrFloatVector`).
decl_float_vector_methods!(Float2, f32);
decl_float_vector_methods!(Float3, f32);
decl_float_vector_methods!(Float4, f32);
decl_float_vector_methods!(Float8, f32);
decl_float_vector_methods!(Float16, f32);

// `cross` is only well-defined for widths 3 and 4.
decl_float_vector_methods_cross!(Float3);
decl_float_vector_methods_cross!(Float4);
