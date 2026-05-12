//! Double-precision `OpenCL` float vector types: `Double2`, `Double3`, `Double4`,
//! `Double8`, `Double16`.
//!
//! Layouts match `OpenCL` C: `double3` is sized and aligned as `double4`
//! (32 bytes), and the wider types follow the `power-of-two * scalar size`
//! rule. Requires the `Float64` SPIR-V capability — opt in with
//! `-C target-feature=+Float64`. `Double8`/`Double16` additionally need
//! `Vector16`, which is auto-enabled on Kernel targets.

use crate::cl::macros::{
    decl_common_constants, decl_extend_w2_to_w3, decl_extend_w3_to_w4, decl_float_vector,
    decl_float_vector_methods, decl_float_vector_methods_cross, decl_xyzw_w2, decl_xyzw_w3,
    decl_xyzw_w4,
};

// OpenCL C alignment / size table for `doubleN`:
//   N=2 → align 16,  size 16
//   N=3 → align 32,  size 32  (padded to N=4)
//   N=4 → align 32,  size 32
//   N=8 → align 64,  size 64
//   N=16→ align 128, size 128
decl_float_vector!(Double2,  scalar = f64, count = 2,  align = 16,  size = 16,
    fields = [s0: 0, s1: 1],);
decl_float_vector!(Double3,  scalar = f64, count = 3,  align = 32,  size = 32,
    fields = [s0: 0, s1: 1, s2: 2],);
decl_float_vector!(Double4,  scalar = f64, count = 4,  align = 32,  size = 32,
    fields = [s0: 0, s1: 1, s2: 2, s3: 3],);
decl_float_vector!(Double8,  scalar = f64, count = 8,  align = 64,  size = 64,
    fields = [s0: 0, s1: 1, s2: 2, s3: 3, s4: 4, s5: 5, s6: 6, s7: 7],);
decl_float_vector!(Double16, scalar = f64, count = 16, align = 128, size = 128,
    fields = [s0: 0, s1: 1, s2: 2, s3: 3, s4: 4, s5: 5, s6: 6, s7: 7,
              s8: 8, s9: 9, sa: 10, sb: 11, sc: 12, sd: 13, se: 14, sf: 15],);

decl_common_constants!(Double2, f64);
decl_common_constants!(Double3, f64);
decl_common_constants!(Double4, f64);
decl_common_constants!(Double8, f64);
decl_common_constants!(Double16, f64);

decl_xyzw_w2!(Double2, f64);
decl_xyzw_w3!(Double3, f64);
decl_xyzw_w4!(Double4, f64);

decl_extend_w2_to_w3!(Double2, Double3, f64);
decl_extend_w3_to_w4!(Double3, Double4, f64);

// Glam-style ergonomic methods (see `cl::float` for rationale).
decl_float_vector_methods!(Double2, f64);
decl_float_vector_methods!(Double3, f64);
decl_float_vector_methods!(Double4, f64);
decl_float_vector_methods!(Double8, f64);
decl_float_vector_methods!(Double16, f64);

decl_float_vector_methods_cross!(Double3);
decl_float_vector_methods_cross!(Double4);
