//! `OpenCL` integer vector types.
//!
//! - `Char{2,3,4,8,16}` (i8), `UChar{2,3,4,8,16}` (u8)
//! - `Short{2,3,4,8,16}` (i16), `UShort{2,3,4,8,16}` (u16)
//! - `Int{2,3,4,8,16}` (i32), `UInt{2,3,4,8,16}` (u32)
//! - `Long{2,3,4,8,16}` (i64), `ULong{2,3,4,8,16}` (u64)
//!
//! Layout follows `OpenCL` C: `T3` is sized and aligned as `T4`, and the wider
//! types follow the `power-of-two * scalar size` rule. `Char*`/`UChar*`
//! require `Int8`, `Short*`/`UShort*` require `Int16`, `Long*`/`ULong*`
//! require `Int64` — all auto-enabled on Kernel targets. Widths 8 and 16
//! additionally require `Vector16`, also auto-enabled.

use crate::cl::macros::{
    decl_common_constants, decl_extend_w2_to_w3, decl_extend_w3_to_w4, decl_integer_vector,
    decl_integer_vector_methods, decl_xyzw_w2, decl_xyzw_w3, decl_xyzw_w4,
};

/// Declares all 5 widths (2/3/4/8/16) for one integer scalar family.
/// Sizes/aligns are passed directly because the `OpenCL` C ABI rule
/// (`align = size = next_pow2(N) * scalar_size`) is uniform but the macro
/// does not have ident-concatenation — the type names are spelled out.
macro_rules! decl_integer_family {
    (
        scalar = $scalar:ty, signed = $signed:tt,
        $n2:ident,  align2  = $a2:literal,
        $n3:ident,  align3  = $a3:literal,
        $n4:ident,  align4  = $a4:literal,
        $n8:ident,  align8  = $a8:literal,
        $n16:ident, align16 = $a16:literal,
    ) => {
        decl_integer_vector!($n2,  scalar = $scalar, signed = $signed,
            count = 2,  align = $a2,  size = $a2,
            fields = [s0: 0, s1: 1],);
        decl_integer_vector!($n3,  scalar = $scalar, signed = $signed,
            count = 3,  align = $a3,  size = $a3,
            fields = [s0: 0, s1: 1, s2: 2],);
        decl_integer_vector!($n4,  scalar = $scalar, signed = $signed,
            count = 4,  align = $a4,  size = $a4,
            fields = [s0: 0, s1: 1, s2: 2, s3: 3],);
        decl_integer_vector!($n8,  scalar = $scalar, signed = $signed,
            count = 8,  align = $a8,  size = $a8,
            fields = [s0: 0, s1: 1, s2: 2, s3: 3, s4: 4, s5: 5, s6: 6, s7: 7],);
        decl_integer_vector!($n16, scalar = $scalar, signed = $signed,
            count = 16, align = $a16, size = $a16,
            fields = [s0: 0, s1: 1, s2: 2, s3: 3, s4: 4, s5: 5, s6: 6, s7: 7,
                      s8: 8, s9: 9, sa: 10, sb: 11, sc: 12, sd: 13, se: 14, sf: 15],);

        // ZERO/ONE for every width.
        decl_common_constants!($n2,  $scalar);
        decl_common_constants!($n3,  $scalar);
        decl_common_constants!($n4,  $scalar);
        decl_common_constants!($n8,  $scalar);
        decl_common_constants!($n16, $scalar);

        // xyzw accessors + axis constants (widths 2/3/4 only).
        decl_xyzw_w2!($n2, $scalar);
        decl_xyzw_w3!($n3, $scalar);
        decl_xyzw_w4!($n4, $scalar);

        // extend ladder.
        decl_extend_w2_to_w3!($n2, $n3, $scalar);
        decl_extend_w3_to_w4!($n3, $n4, $scalar);

        // Glam-style methods. The macro routes `min`/`max`/`clamp` (and
        // `abs`, signed-only) to the right `s_*` / `u_*` opcode based
        // on the family's `$signed` tag.
        decl_integer_vector_methods!($n2,  $scalar, $signed);
        decl_integer_vector_methods!($n3,  $scalar, $signed);
        decl_integer_vector_methods!($n4,  $scalar, $signed);
        decl_integer_vector_methods!($n8,  $scalar, $signed);
        decl_integer_vector_methods!($n16, $scalar, $signed);
    };
}

decl_integer_family!(
    scalar = i8,
    signed = signed,
    Char2,
    align2 = 2,
    Char3,
    align3 = 4,
    Char4,
    align4 = 4,
    Char8,
    align8 = 8,
    Char16,
    align16 = 16,
);
decl_integer_family!(
    scalar = u8,
    signed = unsigned,
    UChar2,
    align2 = 2,
    UChar3,
    align3 = 4,
    UChar4,
    align4 = 4,
    UChar8,
    align8 = 8,
    UChar16,
    align16 = 16,
);

decl_integer_family!(
    scalar = i16,
    signed = signed,
    Short2,
    align2 = 4,
    Short3,
    align3 = 8,
    Short4,
    align4 = 8,
    Short8,
    align8 = 16,
    Short16,
    align16 = 32,
);
decl_integer_family!(
    scalar = u16,
    signed = unsigned,
    UShort2,
    align2 = 4,
    UShort3,
    align3 = 8,
    UShort4,
    align4 = 8,
    UShort8,
    align8 = 16,
    UShort16,
    align16 = 32,
);

decl_integer_family!(
    scalar = i32,
    signed = signed,
    Int2,
    align2 = 8,
    Int3,
    align3 = 16,
    Int4,
    align4 = 16,
    Int8,
    align8 = 32,
    Int16,
    align16 = 64,
);
decl_integer_family!(
    scalar = u32,
    signed = unsigned,
    UInt2,
    align2 = 8,
    UInt3,
    align3 = 16,
    UInt4,
    align4 = 16,
    UInt8,
    align8 = 32,
    UInt16,
    align16 = 64,
);

decl_integer_family!(
    scalar = i64,
    signed = signed,
    Long2,
    align2 = 16,
    Long3,
    align3 = 32,
    Long4,
    align4 = 32,
    Long8,
    align8 = 64,
    Long16,
    align16 = 128,
);
decl_integer_family!(
    scalar = u64,
    signed = unsigned,
    ULong2,
    align2 = 16,
    ULong3,
    align3 = 32,
    ULong4,
    align4 = 32,
    ULong8,
    align8 = 64,
    ULong16,
    align16 = 128,
);
