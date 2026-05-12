use core::ops::Range;
use spirv_std::cl::*;

pub struct BumpAlloc(usize);

impl BumpAlloc {
    pub fn inc(&mut self) -> usize {
        let old = self.0;
        self.0 += 1;
        old
    }
}

macro_rules! write_layout {
    ($out:ident, $offset:ident, $name:ident($($member:ident),*)) => {
        {
            $out[$offset.inc()] = size_of::<$name>() as u32;
            $out[$offset.inc()] = align_of::<$name>() as u32;
            $($out[$offset.inc()] = core::mem::offset_of!($name, $member) as u32;)*
        }
    };
}

pub const LAYOUT_RANGE: Range<usize> = 0..30;
pub const LAYOUT_MAX_SIZE: usize = 0x100 / 0x4;
pub const LAYOUT_LEN: usize = LAYOUT_RANGE.end * LAYOUT_MAX_SIZE;

pub fn eval_cl_layouts(gid: u32, out: &mut [u32]) {
    let mut offset = BumpAlloc(gid as usize * LAYOUT_MAX_SIZE);
    match gid {
        // Float
        0x00 => write_layout!(out, offset, Float2(s0, s1)),
        0x01 => write_layout!(out, offset, Float3(s0, s1, s2)),
        0x02 => write_layout!(out, offset, Float4(s0, s1, s2, s3)),
        // Double
        0x03 => write_layout!(out, offset, Double2(s0, s1)),
        0x04 => write_layout!(out, offset, Double3(s0, s1, s2)),
        0x05 => write_layout!(out, offset, Double4(s0, s1, s2, s3)),
        // Char (i8)
        0x06 => write_layout!(out, offset, Char2(s0, s1)),
        0x07 => write_layout!(out, offset, Char3(s0, s1, s2)),
        0x08 => write_layout!(out, offset, Char4(s0, s1, s2, s3)),
        // UChar (u8)
        0x09 => write_layout!(out, offset, UChar2(s0, s1)),
        0x0a => write_layout!(out, offset, UChar3(s0, s1, s2)),
        0x0b => write_layout!(out, offset, UChar4(s0, s1, s2, s3)),
        // Short (i16)
        0x0c => write_layout!(out, offset, Short2(s0, s1)),
        0x0d => write_layout!(out, offset, Short3(s0, s1, s2)),
        0x0e => write_layout!(out, offset, Short4(s0, s1, s2, s3)),
        // UShort (u16)
        0x0f => write_layout!(out, offset, UShort2(s0, s1)),
        0x10 => write_layout!(out, offset, UShort3(s0, s1, s2)),
        0x11 => write_layout!(out, offset, UShort4(s0, s1, s2, s3)),
        // Int (i32)
        0x12 => write_layout!(out, offset, Int2(s0, s1)),
        0x13 => write_layout!(out, offset, Int3(s0, s1, s2)),
        0x14 => write_layout!(out, offset, Int4(s0, s1, s2, s3)),
        // UInt (u32)
        0x15 => write_layout!(out, offset, UInt2(s0, s1)),
        0x16 => write_layout!(out, offset, UInt3(s0, s1, s2)),
        0x17 => write_layout!(out, offset, UInt4(s0, s1, s2, s3)),
        // Long (i64)
        0x18 => write_layout!(out, offset, Long2(s0, s1)),
        0x19 => write_layout!(out, offset, Long3(s0, s1, s2)),
        0x1a => write_layout!(out, offset, Long4(s0, s1, s2, s3)),
        // ULong (u64)
        0x1b => write_layout!(out, offset, ULong2(s0, s1)),
        0x1c => write_layout!(out, offset, ULong3(s0, s1, s2)),
        0x1d => write_layout!(out, offset, ULong4(s0, s1, s2, s3)),
        _ => {}
    }
}
