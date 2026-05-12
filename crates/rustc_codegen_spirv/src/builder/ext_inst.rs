use super::Builder;
use crate::builder_spirv::{SpirvValue, SpirvValueExt};
use crate::custom_insts;
use rspirv::dr::Operand;
use rspirv::spirv::{GlslStd450Op as GLOp, Word};

const GLSL_STD_450: &str = "GLSL.std.450";
const OPENCL_STD: &str = "OpenCL.std";

/// Manager for OpExtInst/OpExtImport instructions
#[derive(Default)]
pub struct ExtInst {
    /// See `crate::custom_insts` for more details on what this entails.
    custom: Option<Word>,

    glsl: Option<Word>,
    opencl: Option<Word>,
}

impl ExtInst {
    pub fn import_custom(&mut self, bx: &Builder<'_, '_>) -> Word {
        if let Some(id) = self.custom {
            id
        } else {
            let id = bx
                .emit_global()
                .ext_inst_import(custom_insts::CUSTOM_EXT_INST_SET.clone());
            self.custom = Some(id);
            id
        }
    }

    pub fn import_glsl(&mut self, bx: &Builder<'_, '_>) -> Word {
        if let Some(id) = self.glsl {
            id
        } else {
            let id = bx.emit_global().ext_inst_import(GLSL_STD_450);
            self.glsl = Some(id);
            id
        }
    }

    pub fn import_opencl(&mut self, bx: &Builder<'_, '_>) -> Word {
        if let Some(id) = self.opencl {
            id
        } else {
            let id = bx.emit_global().ext_inst_import(OPENCL_STD);
            self.opencl = Some(id);
            id
        }
    }
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Emit `OpExtInst <OpenCL.std> <opcode> <args...>` directly, without
    /// going through the [`GLOp`] enum. Use this for `OpenCL.std` ops that
    /// have no GLSL.std.450 counterpart — `fmin_common` (97), `fmax_common`
    /// (98), and other Kernel-only intrinsics — where translating through
    /// the `GLOp` table would either lose semantics (e.g. `f32::min`'s
    /// NaN-ignoring behaviour collapsing into the NaN-undefined `GLOp::FMin`)
    /// or have no entry at all.
    ///
    /// Caller is responsible for choosing the right opcode and ensuring
    /// the call is on a Kernel-cap module — emitting `OpenCL.std` ops on
    /// a Vulkan target would produce invalid SPIR-V.
    pub fn opencl_op(
        &mut self,
        opcode: u32,
        result_type: Word,
        args: impl AsRef<[SpirvValue]>,
    ) -> SpirvValue {
        let args = args.as_ref();
        let opencl = self.ext_inst.borrow_mut().import_opencl(self);
        self.emit()
            .ext_inst(
                result_type,
                None,
                opencl,
                opcode,
                args.iter().map(|a| Operand::IdRef(a.def(self))),
            )
            .unwrap()
            .with_type(result_type)
    }

    pub fn custom_inst(
        &mut self,
        result_type: Word,
        inst: custom_insts::CustomInst<Operand>,
    ) -> SpirvValue {
        let custom_ext_inst_set = self.ext_inst.borrow_mut().import_custom(self);
        self.emit()
            .ext_inst(
                result_type,
                None,
                custom_ext_inst_set,
                inst.op() as u32,
                inst.into_operands(),
            )
            .unwrap()
            .with_type(result_type)
    }

    /// Emit a math intrinsic. On Vulkan/Shader targets this lowers to
    /// `OpExtInst <GLSL.std.450> <op>`. On Kernel targets `GLSL.std.450`
    /// is not part of the `OpenCL` SPIR-V Environment Spec, so we
    /// substitute `OpExtInst <OpenCL.std> <equivalent_op>` instead — see
    /// [`gl_op_to_opencl_std`] for the mapping table.
    ///
    /// Three GLSL ops have a different operand shape in `OpenCL.std` and
    /// are intentionally NOT supported through this path on Kernel
    /// targets:
    ///
    /// - `Fract` — GLSL returns `x - floor(x)`; `OpenCL` writes the
    ///   integer part through a pointer. Currently unreachable from any
    ///   call site in the codegen; a future user reaching for it on a
    ///   Kernel target will see a fatal diagnostic.
    /// - `FrexpStruct` / `ModfStruct` — GLSL returns a `{primary, out}`
    ///   struct directly; `OpenCL` returns the primary scalar and writes
    ///   `out` through a pointer. The shape adaptation needs more than
    ///   a scratch out-pointer (the result type is a struct, not a
    ///   scalar — naive scratch allocation would type-mismatch). For
    ///   Kernel targets, users should reach for
    ///   [`spirv_std::arch::opencl_std::frexp`] /
    ///   [`spirv_std::arch::opencl_std::modf`] directly — those go
    ///   through inline asm with the correct two-pointer shape.
    pub fn gl_op(
        &mut self,
        op: GLOp,
        result_type: Word,
        args: impl AsRef<[SpirvValue]>,
    ) -> SpirvValue {
        let args = args.as_ref();

        // On Vulkan/Shader, emit the GLSL form.
        if !self.cx.builder.is_kernel_mode() {
            let glsl = self.ext_inst.borrow_mut().import_glsl(self);
            return self
                .emit()
                .ext_inst(
                    result_type,
                    None,
                    glsl,
                    op as u32,
                    args.iter().map(|a| Operand::IdRef(a.def(self))),
                )
                .unwrap()
                .with_type(result_type);
        }

        // On Kernel, route to OpenCL.std with a translated opcode.
        let opencl_op = gl_op_to_opencl_std(op).unwrap_or_else(|| {
            self.fatal(format!(
                "GLSL op `{op:?}` has no `OpenCL.std` equivalent or has a \
                 different operand shape — for `Fract`/`FrexpStruct`/`ModfStruct` \
                 use `spirv_std::arch::opencl_std::{{fract,frexp,modf}}` \
                 directly; other ops are not yet supported on Kernel targets"
            ))
        });
        let opencl = self.ext_inst.borrow_mut().import_opencl(self);
        self.emit()
            .ext_inst(
                result_type,
                None,
                opencl,
                opencl_op,
                args.iter().map(|a| Operand::IdRef(a.def(self))),
            )
            .unwrap()
            .with_type(result_type)
    }
}

/// Maps a `GLSL.std.450` opcode to its operand-compatible counterpart in the
/// `OpenCL.std` extended instruction set. Returns `None` for ops that have
/// no `OpenCL` equivalent (e.g. `Reflect`, `Refract`, matrix ops, packing) —
/// callers should report a clear error rather than silently miscompiling.
///
/// Ops where the operand shape differs (`Fract`, `FrexpStruct`, `ModfStruct`)
/// are intentionally NOT in this table — they need bespoke emission with a
/// scratch out-pointer; see [`Builder::gl_op`] for the dispatch.
const fn gl_op_to_opencl_std(op: GLOp) -> Option<u32> {
    use GLOp::*;
    Some(match op {
        // Unary trig / hyperbolic
        Acos => 0,
        Acosh => 1,
        Asin => 3,
        Asinh => 4,
        Atan => 6,
        Atanh => 8,
        Cos => 14,
        Cosh => 15,
        Sin => 57,
        Sinh => 59,
        Tan => 62,
        Tanh => 63,

        // Binary
        Atan2 => 7,

        // Exponential / logarithm
        Exp => 19,
        Exp2 => 20,
        Log => 37,
        Log2 => 38,
        Pow => 48,

        // Rounding / abs / sign
        FAbs => 23,
        Floor => 25,
        Ceil => 12,
        Round => 55,
        RoundEven => 53, // OpenCL `rint` (banker's rounding)
        Trunc => 66,
        FSign => 103,

        // Min/max/clamp/mix (all signed/unsigned-explicit on the OpenCL side;
        // the GLSL "F"-prefixed variants map to the float forms).
        FMin => 28, // see also fmin_common (98) for stricter NaN-undefined
        FMax => 27, // see also fmax_common (97)
        FClamp => 95,
        FMix => 99,
        SAbs => 141,
        SMin => 158,
        UMin => 159,
        SMax => 156,
        UMax => 157,
        SClamp => 149,
        UClamp => 150,

        // Geometric
        Length => 106,
        Distance => 105,
        Normalize => 107,
        Cross => 104,

        // FMA — GLSL opcode 50, OpenCL opcode 26.
        Fma => 26,

        // Misc
        InverseSqrt => 56, // OpenCL `rsqrt`
        Sqrt => 61,
        Step => 101,
        SmoothStep => 102,
        Ldexp => 34,
        FindILsb => 152, // OpenCL `ctz`

        // Ops with no OpenCL.std equivalent OR with a different operand
        // shape (handled separately in `Builder::gl_op`). Caller will hit
        // a fatal diagnostic. Listed explicitly so adding a new GLOp
        // variant in rspirv won't silently fall through.
        //
        // The shape-differing ops (`Fract`, `FrexpStruct`, `ModfStruct`)
        // are filtered out by `Builder::gl_op` before reaching this table;
        // they're listed here as `None` so a future caller that forgets
        // the special-case gets a clear error rather than wrong-shape
        // OpenCL instructions.
        Determinant
        | MatrixInverse
        | Modf
        | Frexp
        | IMix
        | Radians
        | Degrees
        | Reflect
        | Refract
        | FaceForward
        | NMin
        | NMax
        | NClamp
        | PackSnorm4x8
        | PackUnorm4x8
        | PackSnorm2x16
        | PackUnorm2x16
        | PackHalf2x16
        | PackDouble2x32
        | UnpackSnorm2x16
        | UnpackUnorm2x16
        | UnpackHalf2x16
        | UnpackSnorm4x8
        | UnpackUnorm4x8
        | UnpackDouble2x32
        | FindSMsb
        | FindUMsb
        | InterpolateAtCentroid
        | InterpolateAtSample
        | InterpolateAtOffset
        | SSign
        | Fract
        | FrexpStruct
        | ModfStruct => return None,
    })
}
