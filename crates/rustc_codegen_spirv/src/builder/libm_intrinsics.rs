// HACK(eddyb) avoids rewriting all of the imports (see `lib.rs` and `build.rs`).
use crate::maybe_pqp_cg_ssa as rustc_codegen_ssa;

use super::Builder;
use crate::builder_spirv::{SpirvValue, SpirvValueExt};
use crate::spirv_type::SpirvType;
use rspirv::dr::{InsertPoint, Instruction, Operand};
use rspirv::spirv::{Capability, GlslStd450Op as GLOp, Op, StorageClass, Word};
use rustc_codegen_ssa::traits::BuilderMethods;
use std::iter::empty;

#[derive(Copy, Clone, Debug)]
pub enum LibmCustomIntrinsic {
    CopySign,
    Cbrt,
    Erf,
    Erfc,
    Exp10,
    Expm1,
    Fdim,
    Fmod,
    Log10,
    Hypot,
    Ilogb,
    J0,
    Y0,
    J1,
    Y1,
    Jn,
    Yn,
    Lgamma,
    LgammaR,
    Tgamma,
    Log1p,
    NextAfter,
    Powi,
    Remainder,
    RemQuo,
    Scalbn,
    SinCos,
}

#[derive(Copy, Clone, Debug)]
pub enum LibmIntrinsic {
    GLOp(GLOp),
    Custom(LibmCustomIntrinsic),
}

pub const LIBM_TABLE: &[(&str, LibmIntrinsic)] = &[
    ("acos", LibmIntrinsic::GLOp(GLOp::Acos)),
    ("acosf", LibmIntrinsic::GLOp(GLOp::Acos)),
    ("acosh", LibmIntrinsic::GLOp(GLOp::Acosh)),
    ("acoshf", LibmIntrinsic::GLOp(GLOp::Acosh)),
    ("asin", LibmIntrinsic::GLOp(GLOp::Asin)),
    ("asinf", LibmIntrinsic::GLOp(GLOp::Asin)),
    ("asinh", LibmIntrinsic::GLOp(GLOp::Asinh)),
    ("asinhf", LibmIntrinsic::GLOp(GLOp::Asinh)),
    ("atan2", LibmIntrinsic::GLOp(GLOp::Atan2)),
    ("atan2f", LibmIntrinsic::GLOp(GLOp::Atan2)),
    ("atan", LibmIntrinsic::GLOp(GLOp::Atan)),
    ("atanf", LibmIntrinsic::GLOp(GLOp::Atan)),
    ("atanh", LibmIntrinsic::GLOp(GLOp::Atanh)),
    ("atanhf", LibmIntrinsic::GLOp(GLOp::Atanh)),
    ("cbrt", LibmIntrinsic::Custom(LibmCustomIntrinsic::Cbrt)),
    ("cbrtf", LibmIntrinsic::Custom(LibmCustomIntrinsic::Cbrt)),
    ("ceil", LibmIntrinsic::GLOp(GLOp::Ceil)),
    ("ceilf", LibmIntrinsic::GLOp(GLOp::Ceil)),
    (
        "copysign",
        LibmIntrinsic::Custom(LibmCustomIntrinsic::CopySign),
    ),
    (
        "copysignf",
        LibmIntrinsic::Custom(LibmCustomIntrinsic::CopySign),
    ),
    ("cos", LibmIntrinsic::GLOp(GLOp::Cos)),
    ("cosf", LibmIntrinsic::GLOp(GLOp::Cos)),
    ("cosh", LibmIntrinsic::GLOp(GLOp::Cosh)),
    ("coshf", LibmIntrinsic::GLOp(GLOp::Cosh)),
    ("erf", LibmIntrinsic::Custom(LibmCustomIntrinsic::Erf)),
    ("erff", LibmIntrinsic::Custom(LibmCustomIntrinsic::Erf)),
    ("erfc", LibmIntrinsic::Custom(LibmCustomIntrinsic::Erfc)),
    ("erfcf", LibmIntrinsic::Custom(LibmCustomIntrinsic::Erfc)),
    ("exp10", LibmIntrinsic::Custom(LibmCustomIntrinsic::Exp10)),
    ("exp10f", LibmIntrinsic::Custom(LibmCustomIntrinsic::Exp10)),
    ("exp2", LibmIntrinsic::GLOp(GLOp::Exp2)),
    ("exp2f", LibmIntrinsic::GLOp(GLOp::Exp2)),
    ("exp", LibmIntrinsic::GLOp(GLOp::Exp)),
    ("expf", LibmIntrinsic::GLOp(GLOp::Exp)),
    ("expm1", LibmIntrinsic::Custom(LibmCustomIntrinsic::Expm1)),
    ("expm1f", LibmIntrinsic::Custom(LibmCustomIntrinsic::Expm1)),
    ("fabs", LibmIntrinsic::GLOp(GLOp::FAbs)),
    ("fabsf", LibmIntrinsic::GLOp(GLOp::FAbs)),
    ("fdim", LibmIntrinsic::Custom(LibmCustomIntrinsic::Fdim)),
    ("fdimf", LibmIntrinsic::Custom(LibmCustomIntrinsic::Fdim)),
    ("floor", LibmIntrinsic::GLOp(GLOp::Floor)),
    ("floorf", LibmIntrinsic::GLOp(GLOp::Floor)),
    ("fma", LibmIntrinsic::GLOp(GLOp::Fma)),
    ("fmaf", LibmIntrinsic::GLOp(GLOp::Fma)),
    ("fmax", LibmIntrinsic::GLOp(GLOp::FMax)),
    ("fmaxf", LibmIntrinsic::GLOp(GLOp::FMax)),
    ("fmin", LibmIntrinsic::GLOp(GLOp::FMin)),
    ("fminf", LibmIntrinsic::GLOp(GLOp::FMin)),
    ("fmod", LibmIntrinsic::Custom(LibmCustomIntrinsic::Fmod)),
    ("fmodf", LibmIntrinsic::Custom(LibmCustomIntrinsic::Fmod)),
    ("frexp", LibmIntrinsic::GLOp(GLOp::FrexpStruct)),
    ("frexpf", LibmIntrinsic::GLOp(GLOp::FrexpStruct)),
    ("hypot", LibmIntrinsic::Custom(LibmCustomIntrinsic::Hypot)),
    ("hypotf", LibmIntrinsic::Custom(LibmCustomIntrinsic::Hypot)),
    ("ilogb", LibmIntrinsic::Custom(LibmCustomIntrinsic::Ilogb)),
    ("ilogbf", LibmIntrinsic::Custom(LibmCustomIntrinsic::Ilogb)),
    ("j0", LibmIntrinsic::Custom(LibmCustomIntrinsic::J0)),
    ("j0f", LibmIntrinsic::Custom(LibmCustomIntrinsic::J0)),
    ("y0", LibmIntrinsic::Custom(LibmCustomIntrinsic::Y0)),
    ("y0f", LibmIntrinsic::Custom(LibmCustomIntrinsic::Y0)),
    ("j1", LibmIntrinsic::Custom(LibmCustomIntrinsic::J1)),
    ("j1f", LibmIntrinsic::Custom(LibmCustomIntrinsic::J1)),
    ("y1", LibmIntrinsic::Custom(LibmCustomIntrinsic::Y1)),
    ("y1f", LibmIntrinsic::Custom(LibmCustomIntrinsic::Y1)),
    ("jn", LibmIntrinsic::Custom(LibmCustomIntrinsic::Jn)),
    ("jnf", LibmIntrinsic::Custom(LibmCustomIntrinsic::Jn)),
    ("yn", LibmIntrinsic::Custom(LibmCustomIntrinsic::Yn)),
    ("ynf", LibmIntrinsic::Custom(LibmCustomIntrinsic::Yn)),
    ("ldexp", LibmIntrinsic::GLOp(GLOp::Ldexp)),
    ("ldexpf", LibmIntrinsic::GLOp(GLOp::Ldexp)),
    ("lgamma", LibmIntrinsic::Custom(LibmCustomIntrinsic::Lgamma)),
    (
        "lgammaf",
        LibmIntrinsic::Custom(LibmCustomIntrinsic::Lgamma),
    ),
    (
        "lgamma_r",
        LibmIntrinsic::Custom(LibmCustomIntrinsic::LgammaR),
    ),
    (
        "lgammaf_r",
        LibmIntrinsic::Custom(LibmCustomIntrinsic::LgammaR),
    ),
    ("tgamma", LibmIntrinsic::Custom(LibmCustomIntrinsic::Tgamma)),
    (
        "tgammaf",
        LibmIntrinsic::Custom(LibmCustomIntrinsic::Tgamma),
    ),
    ("log10", LibmIntrinsic::Custom(LibmCustomIntrinsic::Log10)),
    ("log10f", LibmIntrinsic::Custom(LibmCustomIntrinsic::Log10)),
    ("log1p", LibmIntrinsic::Custom(LibmCustomIntrinsic::Log1p)),
    ("log1pf", LibmIntrinsic::Custom(LibmCustomIntrinsic::Log1p)),
    ("log2", LibmIntrinsic::GLOp(GLOp::Log2)),
    ("log2f", LibmIntrinsic::GLOp(GLOp::Log2)),
    ("log", LibmIntrinsic::GLOp(GLOp::Log)),
    ("logf", LibmIntrinsic::GLOp(GLOp::Log)),
    ("modf", LibmIntrinsic::GLOp(GLOp::ModfStruct)),
    ("modff", LibmIntrinsic::GLOp(GLOp::ModfStruct)),
    (
        "nextafter",
        LibmIntrinsic::Custom(LibmCustomIntrinsic::NextAfter),
    ),
    (
        "nextafterf",
        LibmIntrinsic::Custom(LibmCustomIntrinsic::NextAfter),
    ),
    ("pow", LibmIntrinsic::GLOp(GLOp::Pow)),
    ("powf", LibmIntrinsic::GLOp(GLOp::Pow)),
    (
        "remainder",
        LibmIntrinsic::Custom(LibmCustomIntrinsic::Remainder),
    ),
    (
        "remainderf",
        LibmIntrinsic::Custom(LibmCustomIntrinsic::Remainder),
    ),
    ("remquo", LibmIntrinsic::Custom(LibmCustomIntrinsic::RemQuo)),
    (
        "remquof",
        LibmIntrinsic::Custom(LibmCustomIntrinsic::RemQuo),
    ),
    ("rint", LibmIntrinsic::GLOp(GLOp::RoundEven)),
    ("rintf", LibmIntrinsic::GLOp(GLOp::RoundEven)),
    ("round", LibmIntrinsic::GLOp(GLOp::Round)),
    ("roundf", LibmIntrinsic::GLOp(GLOp::Round)),
    ("scalbn", LibmIntrinsic::Custom(LibmCustomIntrinsic::Scalbn)),
    (
        "scalbnf",
        LibmIntrinsic::Custom(LibmCustomIntrinsic::Scalbn),
    ),
    ("sin", LibmIntrinsic::GLOp(GLOp::Sin)),
    ("sincos", LibmIntrinsic::Custom(LibmCustomIntrinsic::SinCos)),
    (
        "sincosf",
        LibmIntrinsic::Custom(LibmCustomIntrinsic::SinCos),
    ),
    ("sinf", LibmIntrinsic::GLOp(GLOp::Sin)),
    ("sinh", LibmIntrinsic::GLOp(GLOp::Sinh)),
    ("sinhf", LibmIntrinsic::GLOp(GLOp::Sinh)),
    ("sqrt", LibmIntrinsic::GLOp(GLOp::Sqrt)),
    ("sqrtf", LibmIntrinsic::GLOp(GLOp::Sqrt)),
    ("tan", LibmIntrinsic::GLOp(GLOp::Tan)),
    ("tanf", LibmIntrinsic::GLOp(GLOp::Tan)),
    ("tanh", LibmIntrinsic::GLOp(GLOp::Tanh)),
    ("tanhf", LibmIntrinsic::GLOp(GLOp::Tanh)),
    ("trunc", LibmIntrinsic::GLOp(GLOp::Trunc)),
    ("truncf", LibmIntrinsic::GLOp(GLOp::Trunc)),
];

pub const NUM_TRAITS_TABLE: &[(&str, LibmIntrinsic)] =
    &[("powi", LibmIntrinsic::Custom(LibmCustomIntrinsic::Powi))];

impl Builder<'_, '_> {
    pub fn call_libm_intrinsic(
        &mut self,
        intrinsic: LibmIntrinsic,
        result_type: Word,
        args: &[SpirvValue],
    ) -> SpirvValue {
        match intrinsic {
            LibmIntrinsic::GLOp(op) => self.gl_op(op, result_type, args),
            LibmIntrinsic::Custom(LibmCustomIntrinsic::SinCos) => {
                assert_eq!(args.len(), 1);
                let x = args[0];
                let sin = self.gl_op(GLOp::Sin, x.ty, [x]).def(self);
                let cos = self.gl_op(GLOp::Cos, x.ty, [x]).def(self);
                self.emit()
                    .composite_construct(result_type, None, [sin, cos].iter().copied())
                    .unwrap()
                    .with_type(result_type)
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::Fmod) => {
                assert_eq!(args.len(), 2);
                self.emit()
                    .f_mod(result_type, None, args[0].def(self), args[1].def(self))
                    .unwrap()
                    .with_type(result_type)
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::CopySign) => {
                assert_eq!(args.len(), 2);
                self.copysign(args[0], args[1])
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::Cbrt) => {
                assert_eq!(args.len(), 1);
                self.gl_op(
                    GLOp::Pow,
                    result_type,
                    [args[0], self.constant_float(args[0].ty, 1.0 / 3.0)],
                )
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::Log10) => {
                assert_eq!(args.len(), 1);
                // log10(x) == (1 / ln(10)) * ln(x)
                let mul = self.constant_float(args[0].ty, 1.0 / 10.0f64.ln());
                let ln = self.gl_op(GLOp::Log, result_type, [args[0]]);
                self.fmul(mul, ln)
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::Log1p) => {
                assert_eq!(args.len(), 1);
                let one = self.constant_float(args[0].ty, 1.0);
                let add = self.fadd(args[0], one);
                self.gl_op(GLOp::Log, result_type, [add])
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::Exp10) => {
                assert_eq!(args.len(), 1);
                // exp10(x) == exp(x * log(10));
                let log10 = self.constant_float(args[0].ty, 10.0f64.ln());
                let mul = self.fmul(args[0], log10);
                self.gl_op(GLOp::Exp, result_type, [mul])
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::Expm1) => {
                let exp = self.gl_op(GLOp::Exp, args[0].ty, [args[0]]);
                let one = self.constant_float(exp.ty, 1.0);
                self.fsub(exp, one)
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::Erf) => {
                if self.cx.builder.has_capability(Capability::Kernel) {
                    self.opencl_op(18, result_type, args)
                } else {
                    self.undef_zombie(result_type, "Erf not supported yet")
                }
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::Erfc) => {
                if self.cx.builder.has_capability(Capability::Kernel) {
                    self.opencl_op(17, result_type, args)
                } else {
                    self.undef_zombie(result_type, "Erfc not supported yet")
                }
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::Fdim) => {
                if self.cx.builder.has_capability(Capability::Kernel) {
                    self.opencl_op(24, result_type, args)
                } else {
                    self.undef_zombie(result_type, "Fdim not supported yet")
                }
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::Hypot) => {
                if self.cx.builder.has_capability(Capability::Kernel) {
                    self.opencl_op(32, result_type, args)
                } else {
                    self.undef_zombie(result_type, "Hypot not supported yet")
                }
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::Ilogb) => {
                if self.cx.builder.has_capability(Capability::Kernel) {
                    self.opencl_op(33, result_type, args)
                } else {
                    self.undef_zombie(result_type, "Ilogb not supported yet")
                }
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::J0) => {
                self.undef_zombie(result_type, "J0 not supported yet")
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::Y0) => {
                self.undef_zombie(result_type, "Y0 not supported yet")
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::J1) => {
                self.undef_zombie(result_type, "J1 not supported yet")
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::Y1) => {
                self.undef_zombie(result_type, "Y1 not supported yet")
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::Jn) => {
                self.undef_zombie(result_type, "Jn not supported yet")
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::Yn) => {
                self.undef_zombie(result_type, "Yn not supported yet")
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::Lgamma) => {
                if self.cx.builder.has_capability(Capability::Kernel) {
                    self.opencl_op(35, result_type, args)
                } else {
                    self.undef_zombie(result_type, "Lgamma not supported yet")
                }
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::LgammaR) => {
                if self.cx.builder.has_capability(Capability::Kernel) {
                    self.opencl_op_with_ptr_out(36, args, result_type)
                } else {
                    self.undef_zombie(result_type, "LgammaR not supported yet")
                }
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::Tgamma) => {
                if self.cx.builder.has_capability(Capability::Kernel) {
                    self.opencl_op(65, result_type, args)
                } else {
                    self.undef_zombie(result_type, "Tgamma not supported yet")
                }
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::NextAfter) => {
                if self.cx.builder.has_capability(Capability::Kernel) {
                    self.opencl_op(47, result_type, args)
                } else {
                    self.undef_zombie(result_type, "NextAfter not supported yet")
                }
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::Powi) => {
                assert_eq!(args.len(), 2);
                // Convert integer exponent to float, then use GLOp::Pow
                let float_exp = self.sitofp(args[1], args[0].ty);
                self.gl_op(GLOp::Pow, result_type, [args[0], float_exp])
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::Remainder) => {
                if self.cx.builder.has_capability(Capability::Kernel) {
                    self.opencl_op(51, result_type, args)
                } else {
                    self.undef_zombie(result_type, "Remainder not supported yet")
                }
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::RemQuo) => {
                if self.cx.builder.has_capability(Capability::Kernel) {
                    self.opencl_op_with_ptr_out(52, args, result_type)
                } else {
                    self.undef_zombie(result_type, "RemQuo not supported yet")
                }
            }
            LibmIntrinsic::Custom(LibmCustomIntrinsic::Scalbn) => {
                self.gl_op(GLOp::Ldexp, result_type, args)
            }
        }
    }

    fn opencl_op_with_ptr_out(
        &mut self,
        opcode: u32,
        args: &[SpirvValue],
        result_type: Word,
    ) -> SpirvValue {
        let float_ty = args[0].ty;
        let i32_ty = SpirvType::Integer(32, false).def(self.span(), self);
        let ptr_ty = self.type_ptr_to(i32_ty);

        let out_var = {
            let mut builder = self.emit();
            builder.select_block(Some(0)).unwrap();
            let index = {
                let block = &builder.module_ref().functions[builder.selected_function().unwrap()]
                    .blocks[builder.selected_block().unwrap()];
                block
                    .instructions
                    .iter()
                    .enumerate()
                    .find_map(|(index, inst)| {
                        if inst.class.opcode != Op::Variable {
                            Some(InsertPoint::FromBegin(index))
                        } else {
                            None
                        }
                    })
                    .unwrap_or(InsertPoint::End)
            };
            let result_id = builder.id();
            let inst = Instruction::new(
                Op::Variable,
                Some(ptr_ty),
                Some(result_id),
                vec![Operand::StorageClass(StorageClass::Function)],
            );
            builder.insert_into_block(index, inst).unwrap();
            result_id.with_type(ptr_ty)
        };

        let opencl = self.ext_inst.borrow_mut().import_opencl(self);
        let mut operands: Vec<Operand> = args.iter().map(|a| Operand::IdRef(a.def(self))).collect();
        operands.push(Operand::IdRef(out_var.def(self)));
        let primary = self
            .emit()
            .ext_inst(float_ty, None, opencl, opcode, operands)
            .unwrap()
            .with_type(float_ty);
        let secondary = self
            .emit()
            .load(i32_ty, None, out_var.def(self), None, empty())
            .unwrap()
            .with_type(i32_ty);
        self.emit()
            .composite_construct(
                result_type,
                None,
                [primary.def(self), secondary.def(self)].iter().copied(),
            )
            .unwrap()
            .with_type(result_type)
    }
}
