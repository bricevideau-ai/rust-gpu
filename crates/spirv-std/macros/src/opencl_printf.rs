use std::fmt::Write;

/// `OpenCL` `printf` implementation using the `OpenCL.std` extended instruction set.
///
/// Strategy: The format string is created as a Rust `const [u8; N]` byte array.
/// The codegen places const-promoted globals in `UniformConstant` storage class.
/// We pass a reference via `in(reg)` and use it directly (no `OpLoad`) — the
/// register holds a pointer to the const data, which is what `printf` expects.
pub fn opencl_printf_inner(input: super::DebugPrintfInput) -> proc_macro::TokenStream {
    let super::DebugPrintfInput {
        format_string,
        variables,
        span,
    } = input;

    let format_arguments = match parse_format_specifiers(&format_string, span) {
        Ok(args) => args,
        Err(ts) => return ts,
    };

    if format_arguments.len() != variables.len() {
        return syn::Error::new(
            span,
            format!(
                "{} % arguments were found, but {} variables were given",
                format_arguments.len(),
                variables.len()
            ),
        )
        .to_compile_error()
        .into();
    }

    // Build OpLoad instructions and input registers for each variable argument.
    let mut variable_idents = String::new();
    let mut input_registers = Vec::new();
    let mut op_loads = Vec::new();
    let mut host_drops = Vec::new();

    for (i, (variable, format_argument)) in variables.into_iter().zip(format_arguments).enumerate()
    {
        // On the host, consume the argument to suppress unused-variable warnings.
        host_drops.push(quote::quote! { let _ = &(#variable); });
        let ident = quote::format_ident!("_{}", i);

        let assert_fn = match format_argument {
            FormatType::Scalar { ty } => {
                quote::quote! { spirv_std::debug_printf::assert_is_type::<#ty> }
            }
            FormatType::Float => {
                quote::quote! { spirv_std::debug_printf::assert_is_float }
            }
            FormatType::Pointer => {
                quote::quote! { spirv_std::debug_printf::assert_is_pointer }
            }
            FormatType::Vector { ty, width } => {
                quote::quote! { spirv_std::debug_printf::assert_is_vector::<#ty, _, #width> }
            }
        };

        input_registers.push(quote::quote! {
            #ident = in(reg) &#assert_fn(#variable),
        });

        let op_load = format!("%{ident} = OpLoad _ {{{ident}}}");
        op_loads.push(quote::quote! {
            #op_load,
        });

        let _ = write!(variable_idents, "%{ident} ");
    }

    let input_registers = input_registers
        .into_iter()
        .collect::<proc_macro2::TokenStream>();
    let op_loads = op_loads.into_iter().collect::<proc_macro2::TokenStream>();

    // Create the format string as a null-terminated byte array constant.
    let format_bytes: Vec<u8> = format_string.bytes().chain(std::iter::once(0u8)).collect();
    let len = format_bytes.len();
    let byte_literals: Vec<proc_macro2::TokenStream> = format_bytes
        .iter()
        .map(|b| {
            let lit = proc_macro2::Literal::u8_suffixed(*b);
            quote::quote! { #lit }
        })
        .collect();

    // Pass the format variable directly (no OpLoad) — it's already a pointer
    // to the const data in UniformConstant storage.
    let ext_inst_line =
        format!("%_pf_result = OpExtInst %_pf_u32 %_pf_opencl 184 {{_pf_fmt}} {variable_idents}");

    let output = quote::quote! {
        {
            #[cfg(target_arch = "spirv")]
            {
                const _PRINTF_FMT: [u8; #len] = [#(#byte_literals),*];

                // SAFETY: OpenCL printf is a standard, safe operation. The unsafe
                // block is only required by the asm! macro, not by the operation itself.
                // Format string type checking is enforced at compile time by the macro.
                unsafe {
                    ::core::arch::asm!(
                        "%_pf_u32 = OpTypeInt 32 0",
                        "%_pf_opencl = OpExtInstImport \"OpenCL.std\"",
                        #op_loads
                        #ext_inst_line,
                        _pf_fmt = in(reg) &_PRINTF_FMT,
                        #input_registers
                    )
                }
            }

            // On non-SPIR-V targets, consume the arguments to avoid
            // unused-variable warnings in the caller.
            #[cfg(not(target_arch = "spirv"))]
            {
                #(#host_drops)*
            }
        }
    };

    output.into()
}

enum FormatType {
    Scalar {
        ty: proc_macro2::TokenStream,
    },
    /// Float specifier (`%f` etc.) — accepts both f32 and f64.
    Float,
    /// Pointer specifier (`%p`) — accepts `*const T` and `*mut T`.
    Pointer,
    Vector {
        ty: proc_macro2::TokenStream,
        width: usize,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LengthModifier {
    None,
    Hh,
    H,
    Hl,
    L,
}

fn resolve_scalar_type(
    length: LengthModifier,
    conversion: char,
    span: proc_macro2::Span,
) -> Result<FormatType, proc_macro::TokenStream> {
    match conversion {
        'd' | 'i' => Ok(FormatType::Scalar {
            ty: match length {
                LengthModifier::Hh => quote::quote! { i8 },
                LengthModifier::H => quote::quote! { i16 },
                LengthModifier::None => quote::quote! { i32 },
                LengthModifier::L => quote::quote! { i64 },
                LengthModifier::Hl => {
                    return Err(syn::Error::new(
                        span,
                        "'hl' length modifier is only valid with a vector specifier",
                    )
                    .to_compile_error()
                    .into());
                }
            },
        }),
        'o' | 'u' | 'x' | 'X' => Ok(FormatType::Scalar {
            ty: match length {
                LengthModifier::Hh => quote::quote! { u8 },
                LengthModifier::H => quote::quote! { u16 },
                LengthModifier::None => quote::quote! { u32 },
                LengthModifier::L => quote::quote! { u64 },
                LengthModifier::Hl => {
                    return Err(syn::Error::new(
                        span,
                        "'hl' length modifier is only valid with a vector specifier",
                    )
                    .to_compile_error()
                    .into());
                }
            },
        }),
        'a' | 'A' | 'e' | 'E' | 'f' | 'F' | 'g' | 'G' => match length {
            LengthModifier::None | LengthModifier::L => Ok(FormatType::Float),
            LengthModifier::Hl => Err(syn::Error::new(
                span,
                "'hl' length modifier is only valid with a vector specifier",
            )
            .to_compile_error()
            .into()),
            _ => Err(syn::Error::new(
                span,
                format!(
                    "Length modifier '{}' is not valid with float conversion '{conversion}'",
                    if length == LengthModifier::Hh {
                        "hh"
                    } else {
                        "h"
                    }
                ),
            )
            .to_compile_error()
            .into()),
        },
        'c' => match length {
            LengthModifier::None => Ok(FormatType::Scalar {
                ty: quote::quote! { u32 },
            }),
            _ => Err(
                syn::Error::new(span, "Length modifiers are not valid with '%c'")
                    .to_compile_error()
                    .into(),
            ),
        },
        'p' => match length {
            LengthModifier::None => Ok(FormatType::Pointer),
            _ => Err(
                syn::Error::new(span, "Length modifiers are not valid with '%p'")
                    .to_compile_error()
                    .into(),
            ),
        },
        _ => Err(syn::Error::new(
            span,
            format!("Unrecognised format specifier: '{conversion}'"),
        )
        .to_compile_error()
        .into()),
    }
}

fn resolve_vector_type(
    width: usize,
    length: LengthModifier,
    conversion: char,
    span: proc_macro2::Span,
) -> Result<FormatType, proc_macro::TokenStream> {
    if width > 4 {
        return Err(syn::Error::new(
            span,
            format!("v{width} vectors are not yet supported (only v2, v3, v4)"),
        )
        .to_compile_error()
        .into());
    }

    let ty = match conversion {
        'd' | 'i' => match length {
            LengthModifier::Hh => quote::quote! { i8 },
            LengthModifier::H => quote::quote! { i16 },
            LengthModifier::Hl | LengthModifier::None => quote::quote! { i32 },
            LengthModifier::L => quote::quote! { i64 },
        },
        'o' | 'u' | 'x' | 'X' => match length {
            LengthModifier::Hh => quote::quote! { u8 },
            LengthModifier::H => quote::quote! { u16 },
            LengthModifier::Hl | LengthModifier::None => quote::quote! { u32 },
            LengthModifier::L => quote::quote! { u64 },
        },
        'a' | 'A' | 'e' | 'E' | 'f' | 'F' | 'g' | 'G' => match length {
            LengthModifier::Hl | LengthModifier::None => quote::quote! { f32 },
            LengthModifier::L => quote::quote! { f64 },
            _ => {
                return Err(syn::Error::new(
                    span,
                    format!(
                        "Length modifier '{}' is not valid with float vector conversion \
                         '{conversion}'",
                        if length == LengthModifier::Hh {
                            "hh"
                        } else {
                            "h"
                        }
                    ),
                )
                .to_compile_error()
                .into());
            }
        },
        _ => {
            return Err(syn::Error::new(
                span,
                format!("Unrecognised vector type specifier: '{conversion}'"),
            )
            .to_compile_error()
            .into());
        }
    };

    Ok(FormatType::Vector { ty, width })
}

/// Parse `OpenCL` printf format specifiers following the spec syntax:
/// `%[flags][width][.precision][vector][length]conversion`
fn parse_format_specifiers(
    format_string: &str,
    span: proc_macro2::Span,
) -> Result<Vec<FormatType>, proc_macro::TokenStream> {
    let mut chars = format_string.chars().peekable();
    let mut format_arguments = Vec::new();

    while let Some(ch) = chars.next() {
        if ch != '%' {
            continue;
        }

        // Handle %% escape.
        match chars.peek() {
            Some('%') => {
                chars.next();
                continue;
            }
            None => {
                return Err(syn::Error::new(span, "Unterminated format specifier")
                    .to_compile_error()
                    .into());
            }
            _ => {}
        }

        // 1. Skip flags: any of [-+ #0]
        while matches!(chars.peek(), Some('-' | '+' | ' ' | '#' | '0')) {
            chars.next();
        }

        // 2. Skip width: [0-9]*
        while matches!(chars.peek(), Some('0'..='9')) {
            chars.next();
        }

        // 3. Skip precision: (\.[0-9]*)?
        if matches!(chars.peek(), Some('.')) {
            chars.next();
            while matches!(chars.peek(), Some('0'..='9')) {
                chars.next();
            }
        }

        // 4. Parse optional vector specifier: v(2|3|4|8|16)
        let vector_width = if matches!(chars.peek(), Some('v')) {
            chars.next();
            let width = match chars.next() {
                Some('2') => 2,
                Some('3') => 3,
                Some('4') => 4,
                Some('8') => 8,
                Some('1') => match chars.peek() {
                    Some('6') => {
                        chars.next();
                        16
                    }
                    _ => {
                        return Err(syn::Error::new(
                            span,
                            "Invalid vector width (expected 2, 3, 4, 8, or 16)",
                        )
                        .to_compile_error()
                        .into());
                    }
                },
                Some(other) => {
                    return Err(syn::Error::new(
                        span,
                        format!("Invalid vector width: '{other}' (expected 2, 3, 4, 8, or 16)"),
                    )
                    .to_compile_error()
                    .into());
                }
                None => {
                    return Err(syn::Error::new(span, "Missing vector width after 'v'")
                        .to_compile_error()
                        .into());
                }
            };
            Some(width)
        } else {
            None
        };

        // 5. Parse optional length modifier: hh | hl | h | l
        let length = match chars.peek() {
            Some('h') => {
                chars.next();
                match chars.peek() {
                    Some('h') => {
                        chars.next();
                        LengthModifier::Hh
                    }
                    Some('l') => {
                        chars.next();
                        LengthModifier::Hl
                    }
                    _ => LengthModifier::H,
                }
            }
            Some('l') => {
                chars.next();
                LengthModifier::L
            }
            _ => LengthModifier::None,
        };

        // 6. Parse conversion specifier.
        let conversion = match chars.next() {
            Some(c) => c,
            None => {
                return Err(syn::Error::new(
                    span,
                    "Unterminated format specifier: missing conversion",
                )
                .to_compile_error()
                .into());
            }
        };

        let fmt = if let Some(width) = vector_width {
            resolve_vector_type(width, length, conversion, span)?
        } else {
            resolve_scalar_type(length, conversion, span)?
        };

        format_arguments.push(fmt);
    }

    Ok(format_arguments)
}
