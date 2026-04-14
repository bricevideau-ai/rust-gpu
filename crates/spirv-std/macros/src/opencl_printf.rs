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
    Vector {
        ty: proc_macro2::TokenStream,
        width: usize,
    },
}

fn parse_format_specifiers(
    format_string: &str,
    span: proc_macro2::Span,
) -> Result<Vec<FormatType>, proc_macro::TokenStream> {
    fn map_specifier_to_format_type(
        specifier: char,
        chars: &mut std::str::Chars<'_>,
    ) -> Option<FormatType> {
        let mut peekable = chars.peekable();

        Some(match specifier {
            'd' | 'i' => FormatType::Scalar {
                ty: quote::quote! { i32 },
            },
            'o' | 'x' | 'X' => FormatType::Scalar {
                ty: quote::quote! { u32 },
            },
            'a' | 'A' | 'e' | 'E' | 'f' | 'F' | 'g' | 'G' => FormatType::Float,
            'u' => {
                if matches!(peekable.peek(), Some('l')) {
                    chars.next();
                    FormatType::Scalar {
                        ty: quote::quote! { u64 },
                    }
                } else {
                    FormatType::Scalar {
                        ty: quote::quote! { u32 },
                    }
                }
            }
            'l' => {
                if matches!(peekable.peek(), Some('u' | 'x')) {
                    chars.next();
                    FormatType::Scalar {
                        ty: quote::quote! { u64 },
                    }
                } else {
                    return None;
                }
            }
            _ => return None,
        })
    }

    let mut chars = format_string.chars();
    let mut format_arguments = Vec::new();

    while let Some(mut ch) = chars.next() {
        if ch == '%' {
            ch = match chars.next() {
                Some('%') => continue,
                None => {
                    return Err(syn::Error::new(span, "Unterminated format specifier")
                        .to_compile_error()
                        .into());
                }
                Some(ch) => ch,
            };

            let mut has_precision = false;

            while ch.is_ascii_digit() {
                ch = match chars.next() {
                    Some(ch) => ch,
                    None => {
                        return Err(syn::Error::new(
                            span,
                            "Unterminated format specifier: missing type after precision",
                        )
                        .to_compile_error()
                        .into());
                    }
                };
                has_precision = true;
            }

            if has_precision && ch == '.' {
                ch = match chars.next() {
                    Some(ch) => ch,
                    None => {
                        return Err(syn::Error::new(
                            span,
                            "Unterminated format specifier: missing type after decimal point",
                        )
                        .to_compile_error()
                        .into());
                    }
                };

                while ch.is_ascii_digit() {
                    ch = match chars.next() {
                        Some(ch) => ch,
                        None => {
                            return Err(syn::Error::new(
                                span,
                                "Unterminated format specifier: missing type after fraction precision",
                            )
                            .to_compile_error()
                            .into());
                        }
                    };
                }
            }

            if ch == 'v' {
                let width = match chars.next() {
                    Some('2') => 2,
                    Some('3') => 3,
                    Some('4') => 4,
                    Some(ch) => {
                        return Err(syn::Error::new(
                            span,
                            format!("Invalid width for vector: {ch}"),
                        )
                        .to_compile_error()
                        .into());
                    }
                    None => {
                        return Err(syn::Error::new(span, "Missing vector dimensions specifier")
                            .to_compile_error()
                            .into());
                    }
                };

                ch = match chars.next() {
                    Some(ch) => ch,
                    None => {
                        return Err(syn::Error::new(span, "Missing vector type specifier")
                            .to_compile_error()
                            .into());
                    }
                };

                let fmt = match map_specifier_to_format_type(ch, &mut chars) {
                    Some(FormatType::Scalar { ty }) => FormatType::Vector { ty, width },
                    Some(FormatType::Float) => FormatType::Vector {
                        ty: quote::quote! { f32 },
                        width,
                    },
                    _ => {
                        return Err(syn::Error::new(
                            span,
                            format!("Unrecognised vector type specifier: '{ch}'"),
                        )
                        .to_compile_error()
                        .into());
                    }
                };

                format_arguments.push(fmt);
            } else {
                let fmt = match map_specifier_to_format_type(ch, &mut chars) {
                    Some(fmt) => fmt,
                    _ => {
                        return Err(syn::Error::new(
                            span,
                            format!("Unrecognised format specifier: '{ch}'"),
                        )
                        .to_compile_error()
                        .into());
                    }
                };

                format_arguments.push(fmt);
            }
        }
    }

    Ok(format_arguments)
}
