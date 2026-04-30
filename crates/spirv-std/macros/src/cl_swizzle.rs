//! Proc-macro implementation of `spirv_std::cl::s!`.
//!
//! Parses one of two index forms after a leading expression:
//!
//! ```ignore
//! s!(v, xyzw)    // letter form, valid only with 1-4 indices each in {x,y,z,w}
//! s!(v, sFEDC)   // OpenCL `sN` form, hex digits 0-9a-f, all widths
//! s!(v, s0_1_0_1) // underscores allowed and ignored
//! ```
//!
//! The parsed indices become literal SPIR-V operands of an emitted
//! `OpVectorShuffle` (multi-index) or `OpCompositeExtract` (single
//! index). The result type is inferred from the surrounding context.
//! Source-vector-width validation is left to SPIR-V validation
//! downstream — the macro only validates that result count is in
//! `{1, 2, 3, 4, 8, 16}` and that the chosen form is internally
//! consistent.

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{Expr, Ident, Token};

pub enum SwizzleInput {
    /// Index list parsed from `xyzw` or `sN` ident.
    Indices { source: Expr, indices: Vec<u32> },
    /// Named group: `lo`, `hi`, `even`, `odd` — width-dependent, dispatched
    /// at type-check time via the corresponding `Swizzle*` trait.
    Named { source: Expr, group: NamedGroup },
}

#[derive(Copy, Clone)]
pub enum NamedGroup {
    Lo,
    Hi,
    Even,
    Odd,
}

impl NamedGroup {
    fn parse(ident: &Ident) -> Option<Self> {
        match ident.to_string().as_str() {
            "lo" => Some(Self::Lo),
            "hi" => Some(Self::Hi),
            "even" => Some(Self::Even),
            "odd" => Some(Self::Odd),
            _ => None,
        }
    }
}

impl Parse for SwizzleInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let source: Expr = input.parse()?;
        input.parse::<Token![,]>()?;
        let ident: Ident = input.parse()?;
        if let Some(group) = NamedGroup::parse(&ident) {
            return Ok(Self::Named { source, group });
        }
        let indices = parse_indices(&ident)?;
        Ok(Self::Indices { source, indices })
    }
}

fn parse_indices(ident: &Ident) -> syn::Result<Vec<u32>> {
    let raw = ident.to_string();

    // Strip optional leading `s` / `S` to discriminate letter vs sN form.
    let (kind, rest) = match raw.chars().next() {
        Some('s' | 'S') if raw.len() > 1 => (Form::SN, &raw[1..]),
        _ => (Form::Letter, raw.as_str()),
    };

    let mut indices = Vec::new();
    for ch in rest.chars() {
        if ch == '_' {
            continue;
        }
        let idx = match kind {
            Form::Letter => match ch {
                'x' | 'X' => 0,
                'y' | 'Y' => 1,
                'z' | 'Z' => 2,
                'w' | 'W' => 3,
                _ => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!(
                            "invalid swizzle component `{ch}` in letter form; \
                             expected `x`, `y`, `z`, or `w`. \
                             For widths above 4, use the `sN` form (e.g. `s0123`)."
                        ),
                    ));
                }
            },
            Form::SN => match ch {
                '0'..='9' => ch as u32 - '0' as u32,
                'a'..='f' => 10 + (ch as u32 - 'a' as u32),
                'A'..='F' => 10 + (ch as u32 - 'A' as u32),
                _ => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!(
                            "invalid swizzle index `{ch}` in `sN` form; \
                             expected hex digit 0-9 or a-f."
                        ),
                    ));
                }
            },
        };
        indices.push(idx);
    }

    if indices.is_empty() {
        return Err(syn::Error::new(
            ident.span(),
            "empty swizzle — at least one component required",
        ));
    }

    if !matches!(indices.len(), 1 | 2 | 3 | 4 | 8 | 16) {
        return Err(syn::Error::new(
            ident.span(),
            format!(
                "swizzle result width {} is not a valid OpenCL/SPIR-V vector width; \
                 expected one of 1, 2, 3, 4, 8, or 16",
                indices.len()
            ),
        ));
    }

    if kind == Form::Letter {
        for (i, &idx) in indices.iter().enumerate() {
            if idx > 3 {
                return Err(syn::Error::new(
                    ident.span(),
                    format!(
                        "letter swizzle index #{i} is out of range \
                         (letter form only addresses components 0-3); \
                         use `sN` form for wider sources"
                    ),
                ));
            }
        }
    }

    Ok(indices)
}

#[derive(PartialEq, Eq)]
enum Form {
    Letter,
    SN,
}

fn named_swizzle(source: Expr, group: NamedGroup) -> TokenStream {
    let (trait_name, method) = match group {
        NamedGroup::Lo => ("SwizzleLo", "lo"),
        NamedGroup::Hi => ("SwizzleHi", "hi"),
        NamedGroup::Even => ("SwizzleEven", "even"),
        NamedGroup::Odd => ("SwizzleOdd", "odd"),
    };
    let trait_ident = syn::Ident::new(trait_name, proc_macro2::Span::call_site());
    let method_ident = syn::Ident::new(method, proc_macro2::Span::call_site());
    let expanded = quote! {{
        // Use UFCS so the trait is unambiguous and the user only needs
        // `use spirv_std::cl::*;` (not the trait names individually).
        spirv_std::cl::#trait_ident::#method_ident(#source)
    }};
    expanded.into()
}

pub fn cl_swizzle_inner(input: SwizzleInput) -> TokenStream {
    let (source, indices) = match input {
        SwizzleInput::Indices { source, indices } => (source, indices),
        SwizzleInput::Named { source, group } => return named_swizzle(source, group),
    };

    if indices.len() == 1 {
        let idx = indices[0];
        let host_idx = idx as usize;
        // Single-component → scalar, via OpCompositeExtract.
        let extract = format!("%dst = OpCompositeExtract typeof*{{dst}} %src {idx}");
        let expanded = quote! {{
            let __src = #source;
            let mut __dst = ::core::default::Default::default();
            #[cfg(target_arch = "spirv")]
            unsafe {
                ::core::arch::asm!(
                    "%src = OpLoad _ {src}",
                    #extract,
                    "OpStore {dst} %dst",
                    src = in(reg) &__src,
                    dst = in(reg) &mut __dst,
                );
            }
            #[cfg(not(target_arch = "spirv"))]
            {
                __dst = __src.to_array()[#host_idx];
            }
            __dst
        }};
        return expanded.into();
    }

    // Multi-component → OpVectorShuffle.
    // SPIR-V `OpVectorShuffle` takes two source vectors; we use the same
    // vector for both operands so any index 0..N_SRC-1 is a valid pick.
    let indices_str = indices
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    let shuffle = format!("%dst = OpVectorShuffle typeof*{{dst}} %src %src {indices_str}");

    // Host-side fallback: build via to_array + from_array on the result type.
    // The result type is whatever inference gives __dst.
    let host_indices = indices.iter().map(|&i| {
        let lit = i as usize;
        quote! { __arr[#lit] }
    });

    let expanded = quote! {{
        let __src = #source;
        let mut __dst = ::core::default::Default::default();
        #[cfg(target_arch = "spirv")]
        unsafe {
            ::core::arch::asm!(
                "%src = OpLoad _ {src}",
                #shuffle,
                "OpStore {dst} %dst",
                src = in(reg) &__src,
                dst = in(reg) &mut __dst,
            );
        }
        #[cfg(not(target_arch = "spirv"))]
        {
            let __arr = __src.to_array();
            __dst = ::core::convert::From::from([ #( #host_indices ),* ]);
        }
        __dst
    }};

    expanded.into()
}
