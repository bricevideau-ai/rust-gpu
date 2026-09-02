/// An opaque reference to settings that describe how to access, filter, or
/// sample an image.
#[spirv(sampler)]
#[derive(Copy, Clone)]
// HACK(eddyb) avoids "transparent newtype of `_anti_zst_padding`" misinterpretation.
#[repr(C)]
pub struct Sampler {
    // HACK(eddyb) avoids the layout becoming ZST (and being elided in one way
    // or another, before `#[spirv(sampler)]` can special-case it).
    _anti_zst_padding: core::mem::MaybeUninit<u32>,
}

impl Sampler {
    /// Construct a placeholder `Sampler` value that is never observable at
    /// runtime. Used by the `const_sampler!` macro on non-`SPIR-V` targets so
    /// that kernel crates type-check when compiled for the host (e.g. as part
    /// of workspace tests).
    #[doc(hidden)]
    pub const fn __host_placeholder() -> Self {
        Self {
            _anti_zst_padding: core::mem::MaybeUninit::uninit(),
        }
    }
}

/// Create a constant sampler (`OpConstantSampler`) — used in `OpenCL` Kernel
/// entry points so the kernel doesn't need a `&Sampler` argument.
///
/// Auto-adds the `LiteralSampler` capability when invoked. Currently only
/// usable on `OpenCL` Kernel targets, since the `LiteralSampler` capability
/// is not part of the Vulkan SPIR-V environment.
///
/// # Syntax
///
/// `const_sampler!(addr = <SamplerAddressingMode>, normalized = <bool>, filter = <SamplerFilterMode>)`
///
/// `<SamplerAddressingMode>` is one of `None`, `ClampToEdge`, `Clamp`,
/// `Repeat`, `RepeatMirrored`. `<SamplerFilterMode>` is `Nearest` or `Linear`.
/// All three arguments are required and must appear in this order.
///
/// # Example
///
/// ```rust,ignore
/// use spirv_std::const_sampler;
///
/// #[spirv(kernel)]
/// pub fn k(
///     #[spirv(global_invocation_id)] id: glam::USizeVec3,
///     src: &Image!(2D, type=f32, sampled=true),
///     dst: &mut Image!(2D, type=f32, sampled=false),
/// ) {
///     let sampler = const_sampler!(
///         addr = ClampToEdge,
///         normalized = true,
///         filter = Linear,
///     );
///     let uv = glam::Vec2::new(id.x as f32, id.y as f32);
///     let color: glam::Vec4 = src.sample_by_lod(sampler, uv, 0.0);
///     unsafe { dst.write(glam::IVec2::new(id.x as i32, id.y as i32), color) };
/// }
/// ```
#[macro_export]
macro_rules! const_sampler {
    (addr = $addr:ident, normalized = true, filter = $filter:ident $(,)?) => {
        $crate::const_sampler!(@inner $addr, 1, $filter)
    };
    (addr = $addr:ident, normalized = false, filter = $filter:ident $(,)?) => {
        $crate::const_sampler!(@inner $addr, 0, $filter)
    };
    (@inner $addr:ident, $normalized:literal, $filter:ident) => {{
        // SPIR-V: synthesize an `OpConstantSampler` and store it into a
        // Function-local Sampler variable, then load and return. The constant
        // ends up in `types_global_values` (module-scope), and the inline-asm
        // path auto-adds the `LiteralSampler` capability.
        #[cfg(target_arch = "spirv")]
        let s: $crate::Sampler = unsafe {
            let mut local: ::core::mem::MaybeUninit<$crate::Sampler> =
                ::core::mem::MaybeUninit::uninit();
            ::core::arch::asm! {
                "%sampler_ty = OpTypeSampler",
                concat!(
                    "%const_sampler = OpConstantSampler %sampler_ty ",
                    stringify!($addr),
                    " ",
                    stringify!($normalized),
                    " ",
                    stringify!($filter),
                ),
                "OpStore {local} %const_sampler",
                local = in(reg) local.as_mut_ptr(),
            }
            local.assume_init()
        };
        // Host: kernel crates compile for the host target during workspace
        // tests; samplers can never be used at runtime there, so a
        // placeholder of the right type is enough to type-check.
        #[cfg(not(target_arch = "spirv"))]
        let s: $crate::Sampler = $crate::Sampler::__host_placeholder();
        s
    }};
}
