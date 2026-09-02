# OpenCL Kernel Support for rust-gpu

## Branch: `opencl-kernel-support-v2`

This branch adds OpenCL Kernel execution model support to rust-gpu, enabling Rust compute kernels to compile to OpenCL SPIR-V and run on OpenCL devices.

## Quick orientation

- **PR**: bricevideau-ai/rust-gpu#3 (against bricevideau-ai/rust-gpu main, which tracks Kerilk/rust-gpu)
- **Backups**: `opencl-kernel-support-v2-backup-2026-05-12` (pre-redistribution squash tip), `opencl-kernel-support-v2-redistributed-2026-05-12` (post-redistribution snapshot)
- **Samples repo**: https://github.com/bricevideau-ai/rust-gpu-opencl-samples (tracks the stable `opencl-kernel-support` branch — PR #1)

## Key files we modified

### Codegen (SPIR-V generation)
- `crates/rustc_codegen_spirv/src/target.rs` — OpenCL target with Physical64 addressing; capability validation helpers
- `crates/rustc_codegen_spirv/src/abi.rs` — `[T]` lowered to element type for Kernel (no RuntimeArray); `AccessQualifier::ReadOnly` on `OpTypeImage` for sampled images
- `crates/rustc_codegen_spirv/src/spirv_type.rs` — skip MemberDecorate Offset, ArrayStride for Kernel; strip integer signedness; `AccessQualifier` field on `Image`
- `crates/rustc_codegen_spirv/src/builder_spirv.rs` — mandatory OpenCL capabilities; capability validation; UniformConstant for const-promoted globals
- `crates/rustc_codegen_spirv/src/builder/builder_methods.rs` — allow OpPtrAccessChain for Physical addressing; generalize GEP; fix AccessChain mixed-width index merging
- `crates/rustc_codegen_spirv/src/builder/spirv_asm.rs` — allow UniformConstant in OpTypePointer asm (for printf); per-parameter image AccessQualifier emission
- `crates/rustc_codegen_spirv/src/builder/ext_inst.rs` — `gl_op` dispatches to `OpenCL.std` on Kernel (via `gl_op_to_opencl_std`), `GLSL.std.450` on Shader
- `crates/rustc_codegen_spirv/src/builder/format_args_decompiler.rs` — DebugPrintf format specifiers extended to i8/i16/i64/u8/u16/u64/f64
- `crates/rustc_codegen_spirv/src/codegen_cx/entry.rs` — CrossWorkgroup default for kernel params; slice decomposition into (ptr, len) pairs; per-image AccessQualifier (ReadOnly/WriteOnly/ReadWrite by mutability + capability)
- `crates/rustc_codegen_spirv/src/codegen_cx/declare.rs` — CrossWorkgroup for mutable statics on Kernel
- `crates/rustc_codegen_spirv/src/symbols.rs` — five Kernel-only builtins (`work_dim`, `global_size`, `enqueued_workgroup_size`, `global_offset`, `global_linear_id`)

### Linker
- `crates/rustc_codegen_spirv/src/linker/kernel_arguments.rs` — **NEW**: convert void(void)+OpVariable to OpFunctionParameter; `KernelParamPositionDecoration` to preserve source param order; fix BuiltIn variable types for Physical64 (v3uint→v3ulong + UConvert); per-callee image-AccessQualifier specialization
- `crates/rustc_codegen_spirv/src/linker/specializer.rs` — tolerate inference conflicts for Kernel (warn instead of exit); the `concrete_fallback` mechanism handles unresolved variables
- `crates/rustc_codegen_spirv/src/linker/custom_decorations.rs` — `KernelParamPositionDecoration` definition
- `crates/rustc_codegen_spirv/src/linker/mod.rs` — register kernel_arguments pass
- `crates/rustc_codegen_spirv/src/link.rs` — isolate spirv-opt crashes with fork() for compiled tools (workaround for SPIRV-Tools #6632); post-link DebugPrintf-to-OpenCL.std-printf conversion pass for Kernel abort diagnostics

### spirv_std
- `crates/spirv-std/src/arch/group.rs` — **NEW**: subgroup + work-group operations (Groups capability, OpGroup* instructions, OpenCL 2.0+); reduce/inclusive-scan/exclusive-scan variants for {i,f,u,s}{add,min,max} including f64
- `crates/spirv-std/src/arch/opencl_std.rs` — **NEW**: ~40 math intrinsics from the OpenCL.std extended instruction set (sqrt/sin/cos/exp/log/pow/fma/clamp/etc. + integer min/max/clamp/abs/popcount/clz/ctz + geometric length/distance/normalize/cross + multi-output fract/modf/frexp/sincos returning tuples + `native_*` low-precision fast variants); accepts scalars (`f32`/`f64`, `i8..=i64`, `u8..=u64`) and glam vectors via the `FloatOrFloatVector`, `SignedIntegerOrSignedVector`, `UnsignedIntegerOrUnsignedVector`, `IntegerOrIntegerVector` traits
- `crates/spirv-std/src/arch/atomics.rs` — `atomic_flag_test_and_set` and `atomic_flag_clear` (Kernel-only OpAtomicFlag* instructions) on top of the existing integer/float/bitwise atomics
- `crates/spirv-std/src/cl/` — **NEW**: native OpenCL vector types (`Char/UChar/Short/UShort/Int/UInt/Long/ULong/Float/Double` × widths `2/3/4/8/16`, requires `Vector16`); `cl::s!` swizzle macro (`xyzw` letter form for widths 1–4, `s0..sf` hex form for all widths up to 16); named-group swizzles `lo/hi/even/odd` (widths 2/4/8/16, width 3 intentionally omitted); host-side fallback impls for every operator and swizzle so the same code runs on CPU
- `crates/spirv-std/src/arch.rs` — register group + opencl_std modules
- `crates/spirv-std/macros/src/opencl_printf.rs` — **NEW**: `printf!` / `printfln!` proc macros with compile-time format-string validation (length modifiers `hh`/`h`/`l`, vector specifiers `v2`/`v3`/`v4`, `%p` via `PrintfPointer` trait, `%f` accepts both `f32` and `f64` via `PrintfFloat`)
- `crates/spirv-std/macros/src/cl_swizzle.rs` — **NEW**: `cl::s!` swizzle macro implementation
- `crates/spirv-std/src/debug_printf.rs` — `PrintfFloat` trait for `%f` accepting both f32 and f64

### Examples
- `examples/shaders/kernel-shader/` — Collatz kernel with printf (Kernel entry-point demo + atomic_reduce kernel)
- `examples/shaders/kernel-fp64-shader/` — f32/f64 printf test kernel
- `examples/shaders/kernel-test-shader/` — subgroup/work-group collective + shared memory torture tests (run by the `opencl` runner)
- `examples/shaders/kernel-image-shader/` — storage-image read (4×4 verifying buffer)
- `examples/shaders/kernel-sampler-shader/` — sampler-based image upscale (4×4 sampled image → 16×16 storage image; exercises `sampled=true` read + `sampled=false` write + hardware bilinear filter)
- `examples/runners/opencl-builder/` — compile-only (spirv-val)
- `examples/runners/opencl/` — full runner with OpenCL execution; sections for Collatz, printf, atomic reduce, subgroups, work-group collectives, image read, sampler upscale

### Tests

#### Compile tests (`tests/compiletests/ui/`)
- `spirv-attr/kernel/*.rs` — entry-point shape: empty kernel, `threads(N)`, each builtin, `cross_workgroup` buffer/scalar/slice params, subgroup builtins
- `spirv-attr/kernel-opencl-math.rs` — exercises one of each `opencl_std::*` category
- `lang/kernel/*.rs` — per-type arithmetic + conversions (u32/i32/u64/f32/f64/bool), control flow, ptr deref, glam vec ops, structs, panics, arrays, consts, closures, slices (safe/dynamic-index/fill/len), static (mut + ref), cl::* swizzle/scalar/integer/float-widths
- `arch/kernel_atomics/*.rs` — integer + float + bitwise + flag (test_and_set/clear/spinlock)
- `arch/kernel_work_group/*.rs` — work-group reduce/inclusive-scan/exclusive-scan for i/f/u/s {add,min,max}
- `arch/kernel_shared/*.rs` — workgroup memory write/read/exchange/fill/tree-reduce for u32 and f64
- `lang/kernel/printf/*.rs` — 19 printf shapes; `printf/type_checking.rs` — 11 build-fail cases
- `lang/kernel/image/*.rs`, `lang/kernel/sampler/*.rs` — kernel image + sampler entry points
- `dis/kernel_*.rs` — golden SPIR-V disassembly checks (caps + memory model, cl swizzle, scalar↔vector ops, etc.)

#### Difftests (`tests/difftests/tests/`)
- `lib/src/scaffold/compute/opencl.rs` — **NEW**: OpenCL compute backend (mirrors `wgpu.rs` / `ash.rs`), uses `opencl3` via Linux-only target dep. Skips gracefully if no SPIR-V-capable OpenCL device is present.
- `lib/src/scaffold/compute/backend.rs` — `BufferConfig::element_size` field added so the OpenCL backend can split `cl_mem` buffers into `(ptr, len)` arg pairs matching the codegen's slice-param decomposition
- `runner/src/testcase.rs` — directories without `src/main.rs` (e.g. `<test>-shared` lib crates) are skipped, allowing helper crates to coexist with bin variants in the same test directory
- 22 OpenCL test variants spread across the chain — each one squashed into the commit that first enabled it (linker pass enables 7, mixed-widths enables 2, UniformConstant enables matrix_ops, math intrinsics enables 8, native cl vectors enables 4)

### Documentation
- `docs/src/writing-kernel-crates.md` — full guide (entry attributes, parameter types, builtins, cl::* native vector types, printf, atomics, images and samplers, f64, differences from Vulkan, runner usage)

## How to test

```bash
cargo test -p rustc_codegen_spirv                                    # unit tests
cargo run -p compiletests --release -- --target-env opencl1.2,opencl2.0 kernel  # OpenCL compiletests
cargo run -p compiletests --release -- --target-env vulkan1.2        # Vulkan regression (must pass)
# Full CI compile tests (all target envs):
cargo run -p compiletests --release -- --target-env vulkan1.1,vulkan1.2,vulkan1.3,vulkan1.4,spv1.3,spv1.4,opencl1.2,opencl2.0
cargo run -p example-runner-opencl                                   # run on OpenCL device
cargo run -p example-runner-opencl-builder                           # compile + spirv-val only
# Difftests (compares CPU/wgpu/ash/OpenCL backends end-to-end):
cargo test --release -p difftests --test difftests
# Or with an explicit pocl ICD:
OCL_ICD_VENDORS=$HOME/.local/etc/OpenCL/vendors cargo test --release -p difftests --test difftests
```

## Known issues

- **spirv-opt crash**: `DeadBranchElimPass` crashes on some Kernel SPIR-V (e.g. `is_multiple_of`). Isolated via fork() for compiled tools. See https://github.com/KhronosGroup/SPIRV-Tools/issues/6632
- **spirv-opt SIGABRT on aarch64**: `matrix_ops-opencl` difftest reliably triggers a spirv-opt assertion on Apple-silicon Linux. Codegen is correct; the failure is downstream in spirv-opt's optimization passes.
- **pocl `__alignof__` cap (aarch64)**: pocl's OpenCL C compiler reports `__alignof__` for vector types > 16 bytes as 16. Affects only the `vector_layout_opencl-c` arm of the `vector_layout_opencl` difftest; the rust-gpu kernel and CPU paths agree on the spec values. See `pocl-vector-align-repro/` and the upstream pocl issue.
- **CI**: Compiletests are compile + spirv-val only. Difftests run end-to-end (lavapipe + mesa-vulkan-drivers from `ppa:kisak/turtle` on Ubuntu, plus the bundled lavapipe on Windows/macOS via `install-vulkan-sdk-action`), but no OpenCL ICD is installed in CI yet — the OpenCL backend skips gracefully, so the OpenCL difftest variants are not actually exercised by CI. Example runners are also not run in CI.
- **Specializer**: Inference conflicts on Kernel targets are warned (not fatal). The `concrete_fallback` mechanism handles unresolved variables.

## Architecture decisions

- **`[T]` → element_type for Kernel**: In Physical64 addressing, `*[T]` = `*T`. No RuntimeArray needed; slices are decomposed at the entry point into `(ptr, len)` kernel-arg pairs.
- **Linker pass for kernel args**: OpenCL consumers expect OpFunctionParameter, not global OpVariable. The pass runs after inlining; source-parameter order is preserved via the `KernelParamPositionDecoration` custom decoration tagged on each codegen-emitted global.
- **BuiltIn type widening**: Physical64 requires v3ulong for GlobalInvocationId. The linker converts v3uint→v3ulong and inserts UConvert for downstream code.
- **Storage classes**: `UniformConstant` for `&'static`, `CrossWorkgroup` for `static mut`. Function storage class is NOT valid at module scope.
- **Specializer tolerance**: The `process::exit(1)` was replaced with `warn!()` for Kernel only. Shader targets still exit on conflicts.
- **Mandatory capabilities**: All capabilities required by the OpenCL SPIR-V Environment Specification are declared by default. User-requested capabilities are validated against the target environment.
- **printf format string**: Created as a const `[u8; N]` byte array in UniformConstant storage, matching clang/llvm-spirv output. `%f` accepts both f32 and f64 via the `PrintfFloat` trait.
- **DebugPrintf for abort diagnostics**: Panic format-args (used by abort intrinsics) emit `NonSemantic.DebugPrintf` from SPIR-T; a post-link pass converts those to `OpenCL.std` printf for Kernel targets, since OpenCL runtimes don't generally support `NonSemantic` extensions.
- **Image AccessQualifier**: derived per-parameter. Default rule: `&Image` → `ReadOnly`, `&mut Image` → `ReadWrite`. The `ImageReadWrite` capability is auto-declared when any `ReadWrite OpTypeImage` is emitted. On OpenCL 1.2 (where `ImageReadWrite` isn't allowed by the env spec), `&mut Image` without an explicit override is a compile error that points the user at `#[spirv(image_access = "write_only")]` — the explicit override that produces a `WriteOnly` qualifier instead. The override accepts `read_only`/`write_only`/`read_write` and is coherence-checked against `&`/`&mut` (mismatch is a hard error). All access-qualifier handling is Kernel-only and bypasses the Vulkan/Shader path.
- **`cl::*` native vectors**: distinct from glam — fills the gaps for real OpenCL kernel authoring (widths 8/16, full integer family, doubleN ABI). Same `*OrVector` traits as the math module so `opencl_std::sqrt(cl::Double8)` works.
- **gl_op dispatch**: A single `gl_op` codegen helper dispatches to `OpenCL.std` when the Kernel capability is present and `GLSL.std.450` otherwise — so `f32::sqrt`, `(*v).length()`, etc. work on both Kernel and Shader targets without per-call-site changes.

## Conventions

- All codegen changes are gated on `BuilderSpirv::is_kernel_mode()` (target-derived, valid under post-link capability pruning) to avoid affecting Vulkan
- Run `cargo fmt --all` before committing
- Run `cargo clippy --workspace --exclude "cargo-gpu*" -- -D warnings` before pushing
- Run compiletests for both OpenCL and Vulkan, and the difftests, before pushing
