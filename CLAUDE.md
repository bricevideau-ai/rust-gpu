# OpenCL Kernel Support for rust-gpu

## Branch: `opencl-kernel-support`

This branch adds OpenCL Kernel execution model support to rust-gpu, enabling Rust compute kernels to compile to OpenCL SPIR-V and run on OpenCL devices (pocl, Intel, etc.).

## Quick orientation

- **PR**: bricevideau-ai/rust-gpu#1 (against bricevideau-ai/rust-gpu main, which tracks Kerilk/rust-gpu)
- **Backup**: `opencl-kernel-support-backup` branch
- **Samples repo**: https://github.com/bricevideau-ai/rust-gpu-opencl-samples

## Key files we modified

### Codegen (SPIR-V generation)
- `crates/rustc_codegen_spirv/src/target.rs` — OpenCL target with Physical64 addressing
- `crates/rustc_codegen_spirv/src/abi.rs` — `[T]` lowered to element type for Kernel (no RuntimeArray)
- `crates/rustc_codegen_spirv/src/spirv_type.rs` — skip MemberDecorate Offset, ArrayStride for Kernel; strip integer signedness
- `crates/rustc_codegen_spirv/src/builder_spirv.rs` — UniformConstant for const-promoted globals on Kernel
- `crates/rustc_codegen_spirv/src/builder/builder_methods.rs` — allow OpPtrAccessChain for Physical addressing; generalize GEP; fix AccessChain index widths
- `crates/rustc_codegen_spirv/src/codegen_cx/entry.rs` — CrossWorkgroup default for kernel params; slice decomposition
- `crates/rustc_codegen_spirv/src/codegen_cx/declare.rs` — CrossWorkgroup for mutable statics on Kernel

### Linker
- `crates/rustc_codegen_spirv/src/linker/kernel_arguments.rs` — **NEW**: convert void(void)+OpVariable to OpFunctionParameter; fix BuiltIn types for Physical64
- `crates/rustc_codegen_spirv/src/linker/specializer.rs` — tolerate inference conflicts for Kernel (warn instead of exit)
- `crates/rustc_codegen_spirv/src/linker/mod.rs` — register kernel_arguments pass
- `crates/rustc_codegen_spirv/src/link.rs` — isolate spirv-opt crashes with fork() for compiled tools

### spirv_std
- `crates/spirv-std/src/arch/group.rs` — **NEW**: Kernel subgroup operations (Groups capability)
- `crates/spirv-std/src/arch.rs` — register group module

### Examples
- `examples/shaders/kernel-shader/` — Collatz kernel (mirrors compute-shader for Vulkan)
- `examples/runners/opencl-builder/` — compile-only (spirv-val)
- `examples/runners/opencl/` — full runner with pocl execution

### Tests
- `tests/compiletests/ui/spirv-attr/kernel-*.rs` — 15 test files, ~100 kernels
- Covers: control flow, integer ops, structs, slices, pointers, math, closures, panics, arrays, consts, glam, f64, subgroups, workgroup memory, statics

### Documentation
- `docs/src/writing-kernel-crates.md` — full guide

## How to test

```bash
cargo test -p rustc_codegen_spirv                                    # unit tests
cargo run -p compiletests -- --target-env opencl1.2,opencl2.0 kernel # OpenCL compiletests
cargo run -p compiletests -- --target-env vulkan1.2                  # Vulkan regression (must pass)
cargo run -p example-runner-opencl                                   # run on pocl (needs pocl 6.0+)
cargo run -p example-runner-opencl-builder                           # compile + spirv-val only
```

## Known issues

- **spirv-opt crash**: `DeadBranchElimPass` crashes on some Kernel SPIR-V (e.g., `is_multiple_of`). Isolated via fork() for compiled tools. CLI spirv-opt (`use-installed-tools`) cannot optimize these patterns at all.
- **CI pocl**: Ubuntu 24.04's pocl 5.0 lacks SPIR-V IL support. macOS Apple OpenCL driver is broken. Runner execution is not in CI — only compilation + spirv-val.
- **Enum discriminants**: Default `u8` discriminant requires Int8 capability. Use `#[repr(u32)]`.
- **Specializer**: Inference conflicts on Kernel targets are warned (not fatal). The `concrete_fallback` mechanism handles unresolved variables.

## Architecture decisions

- **[T] → element_type for Kernel**: In Physical64 addressing, `*[T]` = `*T`. No RuntimeArray needed.
- **Linker pass for kernel args**: OpenCL consumers need OpFunctionParameter, not global OpVariable. The pass runs after inlining.
- **BuiltIn type widening**: Physical64 requires v3ulong for GlobalInvocationId. The linker converts v3uint→v3ulong and inserts UConvert.
- **Storage classes**: UniformConstant for `&'static`, CrossWorkgroup for `static mut`, Function storage class is NOT valid at module scope.
- **Specializer tolerance**: The `process::exit(1)` was replaced with `warn!()` for Kernel only. Shader targets still exit on conflicts.

## Conventions

- Use `clang-format-17` (not `clang-format`) for any C/C++ code
- Always rebase on upstream after a PR merge
- Don't modify upstream files unless strictly necessary for OpenCL support
- All codegen changes are gated on `has_capability(Capability::Kernel)` to avoid affecting Vulkan
- Run `cargo fmt --all` AND `rustfmt tests/compiletests/ui/spirv-attr/kernel-*.rs` before committing
