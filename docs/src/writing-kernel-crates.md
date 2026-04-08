# Writing OpenCL Kernel Crates

This section walks you through writing an OpenCL compute kernel in Rust
using rust-gpu. If you're familiar with writing Vulkan shaders with
rust-gpu, the transition is straightforward — the main differences are
entry point attributes and storage class annotations.

## Quick comparison with Vulkan compute shaders

| Aspect | Vulkan Compute | OpenCL Kernel |
|--------|---------------|---------------|
| Entry point | `#[spirv(compute(threads(64)))]` | `#[spirv(kernel)]` |
| Buffer binding | `#[spirv(storage_buffer, descriptor_set = 0, binding = 0)]` | `#[spirv(cross_workgroup)]` |
| Local size | Required at compile time | Set at dispatch time by host |
| Global ID type | `UVec3` (32-bit) | `USizeVec3` (platform-native) |
| Shared memory | `#[spirv(workgroup)]` | `#[spirv(workgroup)]` (same) |
| Target | `spirv-unknown-vulkan1.2` | `spirv-unknown-opencl1.2` |

## Setting up a kernel crate

A kernel crate is structured exactly like a shader crate:

```toml
# Cargo.toml
[package]
name = "my-kernel"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["dylib"]

[dependencies]
spirv-std = "0.9"
glam = "0.31"
```

```rust
// src/lib.rs
#![cfg_attr(target_arch = "spirv", no_std)]

use glam::USizeVec3;
use spirv_std::{glam, spirv};

#[spirv(kernel)]
pub fn my_kernel(
    #[spirv(global_invocation_id)] id: USizeVec3,
    #[spirv(cross_workgroup)] data: &mut [u32],
) {
    let index = id.x;
    data[index] = data[index] * 2;
}
```

## Building with `spirv-builder`

```rust
// build.rs or runner
use spirv_builder::SpirvBuilder;

let result = SpirvBuilder::new("path/to/kernel-crate", "spirv-unknown-opencl1.2")
    .build()?;
let spv_path = result.module.unwrap_single();
```

The generated `.spv` file is an OpenCL SPIR-V binary that can be loaded
with `clCreateProgramWithIL`.

## Entry point attributes

### `#[spirv(kernel)]`

Marks a function as an OpenCL kernel entry point. Unlike Vulkan compute
shaders, the local work size (threads per workgroup) is optional — it can
be set at dispatch time by the host:

```rust
#[spirv(kernel)]                    // local size set at dispatch
pub fn flexible_kernel(...) {}

#[spirv(kernel(threads(64)))]       // fixed local size of 64
pub fn fixed_kernel(...) {}

#[spirv(kernel(threads(8, 8, 1)))]  // 2D local size
pub fn kernel_2d(...) {}
```

## Parameter types

### Global memory buffers

Use `#[spirv(cross_workgroup)]` for global memory (equivalent to
`__global` in OpenCL C):

```rust
#[spirv(kernel)]
pub fn my_kernel(
    #[spirv(cross_workgroup)] input: &[f32],       // read-only global buffer
    #[spirv(cross_workgroup)] output: &mut [f32],   // read-write global buffer
    #[spirv(cross_workgroup)] scalar: &mut u32,     // single value in global memory
) { ... }
```

Slices (`&[T]`, `&mut [T]`) are decomposed into two kernel arguments:
a pointer and a length. The host sets both via `clSetKernelArg`.

### Scalar parameters

By-value parameters become direct kernel arguments:

```rust
#[spirv(kernel)]
pub fn scale(
    #[spirv(cross_workgroup)] data: &mut [f32],
    factor: f32,     // scalar kernel argument
    offset: u32,     // another scalar argument
) { ... }
```

### Workgroup (shared/local) memory

Use `#[spirv(workgroup)]` for workgroup-local memory (equivalent to
`__local` in OpenCL C). Requires a fixed-size array:

```rust
#[spirv(kernel(threads(32)))]
pub fn reduction(
    #[spirv(global_invocation_id)] id: USizeVec3,
    #[spirv(local_invocation_id)] local_id: USizeVec3,
    #[spirv(cross_workgroup)] input: &[u32],
    #[spirv(cross_workgroup)] output: &mut u32,
    #[spirv(workgroup)] shared: &mut [u32; 32],
) {
    use spirv_std::arch::workgroup_memory_barrier_with_group_sync;

    let lid = local_id.x;
    // Note: safe slice indexing works here — no unsafe needed.
    shared[lid] = input[id.x];
    workgroup_memory_barrier_with_group_sync(); // safe function

    // Tree reduction
    let mut stride = 16;
    while stride > 0 {
        if lid < stride {
            shared[lid] += shared[lid + stride];
        }
        workgroup_memory_barrier_with_group_sync();
        stride /= 2;
    }
    if lid == 0 {
        *output = shared[0];
    }
}
```

### Built-in variables

```rust
#[spirv(kernel)]
pub fn my_kernel(
    #[spirv(global_invocation_id)] global_id: USizeVec3,
    #[spirv(local_invocation_id)] local_id: USizeVec3,
    #[spirv(workgroup_id)] workgroup_id: USizeVec3,
    #[spirv(num_workgroups)] num_workgroups: USizeVec3,
    #[spirv(local_invocation_index)] local_index: u32,
    #[spirv(subgroup_id)] subgroup_id: u32,
    #[spirv(subgroup_local_invocation_id)] subgroup_local_id: u32,
    ...
) { ... }
```

Use `USizeVec3` (from `glam`) for vector builtins — this maps to 32-bit
on Vulkan and 64-bit on OpenCL Physical64, making code portable.

## Subgroup operations

OpenCL kernels can use subgroup operations via the `Groups` capability
(requires OpenCL 2.0+). The `spirv_std::arch` module provides wrappers:

```rust
use spirv_std::arch;

// Vote operations
let all_true = arch::group_all(predicate);
let any_true = arch::group_any(predicate);

// Broadcast
let value = arch::group_broadcast_u32(my_value, 0);

// Reductions
let sum = arch::group_i_add(my_value);
let min = arch::group_u_min(my_value);
let max = arch::group_f_max(my_float);

// Scans (prefix sums)
let prefix = arch::group_inclusive_i_add(my_value);
let exclusive = arch::group_exclusive_i_add(my_value);
```

Note: These use the `OpGroup*` instructions (Kernel execution model),
which are distinct from the `subgroup_*` functions that use
`OpGroupNonUniform*` instructions (Shader execution model).

## Math intrinsics

The `spirv_std::arch::opencl_std` module exposes the `OpenCL.std`
extended instruction set (transcendentals, fast paths, integer
min/max/clamp, …). These map to `OpExtInst %"OpenCL.std" <op>` and
require no extra capability beyond `Kernel`.

```rust
use spirv_std::arch::opencl_std as ocl;

#[spirv(kernel)]
pub fn lighting(
    #[spirv(global_invocation_id)] id: USizeVec3,
    #[spirv(cross_workgroup)] phase: &mut [f32],
) {
    let t = id.x as f32 * 0.01;
    phase[id.x] = ocl::sqrt(ocl::fma(t, t, 1.0))     // distance
        * ocl::clamp(ocl::sin(t), 0.0, 1.0);         // smooth pulse
}
```

### Idiomatic Rust math also works

You can equivalently call the standard `num_traits::Float` methods
(`x.sqrt()`, `x.cos()`, `x.powf(n)`, etc.) on bare `f32`/`f64` — the
codegen detects `Capability::Kernel` modules and rewrites those calls
to `OpExtInst <OpenCL.std> <op>` rather than the GLSL.std.450 form
used on Vulkan/shader targets. Three ops have a different operand
shape in OpenCL.std (`fract`, `frexp`, `modf`); for these the
codegen materialises a function-local scratch out-pointer and discards
the secondary write, so the Rust-shaped `(self) -> Self` API is preserved.

```rust
use spirv_std::num_traits::Float;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn distance_squared(
    #[spirv(cross_workgroup)] x: &f32,
    #[spirv(cross_workgroup)] y: &mut f32,
) {
    *y = (*x).sqrt().powf(2.0); // both lower to OpExtInst <OpenCL.std>
}
```

The same applies to `cl::Float3::sqrt()` / `.dot()` / etc. (which wrap
`opencl_std::*` directly) and to glam's componentwise inherent methods.
**Choose by readability** — there is no codegen difference. The
`opencl_std::*` free functions remain useful when you want the SPIR-V
provenance to be visible in the source, when calling a precision-trade
variant (`native_sqrt`/`fast_length`), or when writing generic kernel
code over `FloatOrFloatVector`.

### Type bounds

All ops accept both scalar and `glam`-vector arguments; on a vector
the op is applied componentwise. The bounds are:

- Float ops: `FloatOrFloatVector` — `f32`, `f64`, `Vec2`, `Vec3`,
  `Vec3A`, `Vec4`, `DVec2`, `DVec3`, `DVec4`
- Signed integer ops (`s_*`): `SignedIntegerOrSignedVector` —
  `i8`/`i16`/`i32`/`i64`, `IVec2`/`IVec3`/`IVec4`
- Unsigned integer ops (`u_*`): `UnsignedIntegerOrUnsignedVector`
  — `u8`/`u16`/`u32`/`u64`, `UVec2`/`UVec3`/`UVec4`
- Sign-agnostic integer ops (`popcount`, `clz`, `ctz`):
  `IntegerOrIntegerVector` — any of the above integer types

Available functions:

- **Trig:** `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`,
  `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- **Exp/log:** `exp`, `exp2`, `exp10`, `log`, `log2`, `log10`, `pow`
- **Roots:** `sqrt`, `rsqrt`, `cbrt`
- **Rounding:** `floor`, `ceil`, `round`, `trunc`
- **Misc float:** `fabs`, `sign`, `copysign`, `fmin`, `fmax`, `fmod`,
  `hypot`, `fma`, `mad`, `clamp`, `mix`, `smoothstep`
- **Native (faster, lower precision):** `native_sqrt`, `native_sin`,
  `native_cos`, `native_exp`, `native_log`
- **Integer:** `s_abs`, `s_min`, `s_max`, `u_min`, `u_max`, `s_clamp`,
  `u_clamp`, `popcount`, `clz`, `ctz`
- **Geometric:** `length`, `distance`, `normalize`, `cross`,
  `fast_length`, `fast_distance`, `fast_normalize`. `length`/`distance`
  collapse a vector to its component scalar (e.g. `length(Vec3) -> f32`);
  `normalize`/`cross` keep the vector type. `cross` is restricted to
  `Vec3`/`Vec4`/`DVec3`/`DVec4` per the OpenCL.std spec.
- **Multi-output:** `fract(value) -> (fractional, integer_part)`,
  `modf(value) -> (fractional, trunc_integer_part)`,
  `frexp(value) -> (mantissa, exponent_i32)`,
  `sincos(value) -> (sin, cos)`. These map to `OpenCL.std` ops that
  produce two outputs (a return value plus one written through a
  Function-storage pointer); the wrapper allocates the out-pointer's
  slot internally so callers get a clean tuple-returning API.

Note: `mad` may use unconstrained intermediate precision (the GPU may
fuse it differently than `fma`); use `fma` for IEEE-754 determinism.
`native_*` and `fast_*` ops trade precision for speed — ULP error is
implementation-defined.

For Vulkan/shader targets, do not use this module — use the GLSL.std.450
wrappers in `spirv_std::arch::*` (e.g. `arch::unsigned_min`) instead.

## Printf

OpenCL kernels can print to the host console using `spirv_std::printf!`
and `spirv_std::printfln!`. These emit the SPIR-V `OpenCL.std` printf
instruction (opcode 184):

```rust
use spirv_std::{printf, printfln};

#[spirv(kernel)]
pub fn debug_kernel(
    #[spirv(global_invocation_id)] id: USizeVec3,
    #[spirv(cross_workgroup)] data: &[u32],
) {
    let i = id.x;
    printf!("work item %u: value = %u\n", i as u32, data[i]);
    printfln!("same with auto-newline: %u", data[i]); // appends \n
}
```

Format specifiers follow the OpenCL C printf spec:
`%[flags][width][.precision][vector][length]conversion`

| Specifier | Type | Example |
|-----------|------|---------|
| `%d`, `%i` | `i32` | `printf!("%d", -42i32)` |
| `%u`, `%x`, `%X`, `%o` | `u32` | `printf!("%08x", val)` |
| `%f`, `%e`, `%g`, `%a` | `f32` or `f64` | `printf!("%.2f", 3.14f32)` |
| `%c` | `u32` (as char) | `printf!("%c", 65u32)` |
| `%p` | `*const T` / `*mut T` | `printf!("%p", data.as_ptr())` |
| `%ld`, `%lu`, `%lx` | `i64` / `u64` | `printf!("%ld", big)` |
| `%hd`, `%hu` | `i16` / `u16` | `printf!("%hd", small)` |
| `%hhd`, `%hhu` | `i8` / `u8` | `printf!("%hhx", byte)` |
| `%v4hlf` | `Vec4` (f32×4) | `printf!("%v4hlf", vec)` |
| `%v2hld` | `IVec2` (i32×2) | `printf!("%v2hld", ivec)` |

Vector specifiers (`v2`, `v3`, `v4`) can be used with length modifiers
(`hl` for 32-bit, `l` for 64-bit). The shorthand `%v4f` (without length
modifier) also works for backward compatibility.

All format specifiers are validated at compile time — type mismatches
produce clear compiler errors.

## Atomic operations

The `spirv_std::arch` module provides atomic operations for
synchronization across work items:

```rust
use spirv_std::arch::atomic_i_add;
use spirv_std::memory::{Scope, Semantics};

#[spirv(kernel)]
pub fn atomic_reduce(
    #[spirv(global_invocation_id)] id: USizeVec3,
    #[spirv(cross_workgroup)] input: &[u32],
    #[spirv(cross_workgroup)] output: &mut [u32],
) {
    unsafe {
        atomic_i_add::<
            _,
            { Scope::Device as u32 },
            { Semantics::NONE.bits() },
        >(&mut output[0], input[id.x]);
    }
}
```

Available atomic operations:

| Category | Functions |
|----------|-----------|
| Load/Store | `atomic_load`, `atomic_store`, `atomic_exchange` |
| Integer arithmetic | `atomic_i_add`, `atomic_i_sub`, `atomic_i_increment`, `atomic_i_decrement` |
| Integer min/max | `atomic_s_min`, `atomic_s_max`, `atomic_u_min`, `atomic_u_max` |
| Float arithmetic | `atomic_f_add`, `atomic_f_min`, `atomic_f_max` |
| Bitwise | `atomic_and`, `atomic_or`, `atomic_xor` |
| Compare-exchange | `atomic_compare_exchange` |
| Flags | `atomic_flag_test_and_set`, `atomic_flag_clear` |

Atomics require `unsafe` and take const generic parameters for scope
(`Device`, `Workgroup`, etc.) and memory semantics
(`NONE`, `ACQUIRE`, `RELEASE`, etc.).

## Double precision (f64)

OpenCL supports double-precision floats. Enable the `Float64` capability
via `SpirvBuilder` or compile flags:

```rust
// In your build script or runner
SpirvBuilder::new(crate_path, "spirv-unknown-opencl1.2")
    .capability(Capability::Float64)
    .build()?;
```

```rust
// Or via compile flags (e.g., in compiletests)
// compile-flags: -C target-feature=+Float64
```

```rust
#[spirv(kernel)]
pub fn daxpy(
    #[spirv(global_invocation_id)] id: USizeVec3,
    #[spirv(cross_workgroup)] x: &[f64],
    #[spirv(cross_workgroup)] y: &mut [f64],
    alpha: f64,
) {
    let i = id.x;
    y[i] = alpha * x[i] + y[i];
}
```

## Differences from Vulkan shaders

### No descriptor sets or bindings

OpenCL kernel arguments are positional, not bound to descriptor sets.
Each buffer parameter becomes a direct `clSetKernelArg` call:

```c
// Host code (C)
clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
clSetKernelArg(kernel, 1, sizeof(cl_ulong), &input_length);
clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_buffer);
clSetKernelArg(kernel, 3, sizeof(cl_ulong), &output_length);
clSetKernelArg(kernel, 4, sizeof(cl_float), &scale_factor);
```

### Integer signedness

OpenCL SPIR-V strips signedness from integer types (`OpTypeInt` always
has signedness=0). Sign is encoded in operations (`OpSDiv` vs `OpUDiv`).
This is handled automatically — you can use `i32` and `u32` normally.

### Struct layouts

The `Offset` and `ArrayStride` decorations (used by Vulkan for explicit
struct layout) are not emitted for Kernel targets. Struct layout follows
the platform's default C-style rules.

## Running kernels

See the `examples/runners/opencl/` directory for a complete example of
compiling and running a kernel using the `opencl3` crate.

The basic flow:

1. Compile with `SpirvBuilder::new(crate, "spirv-unknown-opencl1.2").build()`
2. Load with `Program::create_from_il(&context, &spv_bytes)`
3. Build with `program.build(devices, "")`
4. Create kernel with `Kernel::create(&program, "kernel_name")`
5. Set arguments with `clSetKernelArg` or `ExecuteKernel::set_arg`
6. Enqueue with `clEnqueueNDRangeKernel`

## Known limitations

- **`static` variables**: Immutable statics (`&'static T`) use
  `UniformConstant` storage class. Mutable statics (`static mut`)
  use `CrossWorkgroup` (program-scope global memory).
- **Enums**: Default enum discriminants use `u8`. While `Int8` is
  enabled by default for OpenCL targets, `#[repr(u32)]` may still be
  preferred for portability.
- **Panic paths**: Panic messages compile but are not meaningful on
  GPU — they become abort intrinsics.
