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

## Double precision (f64)

OpenCL supports double-precision floats. Enable with the `Float64`
target feature:

```toml
# In your build script or compile flags
# compile-flags: -C target-feature=+Float64
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
