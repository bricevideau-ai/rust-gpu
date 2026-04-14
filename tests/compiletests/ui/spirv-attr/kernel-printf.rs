// build-pass
// only-opencl1.2

// Test OpenCL printf using the OpenCL.std extended instruction set.

use spirv_std::glam::*;
use spirv_std::spirv;

/// printf with no arguments — just a format string.
#[spirv(kernel)]
pub fn test_printf_no_args() {
    spirv_std::printf!("hello from kernel\n");
}

/// printf with a single integer argument.
#[spirv(kernel)]
pub fn test_printf_int(#[spirv(global_invocation_id)] id: USizeVec3) {
    let i = id.x as u32;
    spirv_std::printf!("work item %u\n", i);
}

/// printf with multiple arguments of different types.
#[spirv(kernel)]
pub fn test_printf_multi(
    #[spirv(global_invocation_id)] id: USizeVec3,
    #[spirv(cross_workgroup)] data: &[u32],
) {
    let i = id.x as u32;
    let val = data[id.x];
    spirv_std::printf!("id=%u value=%u\n", i, val);
}

/// printf with float formatting.
#[spirv(kernel)]
pub fn test_printf_float(value: f32) {
    spirv_std::printf!("float: %f\n", value);
}

/// printf with signed integer.
#[spirv(kernel)]
pub fn test_printf_signed(value: i32) {
    spirv_std::printf!("signed: %d\n", value);
}

/// printf with hex formatting.
#[spirv(kernel)]
pub fn test_printf_hex(value: u32) {
    spirv_std::printf!("hex: 0x%x\n", value);
}

/// printfln (auto-appends newline).
#[spirv(kernel)]
pub fn test_printfln(value: u32) {
    spirv_std::printfln!("value = %u", value);
}
