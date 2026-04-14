// build-fail
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// normalize-stderr-test "\S*/crates/spirv-std/src/" -> "$$SPIRV_STD_SRC/"
// normalize-stderr-test "crates/spirv-std/src/" -> "$$SPIRV_STD_SRC/"

// Test compile-time type checking for OpenCL printf format specifiers.

use spirv_std::spirv;

#[spirv(kernel)]
pub fn test_wrong_type_hhd(value: u32) {
    spirv_std::printf!("%hhd", value);
}

#[spirv(kernel)]
pub fn test_wrong_type_hu(value: i32) {
    spirv_std::printf!("%hu", value);
}

#[spirv(kernel)]
pub fn test_wrong_type_ld(value: u32) {
    spirv_std::printf!("%ld", value);
}

#[spirv(kernel)]
pub fn test_invalid_specifier(value: i32) {
    spirv_std::printf!("%r", value);
}

#[spirv(kernel)]
pub fn test_float_with_int(value: u32) {
    spirv_std::printf!("%f", value);
}

#[spirv(kernel)]
pub fn test_vector_with_scalar(value: f32) {
    spirv_std::printf!("%v2f", value);
}

#[spirv(kernel)]
pub fn test_v8_unsupported(value: u32) {
    spirv_std::printf!("%v8hlu", value);
}

#[spirv(kernel)]
pub fn test_v16_unsupported(value: u32) {
    spirv_std::printf!("%v16hld", value);
}

#[spirv(kernel)]
pub fn test_hl_without_vector(value: i32) {
    spirv_std::printf!("%hld", value);
}

#[spirv(kernel)]
pub fn test_hh_with_float(value: f32) {
    spirv_std::printf!("%hhf", value);
}

#[spirv(kernel)]
pub fn test_pointer_with_int(value: u32) {
    spirv_std::printf!("%p", value);
}
