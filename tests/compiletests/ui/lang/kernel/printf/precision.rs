// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(value: f32) {
    spirv_std::printf!("%.2f\n", value);
    spirv_std::printf!("%.4e\n", value);
    spirv_std::printf!("%8.3f\n", value);
    spirv_std::printf!("%.0f\n", value);
    spirv_std::printf!("%f\n", value);
}
