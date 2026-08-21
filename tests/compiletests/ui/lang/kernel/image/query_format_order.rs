// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

#![cfg_attr(target_arch = "spirv", no_std)]

// Exercise the OpenCL Kernel-only query ops: `OpImageQueryFormat`
// and `OpImageQueryOrder`. These return integer values that map to
// the OpenCL host API's `cl_channel_data_type` and
// `cl_channel_order` enums per the OpenCL SPIR-V env spec's
// "Image Channel Data Type Mapping" / "Image Channel Order Mapping"
// tables. Useful for kernels that want to specialise per actual
// storage format — OpenCL Kernel images carry only sampled-type
// info at compile time, so the channel data type is only available
// at runtime via these queries.

use glam::*;
use spirv_std::{Image, glam, spirv};

#[spirv(kernel)]
pub fn main(
    #[spirv(global_invocation_id)] id: USizeVec3,
    image: &Image!(2D, type=f32, sampled=false),
    #[spirv(cross_workgroup)] out_format: &mut [u32],
    #[spirv(cross_workgroup)] out_order: &mut [u32],
) {
    let i = id.x;
    out_format[i] = image.query_format();
    out_order[i] = image.query_order();
}
