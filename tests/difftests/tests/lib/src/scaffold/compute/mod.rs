mod ash;
mod backend;
#[cfg(target_os = "linux")]
mod opencl;
mod wgpu;

pub use crate::scaffold::shader::*;
pub use ash::AshBackend;
pub use backend::{BufferConfig, BufferUsage, ComputeBackend, ComputeShaderTest, ComputeTest};
#[cfg(target_os = "linux")]
pub use opencl::{OpenClBackend, run_opencl_test, run_opencl_test_default};
pub use wgpu::{
    WgpuBackend, WgpuComputeTest, WgpuComputeTestMultiBuffer, WgpuComputeTestPushConstants,
};
