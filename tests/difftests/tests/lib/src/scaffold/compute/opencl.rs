use super::backend::{BufferConfig, BufferUsage, ComputeBackend, ComputeShaderTest};
use crate::config::Config;
use crate::scaffold::Skip;
use crate::scaffold::shader::RustComputeShader;
use anyhow::{Context as _, Result, anyhow};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_ALL, Device, get_all_devices};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, cl_device_id};
use std::ptr;

pub struct OpenClBackend {
    device_id: cl_device_id,
    context: Context,
    queue: CommandQueue,
}

/// Run the given Rust shader as an `OpenCL` Kernel against `OpenClBackend`,
/// gracefully `Skip`-ing if no SPIR-V-capable `OpenCL` runtime is available.
///
/// Centralises the "try to init `OpenCL`, otherwise mark Skip" boilerplate that
/// every `*-opencl` variant would otherwise repeat verbatim. The shader is
/// always compiled for `spirv-unknown-opencl1.2`; pass a different target via
/// [`run_opencl_test_with_target`] if you need `OpenCL` 2.0 (subgroups, etc.).
pub fn run_opencl_test(
    config: &Config,
    shader: RustComputeShader,
    dispatch: [u32; 3],
    buffers: Vec<BufferConfig>,
) -> Result<()> {
    let test = match ComputeShaderTest::<OpenClBackend, _>::new(shader, dispatch, buffers) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("OpenCL backend init failed: {e:?}");
            return Skip::new("OpenCL runtime unavailable on this host").run_test(config);
        }
    };
    test.run_test(config)
}

/// Convenience: build a `RustComputeShader` for `spirv-unknown-opencl1.2`
/// from the current crate, then dispatch through [`run_opencl_test`].
pub fn run_opencl_test_default(
    config: &Config,
    dispatch: [u32; 3],
    buffers: Vec<BufferConfig>,
) -> Result<()> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let shader = RustComputeShader::with_target(manifest_dir, "spirv-unknown-opencl1.2");
    run_opencl_test(config, shader, dispatch, buffers)
}

impl ComputeBackend for OpenClBackend {
    fn init() -> Result<Self> {
        let device_ids = get_all_devices(CL_DEVICE_TYPE_ALL)
            .map_err(|e| anyhow!("OpenCL: failed to enumerate devices: {e}"))?;
        if device_ids.is_empty() {
            return Err(anyhow!("OpenCL: no devices available"));
        }
        let device_id = device_ids
            .into_iter()
            .find(|&id| {
                let dev = Device::new(id);
                dev.il_version().is_ok_and(|s| !s.is_empty())
            })
            .ok_or_else(|| {
                anyhow!("OpenCL: no device with cl_khr_il_program (SPIR-V IL) support")
            })?;
        let device = Device::new(device_id);
        let context = Context::from_device(&device)
            .map_err(|e| anyhow!("OpenCL: Context::from_device: {e}"))?;
        let queue = CommandQueue::create_default_with_properties(&context, 0, 0)
            .map_err(|e| anyhow!("OpenCL: CommandQueue::create_default: {e}"))?;
        Ok(Self {
            device_id,
            context,
            queue,
        })
    }

    fn run_compute(
        &self,
        spirv_bytes: &[u8],
        entry_point: &str,
        dispatch: [u32; 3],
        buffers: Vec<BufferConfig>,
    ) -> Result<Vec<Vec<u8>>> {
        let mut program = Program::create_from_il(&self.context, spirv_bytes)
            .map_err(|e| anyhow!("OpenCL: Program::create_from_il: {e}"))?;
        if let Err(e) = program.build(self.context.devices(), "") {
            let log = program
                .get_build_log(self.device_id)
                .unwrap_or_else(|_| "no build log".into());
            return Err(anyhow!("OpenCL: Program::build: {e}\nbuild log:\n{log}"));
        }
        let kernel = Kernel::create(&program, entry_point)
            .map_err(|e| anyhow!("OpenCL: Kernel::create({entry_point}): {e}"))?;

        // Allocate device buffers and upload initial data.
        let mut cl_buffers: Vec<Buffer<u8>> = Vec::with_capacity(buffers.len());
        for cfg in &buffers {
            let flags = match cfg.usage {
                BufferUsage::Storage => CL_MEM_READ_WRITE,
                BufferUsage::StorageReadOnly | BufferUsage::Uniform => CL_MEM_READ_ONLY,
            };
            let mut buf = unsafe {
                Buffer::<u8>::create(&self.context, flags, cfg.size as usize, ptr::null_mut())
            }
            .with_context(|| format!("OpenCL: Buffer::create (size={})", cfg.size))?;
            // Always initialise the buffer contents — either from caller-provided
            // data, or to all-zeros otherwise. Without this, kernels that don't
            // touch every output slot leave that memory uninitialised, and the
            // downstream byte-compare against zero-initialised wgpu / ash buffers
            // fails on whatever garbage the OpenCL driver allocated.
            let zeros: Vec<u8>;
            let init_bytes: &[u8] = if let Some(initial) = &cfg.initial_data {
                initial.as_slice()
            } else {
                zeros = vec![0u8; cfg.size as usize];
                &zeros
            };
            unsafe {
                self.queue
                    .enqueue_write_buffer(&mut buf, CL_BLOCKING, 0, init_bytes, &[])
                    .context("OpenCL: enqueue_write_buffer (initial data)")?
                    .wait()
                    .context("OpenCL: wait on initial-data write")?;
            }
            cl_buffers.push(buf);
        }

        // Set kernel args. Each `&[T]` parameter on the Rust-GPU side is
        // decomposed by the codegen into two Kernel args: a pointer and a
        // `usize` length. We mirror that here: every buffer becomes
        // (cl_mem, length-in-bytes-as-usize).
        let mut exec = ExecuteKernel::new(&kernel);
        for (buf, cfg) in cl_buffers.iter().zip(&buffers) {
            let len: usize = cfg.size as usize / cfg.element_size.max(1);
            unsafe {
                exec.set_arg(buf).set_arg(&len);
            }
        }

        // Dispatch.
        //
        // `dispatch` follows the Vulkan / wgpu convention used by the rest of
        // the `ComputeBackend` impls: it counts *workgroups*, not work items.
        // OpenCL `clEnqueueNDRangeKernel` instead takes a global count of work
        // items plus an optional local work-group size. The Rust-GPU
        // `kernel(threads(N))` attribute lowers to `OpExecutionMode LocalSize`,
        // which OpenCL surfaces as the kernel's compile-time required work-
        // group size — so when it's set, multiply through to get the OpenCL
        // global work size. When unset (no `threads(...)`) we treat dispatch as
        // global directly so single-thread kernels still work.
        let local = kernel
            .get_compile_work_group_size(self.device_id)
            .ok()
            .filter(|v| v.len() == 3 && v.iter().any(|&x| x != 0));
        let global: [usize; 3] = match local.as_deref() {
            Some(l) => [
                dispatch[0] as usize * l[0].max(1),
                dispatch[1] as usize * l[1].max(1),
                dispatch[2] as usize * l[2].max(1),
            ],
            None => [
                dispatch[0] as usize,
                dispatch[1] as usize,
                dispatch[2] as usize,
            ],
        };
        exec.set_global_work_sizes(&global);
        if let Some(local) = local.as_deref() {
            exec.set_local_work_sizes(local);
        }
        let event = unsafe {
            exec.enqueue_nd_range(&self.queue)
                .map_err(|e| anyhow!("OpenCL: enqueue_nd_range: {e}"))?
        };
        event.wait().context("OpenCL: wait on kernel event")?;

        // Read back every buffer (matches AshBackend's "always return all").
        let mut results = Vec::with_capacity(buffers.len());
        for (buf, cfg) in cl_buffers.iter().zip(&buffers) {
            let mut data = vec![0u8; cfg.size as usize];
            unsafe {
                self.queue
                    .enqueue_read_buffer(buf, CL_BLOCKING, 0, &mut data, &[])
                    .context("OpenCL: enqueue_read_buffer")?
                    .wait()
                    .context("OpenCL: wait on read-buffer event")?;
            }
            results.push(data);
        }

        Ok(results)
    }
}
