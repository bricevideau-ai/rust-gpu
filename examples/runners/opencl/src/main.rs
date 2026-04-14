use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_ALL, Device, get_all_devices};
use opencl3::event::Event;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, cl_device_id};
use spirv_builder::{CompileResult, SpirvBuilder};
use std::path::Path;
use std::ptr;
use std::time::{Duration, Instant};

// ── OpenCL helpers ─────────────────────────────────────────────────────

/// An `OpenCL` context with a device and command queue, ready to run kernels.
struct OclContext {
    device_id: cl_device_id,
    context: Context,
    queue: CommandQueue,
}

impl OclContext {
    /// Create a context on the first available `OpenCL` device.
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_ALL)?
            .first()
            .expect("no OpenCL devices found");
        let device = Device::new(device_id);
        println!("Device:  {} ({})", device.name()?, device.vendor()?);
        println!("Version: {}", device.version()?);

        let context = Context::from_device(&device)?;
        let queue =
            CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)?;
        Ok(Self {
            device_id,
            context,
            queue,
        })
    }

    /// Load a SPIR-V binary and build a program.
    fn build_program(&self, spv_bytes: &[u8]) -> Result<Program, Box<dyn std::error::Error>> {
        let mut program = Program::create_from_il(&self.context, spv_bytes)
            .map_err(|e| format!("create_from_il: {e}"))?;
        if let Err(e) = program.build(self.context.devices(), "") {
            let log = program
                .get_build_log(self.device_id)
                .unwrap_or_else(|_| "no build log".into());
            return Err(format!("program.build: {e}\nbuild log: {log}").into());
        }
        Ok(program)
    }

    /// Upload a host slice to a new read-write device slice.
    fn upload<T>(&self, data: &[T]) -> Result<DeviceSlice<T>, Box<dyn std::error::Error>> {
        let mut buffer = unsafe {
            Buffer::<T>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                data.len(),
                ptr::null_mut(),
            )?
        };
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut buffer, CL_BLOCKING, 0, data, &[])?
                .wait()?;
        }
        Ok(DeviceSlice {
            buffer,
            len: data.len(),
        })
    }

    /// Upload a host slice to a new read-only device slice.
    #[allow(dead_code)]
    fn upload_ro<T>(&self, data: &[T]) -> Result<DeviceSlice<T>, Box<dyn std::error::Error>> {
        let mut buffer = unsafe {
            Buffer::<T>::create(&self.context, CL_MEM_READ_ONLY, data.len(), ptr::null_mut())?
        };
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut buffer, CL_BLOCKING, 0, data, &[])?
                .wait()?;
        }
        Ok(DeviceSlice {
            buffer,
            len: data.len(),
        })
    }

    /// Download a device slice into an existing host slice.
    fn download<T>(
        &self,
        src: &DeviceSlice<T>,
        dst: &mut [T],
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            self.queue
                .enqueue_read_buffer(&src.buffer, CL_BLOCKING, 0, dst, &[])?
                .wait()?;
        }
        Ok(())
    }

    /// Execute a kernel with the given arguments on a 1D global work size.
    fn run(
        &self,
        kernel: &Kernel,
        global_work_size: usize,
        args: &[&dyn KernelArg],
    ) -> Result<Event, Box<dyn std::error::Error>> {
        let mut exec = ExecuteKernel::new(kernel);
        for arg in args {
            arg.set(&mut exec);
        }
        let event = unsafe {
            exec.set_global_work_size(global_work_size)
                .enqueue_nd_range(&self.queue)?
        };
        event.wait()?;
        Ok(event)
    }
}

/// A device buffer paired with its element count — the device-side
/// equivalent of a Rust slice.
///
/// Rust-GPU decomposes `&[T]` / `&mut [T]` kernel parameters into two
/// kernel arguments: a `*CrossWorkgroup T` pointer and a `u64` length.
/// `DeviceSlice` tracks both so it can be passed directly as a kernel
/// argument via the [`KernelArg`] trait.
struct DeviceSlice<T> {
    buffer: Buffer<T>,
    len: usize,
}

impl<T> DeviceSlice<T> {
    fn len(&self) -> usize {
        self.len
    }
}

/// Trait for kernel arguments.
trait KernelArg {
    fn set(&self, exec: &mut ExecuteKernel<'_>);
}

/// A [`DeviceSlice`] sets two kernel arguments: the buffer pointer and
/// the u64 length, matching the Rust-GPU slice decomposition.
impl<T> KernelArg for DeviceSlice<T> {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        let len: usize = self.len;
        unsafe {
            exec.set_arg(&self.buffer).set_arg(&len);
        }
    }
}

impl KernelArg for u32 {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        unsafe {
            exec.set_arg(self);
        }
    }
}

impl KernelArg for f32 {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        unsafe {
            exec.set_arg(self);
        }
    }
}

/// Compile a kernel crate to `OpenCL` SPIR-V.
fn compile_kernel(path: &Path) -> Result<(Vec<u8>, Duration), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let result: CompileResult = SpirvBuilder::new(path, "spirv-unknown-opencl1.2").build()?;
    let spv_path = result.module.unwrap_single();
    let spv_bytes = std::fs::read(spv_path)?;
    Ok((spv_bytes, start.elapsed()))
}

/// Extract kernel execution time from profiling events.
fn profiling_duration(event: &Event) -> Option<Duration> {
    let start = event.profiling_command_start().ok()?;
    let end = event.profiling_command_end().ok()?;
    Some(Duration::from_nanos(end - start))
}

// ── Main ───────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Compile kernel.
    let kernel_crate = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shaders/kernel-shader");
    let (spv_bytes, compile_time) = compile_kernel(&kernel_crate)?;
    println!(
        "Compiled kernel ({} bytes SPIR-V, {compile_time:?})",
        spv_bytes.len()
    );

    // 2. Set up device.
    let ocl = OclContext::new()?;

    // 3. Build program and create kernel.
    let program = ocl.build_program(&spv_bytes)?;
    let kernel = Kernel::create(&program, "main_kernel")?;

    // 4. Upload input data.
    let top = 2u32.pow(20);
    let src_range = 1..top;
    let mut data: Vec<u32> = src_range.clone().collect();
    let n = data.len();
    println!(
        "Input:   {n} elements ({}..{})",
        src_range.start,
        src_range.end - 1
    );

    let buf = ocl.upload(&data)?;

    // 5. Run kernel.
    let event = ocl.run(&kernel, buf.len(), &[&buf])?;

    // 6. Download results.
    ocl.download(&buf, &mut data)?;

    if let Some(duration) = profiling_duration(&event) {
        println!("Kernel:  {duration:?}");
    }

    // 7. Validate.
    let checks: &[(u32, u32)] = &[(1, 0), (2, 1), (3, 7), (27, 111)];
    let mut all_ok = true;
    for &(input, expected) in checks {
        let got = data[(input - 1) as usize];
        if got != expected {
            eprintln!("FAIL: collatz({input}) = {got}, expected {expected}");
            all_ok = false;
        }
    }
    if all_ok {
        println!("Verify:  all spot checks passed");
    }

    // 8. Print record-holders.
    println!("\nCollatz record-holders (starting value: steps):");
    println!("1: 0");
    let mut max = 0;
    for (src, out) in src_range.zip(data.iter().copied()) {
        if out == u32::MAX {
            println!("{src}: overflowed");
            break;
        } else if out > max {
            max = out;
            println!("{src}: {out}");
        }
    }

    // 9. Test OpenCL printf.
    println!("\n═══ printf test ═══");
    let printf_kernel = Kernel::create(&program, "printf_test")?;
    let printf_data: Vec<u32> = vec![10, 20, 30, 40];
    let printf_buf = ocl.upload(&printf_data)?;
    println!(
        "Running printf_test with {} work items...",
        printf_data.len()
    );
    println!("--- device output ---");
    ocl.run(&printf_kernel, printf_buf.len(), &[&printf_buf])?;
    println!("--- end device output ---");

    Ok(())
}
