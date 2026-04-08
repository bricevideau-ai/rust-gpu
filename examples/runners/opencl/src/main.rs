use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_ALL, Device, get_all_devices};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, cl_ulong};
use spirv_builder::SpirvBuilder;
use std::path::Path;
use std::ptr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Compile the kernel-shader to OpenCL SPIR-V.
    let path_to_crate = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shaders/kernel-shader");
    let compile_result = SpirvBuilder::new(path_to_crate, "spirv-unknown-opencl1.2").build()?;
    let spv_path = compile_result.module.unwrap_single();
    let spv_bytes = std::fs::read(spv_path)?;

    // 2. Find an OpenCL device and create a context + queue.
    let device_id = *get_all_devices(CL_DEVICE_TYPE_ALL)?
        .first()
        .expect("no OpenCL devices found");
    let device = Device::new(device_id);
    println!("Using device: {} ({})", device.name()?, device.vendor()?);

    let context = Context::from_device(&device)?;
    let queue =
        CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)?;

    // 3. Create a program from the SPIR-V IL binary.
    let mut program = Program::create_from_il(&context, &spv_bytes)?;
    program.build(context.devices(), "")?;

    let kernel = Kernel::create(&program, "main_kernel")?;

    // 4. Prepare input data: integers 1..2^20 (same as the wgpu runner).
    let top = 2u32.pow(20);
    let src_range = 1..top;
    let mut data: Vec<u32> = src_range.clone().collect();
    let n = data.len();

    // 5. Create device buffer and upload data.
    let mut buffer =
        unsafe { Buffer::<u32>::create(&context, CL_MEM_READ_WRITE, n, ptr::null_mut())? };

    unsafe {
        queue
            .enqueue_write_buffer(&mut buffer, CL_BLOCKING, 0, &data, &[])?
            .wait()?;
    }

    // 6. Set kernel arguments and execute.
    //    Kernel params (after linker conversion):
    //      arg 0 = buffer pointer (*CrossWorkgroup u32)
    //      arg 1 = slice length (u64)
    let len: cl_ulong = n as cl_ulong;

    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&buffer)
            .set_arg(&len)
            .set_global_work_size(n)
            .enqueue_nd_range(&queue)?
    };
    kernel_event.wait()?;

    // 7. Read back results.
    unsafe {
        queue
            .enqueue_read_buffer(&buffer, CL_BLOCKING, 0, &mut data, &[])?
            .wait()?;
    }

    // 8. Print the Collatz sequence record-holders (matching wgpu output).
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

    Ok(())
}
