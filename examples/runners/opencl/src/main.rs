use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_ALL, Device, get_all_devices};
use opencl3::event::Event;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, cl_ulong};
use spirv_builder::SpirvBuilder;
use std::path::Path;
use std::ptr;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Compile the kernel-shader to OpenCL SPIR-V ──────────────────
    let compile_start = Instant::now();
    let path_to_crate = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shaders/kernel-shader");
    let compile_result = SpirvBuilder::new(path_to_crate, "spirv-unknown-opencl1.2").build()?;
    let spv_path = compile_result.module.unwrap_single();
    let spv_bytes = std::fs::read(spv_path)?;
    println!(
        "Compiled kernel ({} bytes SPIR-V, {:?})",
        spv_bytes.len(),
        compile_start.elapsed()
    );

    // ── 2. Set up OpenCL device, context, queue ────────────────────────
    let device_id = *get_all_devices(CL_DEVICE_TYPE_ALL)?
        .first()
        .expect("no OpenCL devices found");
    let device = Device::new(device_id);
    println!("Device:  {} ({})", device.name()?, device.vendor()?);
    println!("Version: {}", device.version()?);

    let context = Context::from_device(&device)?;
    let queue =
        CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)?;

    // ── 3. Build program from SPIR-V IL ────────────────────────────────
    let mut program = Program::create_from_il(&context, &spv_bytes)
        .map_err(|e| format!("create_from_il: {e}"))?;
    program
        .build(context.devices(), "")
        .map_err(|e| format!("program.build: {e}"))?;
    let kernel =
        Kernel::create(&program, "main_kernel").map_err(|e| format!("create kernel: {e}"))?;

    // ── 4. Prepare input data ──────────────────────────────────────────
    let top = 2u32.pow(20);
    let src_range = 1..top;
    let mut data: Vec<u32> = src_range.clone().collect();
    let n = data.len();
    println!(
        "Input:   {} elements ({}..{})",
        n,
        src_range.start,
        src_range.end - 1
    );

    // ── 5. Upload to device ────────────────────────────────────────────
    let mut buffer =
        unsafe { Buffer::<u32>::create(&context, CL_MEM_READ_WRITE, n, ptr::null_mut())? };
    unsafe {
        queue
            .enqueue_write_buffer(&mut buffer, CL_BLOCKING, 0, &data, &[])?
            .wait()?;
    }

    // ── 6. Execute kernel ──────────────────────────────────────────────
    let len: cl_ulong = n as cl_ulong;
    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&buffer)
            .set_arg(&len)
            .set_global_work_size(n)
            .enqueue_nd_range(&queue)?
    };
    kernel_event.wait()?;

    // Read profiling timestamps.
    let kernel_duration = profiling_duration(&kernel_event);

    // ── 7. Read back results ───────────────────────────────────────────
    unsafe {
        queue
            .enqueue_read_buffer(&buffer, CL_BLOCKING, 0, &mut data, &[])?
            .wait()?;
    }

    // ── 8. Validate and print ──────────────────────────────────────────
    if let Some(duration) = kernel_duration {
        println!("Kernel:  {duration:?}");
    }

    // Verify a few known Collatz values.
    let checks: &[(u32, u32)] = &[
        (1, 0),    // collatz(1) = 0 steps
        (2, 1),    // collatz(2) = 1 step
        (3, 7),    // collatz(3) = 7 steps
        (27, 111), // collatz(27) = 111 steps
    ];
    let mut all_ok = true;
    for &(input, expected) in checks {
        let got = data[(input - 1) as usize]; // data is 0-indexed, inputs start at 1
        if got != expected {
            eprintln!("FAIL: collatz({input}) = {got}, expected {expected}");
            all_ok = false;
        }
    }
    if all_ok {
        println!("Verify:  all spot checks passed");
    }

    // Print the Collatz sequence record-holders (OEIS A006877).
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

    Ok(())
}

/// Extract kernel execution time from profiling events.
fn profiling_duration(event: &Event) -> Option<std::time::Duration> {
    let start = event.profiling_command_start().ok()?;
    let end = event.profiling_command_end().ok()?;
    Some(std::time::Duration::from_nanos(end - start))
}
