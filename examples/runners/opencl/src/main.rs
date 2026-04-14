use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_ALL, Device, get_all_devices};
use opencl3::event::Event;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, cl_device_id};
use spirv_builder::{Capability, CompileResult, SpirvBuilder};
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
        self.build_program_with_options(spv_bytes, "")
    }

    /// Load a SPIR-V binary and build a program with the given `clBuildProgram`
    /// options string (e.g. `-cl-kernel-arg-info` to retain reflection metadata).
    fn build_program_with_options(
        &self,
        spv_bytes: &[u8],
        options: &str,
    ) -> Result<Program, Box<dyn std::error::Error>> {
        let mut program = Program::create_from_il(&self.context, spv_bytes)
            .map_err(|e| format!("create_from_il: {e}"))?;
        if let Err(e) = program.build(self.context.devices(), options) {
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

/// Compile a kernel crate with `OpName` reflection metadata preserved on
/// kernel arguments. Pair with `build_program_with_options(.., "-cl-kernel-arg-info")`
/// for `clGetKernelArgInfo` to return source argument names.
fn compile_kernel_with_arg_info(
    path: &Path,
) -> Result<(Vec<u8>, Duration), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let result: CompileResult = SpirvBuilder::new(path, "spirv-unknown-opencl1.2")
        .spirv_metadata(spirv_builder::SpirvMetadata::NameVariables)
        .build()?;
    let spv_path = result.module.unwrap_single();
    let spv_bytes = std::fs::read(spv_path)?;
    Ok((spv_bytes, start.elapsed()))
}

/// Compile a kernel crate with `Float64` capability.
fn compile_kernel_fp64(path: &Path) -> Result<(Vec<u8>, Duration), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let result: CompileResult = SpirvBuilder::new(path, "spirv-unknown-opencl1.2")
        .capability(Capability::Float64)
        .build()?;
    let spv_path = result.module.unwrap_single();
    let spv_bytes = std::fs::read(spv_path)?;
    Ok((spv_bytes, start.elapsed()))
}

/// Compile a kernel crate with `Groups` capability (`OpenCL` 2.0 for subgroup ops).
fn compile_kernel_groups(path: &Path) -> Result<(Vec<u8>, Duration), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let result: CompileResult = SpirvBuilder::new(path, "spirv-unknown-opencl2.0")
        .capability(Capability::Groups)
        .build()?;
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

// ── Test sections ─────────────────────────────────────────────────────

fn run_collatz(ocl: &OclContext, program: &Program) -> Result<(), Box<dyn std::error::Error>> {
    let kernel = Kernel::create(program, "main_kernel")?;

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
    let event = ocl.run(&kernel, buf.len(), &[&buf])?;
    ocl.download(&buf, &mut data)?;

    if let Some(duration) = profiling_duration(&event) {
        println!("Kernel:  {duration:?}");
    }

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

/// Regression test for kernel argument ordering. The kernel's signature is
/// `(builtin_id, &mut [u32], scalar_a, scalar_b)` — a slice (which expands
/// to a `(ptr, len)` pair of SPIR-V kernel args) interleaved with two
/// scalars. We set the args in source order and verify the kernel sees
/// the right values. If the linker's `kernel_arguments` pass mis-orders
/// the parameters, the kernel reads `scalar_b` into the `scalar_a` slot or
/// vice versa, and the verification fails.
fn run_arg_ordering_test(
    ocl: &OclContext,
    program: &Program,
) -> Result<(), Box<dyn std::error::Error>> {
    let kernel = Kernel::create(program, "arg_ordering_test")?;
    let buf = ocl.upload(&[0u32; 4])?;
    let scalar_a: u32 = 0xAAAA_AAAA;
    let scalar_b: u32 = 0xBBBB_BBBB;
    let _event = ocl.run(&kernel, 1, &[&buf, &scalar_a, &scalar_b])?;
    let mut out = vec![0u32; 4];
    ocl.download(&buf, &mut out)?;
    if out[0] != scalar_a || out[1] != scalar_b {
        return Err(format!(
            "FAIL: data[0]=0x{:08x} (expected 0x{:08x}), data[1]=0x{:08x} (expected 0x{:08x})",
            out[0], scalar_a, out[1], scalar_b
        )
        .into());
    }
    println!("Verify:  scalar args arrived in source order");
    Ok(())
}

/// End-to-end check that source-level kernel argument names are preserved
/// through codegen, the kernel-arguments linker pass, SPIR-T round-trip,
/// and the `name_variables_pass` strip; and surface via `clGetKernelArgInfo`
/// when the program is built with `-cl-kernel-arg-info`.
///
/// The Rust signature
/// ```ignore
/// fn arg_ordering_test(
///     #[spirv(global_invocation_id)] _id: USizeVec3,
///     #[spirv(cross_workgroup)] data: &mut [u32],
///     scalar_a: u32,
///     scalar_b: u32,
/// )
/// ```
/// becomes the kernel arg list `data, data.len, scalar_a, scalar_b` (slice
/// decomposition appends the length as a separate `usize` arg).
fn run_arg_info_test(ocl: &OclContext) -> Result<(), Box<dyn std::error::Error>> {
    let kernel_crate = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shaders/kernel-shader");
    let (spv_bytes, _) = compile_kernel_with_arg_info(&kernel_crate)?;
    let program = ocl.build_program_with_options(&spv_bytes, "-cl-kernel-arg-info")?;
    let kernel = Kernel::create(&program, "arg_ordering_test")?;

    let num_args = kernel.num_args()?;

    // Probe support: drivers that don't surface arg info from SPIR-V binaries
    // return `CL_KERNEL_ARG_INFO_NOT_AVAILABLE`. Treat that as "skip", not
    // "fail": our codegen is still correct, the driver just doesn't expose it.
    match kernel.get_arg_name(0) {
        Ok(_) => {}
        Err(e) if e.to_string().contains("CL_KERNEL_ARG_INFO_NOT_AVAILABLE") => {
            println!("Skipped: driver does not surface kernel arg info from SPIR-V");
            return Ok(());
        }
        Err(e) => return Err(e.into()),
    }

    let names: Vec<String> = (0..num_args)
        .map(|i| kernel.get_arg_name(i))
        .collect::<Result<_, _>>()?;
    let expected = ["data", "data.len", "scalar_a", "scalar_b"];

    if num_args as usize != expected.len() {
        return Err(format!("expected {} args, got {num_args}", expected.len()).into());
    }
    for (i, (got, want)) in names.iter().zip(expected.iter()).enumerate() {
        if got != want {
            return Err(format!("arg {i}: got {got:?}, expected {want:?}").into());
        }
    }
    println!("Verify:  arg names = {names:?}");

    // Bonus: the address qualifier should distinguish the slice pointer
    // (global / `CrossWorkgroup`) from the scalar args (private).
    use opencl3::kernel::{CL_KERNEL_ARG_ADDRESS_GLOBAL, CL_KERNEL_ARG_ADDRESS_PRIVATE};
    let aq: Vec<u32> = (0..num_args)
        .map(|i| kernel.get_arg_address_qualifier(i))
        .collect::<Result<_, _>>()?;
    let expected_aq = [
        CL_KERNEL_ARG_ADDRESS_GLOBAL,  // data: &mut [u32]
        CL_KERNEL_ARG_ADDRESS_PRIVATE, // data.len: usize
        CL_KERNEL_ARG_ADDRESS_PRIVATE, // scalar_a: u32
        CL_KERNEL_ARG_ADDRESS_PRIVATE, // scalar_b: u32
    ];
    if aq != expected_aq {
        return Err(format!("address qualifiers: got {aq:?}, expected {expected_aq:?}").into());
    }
    println!("Verify:  address qualifiers = [global, private, private, private]");
    Ok(())
}

fn run_printf(ocl: &OclContext, program: &Program) -> Result<(), Box<dyn std::error::Error>> {
    let printf_kernel = Kernel::create(program, "printf_test")?;
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

fn run_printf_float(ocl: &OclContext, program: &Program) -> Result<(), Box<dyn std::error::Error>> {
    let float_kernel = Kernel::create(program, "printf_float_test")?;
    println!("--- device output ---");
    ocl.run(&float_kernel, 1, &[])?;
    println!("--- end device output ---");
    Ok(())
}

fn run_printf_fp64(ocl: &OclContext) -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(ocl.device_id);
    let has_fp64 = device.double_fp_config().unwrap_or(0) != 0;
    if !has_fp64 {
        println!("Skipped: device does not support fp64");
        return Ok(());
    }

    let fp64_crate = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shaders/kernel-fp64-shader");
    let (fp64_spv, fp64_time) = compile_kernel_fp64(&fp64_crate)?;
    println!(
        "Compiled fp64 kernel ({} bytes, {fp64_time:?})",
        fp64_spv.len()
    );
    let fp64_program = ocl.build_program(&fp64_spv)?;
    let fp64_kernel = Kernel::create(&fp64_program, "printf_fp64_test")?;

    let floats: Vec<f32> = vec![1.234, 5.678, 9.012, 0.345];
    let doubles: Vec<f64> = vec![1.5, 2.25, 4.125, 8.0625];
    let float_buf = ocl.upload(&floats)?;
    let double_buf = ocl.upload(&doubles)?;
    println!("--- device output ---");
    ocl.run(&fp64_kernel, floats.len(), &[&float_buf, &double_buf])?;
    println!("--- end device output ---");
    Ok(())
}

fn run_subgroup_tests(ocl: &OclContext) -> Result<(), Box<dyn std::error::Error>> {
    let test_crate = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shaders/kernel-test-shader");
    let (test_spv, test_time) = compile_kernel_groups(&test_crate)?;
    println!(
        "Compiled test kernel ({} bytes, {test_time:?})",
        test_spv.len()
    );
    let test_program = ocl.build_program(&test_spv)?;

    const WG: usize = 32;
    const NUM_WG: usize = 4;
    const N: usize = WG * NUM_WG;
    let mut pass_count = 0u32;
    let mut fail_count = 0u32;

    fn check(name: &str, result: &[u32], expected: &[u32], pass: &mut u32, fail: &mut u32) {
        let mut ok = true;
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            if got != exp {
                if ok {
                    println!("FAIL");
                }
                eprintln!("  [{i}]: got {got}, expected {exp}");
                ok = false;
            }
        }
        if ok {
            println!("PASS");
            *pass += 1;
        } else {
            *fail += 1;
        }
        let _ = name;
    }

    // Test 1: subgroup builtins (multi-workgroup)
    {
        let k = Kernel::create(&test_program, "test_subgroup_builtins")?;
        let sg_id = ocl.upload(&vec![0u32; N])?;
        let sg_lid = ocl.upload(&vec![0u32; N])?;
        let num_sg = ocl.upload(&vec![0u32; N])?;
        let sg_size_buf = ocl.upload(&vec![0u32; N])?;
        ocl.run(&k, N, &[&sg_id, &sg_lid, &num_sg, &sg_size_buf])?;
        let mut r_size = vec![0u32; N];
        let mut r_num = vec![0u32; N];
        let mut r_id = vec![0u32; N];
        let mut r_lid = vec![0u32; N];
        ocl.download(&sg_size_buf, &mut r_size)?;
        ocl.download(&num_sg, &mut r_num)?;
        ocl.download(&sg_id, &mut r_id)?;
        ocl.download(&sg_lid, &mut r_lid)?;

        let sg_sz = r_size[0];
        let n_sg = r_num[0];
        print!("Test 1 (subgroup builtins, {NUM_WG} WGs): sg_size={sg_sz}, num_sg={n_sg} ... ");

        let mut expected_id = vec![0u32; N];
        let mut expected_lid = vec![0u32; N];
        for i in 0..N {
            let lid = (i % WG) as u32;
            expected_id[i] = lid / sg_sz;
            expected_lid[i] = lid % sg_sz;
        }
        let mut ok = true;
        for i in 0..N {
            if r_id[i] != expected_id[i] || r_lid[i] != expected_lid[i] {
                if ok {
                    println!("FAIL");
                }
                eprintln!(
                    "  [{i}]: sg_id={} (exp {}), sg_lid={} (exp {})",
                    r_id[i], expected_id[i], r_lid[i], expected_lid[i]
                );
                ok = false;
            }
        }
        if ok {
            println!("PASS");
            pass_count += 1;
        } else {
            fail_count += 1;
        }
    }

    // Test 2: shared memory + barrier (multi-workgroup)
    {
        let k = Kernel::create(&test_program, "test_shared_barrier")?;
        let out = ocl.upload(&vec![0u32; N])?;
        ocl.run(&k, N, &[&out])?;
        let mut r = vec![0u32; N];
        ocl.download(&out, &mut r)?;

        let mut expected = vec![0u32; N];
        for (i, exp) in expected.iter_mut().enumerate() {
            let lid = i % WG;
            *exp = (31 - lid) as u32;
        }
        print!("Test 2 (shared + barrier, {NUM_WG} WGs): ");
        check("test2", &r, &expected, &mut pass_count, &mut fail_count);
    }

    // Test 3: group_i_add reduce (multi-workgroup)
    {
        let k = Kernel::create(&test_program, "test_group_reduce")?;
        let out = ocl.upload(&vec![0u32; N])?;
        ocl.run(&k, N, &[&out])?;
        let mut r = vec![0u32; N];
        ocl.download(&out, &mut r)?;

        print!("Test 3 (group_i_add reduce, {NUM_WG} WGs): ");
        for wg in 0..NUM_WG {
            print!("wg{wg}={} ", r[wg * WG]);
        }
        let expected = vec![528u32; N];
        check("test3", &r, &expected, &mut pass_count, &mut fail_count);
    }

    // Test 4: group_exclusive_i_add scan (multi-workgroup)
    {
        let k = Kernel::create(&test_program, "test_group_scan")?;
        let out = ocl.upload(&vec![0u32; N])?;
        ocl.run(&k, N, &[&out])?;
        let mut r = vec![0u32; N];
        ocl.download(&out, &mut r)?;

        let mut expected = vec![0u32; N];
        for wg in 0..NUM_WG {
            let mut acc = 0u32;
            for lid in 0..WG {
                expected[wg * WG + lid] = acc;
                acc += lid as u32 + 1;
            }
        }
        print!("Test 4 (group_exclusive_i_add scan, {NUM_WG} WGs): ");
        for wg in 0..NUM_WG {
            print!("wg{wg}=[{}..{}] ", r[wg * WG], r[wg * WG + WG - 1]);
        }
        check("test4", &r, &expected, &mut pass_count, &mut fail_count);
    }

    // Test 5: shared + subgroup builtins (multi-workgroup)
    {
        let k = Kernel::create(&test_program, "test_shared_with_subgroup_builtins")?;
        let out = ocl.upload(&vec![0u32; N])?;
        ocl.run(&k, N, &[&out])?;
        let mut r = vec![0u32; N];
        ocl.download(&out, &mut r)?;

        let expected = vec![0u32; N];
        print!("Test 5 (shared + subgroup builtins, {NUM_WG} WGs): ");
        for wg in 0..NUM_WG {
            print!("wg{wg}={} ", r[wg * WG]);
        }
        check("test5", &r, &expected, &mut pass_count, &mut fail_count);
    }

    // Test 6: subgroup ops + shared (no builtins, multi-workgroup)
    {
        let k = Kernel::create(&test_program, "test_subgroup_ops_with_shared")?;
        let out = ocl.upload(&vec![0u32; N])?;
        ocl.run(&k, N, &[&out])?;
        let mut r = vec![0u32; N];
        ocl.download(&out, &mut r)?;

        let expected = vec![528u32; N];
        print!("Test 6 (subgroup ops + shared, {NUM_WG} WGs): ");
        for wg in 0..NUM_WG {
            print!("wg{wg}={} ", r[wg * WG]);
        }
        check("test6", &r, &expected, &mut pass_count, &mut fail_count);
    }

    // Test 7: all three combined (multi-workgroup)
    {
        let k = Kernel::create(&test_program, "test_all_combined")?;
        let out = ocl.upload(&vec![0u32; N])?;
        ocl.run(&k, N, &[&out])?;
        let mut r = vec![0u32; N];
        ocl.download(&out, &mut r)?;

        let expected = vec![528u32; N];
        print!("Test 7 (all combined, {NUM_WG} WGs): ");
        for wg in 0..NUM_WG {
            print!("wg{wg}={} ", r[wg * WG]);
        }
        check("test7", &r, &expected, &mut pass_count, &mut fail_count);
    }

    println!("\nTotal: {pass_count} passed, {fail_count} failed");
    if fail_count > 0 {
        return Err(format!("{fail_count} subgroup test(s) failed").into());
    }
    Ok(())
}

fn run_workgroup_collective_tests(ocl: &OclContext) -> Result<(), Box<dyn std::error::Error>> {
    // Reuse the kernel-test-shader crate (already compiled with Groups capability).
    let test_crate = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shaders/kernel-test-shader");
    let (test_spv, test_time) = compile_kernel_groups(&test_crate)?;
    println!(
        "Compiled test kernel ({} bytes, {test_time:?})",
        test_spv.len()
    );
    let test_program = ocl.build_program(&test_spv)?;

    const WG: usize = 32;
    const NUM_WG: usize = 4;
    const N: usize = WG * NUM_WG;
    // Sum of 1..=WG (constant-folded): 528 for WG=32. Each work-group
    // computes the full sum independently, so every output slot equals it.
    const WG_SUM: u32 = (WG as u32) * (WG as u32 + 1) / 2;

    let mut pass_count = 0u32;
    let mut fail_count = 0u32;

    fn check(result: &[u32], expected: &[u32], pass: &mut u32, fail: &mut u32) {
        let mut ok = true;
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            if got != exp {
                if ok {
                    println!("FAIL");
                }
                eprintln!("  [{i}]: got {got}, expected {exp}");
                ok = false;
            }
        }
        if ok {
            println!("PASS");
            *pass += 1;
        } else {
            *fail += 1;
        }
    }

    // Test 1: work_group_i_add reduce — every lane in the WG gets the WG-wide sum.
    {
        let k = Kernel::create(&test_program, "test_work_group_reduce")?;
        let out = ocl.upload(&vec![0u32; N])?;
        ocl.run(&k, N, &[&out])?;
        let mut r = vec![0u32; N];
        ocl.download(&out, &mut r)?;

        let expected = vec![WG_SUM; N];
        print!("Test 1 (work_group_i_add reduce, {NUM_WG} WGs, expect {WG_SUM} per lane): ");
        check(&r, &expected, &mut pass_count, &mut fail_count);
    }

    // Test 2: work_group_inclusive_i_add scan — out[lid] = sum(1..=lid+1).
    {
        let k = Kernel::create(&test_program, "test_work_group_inclusive_scan")?;
        let out = ocl.upload(&vec![0u32; N])?;
        ocl.run(&k, N, &[&out])?;
        let mut r = vec![0u32; N];
        ocl.download(&out, &mut r)?;

        let mut expected = vec![0u32; N];
        for wg in 0..NUM_WG {
            let mut acc = 0u32;
            for lid in 0..WG {
                acc += lid as u32 + 1;
                expected[wg * WG + lid] = acc;
            }
        }
        print!("Test 2 (work_group_inclusive_i_add scan, {NUM_WG} WGs): ");
        check(&r, &expected, &mut pass_count, &mut fail_count);
    }

    // Test 3: work_group_exclusive_i_add scan — out[lid] = sum(1..lid+1), out[0]=0.
    {
        let k = Kernel::create(&test_program, "test_work_group_exclusive_scan")?;
        let out = ocl.upload(&vec![0u32; N])?;
        ocl.run(&k, N, &[&out])?;
        let mut r = vec![0u32; N];
        ocl.download(&out, &mut r)?;

        let mut expected = vec![0u32; N];
        for wg in 0..NUM_WG {
            let mut acc = 0u32;
            for lid in 0..WG {
                expected[wg * WG + lid] = acc;
                acc += lid as u32 + 1;
            }
        }
        print!("Test 3 (work_group_exclusive_i_add scan, {NUM_WG} WGs): ");
        check(&r, &expected, &mut pass_count, &mut fail_count);
    }

    // Test 4: work_group_broadcast — every lane sees lid=7's contribution (700).
    {
        let k = Kernel::create(&test_program, "test_work_group_broadcast_kernel")?;
        let out = ocl.upload(&vec![0u32; N])?;
        ocl.run(&k, N, &[&out])?;
        let mut r = vec![0u32; N];
        ocl.download(&out, &mut r)?;

        let expected = vec![7u32 * 100; N];
        print!("Test 4 (work_group_broadcast from lid=7, {NUM_WG} WGs, expect 700): ");
        check(&r, &expected, &mut pass_count, &mut fail_count);
    }

    // Test 5: work_group_all / work_group_any — three predicates checked at once.
    {
        let k = Kernel::create(&test_program, "test_work_group_vote")?;
        let out_all_true = ocl.upload(&vec![0u32; N])?;
        let out_all_one = ocl.upload(&vec![0u32; N])?;
        let out_any_one = ocl.upload(&vec![0u32; N])?;
        ocl.run(&k, N, &[&out_all_true, &out_all_one, &out_any_one])?;
        let mut r_all_true = vec![0u32; N];
        let mut r_all_one = vec![0u32; N];
        let mut r_any_one = vec![0u32; N];
        ocl.download(&out_all_true, &mut r_all_true)?;
        ocl.download(&out_all_one, &mut r_all_one)?;
        ocl.download(&out_any_one, &mut r_any_one)?;

        let expect_all_true = vec![1u32; N];
        let expect_all_one = vec![0u32; N];
        let expect_any_one = vec![1u32; N];
        print!("Test 5a (work_group_all of `lid<32`, expect 1): ");
        check(
            &r_all_true,
            &expect_all_true,
            &mut pass_count,
            &mut fail_count,
        );
        print!("Test 5b (work_group_all of `lid==5`, expect 0): ");
        check(
            &r_all_one,
            &expect_all_one,
            &mut pass_count,
            &mut fail_count,
        );
        print!("Test 5c (work_group_any of `lid==5`, expect 1): ");
        check(
            &r_any_one,
            &expect_any_one,
            &mut pass_count,
            &mut fail_count,
        );
    }

    println!("\nTotal: {pass_count} passed, {fail_count} failed");
    if fail_count > 0 {
        return Err(format!("{fail_count} work-group collective test(s) failed").into());
    }
    Ok(())
}

// ── Main ───────────────────────────────────────────────────────────────

/// Run a named test section, catching and reporting any errors.
/// Returns `true` on success, `false` on failure.
fn section(name: &str, f: impl FnOnce() -> Result<(), Box<dyn std::error::Error>>) -> bool {
    println!("\n═══ {name} ═══");
    match f() {
        Ok(()) => true,
        Err(e) => {
            eprintln!("ERROR: {e}");
            false
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile kernel-shader crate (shared by collatz and printf).
    let kernel_crate = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shaders/kernel-shader");
    let (spv_bytes, compile_time) = compile_kernel(&kernel_crate)?;
    println!(
        "Compiled kernel ({} bytes SPIR-V, {compile_time:?})",
        spv_bytes.len()
    );

    // Set up device.
    let ocl = OclContext::new()?;

    // Build program for kernel-shader.
    let program = ocl.build_program(&spv_bytes)?;

    let mut errors = 0u32;

    // Each section runs independently — a failure in one does not prevent
    // the others from running.

    if !section("Collatz", || run_collatz(&ocl, &program)) {
        errors += 1;
    }
    if !section("Kernel arg ordering", || {
        run_arg_ordering_test(&ocl, &program)
    }) {
        errors += 1;
    }
    if !section("Kernel arg info reflection", || run_arg_info_test(&ocl)) {
        errors += 1;
    }
    if !section("printf test", || run_printf(&ocl, &program)) {
        errors += 1;
    }
    if !section("printf float test", || run_printf_float(&ocl, &program)) {
        errors += 1;
    }
    if !section("printf fp64 test", || run_printf_fp64(&ocl)) {
        errors += 1;
    }
    if !section("Subgroup & shared memory tests", || {
        run_subgroup_tests(&ocl)
    }) {
        errors += 1;
    }
    if !section("Work-group collective tests", || {
        run_workgroup_collective_tests(&ocl)
    }) {
        errors += 1;
    }

    if errors > 0 {
        eprintln!("\n{errors} section(s) failed");
        std::process::exit(1);
    }

    Ok(())
}
