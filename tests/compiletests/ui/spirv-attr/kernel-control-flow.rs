// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::spirv;

#[spirv(kernel)]
pub fn test_if(#[spirv(cross_workgroup)] buf: &mut u32, cond: u32) {
    if cond != 0 {
        *buf = 1;
    }
}

#[spirv(kernel)]
pub fn test_if_else(#[spirv(cross_workgroup)] buf: &mut u32, cond: u32) {
    if cond != 0 {
        *buf = 1;
    } else {
        *buf = 0;
    }
}

#[spirv(kernel)]
pub fn test_if_else_if(#[spirv(cross_workgroup)] buf: &mut u32, val: u32) {
    if val > 100 {
        *buf = 3;
    } else if val > 10 {
        *buf = 2;
    } else if val > 0 {
        *buf = 1;
    } else {
        *buf = 0;
    }
}

#[spirv(kernel)]
pub fn test_while_loop(#[spirv(cross_workgroup)] buf: &mut u32, n: u32) {
    let mut i = 0u32;
    while i < n {
        i += 1;
    }
    *buf = i;
}

#[spirv(kernel)]
pub fn test_for_range(#[spirv(cross_workgroup)] buf: &mut u32, n: u32) {
    let mut sum = 0u32;
    for i in 0..n {
        sum += i;
    }
    *buf = sum;
}

#[spirv(kernel)]
pub fn test_loop_break(#[spirv(cross_workgroup)] buf: &mut u32, limit: u32) {
    let mut i = 0u32;
    loop {
        if i >= limit {
            break;
        }
        i += 1;
    }
    *buf = i;
}

#[spirv(kernel)]
pub fn test_loop_continue(#[spirv(cross_workgroup)] buf: &mut u32, n: u32) {
    let mut sum = 0u32;
    for i in 0..n {
        if i.is_multiple_of(2) {
            continue;
        }
        sum += i;
    }
    *buf = sum;
}

#[spirv(kernel)]
pub fn test_nested_loops(#[spirv(cross_workgroup)] buf: &mut u32, n: u32) {
    let mut sum = 0u32;
    for i in 0..n {
        for j in 0..n {
            sum += i * n + j;
        }
    }
    *buf = sum;
}

#[spirv(kernel)]
pub fn test_match(#[spirv(cross_workgroup)] buf: &mut u32, val: u32) {
    *buf = match val {
        0 => 10,
        1 => 20,
        2 => 30,
        _ => 0,
    };
}
