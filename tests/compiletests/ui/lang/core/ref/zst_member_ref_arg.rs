// build-pass
// ignore-opencl1.2
// ignore-opencl2.0

use spirv_std::spirv;
struct A;
struct B;

struct S {
    x: A,
    y: B,
}

fn f(x: &B) {}

#[spirv(fragment)]
pub fn main() {
    let s = S { x: A, y: B };
    f(&s.y);
}
