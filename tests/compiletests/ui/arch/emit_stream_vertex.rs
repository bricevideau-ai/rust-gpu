// build-pass
// compile-flags: -C target-feature=+Int64,+Geometry,+GeometryStreams
// ignore-opencl1.2
// ignore-opencl2.0

use spirv_std::spirv;

#[spirv(geometry(input_lines = 2, output_points = 2))]
pub fn main() {
    unsafe {
        spirv_std::arch::emit_stream_vertex::<2>();
    };
}
