@group(0) @binding(0) var raw_depth: texture_2d<f32>;
@group(0) @binding(1) var display_depth: texture_2d<f32>;
struct Counts { raw: atomic<u32>, display: atomic<u32> };
@group(0) @binding(2) var<storage, read_write> counts: Counts;
var<workgroup> raw_count: atomic<u32>;
var<workgroup> display_count: atomic<u32>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) local: u32) {
    if local == 0u { atomicStore(&raw_count, 0u); atomicStore(&display_count, 0u); }
    workgroupBarrier();
    let size = textureDimensions(raw_depth);
    if all(id.xy < size) {
        let a = textureLoad(raw_depth, vec2<i32>(id.xy), 0);
        let b = textureLoad(display_depth, vec2<i32>(id.xy), 0);
        if a.a > 0.5 && a.a < 1.5 && a.r > 0.0 { atomicAdd(&raw_count, 1u); }
        if b.a > 0.5 && b.r > 0.0 { atomicAdd(&display_count, 1u); }
    }
    workgroupBarrier();
    if local == 0u {
        atomicAdd(&counts.raw, atomicLoad(&raw_count));
        atomicAdd(&counts.display, atomicLoad(&display_count));
    }
}
