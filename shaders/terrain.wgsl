struct TileUniform {
    delta_hi: vec3<f32>, _pad0: f32,
    delta_lo: vec3<f32>, _pad1: f32,
    view_proj: mat4x4<f32>, viewport_size: vec2<f32>, point_size_px: f32, splat_radius_m: f32,
};
@group(0) @binding(0) var<uniform> U: TileUniform;
struct Out { @builtin(position) clip: vec4<f32>, @location(0) eye_depth: f32 };
@vertex fn vs_main(@location(0) position: vec3<f32>) -> Out {
    let p = U.delta_hi + (position + U.delta_lo);
    var o: Out; o.clip = U.view_proj * vec4<f32>(p, 1.0); o.eye_depth = o.clip.w;
    // Reserve a far reversed-depth band for the coarse support surface. It
    // self-occludes, but every real point in the 300 km view range wins over it.
    o.clip.z *= 1e-7;
    return o;
}
struct Fragment { @location(0) color: vec4<f32>, @location(1) depth: vec4<f32> };
@fragment fn fs_main(i: Out) -> Fragment {
    var o: Fragment; o.color = vec4<f32>(0.68, 0.68, 0.68, 1.0);
    o.depth = vec4<f32>(i.eye_depth, 0.0, 0.0, 2.0); return o;
}
