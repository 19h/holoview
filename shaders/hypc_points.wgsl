 // HYPC camera‑relative point renderer with MRT and billboarding.
 // RT0: color (rgb = base, a = coverage alpha)
 // RT1: depth‑linear proxy (r = z_ndc [0..1], g = semantic label / 255, a = tag)

 struct TileUniform {
     delta_hi      : vec3<f32>,
     _pad0         : f32,
     delta_lo      : vec3<f32>,
     _pad1         : f32,
     view_proj     : mat4x4<f32>,
     viewport_size : vec2<f32>,
     point_size_px : f32,
     _pad2         : f32,
 };

 @group(0) @binding(0) var<uniform> U : TileUniform;

 struct VSOut {
     @builtin(position) clip     : vec4<f32>,
     @location(0)       label    : u32,
     @location(1)       zndc     : f32,
     @location(2)       local_uv : vec2<f32>,
 };

 @vertex
 fn vs_main(
     @location(0) corner : vec2<f32>, // Quad vertex buffer (step_mode = Vertex)
     @location(1) ofs_m  : vec3<f32>, // Instance buffer (step_mode = Instance)
     @location(2) label  : u32,       // Instance buffer (step_mode = Instance)
 ) -> VSOut {
     // 1. Reconstruct high‑precision camera‑relative position.
     let world_rel = (U.delta_hi + U.delta_lo) + ofs_m;

     // 2. Project to clip space to find the centre point.
     let clip_center = U.view_proj * vec4<f32>(world_rel, 1.0);

     // 3. Perform billboarding in clip space.
     // Convert point size from pixels to normalized device coordinates (NDC).
     let point_size_ndc = (U.point_size_px / U.viewport_size) * 2.0;

     // Clamp the perspective scaling factor to be non-negative. This prevents
     // billboards from inverting and exploding when behind the camera.
     let perspective_scale = max(clip_center.w, 0.0);

     let offset = vec2<f32>(corner.x * point_size_ndc.x,
                            corner.y * point_size_ndc.y) * perspective_scale;

     var o : VSOut;
     o.clip     = vec4<f32>(clip_center.xy + offset, clip_center.z, clip_center.w);
     o.label    = label;
     // With Mat4::perspective_rh() we are already in 0..1 NDC after divide.
     o.zndc     = clamp(o.clip.z / o.clip.w, 0.0, 1.0);
     o.local_uv = corner; // Pass corner to fragment for circular alpha mask.
     return o;
 }

 struct FSOut {
     @location(0) color : vec4<f32>,
     @location(1) dlin  : vec4<f32>,
 };

 fn base_color(_label : u32) -> vec3<f32> {
     // Slightly darker neutral; semantic tint happens in post.
     return vec3<f32>(0.70, 0.70, 0.70);
 }

 @fragment
 fn fs_main(in : VSOut) -> FSOut {
     // Circular alpha mask.
     let dist_sq = dot(in.local_uv, in.local_uv);
     if (dist_sq > 1.0) {
         discard;
     }
     let alpha = 1.0 - smoothstep(0.8, 1.0, dist_sq);

     var out : FSOut;
     // Color carries only coverage alpha (for blending‑based AA).
     out.color = vec4<f32>(base_color(in.label), alpha);
     // Depth‑linear proxy + semantic label + tag (1 = not grid).
     out.dlin = vec4<f32>(clamp(in.zndc, 0.0, 1.0),
                         f32(in.label) / 255.0,
                         0.0,
                         1.0);
     return out;
 }
