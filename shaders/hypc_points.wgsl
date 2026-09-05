 // HYPC camera-relative point renderer with MRT and billboarding.
 // RT0: color (rgb = base, a = coverage alpha)
 // RT1: linear eye depth (r = distance in metres, g = semantic label / 255, a = tag)

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
     @location(1)       depth_m     : f32,
     @location(2)       local_uv : vec2<f32>,
     @location(3)       visible  : u32,
 };

 @vertex
 fn vs_main(
     @location(0) corner : vec2<f32>,
     @location(1) ofs_m  : vec3<f32>,
     @location(2) label  : u32,
 ) -> VSOut {
     let world_rel   = U.delta_hi + (ofs_m + U.delta_lo);
     let clip_center = U.view_proj * vec4<f32>(world_rel, 1.0);

     // 🚫 Hard-kill billboards whose center is behind the camera.
     if (clip_center.w <= 0.0) {
         var o: VSOut;
         // Push entirely outside the clip volume; rasterizer drops it.
         o.clip     = vec4<f32>(-2.0, -2.0, 1.0, 1.0);
         o.label    = label;
         o.depth_m     = 0.0;
         o.local_uv = vec2<f32>(2.0, 2.0);
         o.visible  = 0u;
         return o;
     }

     // Normal billboarding path
     let point_size_ndc   = (U.point_size_px / U.viewport_size) * 2.0;
     let perspective_scale = clip_center.w; // w > 0 guaranteed here
     let offset = vec2<f32>(corner.x * point_size_ndc.x,
                            corner.y * point_size_ndc.y) * perspective_scale;

     var o : VSOut;
     o.clip     = vec4<f32>(clip_center.xy + offset, clip_center.z, clip_center.w);
     o.label    = label;
     o.depth_m     = clip_center.w;
     o.local_uv = corner;
     o.visible  = 1u;
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
     // Entire billboard is culled if its centre is behind the camera.
     if (in.visible == 0u) {
         discard;
     }

     // Circular alpha mask.
     let dist_sq = dot(in.local_uv, in.local_uv);
     if (dist_sq > 1.0) {
         discard;
     }
     let alpha = 1.0 - smoothstep(0.8, 1.0, dist_sq);

     var out : FSOut;
     // Color carries only coverage alpha (for blending-based AA).
     out.color = vec4<f32>(base_color(in.label), alpha);
     // Depth-linear proxy + semantic label + tag (1 = not grid).
     out.dlin = vec4<f32>(in.depth_m,
                         f32(in.label) / 255.0,
                         0.0,
                         1.0);
     return out;
 }
