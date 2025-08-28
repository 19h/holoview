// CRT effect shader
// Adds scanlines, vignetting, and film grain for a retro display look

struct Uniforms {
    inv_size: vec2<f32>,
    time: f32,
    intensity: f32,
    vignette: f32,
}

@group(0) @binding(0) var tSrc: texture_2d<f32>;
@group(0) @binding(1) var tDepthLin: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> UBO: Uniforms;

struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0)          uv:   vec2<f32>,
}

@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> VSOut {
    var out: VSOut;
    out.clip = vec4<f32>(pos, 0.0, 1.0);
    out.uv   = 0.5 * (pos + vec2<f32>(1.0, 1.0));
    return out;
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    // Clamp FS‑triangle UVs to [0,1] and sample with the NonFiltering sampler
    let uv = clamp(in.uv, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));

    // Sample source texture (LOD 0.0 required for non‑filtering sampler)
    let uv_c = in.uv;
    let src_color = textureSampleLevel(tSrc, samp, uv_c, 0.0).rgb;

    // Depth texture information

    // Base background colour (dark gray)
    let bg_base = vec3<f32>(0.01, 0.01, 0.01);

    // --------------------------------------------------------------------
    // Scanline effect
    // --------------------------------------------------------------------
    let y_pixel   = uv.y / UBO.inv_size.y;
    let scan_sine = 0.5 + 0.5 * sin(6.28318 * (y_pixel * 0.5 + UBO.time * 12.0));
    let scan_bg   = 1.0 - UBO.intensity * scan_sine;
    let scan_fg   = 1.0 - (UBO.intensity * 0.25) * scan_sine;

    // --------------------------------------------------------------------
    // Film grain
    // --------------------------------------------------------------------
    let grain = (hash(uv * 2048.0 + vec2<f32>(UBO.time, -UBO.time)) - 0.5) * 0.02;

    // --------------------------------------------------------------------
    // Vignette
    // --------------------------------------------------------------------
    let vignette_dist = length((in.uv - vec2<f32>(0.5, 0.5)) / vec2<f32>(1.1, 0.95));
    let vignette      = 1.0 - UBO.vignette * smoothstep(0.6, 1.0, vignette_dist);

    // --------------------------------------------------------------------
    // Background mask (treat as background when depth ≈ 1 and alpha ≥ 0.5)
    // --------------------------------------------------------------------
    let uv_d = uv_c; // same orientation as color
    // Normalized sampling with NonFiltering sampler
    let depth_sample = textureSampleLevel(tDepthLin, samp, uv_d, 0.0);
    let depth_z      = depth_sample.r;
    let depth_alpha  = depth_sample.a; // 1 = real background, 0 = overlay (grid)

    // Keep thresholds consistent with other passes
    let bg_mask      = step(0.9999, depth_z) * step(0.5, depth_alpha);

    // Combine colours
    let bg_color = (bg_base * 0.85 + grain) * scan_bg;
    let fg_color = (src_color + grain * 0.4) * scan_fg;

    let final_rgb = mix(fg_color, bg_color, bg_mask) * vignette;
    return vec4<f32>(final_rgb, 1.0);
}
