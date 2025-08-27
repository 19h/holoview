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
    @location(0) uv: vec2<f32>
}

@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> VSOut {
    var o: VSOut;
    o.clip = vec4<f32>(pos, 0.0, 1.0);
    o.uv = 0.5 * (pos + vec2<f32>(1.0, 1.0));
    return o;
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    // Use textureSampleLevel with LOD 0.0 to be compliant with the NonFiltering sampler.
    let col = textureSampleLevel(tSrc, samp, in.uv, 0.0).rgb;
    let dims = textureDimensions(tDepthLin);
    let coord = vec2<i32>(in.uv * vec2<f32>(dims));

    // Dark background to match reference
    let bg_base = vec3<f32>(0.01, 0.01, 0.01);

    // Scanlines (strong on background, subtle on content)
    let y_pix = in.uv.y / UBO.inv_size.y;
    let scan_sine = 0.5 + 0.5 * sin(6.28318 * (y_pix * 0.5 + UBO.time * 12.0));
    let scan_bg = 1.0 - UBO.intensity * scan_sine;
    let scan_fg = 1.0 - (UBO.intensity * 0.25) * scan_sine;

    // Film grain
    let g = (hash(in.uv * 2048.0 + vec2<f32>(UBO.time, -UBO.time)) - 0.5) * 0.02;

    // Vignette
    let d = length((in.uv - vec2<f32>(0.5, 0.5)) / vec2<f32>(1.1, 0.95));
    let vig = 1.0 - UBO.vignette * smoothstep(0.6, 1.0, d);

    // Background mask: only treat as background if z≈1 AND alpha≥0.5
    let dlin = textureLoad(tDepthLin, coord, 0);
    let z = dlin.r;
    let tag = dlin.a;  // 1 = real background, 0 = overlay (grid)
    let bgm = step(0.9995, z) * step(0.5, tag);

    let bg_col = (bg_base * 0.85 + g) * scan_bg;
    let fg_col = (col + g * 0.4) * scan_fg;

    let out_rgb = mix(fg_col, bg_col, bgm) * vig;
    return vec4<f32>(out_rgb, 1.0);
}
