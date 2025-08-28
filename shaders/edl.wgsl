// Eye‑dome lighting (EDL) post‑processing shader
// Enhances depth perception by darkening occluded areas.
// NOTE: preserves incoming alpha so SMC1 labels (in RT0.a) survive the next pass.

struct Uniforms {
    inv_size  : vec2<f32>,
    strength  : f32,
    radius_px : f32,
}

@group(0) @binding(0) var tColor    : texture_2d<f32>;
@group(0) @binding(1) var tDepthLin : texture_2d<f32>;
@group(0) @binding(2) var samp      : sampler;
@group(0) @binding(3) var<uniform> UBO : Uniforms;

struct VSOut {
    @builtin(position) clip : vec4<f32>,
    @location(0)      uv   : vec2<f32>,
};

@vertex
fn vs_main(@location(0) pos : vec2<f32>) -> VSOut {
    var out : VSOut;
    out.clip = vec4<f32>(pos, 0.0, 1.0);
    out.uv   = 0.5 * (pos + vec2<f32>(1.0, 1.0));
    return out;
}

@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
    let uv_c = in.uv;
    // Same orientation as color; no vertical flip.
    let uv_d = uv_c;

    // Sample source colour and linear depth (LOD 0 required for a non‑filtering sampler).
    let src = textureSampleLevel(tColor,    samp, uv_c, 0.0);
    let col = src.rgb;
    let z0  = textureSampleLevel(tDepthLin, samp, uv_d, 0.0).r;

    // Preserve background pixels (alpha carries the label).
    if (z0 >= 0.9999) {
        return src;
    }

    let px  = UBO.inv_size;
    let r   = UBO.radius_px;
    let eps = 1e-6;
    let lz0 = log(z0 + eps); // pre‑compute log(z0)

    // Offsets in screen space.
    let offsets = array<vec2<f32>, 8>(
        vec2<f32>( 1.0,  0.0),
        vec2<f32>(-1.0,  0.0),
        vec2<f32>( 0.0,  1.0),
        vec2<f32>( 0.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0, -1.0)
    );

    var s : f32 = 0.0;

    // Unrolled neighbourhood scan.
    let zi0 = textureSampleLevel(tDepthLin, samp, uv_d + offsets[0] * px * r, 0.0).r;
    s += max(0.0, log(zi0 + eps) - lz0);

    let zi1 = textureSampleLevel(tDepthLin, samp, uv_d + offsets[1] * px * r, 0.0).r;
    s += max(0.0, log(zi1 + eps) - lz0);

    let zi2 = textureSampleLevel(tDepthLin, samp, uv_d + offsets[2] * px * r, 0.0).r;
    s += max(0.0, log(zi2 + eps) - lz0);

    let zi3 = textureSampleLevel(tDepthLin, samp, uv_d + offsets[3] * px * r, 0.0).r;
    s += max(0.0, log(zi3 + eps) - lz0);

    let zi4 = textureSampleLevel(tDepthLin, samp, uv_d + offsets[4] * px * r, 0.0).r;
    s += max(0.0, log(zi4 + eps) - lz0);

    let zi5 = textureSampleLevel(tDepthLin, samp, uv_d + offsets[5] * px * r, 0.0).r;
    s += max(0.0, log(zi5 + eps) - lz0);

    let zi6 = textureSampleLevel(tDepthLin, samp, uv_d + offsets[6] * px * r, 0.0).r;
    s += max(0.0, log(zi6 + eps) - lz0);

    let zi7 = textureSampleLevel(tDepthLin, samp, uv_d + offsets[7] * px * r, 0.0).r;
    s += max(0.0, log(zi7 + eps) - lz0);

    let shade = exp(-UBO.strength * s);
    return vec4<f32>(col * shade, src.a); // preserve alpha (coverage)
}
