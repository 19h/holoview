// Eye-dome lighting (EDL) post-processing shader
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
    // Correctly map clip space to UV space for WebGPU/Vulkan/Metal.
    // Clip space Y is -1 (bottom) to +1 (top).
    // UV space Y is  0 (top)    to  1 (bottom).
    // The Y-coordinate must be flipped.
    out.uv = vec2<f32>(0.5 * (pos.x + 1.0), 0.5 * (-pos.y + 1.0));
    return out;
}

// Helper: accumulate only from valid neighbor (tag>=0.5 AND z<1)
fn acc(uv_n: vec2<f32>, lz0: f32) -> f32 {
    let dl = textureSampleLevel(tDepthLin, samp, uv_n, 0.0);
    let z  = dl.r;
    let a  = dl.a;
    // mask: 1 for real geometry neighbor, else 0
    let m  = select(0.0, 1.0, a >= 0.5 && z < 0.9999);
    return m * max(0.0, log(z + 1e-6) - lz0);
}

@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
    let uv_c = in.uv;
    let uv_d = uv_c;

    // Source
    let src = textureSampleLevel(tColor,    samp, uv_c, 0.0);
    let col = src.rgb;

    // Center depth + tag
    let dl0 = textureSampleLevel(tDepthLin, samp, uv_d, 0.0);
    let z0  = dl0.r;
    let a0  = dl0.a;

    // Skip non-geometry (grid/tag==0) and true background (z≈1)
    if (a0 < 0.5 || z0 >= 0.9999) {
        return src;
    }

    let px  = UBO.inv_size;
    let r   = UBO.radius_px;
    let eps = 1e-6;
    let lz0 = log(z0 + eps);

    let offsets = array<vec2<f32>, 8>(
        vec2<f32>( 1.0,  0.0), vec2<f32>(-1.0,  0.0),
        vec2<f32>( 0.0,  1.0), vec2<f32>( 0.0, -1.0),
        vec2<f32>( 1.0,  1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0, -1.0)
    );

    var s : f32 = 0.0;

    s += acc(uv_d + offsets[0] * px * r, lz0);
    s += acc(uv_d + offsets[1] * px * r, lz0);
    s += acc(uv_d + offsets[2] * px * r, lz0);
    s += acc(uv_d + offsets[3] * px * r, lz0);
    s += acc(uv_d + offsets[4] * px * r, lz0);
    s += acc(uv_d + offsets[5] * px * r, lz0);
    s += acc(uv_d + offsets[6] * px * r, lz0);
    s += acc(uv_d + offsets[7] * px * r, lz0);

    let shade = exp(-UBO.strength * s);
    return vec4<f32>(col * shade, src.a); // preserve point coverage alpha
}
