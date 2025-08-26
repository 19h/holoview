// Eye-dome lighting (EDL) post-processing shader
// Enhances depth perception by darkening occluded areas

struct Uniforms {
    inv_size: vec2<f32>,
    strength: f32,
    radius_px: f32,
}

@group(0) @binding(0) var tColor: texture_2d<f32>;
@group(0) @binding(1) var tDepthLin: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> UBO: Uniforms;

struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> VSOut {
    var o: VSOut;
    o.clip = vec4<f32>(pos, 0.0, 1.0);
    o.uv = 0.5 * (pos + vec2<f32>(1.0, 1.0));
    return o;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let col = textureSampleLevel(tColor, samp, in.uv, 0.0).rgb;
    let z0 = textureSampleLevel(tDepthLin, samp, in.uv, 0.0).r;
    
    // Skip background pixels
    if (z0 >= 0.9999) {
        return vec4<f32>(col, 1.0);
    }

    let px = UBO.inv_size;
    let r = UBO.radius_px;
    let eps = 1e-6;
    let lz0 = log(z0 + eps); // Precompute log(z0)

    // Sample 8 neighboring pixels
    let o = array<vec2<f32>, 8>(
        vec2<f32>( 1.0, 0.0), vec2<f32>(-1.0, 0.0),
        vec2<f32>( 0.0, 1.0), vec2<f32>( 0.0,-1.0),
        vec2<f32>( 1.0, 1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>( 1.0,-1.0), vec2<f32>(-1.0,-1.0)
    );
    
    var s: f32 = 0.0;
    
    // Unrolled loop for better performance
    let zi0 = textureSampleLevel(tDepthLin, samp, in.uv + o[0] * px * r, 0.0).r;
    s = s + max(0.0, log(zi0 + eps) - lz0);
    let zi1 = textureSampleLevel(tDepthLin, samp, in.uv + o[1] * px * r, 0.0).r;
    s = s + max(0.0, log(zi1 + eps) - lz0);
    let zi2 = textureSampleLevel(tDepthLin, samp, in.uv + o[2] * px * r, 0.0).r;
    s = s + max(0.0, log(zi2 + eps) - lz0);
    let zi3 = textureSampleLevel(tDepthLin, samp, in.uv + o[3] * px * r, 0.0).r;
    s = s + max(0.0, log(zi3 + eps) - lz0);
    let zi4 = textureSampleLevel(tDepthLin, samp, in.uv + o[4] * px * r, 0.0).r;
    s = s + max(0.0, log(zi4 + eps) - lz0);
    let zi5 = textureSampleLevel(tDepthLin, samp, in.uv + o[5] * px * r, 0.0).r;
    s = s + max(0.0, log(zi5 + eps) - lz0);
    let zi6 = textureSampleLevel(tDepthLin, samp, in.uv + o[6] * px * r, 0.0).r;
    s = s + max(0.0, log(zi6 + eps) - lz0);
    let zi7 = textureSampleLevel(tDepthLin, samp, in.uv + o[7] * px * r, 0.0).r;
    s = s + max(0.0, log(zi7 + eps) - lz0);
    
    let shade = exp(-UBO.strength * s);
    return vec4<f32>(col * shade, 1.0);
}
