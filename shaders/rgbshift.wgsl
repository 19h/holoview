// RGB shift / chromatic aberration shader
// Creates a holographic effect by shifting color channels.
// NOTE: Uses textureSampleLevel(..., 0.0) per NonFiltering sampler binding.

struct Uniforms {
    inv_size: vec2<f32>,
    amount:   f32,
    angle:    f32,
};

@group(0) @binding(0) var tSrc:      texture_2d<f32>;
@group(0) @binding(1) var tDepthLin: texture_2d<f32>;
@group(0) @binding(2) var samp:      sampler;
@group(0) @binding(3) var<uniform> UBO: Uniforms;

struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0)      uv:   vec2<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> VSOut {
    var out: VSOut;
    out.clip = vec4<f32>(pos, 0.0, 1.0);
    // Correctly map clip space to UV space for WebGPU/Vulkan/Metal.
    // Clip space Y is -1 (bottom) to +1 (top).
    // UV space Y is  0 (top)    to  1 (bottom).
    // The Y-coordinate must be flipped.
    out.uv = vec2<f32>(0.5 * (pos.x + 1.0), 0.5 * (-pos.y + 1.0));
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    // Clamp FS-triangle UVs to [0,1] and sample with the NonFiltering sampler
    let uv = clamp(in.uv, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));

    // Discriminator tag stored in the depth-linear buffer's alpha channel.
    // tag < 0.5 indicates a grid fragment.
    let uv_c = in.uv;
    let uv_d = uv_c;
    let dl   = textureSampleLevel(tDepthLin, samp, uv_d, 0.0);

    // Skip grid (tag<0.5) and true background (z≈1).
    if (dl.a < 0.5 || dl.r >= 0.9999) {
        return textureSampleLevel(tSrc, samp, uv_c, 0.0);
    }

    // Point-cloud fragments: apply RGB shift.
    let offset: vec2<f32> = UBO.amount * vec2<f32>(cos(UBO.angle), sin(UBO.angle));

    let shifted_r  = textureSampleLevel(tSrc, samp, uv_c + offset, 0.0);
    let shifted_gb = textureSampleLevel(tSrc, samp, uv_c - offset, 0.0);
    let a_src      = textureSampleLevel(tSrc, samp, uv_c, 0.0).a;

    return vec4<f32>(shifted_r.r, shifted_gb.g, shifted_gb.b, a_src);
}
