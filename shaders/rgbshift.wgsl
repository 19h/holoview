// RGB shift / chromatic aberration shader
// Creates a holographic effect by shifting color channels

struct Uniforms {
    inv_size: vec2<f32>,
    amount: f32,
    angle: f32,
}

@group(0) @binding(0) var tSrc: texture_2d<f32>;
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
    // Read the discriminator tag from the depth-linear buffer's alpha channel.
    // tag < 0.5 indicates a grid fragment.
    let tag = textureSample(tDepthLin, samp, in.uv).a;
    
    if (tag < 0.5) {
        // This is the grid. Bypass the shift effect to prevent moiré.
        return textureSample(tSrc, samp, in.uv);
    }

    // This is the point cloud. Apply the shift effect.
    let ofs = UBO.amount * vec2<f32>(cos(UBO.angle), sin(UBO.angle));
    let cr = textureSample(tSrc, samp, in.uv + ofs);
    let cgb = textureSample(tSrc, samp, in.uv - ofs);
    
    return vec4<f32>(cr.r, cgb.g, cgb.b, 1.0);
}
