// RGB shift / chromatic aberration shader
// Creates a holographic effect by shifting color channels
// NOTE: uses textureSampleLevel(..., 0.0) per NonFiltering sampler binding.

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
    let tag = textureSampleLevel(tDepthLin, samp, in.uv, 0.0).a;

    if (tag < 0.5) {
        // This is the grid. Bypass the shift effect to prevent moiré.
        return textureSampleLevel(tSrc, samp, in.uv, 0.0);
    }

    // This is the point cloud. Apply the shift effect.
    let ofs = UBO.amount * vec2<f32>(cos(UBO.angle), sin(UBO.angle));
    let cr  = textureSampleLevel(tSrc, samp, in.uv + ofs, 0.0);
    let cgb = textureSampleLevel(tSrc, samp, in.uv - ofs, 0.0);

    return vec4<f32>(cr.r, cgb.g, cgb.b, 1.0);
}
