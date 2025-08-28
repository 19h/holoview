struct Uniforms {
    mode: u32,
    _pad: vec3<u32>,
};

@group(0) @binding(0) var tSrc      : texture_2d<f32>;
@group(0) @binding(1) var tDepthLin : texture_2d<f32>;
@group(0) @binding(2) var samp      : sampler;
@group(0) @binding(3) var<uniform>  U : Uniforms;

struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> VSOut {
    var o: VSOut;
    o.clip = vec4<f32>(pos, 0.0, 1.0);
    // Correctly map clip space to UV space for WebGPU/Vulkan/Metal.
    // Clip space Y is -1 (bottom) to +1 (top).
    // UV space Y is  0 (top)    to  1 (bottom).
    // The Y-coordinate must be flipped.
    o.uv = vec2<f32>(0.5 * (pos.x + 1.0), 0.5 * (-pos.y + 1.0));
    return o;
}

fn class_color(label: u32) -> vec3<f32> {
    switch (label) {
        case 1u: { return vec3<f32>(1.00, 0.82, 0.40); }
        case 2u: { return vec3<f32>(1.00, 0.92, 0.20); }
        case 3u: { return vec3<f32>(0.80, 0.80, 0.80); }
        case 4u: { return vec3<f32>(0.70, 0.70, 0.70); }
        case 5u: { return vec3<f32>(0.20, 0.55, 0.95); }
        case 6u: { return vec3<f32>(0.40, 0.85, 0.40); }
        case 7u: { return vec3<f32>(0.17, 0.55, 0.30); }
        case 8u: { return vec3<f32>(0.85, 0.30, 0.55); }
        case 9u: { return vec3<f32>(0.55, 0.55, 0.95); }
        default: { return vec3<f32>(0.85, 0.85, 0.85); }
    }
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let uv = clamp(in.uv, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
    let c  = textureSampleLevel(tSrc, samp, uv, 0.0);
    let dl = textureSampleLevel(tDepthLin, samp, uv, 0.0);

    if (U.mode == 1u) {
        // Depth: near = white
        let z = 1.0 - clamp(dl.r, 0.0, 1.0);
        return vec4<f32>(z, z, z, 1.0);
    } else if (U.mode == 2u) {
        // Labels (class colors), background dimmed
        let z = dl.r;
        if (z >= 0.9999) {
            return vec4<f32>(0.05, 0.05, 0.05, 1.0);
        }
        let lbl = u32(round(clamp(dl.g, 0.0, 1.0) * 255.0));
        let color = class_color(lbl);
        return vec4<f32>(color.r, color.g, color.b, 1.0);
    } else if (U.mode == 3u) {
        // Tag: points=white (a>=0.5), grid/background=black
        let v = step(0.5, dl.a);
        return vec4<f32>(v, v, v, 1.0);
    }

    // Passthrough
    return vec4<f32>(c.r, c.g, c.b, 1.0);
}
