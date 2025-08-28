// Semantic color grading post pass.
// Expects:
//   tSrc.rgb  = base color
//   tDepthLin = (r = linearized depth proxy [0..1], g = semantic label / 255, a = tag)
//
// For background pixels (z >= ~1), output original color.
// Otherwise mix towards a class color by `amount`.

struct Uniforms {
    amount: f32,
};

@group(0) @binding(0) var tSrc: texture_2d<f32>;
@group(0) @binding(1) var tDepthLin: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> UBO: Uniforms;

struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0)         uv:   vec2<f32>,
};

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
    switch label {
        case 1u: { return vec3<f32>(1.00, 0.82, 0.40); } // Building (amber)
        case 2u: { return vec3<f32>(1.00, 0.92, 0.20); } // RoadMajor (yellow)
        case 3u: { return vec3<f32>(0.80, 0.80, 0.80); } // RoadMinor (light gray)
        case 4u: { return vec3<f32>(0.70, 0.70, 0.70); } // Path
        case 5u: { return vec3<f32>(0.20, 0.55, 0.95); } // Water (blue)
        case 6u: { return vec3<f32>(0.40, 0.85, 0.40); } // Park
        case 7u: { return vec3<f32>(0.17, 0.55, 0.30); } // Woodland
        case 8u: { return vec3<f32>(0.85, 0.30, 0.55); } // Railway
        case 9u: { return vec3<f32>(0.55, 0.55, 0.95); } // Parking
        default: { return vec3<f32>(0.85, 0.85, 0.85); } // Unknown -> neutral
    }
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    // Clamp FS-triangle UVs to [0,1] and sample with the NonFiltering sampler
    let uv = clamp(in.uv, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));

    // Use LOD 0.0 for non-filtering sampler compliance.
    let uv_c = in.uv;
    let uv_d = uv_c;
    let c  = textureSampleLevel(tSrc,      samp, uv_c, 0.0);
    let dl = textureSampleLevel(tDepthLin, samp, uv_d, 0.0);

    // Background: z ≈ 1 retains the original color.
    let z = dl.r;
    if (z >= 0.9999) {
        // Preserve incoming coverage alpha for later passes if needed.
        return vec4<f32>(c.rgb, c.a);
    }

    // Semantic label is stored in the green channel.
    let lbl = u32(round(clamp(dl.g, 0.0, 1.0) * 255.0));
    if (lbl == 0u) {
        // No valid label – pass through original color.
        return vec4<f32>(c.rgb, c.a);
    }

    let target_color = class_color(lbl);
    let amt          = clamp(UBO.amount, 0.0, 1.0);
    let out_rgb      = mix(c.rgb, target_color, amt);

    return vec4<f32>(out_rgb, c.a);
}
