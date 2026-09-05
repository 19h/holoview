// Conservative screen-space surface reconstruction. Only small interior gaps
// supported from opposing directions are filled; geometry bytes are unchanged.
@group(0) @binding(0) var color_tex: texture_2d<f32>;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
struct VertexOut { @builtin(position) position: vec4<f32> };
@vertex fn vs_main(@builtin(vertex_index) id: u32) -> VertexOut {
    let x = f32((id << 1u) & 2u);
    let y = f32(id & 2u);
    var o: VertexOut; o.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0); return o;
}
struct Out { @location(0) color: vec4<f32>, @location(1) depth: vec4<f32> };
fn sample_depth(p: vec2<i32>) -> vec4<f32> {
    let size = vec2<i32>(textureDimensions(depth_tex));
    if any(p < vec2<i32>(0)) || any(p >= size) { return vec4<f32>(0.0); }
    return textureLoad(depth_tex, p, 0);
}
struct Support { depth: vec4<f32>, distance: f32, pixel: vec2<i32> };
fn support(p: vec2<i32>, direction: vec2<i32>) -> Support {
    var found: Support;
    found.depth = vec4<f32>(0.0); found.distance = 0.0; found.pixel = p;
    for (var distance = 1; distance <= 8; distance *= 2) {
        let q = p + direction * distance;
        let d = sample_depth(q);
        if d.a > 0.5 && d.r > 0.0 {
            found.depth = d; found.distance = f32(distance); found.pixel = q; return found;
        }
    }
    return found;
}
@fragment fn fs_main(@builtin(position) screen: vec4<f32>) -> Out {
    let p = vec2<i32>(screen.xy);
    let original = sample_depth(p);
    var out: Out;
    out.color = textureLoad(color_tex, p, 0); out.depth = original;
    if original.a > 0.5 && original.r > 0.0 {
        // Bilateral shading depth: smooth subpixel splat steps without blending
        // across a foreground/background discontinuity or semantic boundary.
        var sum = original.r * 4.0; var weight = 4.0;
        let limit = max(0.02, original.r * 0.002);
        for (var axis = 0; axis < 4; axis++) {
            var direction = vec2<i32>(1, 0);
            if axis == 1 { direction = vec2<i32>(-1, 0); }
            if axis == 2 { direction = vec2<i32>(0, 1); }
            if axis == 3 { direction = vec2<i32>(0, -1); }
            let d = sample_depth(p + direction);
            if d.a > 0.5 && abs(d.r - original.r) < limit && d.g == original.g {
                sum += d.r; weight += 1.0;
            }
        }
        out.depth.r = sum / weight;
        return out;
    }
    var pairs = 0;
    var inverse_depth = 0.0;
    var reference_depth = 0.0;
    var best_span = 100.0;
    var label = 0.0;
    var best_pixel = p;
    for (var axis = 0; axis < 4; axis++) {
        var direction = vec2<i32>(1, 0);
        if axis == 1 { direction = vec2<i32>(0, 1); }
        if axis == 2 { direction = vec2<i32>(1, 1); }
        if axis == 3 { direction = vec2<i32>(1, -1); }
        let a = support(p, direction);
        let b = support(p, -direction);
        if a.distance > 0.0 && b.distance > 0.0 {
            let za = a.depth.r; let zb = b.depth.r;
            if abs(za - zb) <= max(0.5, min(za, zb) * 0.08) {
                let t = a.distance / (a.distance + b.distance);
                let zi = 1.0 / mix(1.0 / za, 1.0 / zb, t);
                if pairs == 0 || abs(zi - reference_depth) < max(0.5, reference_depth * 0.04) {
                    pairs += 1; inverse_depth += 1.0 / zi; reference_depth = zi;
                    if a.distance + b.distance < best_span {
                        best_span = a.distance + b.distance;
                        label = select(b.depth.g, a.depth.g, a.distance < b.distance);
                        best_pixel = select(b.pixel, a.pixel, a.distance < b.distance);
                    }
                }
            }
        }
    }
    if pairs >= 2 {
        out.color = textureLoad(color_tex, best_pixel, 0); out.color.a = 1.0;
        out.depth = vec4<f32>(f32(pairs) / inverse_depth, label, 0.0, 1.0);
    }
    return out;
}
