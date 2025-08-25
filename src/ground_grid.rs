// src/ground_grid.rs
use wgpu::util::DeviceExt;
use glam::{Mat4, Vec3};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridUniforms {
    pub view: Mat4,
    pub proj: Mat4,

    pub decode_min: Vec3,
    pub _pad0: f32,
    pub decode_max: Vec3,
    pub _pad1: f32,

    // GEOT (CRS:84) bbox
    pub geot_lon_min: f32,
    pub geot_lat_min: f32,
    pub geot_lon_span: f32,
    pub geot_lat_span: f32,

    // rendering controls
    pub extent_mul: f32,  // how much larger than the tile AABB to draw (e.g. 12.0)
    pub z_offset:  f32,   // push plane a little below decode_min.z (in world units)
    pub opacity:   f32,   // 0..1
    pub enabled:   u32,   // 1 = on, 0 = off

    pub color_minor: Vec3, pub _pad2: f32,
    pub color_major: Vec3, pub _pad3: f32,
}

pub struct GroundGrid {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
    pub quad_vb: wgpu::Buffer,
    pub uniforms: GridUniforms,
    pub bgl: wgpu::BindGroupLayout,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct QuadVertex { corner: [f32; 2] }

impl QuadVertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<QuadVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
        }
    }
}

impl GroundGrid {
    pub fn new(device: &wgpu::Device, scene_format: wgpu::TextureFormat, depthlin_format: wgpu::TextureFormat) -> Self {
        let uniforms = GridUniforms {
            view: Mat4::IDENTITY,
            proj: Mat4::IDENTITY,
            decode_min: Vec3::ZERO, _pad0: 0.0,
            decode_max: Vec3::ONE,  _pad1: 0.0,
            geot_lon_min: 0.0, geot_lat_min: 0.0,
            geot_lon_span: 1.0, geot_lat_span: 1.0,
            extent_mul: 12.0, z_offset: 0.02, opacity: 0.70, enabled: 0,
            color_minor: Vec3::new(0.90, 0.15, 0.15), _pad2: 0.0,
            color_major: Vec3::new(1.00, 0.20, 0.20), _pad3: 0.0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Grid BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Grid BG"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // big quad (two triangles) in local [-1,1]^2, expanded in VS by extent_mul
        let corners: [QuadVertex; 6] = [
            QuadVertex { corner: [-1.0, -1.0] }, QuadVertex { corner: [ 1.0, -1.0] }, QuadVertex { corner: [ 1.0,  1.0] },
            QuadVertex { corner: [-1.0, -1.0] }, QuadVertex { corner: [ 1.0,  1.0] }, QuadVertex { corner: [-1.0,  1.0] },
        ];
        let quad_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Quad"),
            contents: bytemuck::cast_slice(&corners),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Grid WGSL"),
            source: wgpu::ShaderSource::Wgsl(GRID_WGSL.into()),
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Grid PL"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Ground Grid Pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[QuadVertex::layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[
                    // scene color
                    Some(wgpu::ColorTargetState {
                        format: scene_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // linear depth buffer
                    Some(wgpu::ColorTargetState {
                        format: depthlin_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,               // << do NOT write depth (points will)
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 4, // Must match SAMPLE_COUNT in main.rs
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        Self { pipeline, bind_group, uniform_buffer, quad_vb, uniforms, bgl }
    }

    pub fn update_uniforms(&mut self, queue: &wgpu::Queue, u: GridUniforms) {
        self.uniforms = u;
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms));
    }
}

pub const GRID_WGSL: &str = r#"
struct U {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    decode_min: vec3<f32>, _pad0: f32,
    decode_max: vec3<f32>, _pad1: f32,

    geot_lon_min: f32,
    geot_lat_min: f32,
    geot_lon_span: f32,
    geot_lat_span: f32,

    extent_mul: f32,
    z_offset:  f32,
    opacity:   f32,
    enabled:   u32,

    color_minor: vec3<f32>, _pad2: f32,
    color_major: vec3<f32>, _pad3: f32,
};
@group(0) @binding(0) var<uniform> UBO: U;

struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_xy: vec2<f32>,
}

@vertex
fn vs_main(@location(0) corner: vec2<f32>) -> VSOut {
    // plane anchored to decode-XY, slightly below the data to avoid z-fighting
    let center = 0.5 * (UBO.decode_min.xy + UBO.decode_max.xy);
    let half   = 0.5 * (UBO.decode_max.xy - UBO.decode_min.xy) * max(1.0, UBO.extent_mul);
    let world_xy = center + corner * half;
    let world_z  = UBO.decode_min.z - UBO.z_offset;

    let world = vec4<f32>(world_xy, world_z, 1.0);
    var o: VSOut;
    o.clip = UBO.proj * (UBO.view * world);
    o.world_xy = world_xy;
    return o;
}

// === helpers ================================================================

// Returns (s_lo, s_hi, t) where t∈[0,1] blends between lower and higher step in log-space.
fn pick_steps_blended(goal: f32) -> vec3<f32> {
    // Same "nice" set as before - defined as individual constants to avoid dynamic indexing
    let s0  = 5.0/3600.0;
    let s1  = 10.0/3600.0;
    let s2  = 30.0/3600.0;
    let s3  = 1.0/60.0;
    let s4  = 2.0/60.0;
    let s5  = 5.0/60.0;
    let s6  = 10.0/60.0;
    let s7  = 30.0/60.0;
    let s8  = 1.0;
    let s9  = 2.0;
    let s10 = 5.0;
    let s11 = 10.0;

    // Clamp goal into covered range
    let g = clamp(goal, s0, s11);

    // Find enclosing interval [s_lo, s_hi] using unrolled comparisons
    var s_lo = s0;
    var s_hi = s1;
    
    if (g > s1) {
        s_lo = s1;
        s_hi = s2;
    }
    if (g > s2) {
        s_lo = s2;
        s_hi = s3;
    }
    if (g > s3) {
        s_lo = s3;
        s_hi = s4;
    }
    if (g > s4) {
        s_lo = s4;
        s_hi = s5;
    }
    if (g > s5) {
        s_lo = s5;
        s_hi = s6;
    }
    if (g > s6) {
        s_lo = s6;
        s_hi = s7;
    }
    if (g > s7) {
        s_lo = s7;
        s_hi = s8;
    }
    if (g > s8) {
        s_lo = s8;
        s_hi = s9;
    }
    if (g > s9) {
        s_lo = s9;
        s_hi = s10;
    }
    if (g > s10) {
        s_lo = s10;
        s_hi = s11;
    }

    // Log-space blend weight (scale-invariant)
    let lg  = log(g);
    let llo = log(s_lo);
    let lhi = log(s_hi);
    let t   = clamp((lg - llo) / max(1e-6, lhi - llo), 0.0, 1.0);

    return vec3<f32>(s_lo, s_hi, t);
}

// Pattern evaluation at a given angular step.
// Returns (minor, major) in [0,1], as in the original shader.
fn grid_patterns(lon: f32, lat: f32, step_minor: f32) -> vec2<f32> {
    let step_major = step_minor * 6.0;
    let m_lon = line_mask(lon, step_minor);
    let m_lat = line_mask(lat, step_minor);
    let M_lon = line_mask(lon, step_major);
    let M_lat = line_mask(lat, step_major);
    let minor = clamp(max(m_lon, m_lat), 0.0, 1.0);
    let major = clamp(max(M_lon, M_lat), 0.0, 1.0);
    return vec2<f32>(minor, major);
}

// degrees per pixel -> pick the SMALLEST "nice" step that is >= the desired spacing (~100px)
// (renamed 'goal' to avoid WGSL reserved word 'target')
fn pick_step(degpp: f32) -> f32 {
    // desired angular spacing for ~100 pixels, but never smaller than 5 arcsec
    let goal = max(degpp * 100.0, 5.0/3600.0);

    // evaluate thresholds in ascending order
    if (goal <= 5.0/3600.0)  { return 5.0/3600.0;  } // 5"
    if (goal <= 10.0/3600.0) { return 10.0/3600.0; } // 10"
    if (goal <= 30.0/3600.0) { return 30.0/3600.0; } // 30"
    if (goal <= 1.0/60.0)    { return 1.0/60.0;    } // 1'
    if (goal <= 2.0/60.0)    { return 2.0/60.0;    } // 2'
    if (goal <= 5.0/60.0)    { return 5.0/60.0;    } // 5'
    if (goal <= 10.0/60.0)   { return 10.0/60.0;   } // 10'
    if (goal <= 30.0/60.0)   { return 30.0/60.0;   } // 30'
    if (goal <= 1.0)         { return 1.0;         } // 1°
    if (goal <= 2.0)         { return 2.0;         } // 2°
    if (goal <= 5.0)         { return 5.0;         } // 5°
    return 10.0;                                     // 10°
}

// AA’d line mask at INTEGER multiples of 'step'
fn line_mask(coord: f32, step: f32) -> f32 {
    let u   = coord / step;
    let aaw = max(fwidth(u) * 1.5, 1e-5);
    let f   = fract(u);
    let d   = min(f, 1.0 - f);   // distance to nearest integer
    return 1.0 - smoothstep(0.0, aaw, d);
}

struct FSOut {
    @location(0) color: vec4<f32>,
    @location(1) depthlin: vec4<f32>,
}

@fragment
fn fs_main(in: VSOut) -> FSOut {
    // If disabled or missing GEOT, output transparent (no lines).
    if (UBO.enabled == 0u || UBO.geot_lon_span <= 0.0 || UBO.geot_lat_span <= 0.0) {
        var o: FSOut;
        o.color    = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        // IMPORTANT: set alpha=0 in depthlin to tag as "overlay" (see CRT pass).
        o.depthlin = vec4<f32>(1.0, 1.0, 1.0, 0.0);
        return o;
    }

    // decode-XY -> lon/lat by linearizing the GEOT bbox
    let lon = UBO.geot_lon_min +
              (in.world_xy.x - UBO.decode_min.x) / max(1e-9, (UBO.decode_max.x - UBO.decode_min.x)) * UBO.geot_lon_span;
    let lat = UBO.geot_lat_min +
              (in.world_xy.y - UBO.decode_min.y) / max(1e-9, (UBO.decode_max.y - UBO.decode_min.y)) * UBO.geot_lat_span;

    // Original goal (≥ 5 arcsec; ~100 px target)
    let goal_lon = max(fwidth(lon) * 100.0, 5.0/3600.0);
    let goal_lat = max(fwidth(lat) * 100.0, 5.0/3600.0);
    let goal     = min(goal_lon, goal_lat);

    // Pick enclosing steps and blend weight t
    let ps   = pick_steps_blended(goal);
    let s_lo = ps.x;
    let s_hi = ps.y;
    let t    = ps.z;

    // Evaluate patterns at both steps and blend
    let p_lo = grid_patterns(lon, lat, s_lo);
    let p_hi = grid_patterns(lon, lat, s_hi);
    let minor = mix(p_lo.x, p_hi.x, t);
    let major = mix(p_lo.y, p_hi.y, t);

    // Major dominates as before
    let c = UBO.color_minor * (minor * 0.65) + UBO.color_major * major;
    let a = UBO.opacity * max(major, minor);

    var o: FSOut;
    o.color    = vec4<f32>(c, a);
    o.depthlin = vec4<f32>(1.0, 1.0, 1.0, 0.0); // keep overlay tag

    return o;
}
"#;
