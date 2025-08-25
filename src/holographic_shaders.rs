// src/holographic_shaders.rs
use wgpu::util::DeviceExt;
use glam::{Mat4, Vec3};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HoloUniforms {
    pub view: Mat4,
    pub proj: Mat4,
    pub viewport: [f32; 2],   // (width, height)
    pub base_size_px: f32,    // ~1.0
    pub size_atten: f32,      // ~1.25
    pub time: f32,
    pub near: f32,
    pub far: f32,
    pub _pad: f32,
    pub decode_min: Vec3, pub _pad0: f32,
    pub decode_max: Vec3, pub _pad1: f32,
    pub cyan:       Vec3, pub _pad2: f32,
    pub red:        Vec3, pub _pad3: f32,
}
impl Default for HoloUniforms {
    fn default() -> Self {
        Self {
            view: Mat4::IDENTITY,
            proj: Mat4::IDENTITY,
            viewport: [1280.0, 720.0],
            base_size_px: 4.0,
            size_atten: 1.0,
            time: 0.0,
            near: 0.5,
            far: 8000.0,
            _pad: 0.0,
            decode_min: Vec3::ZERO,
            decode_max: Vec3::ONE,
            cyan: Vec3::new(1.1, 0.5, 0.2),     // uniform warm orange
            red:  Vec3::new(0.5, 0.5, 0.2),      // very similar warm orange
            _pad0: 0.0, _pad1: 0.0, _pad2: 0.0, _pad3: 0.0,
        }
    }
}

pub struct HoloPipelines {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
    pub quad_vb: wgpu::Buffer,
    pub uniforms: HoloUniforms,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Instance {
    pub position: [f32; 3],
}

impl Instance {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Instance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x3,
            }],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct QuadVertex {
    // Local corners in [-1, 1]^2
    corner: [f32; 2],
}

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

impl HoloPipelines {
    pub fn new(
        device: &wgpu::Device,
        scene_format: wgpu::TextureFormat,
        depthlin_format: wgpu::TextureFormat,
    ) -> Self {
        // ---------- Uniform buffer ----------
        let uniforms = HoloUniforms::default();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Holo Uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ---------- Bind group layout ----------
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Holo BGL"),
            entries: &[
                // Uniform buffer object
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // SMC1 label texture (R8Unorm → Float sample type)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                // Nearest sampler for labels
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        // ---------- Dummy texture (1×1 R8Unorm) ----------
        let sem_dummy = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SMC1 Dummy"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let sem_dummy_view = sem_dummy.create_view(&wgpu::TextureViewDescriptor::default());

        let sem_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SMC1 Nearest"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // ---------- Bind group ----------
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Holo BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&sem_dummy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sem_sampler),
                },
            ],
        });

        // ---------- Instanced quad geometry ----------
        let corners = [
            QuadVertex { corner: [-1.0, -1.0] },
            QuadVertex { corner: [ 1.0, -1.0] },
            QuadVertex { corner: [ 1.0,  1.0] },
            QuadVertex { corner: [-1.0, -1.0] },
            QuadVertex { corner: [ 1.0,  1.0] },
            QuadVertex { corner: [-1.0,  1.0] },
        ];
        let quad_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Point Quad VB"),
            contents: bytemuck::cast_slice(&corners),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // ---------- Shader ----------
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Holo WGSL"),
            source: wgpu::ShaderSource::Wgsl(HOLO_WGSL.into()),
        });

        // ---------- Pipeline layout ----------
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Holo PL"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        // ---------- Render pipeline ----------
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Holo Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[QuadVertex::layout(), Instance::layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[
                    // Color target
                    Some(wgpu::ColorTargetState {
                        format: scene_format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // Linearized depth buffer (RGBA16F)
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
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 4, // Must match SAMPLE_COUNT in main.rs
                mask: !0,
                alpha_to_coverage_enabled: false, // Can be enabled for better sprite edges
            },
            multiview: None,
        });

        Self {
            pipeline,
            bind_group,
            uniform_buffer,
            quad_vb,
            uniforms,
        }
    }

    pub fn set_semantics_inputs(
        &mut self,
        device: &wgpu::Device,
        sem_view: &wgpu::TextureView,
        sem_sampler: &wgpu::Sampler,
    ) {
        // Rebuild the bind group with the same UBO but new texture & sampler
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Holo BG (with SMC1)"),
            layout: &device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Holo BGL (mirror)"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            }),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(sem_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sem_sampler),
                },
            ],
        });
    }

    pub fn update_uniforms(&mut self, queue: &wgpu::Queue, u: HoloUniforms) {
        self.uniforms = u;
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms));
    }
}
pub const HOLO_WGSL: &str = r#"
struct UBO {
    view: mat4x4<f32>, proj: mat4x4<f32>,
    viewport: vec2<f32>, base_size_px: f32, size_atten: f32,
    time: f32, near: f32, far: f32, _pad: f32,
    decode_min: vec3<f32>, _pad0: f32,
    decode_max: vec3<f32>, _pad1: f32,
    cyan: vec3<f32>, _pad2: f32,
    red:  vec3<f32>, _pad3: f32,
};
@group(0) @binding(0) var<uniform> U: UBO;

// --- SMC1 bindings ---
@group(0) @binding(1) var tSMC1: texture_2d<f32>;   // R8Unorm labels → [0,1]
@group(0) @binding(2) var sSMC1: sampler;           // nearest, clamp

struct VSIn { @location(0) corner: vec2<f32>, @location(1) center: vec3<f32> };
struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) hfac: f32,
    @location(2) center_ndc_depth: f32,
    @location(3) eye_depth: f32,
    @location(4) uv_sem: vec2<f32>,
}

@vertex
fn vs_main(in: VSIn) -> VSOut {
    let world = in.center;

    // Transform to view space
    let view_pos = U.view * vec4<f32>(world, 1.0);

    // Camera looks down -Z in view space.
    let behind   = view_pos.z >= 0.0;
    let too_close = -view_pos.z < U.near;

    // Cull points that are behind the camera or too close.
    if (behind || too_close) {
        var out: VSOut;
        // Push outside clip space – it will be trivially discarded.
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.local = in.corner;
        out.hfac = 0.0;
        out.center_ndc_depth = 1.0;
        out.eye_depth = U.far;
        out.uv_sem = vec2<f32>(0.0);
        return out;
    }

    // Sprite size attenuation (distance is -view_pos.z because camera looks -Z)
    let dist = -view_pos.z;
    let size_px = U.base_size_px * clamp(200.0 / (U.size_atten * dist), 0.6, 4.5);
    let half_px = 0.5 * size_px;
    let ndc_dx = (half_px * 2.0) / U.viewport.x;
    let ndc_dy = (half_px * 2.0) / U.viewport.y;

    let clip_center = U.proj * view_pos;

    // Extra safety: ensure w is positive before expanding.
    if (clip_center.w <= 0.0) {
        var out: VSOut;
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.local = in.corner;
        out.hfac = 0.0;
        out.center_ndc_depth = 1.0;
        out.eye_depth = dist;
        out.uv_sem = vec2<f32>(0.0);
        return out;
    }

    var clip = clip_center;
    let offset = vec2<f32>(in.corner.x * ndc_dx, in.corner.y * ndc_dy) * clip_center.w;
    clip.x += offset.x;
    clip.y += offset.y;

    let hr = max(1e-6, U.decode_max.z - U.decode_min.z);
    let hfac = clamp((world.z - U.decode_min.z) / hr, 0.0, 1.0);

    // --- decode-XY → [0,1]^2 UV for SMC1 ---
    let span = max(vec2<f32>(1e-6, 1e-6), (U.decode_max.xy - U.decode_min.xy));
    let uv_sem = clamp((world.xy - U.decode_min.xy) / span, vec2<f32>(0.0), vec2<f32>(1.0));

    var out: VSOut;
    out.clip = clip;
    out.local = in.corner;
    out.hfac = hfac;
    out.center_ndc_depth = clip_center.z / clip_center.w;
    out.eye_depth = dist;
    out.uv_sem = uv_sem;
    return out;
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898,78.233))) * 43758.5453);
}

// --- NEW: fixed palette (10 classes) ---
// 0:Unknown 1:Building 2:RoadMajor 3:RoadMinor 4:Path 5:Water 6:Park 7:Woodland 8:Railway 9:Parking
const PAL: array<vec3<f32>, 10> = array<vec3<f32>, 10>(
    vec3<f32>(0.12, 0.12, 0.12), // Unknown
    vec3<f32>(1.00, 0.80, 0.25), // Building
    vec3<f32>(1.00, 1.00, 1.00), // RoadMajor
    vec3<f32>(0.75, 0.75, 0.75), // RoadMinor
    vec3<f32>(0.85, 0.65, 0.45), // Path
    vec3<f32>(0.20, 0.50, 1.00), // Water
    vec3<f32>(0.12, 0.85, 0.35), // Park
    vec3<f32>(0.00, 0.55, 0.00), // Woodland
    vec3<f32>(0.65, 0.25, 0.85), // Railway
    vec3<f32>(0.78, 0.84, 0.92)  // Parking
);

struct FSOut {
    @location(0) color: vec4<f32>,
    @location(1) depthlin: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(in: VSOut) -> FSOut {
    // Anti-aliased circular disc
    let r2 = dot(in.local, in.local);
    let r = sqrt(r2);
    let aaw = fwidth(r); // screen-space derivative of radius
    let circle_alpha = 1.0 - smoothstep(1.0 - aaw, 1.0, r);
    if (circle_alpha < 0.001) { discard; }

    // base holographic look
    let falloff = pow(1.0 - r2, 1.6);
    let n = hash(in.clip.xy + vec2<f32>(U.time, U.time));
    let flicker = 0.92 + 0.16 * (n - 0.5);
    let holo = mix(U.red, U.cyan, in.hfac) * falloff * flicker * 1.25;

    // --- sample SMC1 label (R8Unorm → [0,1]) and use per-class color when label>0
    let lblNorm = textureSampleLevel(tSMC1, sSMC1, in.uv_sem, 0.0).r;
    let lblU = u32(round(clamp(lblNorm, 0.0, 1.0) * 255.0));
    let lbl = min(lblU, 9u); // our palette has 10 entries

    var color = holo;
    if (lblU > 0u) {
        // Slight lighting via falloff to keep dots readable
        var palColor = PAL[0]; // default to Unknown
        switch (lbl) {
            case 0u: { palColor = PAL[0]; } // Unknown
            case 1u: { palColor = PAL[1]; } // Building
            case 2u: { palColor = PAL[2]; } // RoadMajor
            case 3u: { palColor = PAL[3]; } // RoadMinor
            case 4u: { palColor = PAL[4]; } // Path
            case 5u: { palColor = PAL[5]; } // Water
            case 6u: { palColor = PAL[6]; } // Park
            case 7u: { palColor = PAL[7]; } // Woodland
            case 8u: { palColor = PAL[8]; } // Railway
            case 9u: { palColor = PAL[9]; } // Parking
            default: { palColor = PAL[0]; } // fallback to Unknown
        }
        color = palColor * (0.55 + 0.45 * falloff);
    }

    let alpha = falloff * 0.95 * circle_alpha;
    let lin = clamp((in.eye_depth - U.near) / max(1e-6, U.far - U.near), 0.0, 1.0);

    var o: FSOut;
    o.color    = vec4<f32>(color, alpha);
    o.depthlin = vec4<f32>(lin, lin, lin, 1.0);
    o.depth    = in.center_ndc_depth;
    return o;
}
"#;
