use std::time::Instant;
use wgpu::util::DeviceExt;

/// Intermediate texture format
const INTERMEDIATE_FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Full-screen triangle vertices
const FS_TRI: [[f32; 2]; 3] = [
    [-1.0, -1.0],
    [3.0, -1.0],
    [-1.0, 3.0],
];

/// Ping‑pong textures for multi‑pass rendering
pub struct PingPong {
    pub ping: wgpu::TextureView,
    pub pong: wgpu::TextureView,
    size: wgpu::Extent3d,
    _tex_ping: wgpu::Texture,
    _tex_pong: wgpu::Texture,
}

impl PingPong {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        fn make_tex(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("PostStack PingPong"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: INTERMEDIATE_FMT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        }

        let tex_ping = make_tex(device, width, height);
        let tex_pong = make_tex(device, width, height);
        let ping = tex_ping.create_view(&wgpu::TextureViewDescriptor::default());
        let pong = tex_pong.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            ping,
            pong,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            _tex_ping: tex_ping,
            _tex_pong: tex_pong,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.size.width == width && self.size.height == height {
            return;
        }
        *self = Self::new(device, width, height);
    }
}

// -------------------- Uniform Buffers --------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct UboEdl {
    inv_size: [f32; 2],
    strength: f32,
    radius_px: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct UboSem {
    amount: f32,
    _pad: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct UboRgb {
    inv_size: [f32; 2],
    amount: f32,
    angle: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct UboCrt {
    inv_size: [f32; 2],
    time: f32,
    intensity: f32,
    vignette: f32,
    _pad: f32,
}

// -------------------- Pass Types --------------------

struct EdlPass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    ubo: wgpu::Buffer,
    fs_vbo: wgpu::Buffer,
}

struct SemPost {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    ubo: wgpu::Buffer,
    fs_vbo: wgpu::Buffer,
}

struct RgbShiftPass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    ubo: wgpu::Buffer,
    fs_vbo: wgpu::Buffer,
}

struct CrtPass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    ubo: wgpu::Buffer,
    fs_vbo: wgpu::Buffer,
}

// -------------------- Post Parameters & Stack --------------------

#[derive(Clone, Copy, Debug)]
pub struct PostParams {
    pub edl_strength: f32,
    pub edl_radius_px: f32,
    pub sem_amount: f32,
    pub rgb_amount: f32,
    pub rgb_angle: f32,
    pub crt_intensity: f32,
    pub crt_vignette: f32,
}

impl Default for PostParams {
    fn default() -> Self {
        Self {
            edl_strength: 1.4,
            edl_radius_px: 1.0,
            sem_amount: 0.40,
            rgb_amount: 0.0025,
            rgb_angle: 0.9,
            crt_intensity: 0.25,
            crt_vignette: 0.8,
        }
    }
}

pub struct PostStack {
    pingpong: PingPong,
    edl: EdlPass,
    sem: SemPost,
    rgb: RgbShiftPass,
    crt: CrtPass,
    pub params: PostParams,
    start: Instant,
}

impl PostStack {
    pub fn new(
        device: &wgpu::Device,
        out_fmt: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let pingpong = PingPong::new(device, width, height);
        let edl = EdlPass::new(device, INTERMEDIATE_FMT);
        let sem = SemPost::new(device, INTERMEDIATE_FMT);
        let rgb = RgbShiftPass::new(device, INTERMEDIATE_FMT);
        let crt = CrtPass::new(device, out_fmt);
        Self {
            pingpong,
            edl,
            sem,
            rgb,
            crt,
            params: PostParams::default(),
            start: Instant::now(),
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.pingpong.resize(device, width, height);
    }

    /// Run the post‑processing chain: EDL → Semantic → RGB shift → CRT
    pub fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        swapchain_dst: &wgpu::TextureView,
        scene_color_src: &wgpu::TextureView,
        depthlin: &wgpu::TextureView,
    ) {
        let w = self.pingpong.size.width.max(1) as f32;
        let h = self.pingpong.size.height.max(1) as f32;
        let inv_size = [1.0 / w, 1.0 / h];
        let t = self.start.elapsed().as_secs_f32();

        // 1️⃣ EDL
        self.edl.draw(
            device,
            queue,
            encoder,
            &self.pingpong.ping,
            scene_color_src,
            depthlin,
            inv_size,
            self.params.edl_strength,
            self.params.edl_radius_px,
        );

        // 2️⃣ Semantic grading
        self.sem.draw(
            device,
            queue,
            encoder,
            &self.pingpong.pong,
            &self.pingpong.ping,
            depthlin,
            self.params.sem_amount,
        );

        // 3️⃣ RGB shift
        self.rgb.draw(
            device,
            queue,
            encoder,
            &self.pingpong.ping,
            &self.pingpong.pong,
            depthlin,
            inv_size,
            self.params.rgb_amount,
            self.params.rgb_angle,
        );

        // 4️⃣ CRT effect
        self.crt.draw(
            device,
            queue,
            encoder,
            swapchain_dst,
            &self.pingpong.ping,
            depthlin,
            inv_size,
            t,
            self.params.crt_intensity,
            self.params.crt_vignette,
        );
    }
}

// -------------------- Pass Implementations --------------------

macro_rules! create_post_pass {
    ($name:ident, $ubo_type:ty, $shader:expr) => {
        impl $name {
            pub fn new(device: &wgpu::Device, out_fmt: wgpu::TextureFormat) -> Self {
                let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(concat!(stringify!($name), " Layout")),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(
                                    std::mem::size_of::<$ubo_type>() as u64,
                                ),
                            },
                            count: None,
                        },
                    ],
                });

                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some($shader),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!(concat!("../../../shaders/", $shader)).into(),
                    ),
                });

                let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(concat!(stringify!($name), " PipelineLayout")),
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                });

                let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(concat!(stringify!($name), " Pipeline")),
                    layout: Some(&pipe_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: "vs_main",
                        buffers: &[wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<[f32; 2]>() as u64,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[wgpu::VertexAttribute {
                                shader_location: 0,
                                offset: 0,
                                format: wgpu::VertexFormat::Float32x2,
                            }],
                        }],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: "fs_main",
                        targets: &[Some(wgpu::ColorTargetState {
                            format: out_fmt,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                });

                let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some(concat!(stringify!($name), " Sampler")),
                    mag_filter: wgpu::FilterMode::Nearest,
                    min_filter: wgpu::FilterMode::Nearest,
                    ..Default::default()
                });

                let ubo = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(concat!(stringify!($name), " UBO")),
                    size: std::mem::size_of::<$ubo_type>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                let fs_vbo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(concat!(stringify!($name), " FS VBO")),
                    contents: bytemuck::cast_slice(&FS_TRI),
                    usage: wgpu::BufferUsages::VERTEX,
                });

                Self {
                    pipeline,
                    layout,
                    sampler,
                    ubo,
                    fs_vbo,
                }
            }
        }
    };
}

create_post_pass!(EdlPass, UboEdl, "edl.wgsl");
create_post_pass!(SemPost, UboSem, "sem_post.wgsl");
create_post_pass!(RgbShiftPass, UboRgb, "rgbshift.wgsl");
create_post_pass!(CrtPass, UboCrt, "crt.wgsl");

fn execute_pass(
    pipeline: &wgpu::RenderPipeline,
    encoder: &mut wgpu::CommandEncoder,
    bind_group: &wgpu::BindGroup,
    fs_vbo: &wgpu::Buffer,
    dst: &wgpu::TextureView,
    label: &str,
) {
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: dst,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    rpass.set_pipeline(pipeline);
    rpass.set_bind_group(0, bind_group, &[]);
    rpass.set_vertex_buffer(0, fs_vbo.slice(..));
    rpass.draw(0..3, 0..1);
}

impl EdlPass {
    pub fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        dst: &wgpu::TextureView,
        t_color: &wgpu::TextureView,
        t_depthlin: &wgpu::TextureView,
        inv_size: [f32; 2],
        strength: f32,
        radius_px: f32,
    ) {
        queue.write_buffer(
            &self.ubo,
            0,
            bytemuck::bytes_of(&UboEdl {
                inv_size,
                strength,
                radius_px,
            }),
        );
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("EDL Bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(t_color),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(t_depthlin),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.ubo.as_entire_binding(),
                },
            ],
        });
        execute_pass(&self.pipeline, encoder, &bind, &self.fs_vbo, dst, "EDL Pass");
    }
}

impl SemPost {
    pub fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        dst: &wgpu::TextureView,
        t_src: &wgpu::TextureView,
        t_depthlin: &wgpu::TextureView,
        amount: f32,
    ) {
        queue.write_buffer(
            &self.ubo,
            0,
            bytemuck::bytes_of(&UboSem {
                amount,
                _pad: [0.0; 3],
            }),
        );
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SemPost Bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(t_src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(t_depthlin),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.ubo.as_entire_binding(),
                },
            ],
        });
        execute_pass(&self.pipeline, encoder, &bind, &self.fs_vbo, dst, "SemPost Pass");
    }
}

impl RgbShiftPass {
    pub fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        dst: &wgpu::TextureView,
        t_src: &wgpu::TextureView,
        t_depthlin: &wgpu::TextureView,
        inv_size: [f32; 2],
        amount: f32,
        angle: f32,
    ) {
        queue.write_buffer(
            &self.ubo,
            0,
            bytemuck::bytes_of(&UboRgb {
                inv_size,
                amount,
                angle,
            }),
        );
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RgbShift Bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(t_src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(t_depthlin),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.ubo.as_entire_binding(),
                },
            ],
        });
        execute_pass(&self.pipeline, encoder, &bind, &self.fs_vbo, dst, "RgbShift Pass");
    }
}

impl CrtPass {
    pub fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        dst: &wgpu::TextureView,
        t_src: &wgpu::TextureView,
        t_depthlin: &wgpu::TextureView,
        inv_size: [f32; 2],
        time: f32,
        intensity: f32,
        vignette: f32,
    ) {
        queue.write_buffer(
            &self.ubo,
            0,
            bytemuck::bytes_of(&UboCrt {
                inv_size,
                time,
                intensity,
                vignette,
                _pad: 0.0,
            }),
        );
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Crt Bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(t_src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(t_depthlin),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.ubo.as_entire_binding(),
                },
            ],
        });
        execute_pass(&self.pipeline, encoder, &bind, &self.fs_vbo, dst, "Crt Pass");
    }
}
