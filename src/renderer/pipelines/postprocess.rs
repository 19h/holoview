// src/renderer/pipelines/postprocess.rs
//! Contains all post-processing effect pipelines (EDL, RGB Shift, CRT).

use wgpu::util::DeviceExt;
use super::base::{PostPass, impl_post_pass};

// ----------------- EDL -----------------
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EdlUniforms {
    pub inv_size: [f32; 2],
    pub strength: f32,
    pub radius_px: f32,
}

impl Default for EdlUniforms {
    fn default() -> Self {
        Self {
            inv_size: [1.0 / 1280.0, 1.0 / 720.0],
            strength: 1.6,
            radius_px: 1.5,
        }
    }
}

pub struct EdlPass {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub bgl: wgpu::BindGroupLayout,
    pub uniform: EdlUniforms,
    pub uniform_buffer: wgpu::Buffer,
    pub fsq_vb: wgpu::Buffer,
}

impl_post_pass!(EdlPass);

impl EdlPass {
    pub fn new(device: &wgpu::Device, dst_format: wgpu::TextureFormat) -> Self {
        let uniform = EdlUniforms::default();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("EDL Uniforms"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("EDL BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let fsq = [[-1.0_f32, -3.0], [3.0, 1.0], [-1.0, 1.0]];
        let fsq_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FSQ VB"),
            contents: bytemuck::cast_slice(&fsq),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("EDL WGSL"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/edl.wgsl").into()),
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("EDL PL"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("EDL Pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x2,
                    }],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: dst_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let dummy = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("EDL Dummy"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: dst_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = dummy.create_view(&wgpu::TextureViewDescriptor::default());
        let samp = device.create_sampler(&wgpu::SamplerDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("EDL BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&samp),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            bind_group,
            bgl,
            uniform,
            uniform_buffer,
            fsq_vb,
        }
    }

    pub fn set_inputs(
        &mut self,
        device: &wgpu::Device,
        color_view: &wgpu::TextureView,
        depthlin_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) {
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("EDL BG (updated)"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depthlin_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
    }

    pub fn update_uniforms(&mut self, queue: &wgpu::Queue, u: EdlUniforms) {
        self.uniform = u;
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&u));
    }
}

// ----------------- RGB Shift -----------------
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RgbShiftUniforms {
    pub inv_size: [f32; 2],
    pub amount: f32,
    pub angle: f32,
}

impl Default for RgbShiftUniforms {
    fn default() -> Self {
        Self {
            inv_size: [1.0 / 1280.0, 1.0 / 720.0],
            amount: 0.0016,
            angle: 0.0,
        }
    }
}

pub struct RgbShiftPass {
    pub pipeline: wgpu::RenderPipeline,
    pub bgl: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub uniform: RgbShiftUniforms,
    pub uniform_buffer: wgpu::Buffer,
    pub fsq_vb: wgpu::Buffer,
}

impl_post_pass!(RgbShiftPass);

impl RgbShiftPass {
    pub fn new(device: &wgpu::Device, dst_format: wgpu::TextureFormat) -> Self {
        let uniform = RgbShiftUniforms::default();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RGBShift Uniforms"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RGBShift BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let fsq = [[-1.0_f32, -3.0], [3.0, 1.0], [-1.0, 1.0]];
        let fsq_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RGBShift FSQ"),
            contents: bytemuck::cast_slice(&fsq),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RGBShift WGSL"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/rgbshift.wgsl").into()),
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RGBShift PL"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("RGBShift Pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x2,
                    }],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: dst_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let dummy = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RGBShift Dummy"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: dst_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let view = dummy.create_view(&wgpu::TextureViewDescriptor::default());
        let samp = device.create_sampler(&wgpu::SamplerDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RGBShift BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&samp),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            bgl,
            bind_group,
            uniform,
            uniform_buffer,
            fsq_vb,
        }
    }

    pub fn set_input(
        &mut self,
        device: &wgpu::Device,
        src_view: &wgpu::TextureView,
        depthlin_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) {
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RGBShift BG (updated)"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depthlin_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
    }

    pub fn update_uniforms(&mut self, queue: &wgpu::Queue, u: RgbShiftUniforms) {
        self.uniform = u;
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&u));
    }
}

// ----------------- CRT -----------------
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CrtUniforms {
    pub inv_size: [f32; 2],
    pub time: f32,
    pub intensity: f32,
    pub vignette: f32,
    pub _padding: f32,
}

impl Default for CrtUniforms {
    fn default() -> Self {
        Self {
            inv_size: [1.0 / 1280.0, 1.0 / 720.0],
            time: 0.0,
            intensity: 0.08,
            vignette: 0.35,
            _padding: 0.0,
        }
    }
}

pub struct CrtPass {
    pub pipeline: wgpu::RenderPipeline,
    pub bgl: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub uniform: CrtUniforms,
    pub uniform_buffer: wgpu::Buffer,
    pub fsq_vb: wgpu::Buffer,
}

impl_post_pass!(CrtPass);

impl CrtPass {
    pub fn new(device: &wgpu::Device, dst_format: wgpu::TextureFormat) -> Self {
        let uniform = CrtUniforms::default();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CRT Uniforms"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CRT BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let fsq = [[-1.0_f32, -3.0], [3.0, 1.0], [-1.0, 1.0]];
        let fsq_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CRT FSQ"),
            contents: bytemuck::cast_slice(&fsq),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CRT WGSL"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/crt.wgsl").into()),
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("CRT PL"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("CRT Pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x2,
                    }],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: dst_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let dummy = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("CRT Dummy"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: dst_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let view = dummy.create_view(&wgpu::TextureViewDescriptor::default());
        let samp = device.create_sampler(&wgpu::SamplerDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CRT BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&samp),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            bgl,
            bind_group,
            uniform,
            uniform_buffer,
            fsq_vb,
        }
    }

    pub fn set_inputs(
        &mut self,
        device: &wgpu::Device,
        color_view: &wgpu::TextureView,
        depthlin_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) {
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CRT BG (updated)"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depthlin_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
    }

    pub fn update_uniforms(&mut self, queue: &wgpu::Queue, u: CrtUniforms) {
        self.uniform = u;
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&u));
    }
}


// ----------------- WGSL Shaders -----------------

pub const EDL_WGSL: &str = r#"
struct U {
    inv_size: vec2<f32>,
    strength: f32,
    radius_px: f32,
};
@group(0) @binding(0) var tColor: texture_2d<f32>;
@group(0) @binding(1) var tDepthLin: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> UBO: U;

struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

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
    if (z0 >= 0.9999) {
        return vec4<f32>(col, 1.0);
    }

    let px = UBO.inv_size;
    let r = UBO.radius_px;
    let eps = 1e-6;
    let lz0 = log(z0 + eps); // Precompute log(z0)

    let o = array<vec2<f32>, 8>(
        vec2<f32>( 1.0, 0.0), vec2<f32>(-1.0, 0.0),
        vec2<f32>( 0.0, 1.0), vec2<f32>( 0.0,-1.0),
        vec2<f32>( 1.0, 1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>( 1.0,-1.0), vec2<f32>(-1.0,-1.0)
    );
    var s: f32 = 0.0;
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
"#;

pub const RGBSHIFT_WGSL: &str = r#"
struct U {
    inv_size: vec2<f32>,
    amount: f32,
    angle: f32,
};
@group(0) @binding(0) var tSrc: texture_2d<f32>;
@group(0) @binding(1) var tDepthLin: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> UBO: U;

struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

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
"#;

pub const CRT_WGSL: &str = r#"
struct U {
    inv_size: vec2<f32>,
    time: f32,
    intensity: f32,
    vignette: f32,
};
@group(0) @binding(0) var tSrc:      texture_2d<f32>;
@group(0) @binding(1) var tDepthLin: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> UBO: U;

struct VSOut { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> VSOut {
    var o: VSOut;
    o.clip = vec4<f32>(pos, 0.0, 1.0);
    o.uv = 0.5 * (pos + vec2<f32>(1.0, 1.0));
    return o;
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898,78.233))) * 43758.5453);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    // Use textureSampleLevel with LOD 0.0 to be compliant with the NonFiltering sampler.
    let col = textureSampleLevel(tSrc, samp, in.uv, 0.0).rgb;
    let dims = textureDimensions(tDepthLin);
    let coord = vec2<i32>(in.uv * vec2<f32>(dims));

    // dark background to match reference
    let bg_base = vec3<f32>(0.01, 0.01, 0.01);

    // scanlines (strong on bg, subtle on content)
    let y_pix = in.uv.y / UBO.inv_size.y;
    let scan_sine = 0.5 + 0.5 * sin(6.28318 * (y_pix * 0.5 + UBO.time * 12.0));
    let scan_bg = 1.0 - UBO.intensity * scan_sine;
    let scan_fg = 1.0 - (UBO.intensity * 0.25) * scan_sine;

    // film grain
    let g = (hash(in.uv * 2048.0 + vec2<f32>(UBO.time, -UBO.time)) - 0.5) * 0.02;

    // vignette
    let d = length((in.uv - vec2<f32>(0.5,0.5)) / vec2<f32>(1.1,0.95));
    let vig = 1.0 - UBO.vignette * smoothstep(0.6, 1.0, d);

    // background mask: only treat as background if z≈1 AND alpha≥0.5
    let dlin = textureLoad(tDepthLin, coord, 0);
    let z    = dlin.r;
    let tag  = dlin.a;                       // 1 = real background, 0 = overlay (grid)
    let bgm  = step(0.9995, z) * step(0.5, tag);

    let bg_col = (bg_base * 0.85 + g) * scan_bg;
    let fg_col = (col + g * 0.4) * scan_fg;

    let out_rgb = mix(fg_col, bg_col, bgm) * vig;
    return vec4<f32>(out_rgb, 1.0);
}
"#;
