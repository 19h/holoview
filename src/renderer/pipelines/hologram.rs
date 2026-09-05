use crate::data::types::{PointInstance, TileUniformStd140 as TileUniform};
use wgpu::util::DeviceExt;

pub struct HologramPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub tile_layout: wgpu::BindGroupLayout,
    quad_vb: wgpu::Buffer,
}

impl HologramPipeline {
    pub fn new(
        device: &wgpu::Device,
        color_fmt: wgpu::TextureFormat,
        depth_fmt: wgpu::TextureFormat,
        dlin_fmt: wgpu::TextureFormat,
    ) -> Self {
        // Uniform buffer layout for tile data
        let tile_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HYPC Tile UBO Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<TileUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });

        // Vertex/fragment shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaders/hypc_points.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/hypc_points.wgsl").into(),
            ),
        });

        // Full‑screen quad vertices
        let quad_corners: [[f32; 2]; 6] = [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ];

        let quad_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hologram Quad VB"),
            contents: bytemuck::cast_slice(&quad_corners),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Vertex buffer layouts: quad + per‑instance data
        let vbuf_layouts = [
            // Quad vertices
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 2]>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    shader_location: 0,
                    offset: 0,
                    format: wgpu::VertexFormat::Float32x2,
                }],
            },
            // Instance attributes
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<PointInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    // Position offset (vec3)
                    wgpu::VertexAttribute {
                        shader_location: 1,
                        offset: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    // Label (uint)
                    wgpu::VertexAttribute {
                        shader_location: 2,
                        offset: 12,
                        format: wgpu::VertexFormat::Uint32,
                    },
                ],
            },
        ];

        // Pipeline layout with tile uniform bind group
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HYPC Hologram PipelineLayout"),
            bind_group_layouts: &[&tile_layout],
            push_constant_ranges: &[],
        });

        // Render pipeline definition
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("HYPC Hologram Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &vbuf_layouts,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_fmt,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::GreaterEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: color_fmt,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: dlin_fmt,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            pipeline,
            tile_layout,
            quad_vb,
        }
    }

    pub fn draw_tile<'a>(
        &'a self,
        rpass: &mut wgpu::RenderPass<'a>,
        tile: &'a crate::data::types::TileGpu,
    ) {
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &tile.bind, &[0]);
        rpass.set_vertex_buffer(0, self.quad_vb.slice(..));
        rpass.set_vertex_buffer(1, tile.vtx.slice(..));
        rpass.draw(0..6, 0..tile.instances_len);
    }
    pub fn draw_tile_with_uniform<'a>(
        &'a self,
        rpass: &mut wgpu::RenderPass<'a>,
        tile: &'a crate::data::types::TileGpu,
        uniforms: &'a wgpu::BindGroup,
        offset: u32,
    ) {
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, uniforms, &[offset]);
        rpass.set_vertex_buffer(0, self.quad_vb.slice(..));
        rpass.set_vertex_buffer(1, tile.vtx.slice(..));
        rpass.draw(0..6, 0..tile.instances_len);
    }

}

/// One contiguous CPU→GPU update for all draws. Dynamic offsets avoid thousands
/// of individual queue writes and per-frame bind-group allocations.
pub struct DrawUniforms {
    buffer: wgpu::Buffer,
    pub bind: wgpu::BindGroup,
    staging: Vec<u8>,
    capacity: usize,
    pub stride: u32,
}
impl DrawUniforms {
    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, capacity: usize) -> Self {
        let alignment = device.limits().min_uniform_buffer_offset_alignment as usize;
        let size = std::mem::size_of::<TileUniform>();
        let stride = size.div_ceil(alignment) * alignment;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Frame draw uniforms"), size: (capacity * stride) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Frame draw uniforms"), layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer, offset: 0, size: wgpu::BufferSize::new(size as u64),
            }) }],
        });
        Self { buffer, bind, staging: Vec::with_capacity(capacity * stride), capacity, stride: stride as u32 }
    }

    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, layout: &wgpu::BindGroupLayout,
        camera: &crate::camera::Camera, viewport: [f32; 2], draws: &[(&crate::data::types::TileGpu, f32)]) {
        if draws.len() > self.capacity { *self = Self::new(device, layout, draws.len().next_power_of_two()); }
        self.staging.resize(draws.len() * self.stride as usize, 0);
        for (index, (tile, spacing)) in draws.iter().enumerate() {
            let mut u = camera.make_tile_uniform(tile.anchor_units, tile.units_per_meter, viewport, 0.7);
            u.splat_radius_m = spacing * 0.8;
            let start = index * self.stride as usize;
            self.staging[start..start + std::mem::size_of::<TileUniform>()].copy_from_slice(bytemuck::bytes_of(&u));
        }
        if !self.staging.is_empty() { queue.write_buffer(&self.buffer, 0, &self.staging); }
    }
}
