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
        let tile_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HYPC Tile UBO Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<TileUniform>() as u64
                    ),
                },
                count: None,
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaders/hypc_points.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/hypc_points.wgsl").into()),
        });

        let quad_corners: [[f32; 2]; 6] = [
            [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0],
            [-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
        ];
        let quad_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hologram Quad VB"),
            contents: bytemuck::cast_slice(&quad_corners),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let vbuf_layouts = [
            // Quad corner buffer
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 2]>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    shader_location: 0, // Corresponds to @location(0) in WGSL
                    offset: 0,
                    format: wgpu::VertexFormat::Float32x2,
                }],
            },
            // Per-instance data buffer
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<PointInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    // ofs_m
                    wgpu::VertexAttribute {
                        shader_location: 1, // Corresponds to @location(1) in WGSL
                        offset: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    // label
                    wgpu::VertexAttribute {
                        shader_location: 2, // Corresponds to @location(2) in WGSL
                        offset: 12,
                        format: wgpu::VertexFormat::Uint32,
                    },
                ],
            },
        ];

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HYPC Hologram PipelineLayout"),
            bind_group_layouts: &[&tile_layout],
            push_constant_ranges: &[],
        });

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
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: color_fmt,
                        //blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

    pub fn draw_tile<'a>(&'a self, rpass: &mut wgpu::RenderPass<'a>, tile: &'a crate::data::types::TileGpu) {
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &tile.bind, &[]);
        rpass.set_vertex_buffer(0, self.quad_vb.slice(..));
        rpass.set_vertex_buffer(1, tile.vtx.slice(..));
        rpass.draw(0..6, 0..tile.instances_len);
    }
}
