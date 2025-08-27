//! Renders a dynamic, adaptive grid on a local tangent plane to the WGS-84 ellipsoid.

use crate::camera::Camera;
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridUniforms {
    /// Transforms a unit quad in model space to the correct tangent plane position in clip space.
    pub model_view_proj: Mat4,
    /// Camera height above the tangent plane in meters, for fading.
    pub camera_height_m: f32,
    pub _padding: [f32; 3],
}

pub struct GroundGridPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    quad_vb: wgpu::Buffer,
}

impl GroundGridPipeline {
    pub fn new(
        device: &wgpu::Device,
        color_fmt: wgpu::TextureFormat,
        dlin_fmt: wgpu::TextureFormat,
        depth_fmt: wgpu::TextureFormat,
    ) -> Self {
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid Uniform Buffer"),
            size: std::mem::size_of::<GridUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Grid BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Grid Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let corners: [[f32; 2]; 6] = [
            [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0],
            [-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
        ];
        let quad_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Quad VB"),
            contents: bytemuck::cast_slice(&corners),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Grid WGSL"),
            source: wgpu::ShaderSource::Wgsl(GRID_WGSL.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Grid Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Ground Grid Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 2]>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                    }],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: color_fmt,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: dlin_fmt,
                        blend: None, // Write tag directly
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_fmt,
                depth_write_enabled: false, // Don't occlude points
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            pipeline,
            bind_group,
            uniform_buffer,
            quad_vb,
        }
    }

    pub fn draw<'a>(&'a self, rpass: &mut wgpu::RenderPass<'a>, queue: &wgpu::Queue, camera: &Camera) {
        let tangent_point_ecef = hypc::geodetic_to_ecef(camera.lat_deg, camera.lon_deg, 0.0);
        let camera_ecef = camera.ecef_m();

        let view_proj_ecef = camera.view_proj_ecef();
        let view_ecef = camera.view_ecef();
        let camera_relative_tangent_point = Vec3::new(
            (tangent_point_ecef[0] - camera_ecef[0]) as f32,
            (tangent_point_ecef[1] - camera_ecef[1]) as f32,
            (tangent_point_ecef[2] - camera_ecef[2]) as f32,
        );

        // Extract rotation from pure view (not from view*proj)
        let (_scale, view_rotation, _translation) = view_ecef.to_scale_rotation_translation();
        let model_matrix = Mat4::from_translation(camera_relative_tangent_point)
            * Mat4::from_quat(view_rotation.inverse())
            * Mat4::from_scale(Vec3::splat(500_000.0));

        let uniforms = GridUniforms {
            model_view_proj: view_proj_ecef * model_matrix,
            camera_height_m: camera.h_m as f32,
            _padding: [0.0; 3],
        };

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.set_vertex_buffer(0, self.quad_vb.slice(..));
        rpass.draw(0..6, 0..1);
    }
}

pub const GRID_WGSL: &str = r#"
struct GridUniforms {
    model_view_proj: mat4x4<f32>,
    camera_height_m: f32,
};
@group(0) @binding(0) var<uniform> U: GridUniforms;

struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec2<f32>,
}

@vertex
fn vs_main(@location(0) corner: vec2<f32>) -> VSOut {
    var o: VSOut;
    o.clip = U.model_view_proj * vec4<f32>(corner.x, corner.y, 0.0, 1.0);
    o.world_pos = corner * 500000.0; // Pass un-transformed position for grid lines
    return o;
}

struct FSOut {
    @location(0) color: vec4<f32>,
    @location(1) dlin: vec4<f32>,
}

// Anti-aliased line mask
fn line(coord: f32, step: f32) -> f32 {
    let temp = coord / step;
    let aaw = fwidth(temp) * 1.5;
    let f = fract(temp);
    let d = min(f, 1.0 - f);
    return 1.0 - smoothstep(0.0, aaw, d);
}

@fragment
fn fs_main(in: VSOut) -> FSOut {
    let log_h = log2(U.camera_height_m);

    // Determine two levels of grid spacing based on height
    let level0_log = floor(log_h / log2(10.0));
    let level0_step = pow(10.0, level0_log);
    let level1_step = level0_step * 10.0;

    // Blend between the two levels
    let t = fract(log_h / log2(10.0));

    let p = in.world_pos;

    let minor0 = max(line(p.x, level0_step), line(p.y, level0_step));
    let major0 = max(line(p.x, level0_step * 10.0), line(p.y, level0_step * 10.0));
    let grid0 = minor0 * 0.5 + major0 * 0.5;

    let minor1 = max(line(p.x, level1_step), line(p.y, level1_step));
    let major1 = max(line(p.x, level1_step * 10.0), line(p.y, level1_step * 10.0));
    let grid1 = minor1 * 0.5 + major1 * 0.5;

    let grid = mix(grid0, grid1, t);

    // Fade out at extreme heights
    let opacity = grid * (1.0 - smoothstep(20000.0, 200000.0, U.camera_height_m));
    let color = vec3<f32>(0.90, 0.15, 0.15);

    var o: FSOut;
    o.color = vec4<f32>(color, opacity * 0.55);
    // Tag as overlay (alpha=0) and background depth (r=1)
    o.dlin = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    return o;
}
"#;
