// src/renderer/pipelines/base.rs
//! Shared utilities for rendering pipelines.
//!
//! This module provides common functionality used across multiple
//! pipeline implementations, including full-screen quad generation
//! and bind group layout helpers.

use wgpu::util::DeviceExt;

/// Creates a vertex buffer for a full-screen quad (triangle).
///
/// Uses the three-vertex trick: a single triangle that covers the entire screen.
/// Vertices at (-1,-3), (3,1), (-1,1) create a triangle that covers [-1,1]x[-1,1]
/// when clipped.
pub fn fsq_vertex_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    let fsq = [
        [-1.0_f32, -3.0],
        [ 3.0,  1.0],
        [-1.0,  1.0],
    ];
    
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("FSQ Vertex Buffer"),
        contents: bytemuck::cast_slice(&fsq),
        usage: wgpu::BufferUsages::VERTEX,
    })
}

/// Creates a bind group layout for a single uniform buffer.
///
/// # Arguments
/// * `device` - The GPU device
/// * `stages` - Which shader stages can access the uniform
pub fn single_uniform_bind_group_layout(
    device: &wgpu::Device,
    stages: wgpu::ShaderStages,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Single Uniform BGL"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: stages,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

/// Trait for post-processing passes that render to a texture view.
pub trait PostPass {
    /// Draws the post-processing effect to the given texture view.
    fn draw(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView);
}

/// Macro to implement the PostPass trait for a post-processing struct.
///
/// The struct must have:
/// - `pipeline: wgpu::RenderPipeline`
/// - `bind_group: wgpu::BindGroup`
/// - `fsq_vb: wgpu::Buffer`
macro_rules! impl_post_pass {
    ($pass_struct:ty) => {
        impl $crate::renderer::pipelines::base::PostPass for $pass_struct {
            fn draw(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(stringify!($pass_struct)),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
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

                render_pass.set_pipeline(&self.pipeline);
                render_pass.set_bind_group(0, &self.bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.fsq_vb.slice(..));
                render_pass.draw(0..3, 0..1);
            }
        }
    };
}

pub(crate) use impl_post_pass;
