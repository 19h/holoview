// src/renderer/mod.rs
//! The main rendering orchestrator. Owns the GPU context, render targets,
//! and all the individual render pass pipelines.

pub mod context;
pub mod pipelines;
pub mod targets;

use self::{
    context::GpuContext,
    pipelines::{
        ground_grid::GroundGridPipeline, hologram::HologramPipeline, postprocess::CrtPass,
        postprocess::EdlPass, postprocess::PostPass, postprocess::RgbShiftPass,
    },
    targets::RenderTargets,
};
use crate::data::point_cloud::{GeoExtentDeg, SemanticMask};
use glam::{Vec2, Vec3};
use std::sync::Arc;
use winit::window::Window;

/// Holds all GPU resources and data for a single point cloud tile.
pub struct TileDraw {
    pub name: String,
    pub vb: wgpu::Buffer,
    pub count: u32,
    // Semantics
    pub sem_tex: Option<wgpu::Texture>,
    pub sem_view: wgpu::TextureView,
    pub sem_samp: wgpu::Sampler,
    pub holo_bg: wgpu::BindGroup,
    // Per-tile UBO and bind group for SMC1 UV bounds
    pub tile_ubo: wgpu::Buffer,
    pub tile_bg: wgpu::BindGroup,
    // Per‑tile world decode bounds for SMC1 UV & height modulation
    pub dmin_world: Vec3,
    pub dmax_world: Vec3,
    pub geot: GeoExtentDeg,
    // Biases for consistent alignment across tiles
    pub z_bias: f32,
    pub xy_bias: Vec2,
    // (x,y,z) near the tile border, for alignment
    pub edge_samples: Vec<[f32; 3]>,
}

/// Owns all rendering-related state.
pub struct Renderer {
    pub context: GpuContext,
    pub targets: RenderTargets,
    pub pipelines: AllPipelines,
    pub egui_renderer: egui_wgpu::Renderer,
    // Fallback resources for tiles without semantic masks
    sem_fallback_tex: wgpu::Texture,
    // CORRECTED: Removed sem_fallback_samp, as Sampler is not Clone.
}

/// A container for all the render pass pipelines.
pub struct AllPipelines {
    pub hologram: HologramPipeline,
    pub ground_grid: GroundGridPipeline,
    pub post_edl: EdlPass,
    pub post_rgb: RgbShiftPass,
    pub post_crt: CrtPass,
}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> Self {
        let context = GpuContext::new(window).await;
        let targets = RenderTargets::new(&context);

        let hologram = HologramPipeline::new(
            &context.device,
            context.surface_format,
            wgpu::TextureFormat::Rgba16Float,
        );
        let ground_grid = GroundGridPipeline::new(
            &context.device,
            context.surface_format,
            wgpu::TextureFormat::Rgba16Float,
        );
        let mut post_edl = EdlPass::new(&context.device, context.surface_format);
        let mut post_rgb = RgbShiftPass::new(&context.device, context.surface_format);
        let mut post_crt = CrtPass::new(&context.device, context.surface_format);

        // Connect the post-processing chain
        targets.bind_post_processing_inputs(&mut post_edl, &mut post_rgb, &mut post_crt, &context.device);

        let egui_renderer = egui_wgpu::Renderer::new(
            &context.device,
            context.surface_format,
            None,
            1,
        );

        // Fallback SMC1 resources
        let sem_fallback_tex = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SMC1 Fallback 1x1"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        context.queue.write_texture(
            wgpu::ImageCopyTexture { texture: &sem_fallback_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &[0u8], // label=0 → Unknown
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(1), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );

        // CORRECTED: Removed creation of sem_fallback_samp here.

        let pipelines = AllPipelines { hologram, ground_grid, post_edl, post_rgb, post_crt };
        Self { context, targets, pipelines, egui_renderer, sem_fallback_tex }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.context.resize(new_size);
            self.targets = RenderTargets::new(&self.context);
            self.targets.bind_post_processing_inputs(
                &mut self.pipelines.post_edl,
                &mut self.pipelines.post_rgb,
                &mut self.pipelines.post_crt,
                &self.context.device,
            );
        }
    }

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        swap_view: &wgpu::TextureView,
        tiles: &[TileDraw],
    ) {
        // PASS 1: Points -> scene targets (MSAA with resolve)
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Scene"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.targets.scene_color_view,
                        resolve_target: Some(&self.targets.scene_color_resolved_view),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0196, g: 0.0196, b: 0.0275, a: 1.0 }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.targets.scene_depthlin_view,
                        resolve_target: Some(&self.targets.scene_depthlin_resolved_view),
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.targets.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None, occlusion_query_set: None,
            });

            // Draw ground grid first
            rp.set_pipeline(&self.pipelines.ground_grid.pipeline);
            rp.set_bind_group(0, &self.pipelines.ground_grid.bind_group, &[]);
            rp.set_vertex_buffer(0, self.pipelines.ground_grid.quad_vb.slice(..));
            rp.draw(0..6, 0..1);

            // Draw all tiles
            rp.set_pipeline(&self.pipelines.hologram.pipeline);
            rp.set_vertex_buffer(0, self.pipelines.hologram.quad_vb.slice(..));
            for t in tiles {
                rp.set_bind_group(0, &t.holo_bg, &[]);
                rp.set_bind_group(1, &t.tile_bg, &[]);
                rp.set_vertex_buffer(1, t.vb.slice(..));
                rp.draw(0..6, 0..t.count);
            }
        }

        // PASS 2: EDL
        self.pipelines.post_edl.draw(encoder, &self.targets.post_edl_view);
        // PASS 3: RGB Shift
        self.pipelines.post_rgb.draw(encoder, &self.targets.post_rgb_view);
        // PASS 4: CRT
        self.pipelines.post_crt.draw(encoder, swap_view);
    }

    pub fn upload_semantic_mask(
        &self,
        sm: Option<&SemanticMask>,
    ) -> (Option<wgpu::Texture>, wgpu::TextureView, wgpu::Sampler) {
        let Some(sm) = sm else {
            let fallback_view = self.sem_fallback_tex.create_view(&wgpu::TextureViewDescriptor::default());
            // CORRECTED: Create a new sampler here instead of cloning.
            let fallback_sampler = self.context.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("SMC1 Fallback Sampler (instance)"),
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });
            return (None, fallback_view, fallback_sampler);
        };

        let (width, height) = (sm.width as u32, sm.height as u32);
        let texture = self.context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SMC1 Labels"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let bytes_per_row = ((width + 255) / 256) * 256;
        let mut staging_data = vec![0u8; (bytes_per_row * height) as usize];
        for row in 0..height as usize {
            let source_row = &sm.data[row * width as usize..(row + 1) * width as usize];
            let dest_row = &mut staging_data[row * bytes_per_row as usize..row * bytes_per_row as usize + width as usize];
            dest_row.copy_from_slice(source_row);
        }

        self.context.queue.write_texture(
            wgpu::ImageCopyTexture { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &staging_data,
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(bytes_per_row), rows_per_image: Some(height) },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = self.context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SMC1 Nearest"), mag_filter: wgpu::FilterMode::Nearest, min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        (Some(texture), view, sampler)
    }
}
