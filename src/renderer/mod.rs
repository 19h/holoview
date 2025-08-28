//! The main rendering orchestrator. Owns the GPU context, render targets,
//! and all the individual render pass pipelines.

pub mod context;
pub mod pipelines;
pub mod targets;

use self::{
    context::GfxContext,
    pipelines::{ground_grid::GroundGridPipeline, hologram::HologramPipeline, post_stack::PostStack},
    targets::Targets,
};
use crate::{camera::Camera, data::types::TileGpu};
use std::sync::Arc;
use winit::window::Window;

/// Owns all rendering-related state.
pub struct Renderer {
    pub gfx: GfxContext,
    pub targets: Targets,
    pub holo: HologramPipeline,
    pub grid: GroundGridPipeline,
    pub post_stack: PostStack,
    pub egui_renderer: egui_wgpu::Renderer,
}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let gfx = GfxContext::new(window).await?;
        let size = gfx.size;

        let targets = Targets::new(&gfx.device, size);
        let holo = HologramPipeline::new(
            &gfx.device,
            targets.color_fmt,
            targets.depth_fmt,
            targets.dlin_fmt,
        );
        let grid = GroundGridPipeline::new(
            &gfx.device,
            targets.color_fmt,
            targets.dlin_fmt,
            targets.depth_fmt,
        );
        let post_stack = PostStack::new(&gfx.device, gfx.config.format, size.width, size.height);

        let egui_renderer =
            egui_wgpu::Renderer::new(&gfx.device, gfx.config.format, None, 1);

        Ok(Self {
            gfx,
            targets,
            holo,
            grid,
            post_stack,
            egui_renderer,
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.gfx.resize(new_size);
            self.targets.resize(&self.gfx.device, new_size);
            self.post_stack.resize(&self.gfx.device, new_size.width, new_size.height);
        }
    }

    pub fn render(
        &mut self,
        swap_view: &wgpu::TextureView,
        tiles: &[TileGpu],
        camera: &Camera,
    ) {
        let mut encoder = self
            .gfx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Frame Encoder"),
            });

        // Pass 1: Geometry (Points -> MRT)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Geometry Pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.targets.color,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.targets.dlin,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 1.0, g: 0.0, b: 0.0, a: 0.0 }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.targets.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Draw the grid first, so it's behind the points
            if self.post_stack.params.grid_on {
                self.grid.draw(&mut pass, &self.gfx.queue, camera);
            }

            // Draw all point cloud tiles
            for tile in tiles {
                self.holo.draw_tile(&mut pass, tile);
            }
        }

        // Pass 2..N: Post-processing stack
        self.post_stack.run(
            &self.gfx.device,
            &self.gfx.queue,
            &mut encoder,
            swap_view,
            &self.targets.color,
            &self.targets.dlin,
        );

        self.gfx.queue.submit(std::iter::once(encoder.finish()));
    }
}
