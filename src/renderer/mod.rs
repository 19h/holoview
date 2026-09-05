//! The main rendering orchestrator. Owns the GPU context, render targets,
//! and all the individual render pass pipelines.

pub mod context;
pub mod pipelines;
pub mod targets;
pub mod probe;

use self::{
    context::GfxContext,
    pipelines::{
        ground_grid::GroundGridPipeline, hologram::{HologramPipeline, DrawUniforms}, post_stack::PostStack, reconstruct::Reconstruction,
    },
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
    pub draw_uniforms: DrawUniforms,
    pub reconstruction: Reconstruction,
    pub probe: Option<probe::FrameProbe>,
    pub terrain: Option<pipelines::terrain::TerrainGpu>,
    terrain_pipeline: pipelines::terrain::TerrainPipeline,
    frame_number: u64,
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

        let terrain_pipeline = pipelines::terrain::TerrainPipeline::new(&gfx.device, &holo.tile_layout, &targets);
        let reconstruction = Reconstruction::new(&gfx.device, size, &targets);
        let draw_uniforms = DrawUniforms::new(&gfx.device, &holo.tile_layout, 2048);
        let egui_renderer = egui_wgpu::Renderer::new(&gfx.device, gfx.config.format, None, 1);

        Ok(Self {
            gfx,
            targets,
            holo,
            grid,
            post_stack,
            egui_renderer,
            draw_uniforms,
            reconstruction,
            probe: None,
            terrain: None,
            terrain_pipeline,
            frame_number: 0,
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.gfx.resize(new_size);
            self.targets.resize(&self.gfx.device, new_size);
            self.reconstruction.resize(&self.gfx.device, new_size, &self.targets);
            if let Some(probe) = &mut self.probe { probe.resize(&self.gfx.device, &self.targets.dlin, &self.reconstruction.depth, [new_size.width, new_size.height]); }
            self.post_stack
                .resize(&self.gfx.device, new_size.width, new_size.height);
        }
    }

    pub fn render(&mut self, swap_view: &wgpu::TextureView, tiles: &[TileGpu], camera: &Camera) {
        let draws: Vec<_> = tiles.iter().map(|t| (t, 0.0)).collect();
        self.render_streamed(swap_view, &draws, camera);
    }

    pub fn render_streamed(&mut self, swap_view: &wgpu::TextureView, draws: &[(&TileGpu, f32)], camera: &Camera) {
        self.draw_uniforms.update(&self.gfx.device, &self.gfx.queue, &self.holo.tile_layout,
            camera, [self.gfx.size.width as f32, self.gfx.size.height as f32], draws);
        let mut encoder = self
            .gfx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Frame Encoder"),
            });

        if let Some(terrain) = &self.terrain { terrain.update(&self.gfx.queue, camera, [self.gfx.size.width as f32, self.gfx.size.height as f32]); }
        let probe_slot = self.probe.as_mut().and_then(|probe| {
            probe.poll(&self.gfx.device, false);
            // Verification must cover every frame. This wait only occurs if all
            // readback slots are busy, and the probe is disabled in normal use.
            if !probe.has_available_slot() { probe.poll(&self.gfx.device, true); }
            probe.begin(&mut encoder, self.frame_number)
        });
        self.frame_number += 1;

        // Pass 1: Geometry (Points -> MRT)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Geometry Pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.targets.color,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.targets.dlin,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.targets.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: probe_slot.and_then(|slot| self.probe.as_ref().unwrap().timestamp_writes(slot)),
                occlusion_query_set: None,
            });

            // Draw the grid first, so it's behind the points
            if self.post_stack.params.grid_on {
                self.grid.draw(
                    &mut pass,
                    &self.gfx.queue,
                    camera,
                    self.post_stack.params.grid_utm_align,
                );
            }

            if let Some(terrain) = &self.terrain { self.terrain_pipeline.draw(&mut pass, terrain); }

            // Draw all point cloud tiles
            for (index, (tile, _)) in draws.iter().enumerate() {
                self.holo.draw_tile_with_uniform(&mut pass, tile, &self.draw_uniforms.bind,
                    index as u32 * self.draw_uniforms.stride);
            }
        }

        if self.post_stack.params.fill_on { self.reconstruction.run(&mut encoder); }
        let (color, depth) = if self.post_stack.params.fill_on {
            (&self.reconstruction.color, &self.reconstruction.depth)
        } else { (&self.targets.color, &self.targets.dlin) };

        // Pass 2..N: Post-processing stack
        self.post_stack.run(
            &self.gfx.device,
            &self.gfx.queue,
            &mut encoder,
            swap_view,
            color,
            depth,
        );

        if let (Some(probe), Some(slot)) = (&self.probe, probe_slot) { probe.end(&mut encoder, slot, self.post_stack.params.fill_on); }
        self.gfx.queue.submit(std::iter::once(encoder.finish()));
        if let (Some(probe), Some(slot)) = (&self.probe, probe_slot) { probe.submitted(slot); }
    }
}
