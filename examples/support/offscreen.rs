use anyhow::{Context, Result};
use holographic_viewer::{
    camera::Camera, data::types::TileGpu, renderer::pipelines::hologram::HologramPipeline,
};

pub struct Offscreen {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: HologramPipeline,
    pub size: u32,
}

use holographic_viewer::renderer::pipelines::post_stack::{PostParams, PostStack};

impl Offscreen {
    pub fn new(size: u32) -> Result<Self> {
        assert_eq!(size % 64, 0, "RGBA row must be 256-byte aligned");
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .context("No GPU adapter for offscreen verification")?;
            eprintln!("GPU verification adapter: {:?}", adapter.get_info());
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await?;
            let pipeline = HologramPipeline::new(
                &device,
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureFormat::Depth32Float,
                wgpu::TextureFormat::Rgba32Float,
            );
            Ok(Self {
                device,
                queue,
                pipeline,
                size,
            })
        })
    }

    pub fn render(
        &self,
        tiles: &[TileGpu],
        camera: &Camera,
        point_size_px: f32,
    ) -> Result<Vec<u8>> {
        self.render_with_post(tiles, camera, point_size_px, None)
    }

    pub fn render_with_post(
        &self,
        tiles: &[TileGpu],
        camera: &Camera,
        point_size_px: f32,
        post: Option<PostParams>,
    ) -> Result<Vec<u8>> {
        let extent = wgpu::Extent3d {
            width: self.size,
            height: self.size,
            depth_or_array_layers: 1,
        };
        let texture = |format, usage| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Alignment verification"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let color = texture(
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let dlin = texture(
            wgpu::TextureFormat::Rgba32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let depth = texture(
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let cv = color.create_view(&Default::default());
        let dv = dlin.create_view(&Default::default());
        let zv = depth.create_view(&Default::default());
        for tile in tiles {
            let u = camera.make_tile_uniform(
                tile.anchor_units,
                tile.units_per_meter,
                [self.size as f32; 2],
                point_size_px,
            );
            self.queue
                .write_buffer(&tile.ubo, 0, bytemuck::bytes_of(&u));
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let attachment = |view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Production tile shader verification"),
                color_attachments: &[attachment(&cv), attachment(&dv)],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &zv,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            for tile in tiles {
                self.pipeline.draw_tile(&mut pass, tile);
            }
        }
        let post_output = post.map(|params| {
            let result = texture(
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            );
            let rv = result.create_view(&Default::default());
            let mut stack = PostStack::new(
                &self.device,
                wgpu::TextureFormat::Rgba8Unorm,
                self.size,
                self.size,
            );
            stack.params = params;
            stack.run(&self.device, &self.queue, &mut encoder, &rv, &cv, &dv);
            result
        });
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Alignment readback"),
            size: (self.size as u64).pow(2) * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: post_output.as_ref().unwrap_or(&color),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.size * 4),
                    rows_per_image: Some(self.size),
                },
            },
            extent,
        );
        self.queue.submit(Some(encoder.finish()));
        let slice = output.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;
        let bytes = slice.get_mapped_range().to_vec();
        output.unmap();
        Ok(bytes)
    }
}
