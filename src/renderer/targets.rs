//! Manages primary render target textures for the geometry pass.

pub struct Targets {
    // Private textures – keep alive for the lifetime of the views.
    _color_tex: wgpu::Texture,
    _depth_tex: wgpu::Texture,
    _dlin_tex: wgpu::Texture,

    // Public texture views used by render passes and post‑processing.
    pub color: wgpu::TextureView,
    pub depth: wgpu::TextureView,
    pub dlin: wgpu::TextureView,

    // Formats required by pipeline creation.
    pub color_fmt: wgpu::TextureFormat,
    pub depth_fmt: wgpu::TextureFormat,
    pub dlin_fmt: wgpu::TextureFormat,
}

impl Targets {
    pub fn new(device: &wgpu::Device, size: winit::dpi::PhysicalSize<u32>) -> Self {
        // Ensure non‑zero dimensions.
        let width = size.width.max(1);
        let height = size.height.max(1);

        let tex_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        // Define texture formats.
        let color_fmt = wgpu::TextureFormat::Rgba16Float;
        let depth_fmt = wgpu::TextureFormat::Depth32Float;
        // Linear eye depth in metres needs f32 precision for sub-metre surfaces.
        let dlin_fmt = wgpu::TextureFormat::Rgba32Float;

        // Helper to create a texture with the given parameters.
        let create_tex = |label: &str, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: tex_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };

        // Create textures.
        let color_tex = create_tex(
            "Scene Color Target",
            color_fmt,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );

        let depth_tex = create_tex(
            "Scene Depth Target",
            depth_fmt,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );

        let dlin_tex = create_tex(
            "Depth-Linear Proxy Target",
            dlin_fmt,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );

        // Assemble the struct.
        Self {
            color: color_tex.create_view(&wgpu::TextureViewDescriptor::default()),
            depth: depth_tex.create_view(&wgpu::TextureViewDescriptor::default()),
            dlin: dlin_tex.create_view(&wgpu::TextureViewDescriptor::default()),
            _color_tex: color_tex,
            _depth_tex: depth_tex,
            _dlin_tex: dlin_tex,
            color_fmt,
            depth_fmt,
            dlin_fmt,
        }
    }

    /// Resize all render targets to the new window size.
    pub fn resize(&mut self, device: &wgpu::Device, size: winit::dpi::PhysicalSize<u32>) {
        *self = Self::new(device, size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::{pipelines::reconstruct::Reconstruction, probe::FrameProbe};

    #[test]
    fn reconstruction_fills_supported_interior_gaps_without_crossing_depth_edges() {
        let (device, queue) = pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance.request_adapter(&Default::default()).await.expect("GPU required");
            adapter.request_device(&Default::default(), None).await.unwrap()
        });
        let size = winit::dpi::PhysicalSize::new(64, 64);
        let targets = Targets::new(&device, size);
        let reconstruction = Reconstruction::new(&device, size, &targets);
        for discontinuity in [false, true] {
            let mut depth = vec![[0.0f32; 4]; 64 * 64];
            let mut expected_raw = 0;
            for y in 8..56 {
                for x in 8..56 {
                    let hole = if discontinuity { (30..34).contains(&x) } else { (29..35).contains(&x) && (29..35).contains(&y) };
                    if !hole {
                        depth[y * 64 + x] = [if discontinuity && x < 32 { 10.0 } else { 100.0 }, 1.0 / 255.0, 0.0, 1.0];
                        expected_raw += 1;
                    }
                }
            }
            let color: Vec<[u16; 4]> = depth.iter().map(|d| [0x3800, 0x3800, 0x3800, if d[3] > 0.0 { 0x3c00 } else { 0 }]).collect();
            let write = |texture: &wgpu::Texture, bytes: &[u8], row| queue.write_texture(
                wgpu::ImageCopyTexture { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All }, bytes,
                wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(64) },
                wgpu::Extent3d { width: 64, height: 64, depth_or_array_layers: 1 });
            write(&targets._dlin_tex, bytemuck::cast_slice(&depth), 64 * 16);
            write(&targets._color_tex, bytemuck::cast_slice(&color), 64 * 8);
            let mut probe = FrameProbe::new(&device, &queue, &targets.dlin, &reconstruction.depth, [64, 64]);
            let mut encoder = device.create_command_encoder(&Default::default());
            let slot = probe.begin(&mut encoder, 0).unwrap();
            reconstruction.run(&mut encoder); probe.end(&mut encoder, slot, true);
            queue.submit(Some(encoder.finish())); probe.submitted(slot); probe.poll(&device, true);
            let result = &probe.samples[0];
            assert_eq!(result.raw_pixels, expected_raw);
            assert_eq!(result.display_pixels, if discontinuity { expected_raw } else { 48 * 48 });
        }
    }
}
