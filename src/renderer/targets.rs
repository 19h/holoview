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
        let dlin_fmt = wgpu::TextureFormat::Rgba16Float;

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
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let depth_tex = create_tex(
            "Scene Depth Target",
            depth_fmt,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );

        let dlin_tex = create_tex(
            "Depth-Linear Proxy Target",
            dlin_fmt,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
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
