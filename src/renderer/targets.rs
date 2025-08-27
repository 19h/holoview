//! Manages all primary render target textures for the geometry pass.

pub struct Targets {
    // Textures are stored to manage their lifetime.
    _color_tex: wgpu::Texture,
    _depth_tex: wgpu::Texture,
    _dlin_tex: wgpu::Texture,

    // Views are used by render passes and post-processing.
    pub color: wgpu::TextureView,
    pub depth: wgpu::TextureView,
    pub dlin: wgpu::TextureView,

    // Formats are needed by pipeline constructors.
    pub color_fmt: wgpu::TextureFormat,
    pub depth_fmt: wgpu::TextureFormat,
    pub dlin_fmt: wgpu::TextureFormat,
}

impl Targets {
    pub fn new(device: &wgpu::Device, size: winit::dpi::PhysicalSize<u32>) -> Self {
        let width = size.width.max(1);
        let height = size.height.max(1);

        let size_ext = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let color_fmt = wgpu::TextureFormat::Rgba16Float;
        let depth_fmt = wgpu::TextureFormat::Depth32Float;
        let dlin_fmt = wgpu::TextureFormat::Rgba16Float;

        let color_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Scene Color Target"),
            size: size_ext,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color_fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Scene Depth Target"),
            size: size_ext,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: depth_fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let dlin_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth-Linear Proxy Target"),
            size: size_ext,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: dlin_fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

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

    pub fn resize(&mut self, device: &wgpu::Device, size: winit::dpi::PhysicalSize<u32>) {
        *self = Self::new(device, size);
    }
}
