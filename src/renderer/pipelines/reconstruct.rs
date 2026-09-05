use crate::renderer::targets::Targets;

pub struct Reconstruction {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    bind: wgpu::BindGroup,
    _color: wgpu::Texture,
    _depth: wgpu::Texture,
    pub color: wgpu::TextureView,
    pub depth: wgpu::TextureView,
}
impl Reconstruction {
    fn textures(device: &wgpu::Device, size: winit::dpi::PhysicalSize<u32>, targets: &Targets) -> (wgpu::Texture, wgpu::Texture) {
        let texture = |format| device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Reconstructed surface"), size: wgpu::Extent3d { width: size.width.max(1), height: size.height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[],
        });
        (texture(targets.color_fmt), texture(targets.dlin_fmt))
    }
    fn bind(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, targets: &Targets) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("Surface reconstruction input"), layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&targets.color) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&targets.dlin) },
            ] })
    }
    pub fn new(device: &wgpu::Device, size: winit::dpi::PhysicalSize<u32>, targets: &Targets) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Surface reconstruction layout"), entries: &[0, 1].map(|binding| wgpu::BindGroupLayoutEntry {
                binding, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None,
            }),
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Surface reconstruction"), source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/reconstruct.wgsl").into()) });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("Surface reconstruction"), bind_group_layouts: &[&layout], push_constant_ranges: &[] });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Surface reconstruction"), layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &module, entry_point: "vs_main", compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: "fs_main", compilation_options: Default::default(), targets: &[
                Some(wgpu::ColorTargetState { format: targets.color_fmt, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                Some(wgpu::ColorTargetState { format: targets.dlin_fmt, blend: None, write_mask: wgpu::ColorWrites::ALL }),
            ] }), primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview: None,
        });
        let (_color, _depth) = Self::textures(device, size, targets);
        let color = _color.create_view(&Default::default()); let depth = _depth.create_view(&Default::default());
        let bind = Self::bind(device, &layout, targets);
        Self { pipeline, layout, bind, _color, _depth, color, depth }
    }
    pub fn resize(&mut self, device: &wgpu::Device, size: winit::dpi::PhysicalSize<u32>, targets: &Targets) {
        (self._color, self._depth) = Self::textures(device, size, targets);
        self.color = self._color.create_view(&Default::default()); self.depth = self._depth.create_view(&Default::default());
        self.bind = Self::bind(device, &self.layout, targets);
    }
    pub fn run(&self, encoder: &mut wgpu::CommandEncoder) {
        let attachment = |view| Some(wgpu::RenderPassColorAttachment { view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Surface reconstruction"), color_attachments: &[attachment(&self.color), attachment(&self.depth)], depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline); pass.set_bind_group(0, &self.bind, &[]); pass.draw(0..3, 0..1);
    }
}
