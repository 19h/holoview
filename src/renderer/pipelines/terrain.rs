//! Always-resident coarse ground coverage for immediate close-up camera changes.
use crate::{camera::Camera, data::terrain::TerrainGrid};
use anyhow::{ensure, Result};
use glam::DVec3;
use wgpu::util::DeviceExt;

pub struct TerrainGpu {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    count: u32,
    anchor: [i64; 3],
    uniform: wgpu::Buffer,
    bind: wgpu::BindGroup,
    pub bytes: u64,
}
impl TerrainGpu {
    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, terrain: &TerrainGrid) -> Result<Self> {
        terrain.validate()?;
        let vertex_bytes = terrain.width as u64 * terrain.height as u64 * 12;
        let index_max = terrain.width.saturating_sub(1) as u64 * terrain.height.saturating_sub(1) as u64 * 24;
        ensure!(vertex_bytes <= device.limits().max_buffer_size && index_max <= device.limits().max_buffer_size, "Terrain mesh exceeds device buffer limit");
        let east = DVec3::from(terrain.east); let north = DVec3::from(terrain.north); let up = DVec3::from(terrain.up);
        let origin = DVec3::from(terrain.origin);
        let anchor = terrain.origin.map(|v| (v * 2000.0).round() as i64);
        let shift = origin - DVec3::from(anchor.map(|v| v as f64 / 2000.0));
        let mut vertices = Vec::<[f32; 3]>::with_capacity(terrain.heights_q10.len());
        for y in 0..terrain.height { for x in 0..terrain.width {
            let h = terrain.heights_q10[y * terrain.width + x];
            let z = if h == i16::MIN { 0.0 } else { h as f64 * 0.1 - 0.5 };
            vertices.push((shift + east * (terrain.west_m + (x as f64 + 0.5) * terrain.cell_m)
                + north * (terrain.south_m + (y as f64 + 0.5) * terrain.cell_m) + up * z).as_vec3().into());
        } }
        let mut indices = Vec::<u32>::new();
        for y in 0..terrain.height.saturating_sub(1) { for x in 0..terrain.width.saturating_sub(1) {
            let a = y * terrain.width + x; let b = a + 1; let c = a + terrain.width; let d = c + 1;
            let valid = |i| terrain.heights_q10[i] != i16::MIN && (terrain.coverage.is_empty() || terrain.coverage[i] != 0);
            if [a,b,c,d].iter().all(|&i| valid(i)) { indices.extend([a as u32, b as u32, c as u32, c as u32, b as u32, d as u32]); }
        } }
        ensure!(!indices.is_empty(), "Terrain has no supported quads");
        let count = indices.len() as u32;
        let bytes = vertex_bytes + indices.len() as u64 * 4;
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("City support terrain vertices"), contents: bytemuck::cast_slice(&vertices), usage: wgpu::BufferUsages::VERTEX });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("City support terrain indices"), contents: bytemuck::cast_slice(&indices), usage: wgpu::BufferUsages::INDEX });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor { label: Some("City support terrain uniform"), size: std::mem::size_of::<crate::data::TileUniformStd140>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("City support terrain uniform"), layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }] });
        Ok(Self { vertices, indices, count, anchor, uniform, bind, bytes })
    }
    pub fn update(&self, queue: &wgpu::Queue, camera: &Camera, viewport: [f32; 2]) {
        let u = camera.make_tile_uniform(self.anchor, 2000, viewport, 1.0);
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
    }
}

pub struct TerrainPipeline { pipeline: wgpu::RenderPipeline }
impl TerrainPipeline {
    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, targets: &crate::renderer::targets::Targets) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("City support terrain"), source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/terrain.wgsl").into()) });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("City support terrain"), bind_group_layouts: &[layout], push_constant_ranges: &[] });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("City support terrain"), layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &module, entry_point: "vs_main", compilation_options: Default::default(), buffers: &[
                wgpu::VertexBufferLayout { array_stride: 12, step_mode: wgpu::VertexStepMode::Vertex, attributes: &[wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 }] },
            ] },
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: "fs_main", compilation_options: Default::default(), targets: &[
                Some(wgpu::ColorTargetState { format: targets.color_fmt, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                Some(wgpu::ColorTargetState { format: targets.dlin_fmt, blend: None, write_mask: wgpu::ColorWrites::ALL }),
            ] }), primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState { format: targets.depth_fmt, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::GreaterEqual, stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(), multiview: None,
        });
        Self { pipeline }
    }
    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, terrain: &'a TerrainGpu) {
        pass.set_pipeline(&self.pipeline); pass.set_bind_group(0, &terrain.bind, &[0]);
        pass.set_vertex_buffer(0, terrain.vertices.slice(..)); pass.set_index_buffer(terrain.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..terrain.count, 0, 0..1);
    }
}
