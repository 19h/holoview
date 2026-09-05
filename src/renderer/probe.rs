//! Optional GPU coverage and timestamp instrumentation. Three asynchronous
//! readbacks avoid synchronizing normal rendering with CPU measurements.
use std::sync::{Arc, atomic::{AtomicU8, Ordering}};

#[derive(Clone, Debug, serde::Serialize)]
pub struct ProbeSample {
    pub frame: u64,
    pub raw_pixels: u32,
    pub display_pixels: u32,
    pub gpu_ms: Option<f64>,
}
struct Slot { buffer: wgpu::Buffer, state: Arc<AtomicU8>, frame: u64 }
pub struct FrameProbe {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    bind_filled: wgpu::BindGroup,
    bind_raw: wgpu::BindGroup,
    counts: wgpu::Buffer,
    query: Option<wgpu::QuerySet>,
    resolve: wgpu::Buffer,
    slots: Vec<Slot>,
    period_ns: f32,
    pub samples: Vec<ProbeSample>,
    pub skipped: u64,
    size: [u32; 2],
}
impl FrameProbe {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, raw: &wgpu::TextureView, filled: &wgpu::TextureView, size: [u32; 2]) -> Self {
        let texture = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("Coverage probe"), entries: &[
            texture(0), texture(1), wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new(8) }, count: None },
        ] });
        let counts = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Coverage counters"), size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let bind_filled = Self::bind(device, &layout, &counts, raw, filled);
        let bind_raw = Self::bind(device, &layout, &counts, raw, raw);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Coverage probe"), source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/coverage.wgsl").into()) });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("Coverage probe"), bind_group_layouts: &[&layout], push_constant_ranges: &[] });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Coverage probe"), layout: Some(&pipeline_layout), module: &module, entry_point: "main", compilation_options: Default::default() });
        let query = device.features().contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS).then(|| device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("Frame timestamps"), ty: wgpu::QueryType::Timestamp, count: 6 }));
        let resolve = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Frame timestamp resolve"), size: 256,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let slots = (0..3).map(|_| Slot { buffer: device.create_buffer(&wgpu::BufferDescriptor { label: Some("Frame probe readback"), size: 32,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }), state: Arc::new(AtomicU8::new(0)), frame: 0 }).collect();
        Self { pipeline, layout, bind_filled, bind_raw, counts, query, resolve, slots, period_ns: queue.get_timestamp_period(), samples: vec![], skipped: 0, size }
    }
    fn bind(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, counts: &wgpu::Buffer, raw: &wgpu::TextureView, filled: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("Coverage probe inputs"), layout, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(raw) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(filled) },
            wgpu::BindGroupEntry { binding: 2, resource: counts.as_entire_binding() },
        ] })
    }
    pub fn resize(&mut self, device: &wgpu::Device, raw: &wgpu::TextureView, filled: &wgpu::TextureView, size: [u32; 2]) {
        self.bind_filled = Self::bind(device, &self.layout, &self.counts, raw, filled);
        self.bind_raw = Self::bind(device, &self.layout, &self.counts, raw, raw); self.size = size;
    }
    pub fn poll(&mut self, device: &wgpu::Device, wait: bool) {
        device.poll(if wait { wgpu::Maintain::Wait } else { wgpu::Maintain::Poll });
        for slot in &mut self.slots {
            let state = slot.state.load(Ordering::Acquire);
            if state == 2 {
                let bytes = slot.buffer.slice(..).get_mapped_range();
                let raw = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
                let display = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
                let a = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
                let b = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
                self.samples.push(ProbeSample { frame: slot.frame, raw_pixels: raw, display_pixels: display,
                    gpu_ms: self.query.as_ref().map(|_| b.saturating_sub(a) as f64 * self.period_ns as f64 * 1e-6) });
                drop(bytes); slot.buffer.unmap(); slot.state.store(0, Ordering::Release);
            } else if state == 3 {
                log::error!("GPU probe readback failed for frame {}", slot.frame);
                slot.state.store(0, Ordering::Release); self.skipped += 1;
            }
        }
    }
    pub fn begin(&mut self, encoder: &mut wgpu::CommandEncoder, frame: u64) -> Option<usize> {
        let Some(index) = self.slots.iter().position(|s| s.state.load(Ordering::Acquire) == 0) else { self.skipped += 1; return None; };
        self.slots[index].frame = frame;
        if let Some(query) = &self.query { encoder.write_timestamp(query, index as u32 * 2); }
        Some(index)
    }
    pub fn end(&self, encoder: &mut wgpu::CommandEncoder, index: usize, filled: bool) {
        if let Some(query) = &self.query {
            encoder.write_timestamp(query, index as u32 * 2 + 1);
            encoder.resolve_query_set(query, index as u32 * 2..index as u32 * 2 + 2, &self.resolve, 0);
            encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.slots[index].buffer, 16, 16);
        }
        encoder.clear_buffer(&self.counts, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Coverage audit"), timestamp_writes: None });
            pass.set_pipeline(&self.pipeline); pass.set_bind_group(0, if filled { &self.bind_filled } else { &self.bind_raw }, &[]);
            pass.dispatch_workgroups(self.size[0].div_ceil(16), self.size[1].div_ceil(16), 1);
        }
        encoder.copy_buffer_to_buffer(&self.counts, 0, &self.slots[index].buffer, 0, 8);
    }
    pub fn submitted(&self, index: usize) {
        let state = self.slots[index].state.clone(); state.store(1, Ordering::Release);
        self.slots[index].buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| { state.store(if r.is_ok() { 2 } else { 3 }, Ordering::Release); });
    }
}
