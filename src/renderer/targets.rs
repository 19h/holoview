// src/renderer/targets.rs
//! Manages all render target textures (framebuffers).

use super::{
    context::GpuContext,
    pipelines::postprocess::{CrtPass, EdlPass, RgbShiftPass},
};

pub const SAMPLE_COUNT: u32 = 4;

pub struct RenderTargets {
    // Scene (multisampled)
    pub scene_color: wgpu::Texture,
    pub scene_color_view: wgpu::TextureView,
    pub scene_depthlin: wgpu::Texture,
    pub scene_depthlin_view: wgpu::TextureView,
    pub depth: wgpu::Texture,
    pub depth_view: wgpu::TextureView,

    // Resolved (single-sampled) for post-processing
    pub scene_color_resolved: wgpu::Texture,
    pub scene_color_resolved_view: wgpu::TextureView,
    pub scene_depthlin_resolved: wgpu::Texture,
    pub scene_depthlin_resolved_view: wgpu::TextureView,

    // Post-processing chain
    pub post_edl: wgpu::Texture,
    pub post_edl_view: wgpu::TextureView,
    pub post_rgb: wgpu::Texture,
    pub post_rgb_view: wgpu::TextureView,

    // Samplers
    pub linear_samp: wgpu::Sampler,
    pub nearest_samp: wgpu::Sampler,
}

impl RenderTargets {
    pub fn new(gpu: &GpuContext) -> Self {
        let size = wgpu::Extent3d {
            width: gpu.config.width,
            height: gpu.config.height,
            depth_or_array_layers: 1,
        };

        let make_tex = |label, fmt, count, usage| {
            let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: count,
                dimension: wgpu::TextureDimension::D2,
                format: fmt,
                usage,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };

        let (scene_color, scene_color_view) =
            make_tex("SceneColor MS", gpu.surface_format, SAMPLE_COUNT, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let (scene_depthlin, scene_depthlin_view) = make_tex(
            "LinearDepth MS",
            wgpu::TextureFormat::Rgba16Float,
            SAMPLE_COUNT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let (depth, depth_view) =
            make_tex("Depth MS", wgpu::TextureFormat::Depth32Float, SAMPLE_COUNT, wgpu::TextureUsages::RENDER_ATTACHMENT);

        let tex_usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let (scene_color_resolved, scene_color_resolved_view) =
            make_tex("SceneColor Resolved", gpu.surface_format, 1, tex_usage);
        let (scene_depthlin_resolved, scene_depthlin_resolved_view) =
            make_tex("LinearDepth Resolved", wgpu::TextureFormat::Rgba16Float, 1, tex_usage);
        let (post_edl, post_edl_view) = make_tex("PostEDL", gpu.surface_format, 1, tex_usage);
        let (post_rgb, post_rgb_view) = make_tex("PostRGB", gpu.surface_format, 1, tex_usage);

        let linear_samp = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let nearest_samp = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Nearest"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            scene_color,
            scene_color_view,
            scene_depthlin,
            scene_depthlin_view,
            depth,
            depth_view,
            scene_color_resolved,
            scene_color_resolved_view,
            scene_depthlin_resolved,
            scene_depthlin_resolved_view,
            post_edl,
            post_edl_view,
            post_rgb,
            post_rgb_view,
            linear_samp,
            nearest_samp,
        }
    }

    pub fn bind_post_processing_inputs(
        &self,
        edl: &mut EdlPass,
        rgb: &mut RgbShiftPass,
        crt: &mut CrtPass,
        device: &wgpu::Device,
    ) {
        edl.set_inputs(
            device,
            &self.scene_color_resolved_view,
            &self.scene_depthlin_resolved_view,
            &self.nearest_samp,
        );

        rgb.set_input(
            device,
            &self.post_edl_view,
            &self.scene_depthlin_resolved_view,
            &self.linear_samp,
        );

        crt.set_inputs(
            device,
            &self.post_rgb_view,
            &self.scene_depthlin_resolved_view,
            &self.nearest_samp,
        );
    }
}
