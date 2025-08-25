// src/main.rs
mod point_cloud;
mod holographic_shaders;
mod post;
mod ground_grid;

use anyhow::Result;
use glam::{Mat4, Vec3, Vec4, Vec4Swizzles};
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::{
    dpi::LogicalSize,
    event::{ElementState, Event, MouseButton, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

use holographic_shaders::{HoloPipelines, HoloUniforms};
use point_cloud::QuantizedPointCloud;
use post::{EdlPass, EdlUniforms, RgbShiftPass, RgbShiftUniforms, CrtPass, CrtUniforms};

// WGPU (Vulkan/D3D) clip-space conversion for a GL-style projection
const OPENGL_TO_WGPU_MATRIX: Mat4 = Mat4::from_cols_array(&[
    1.0,  0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,  // flip Y
    0.0,  0.0, 0.5, 0.0,  // map z: [-1,1] -> [0,1]
    0.0,  0.0, 0.5, 1.0,
]);

// Multi-sampling sample count for anti-aliasing
const SAMPLE_COUNT: u32 = 4;

struct Gpu {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: wgpu::Adapter,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    surface_format: wgpu::TextureFormat,
}

struct Targets {
    scene_color: wgpu::Texture,
    scene_color_view: wgpu::TextureView,
    scene_color_resolved: wgpu::Texture,
    scene_color_resolved_view: wgpu::TextureView,
    scene_depthlin: wgpu::Texture,
    scene_depthlin_view: wgpu::TextureView,
    scene_depthlin_resolved: wgpu::Texture,
    scene_depthlin_resolved_view: wgpu::TextureView,
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    post_edl: wgpu::Texture,
    post_edl_view: wgpu::TextureView,
    post_rgb: wgpu::Texture,
    post_rgb_view: wgpu::TextureView,
    linear_samp: wgpu::Sampler,
    nearest_samp: wgpu::Sampler,
}

struct App {
    gpu: Gpu,
    window: Arc<Window>,

    // Overlay (egui) — only for HUD (no windows)
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,

    // Draw pipelines
    holo: HoloPipelines,
    edl: EdlPass,
    rgb: RgbShiftPass,
    crt: CrtPass,
    grid: ground_grid::GroundGrid,

    // Data
    instances: Option<wgpu::Buffer>,
    instance_count: u32,
    cloud: Option<QuantizedPointCloud>,

    sem_tex: Option<wgpu::Texture>,
    sem_view: Option<wgpu::TextureView>,
    sem_samp: Option<wgpu::Sampler>,

    // Targets
    tgt: Targets,

    // Camera
    cam_pos: Vec3,
    cam_target: Vec3,
    cam_up: Vec3,
    fov_y: f32,
    near: f32,
    far: f32,

    // Mouse/orbit
    mouse_down: bool,
    last_mouse: Option<(f64, f64)>,

    // Time
    start: Instant,

    // Grid
    grid_enabled: bool,
}

impl Gpu {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance =
            wgpu::Instance::new(
                wgpu::InstanceDescriptor::default(),
            );

        let surface =
            instance
                .create_surface(
                    window.clone(),
                )
                .unwrap();

        let adapter =
            instance
                .request_adapter(
                    &wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: Some(&surface),
                        force_fallback_adapter: false,
                    }
                )
                .await
                .unwrap();

        let (device, queue) =
            adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None
            )
            .await
            .unwrap();

        let caps =
            surface
                .get_capabilities(
                    &adapter,
                );

        let surface_format =
            caps.formats
                .iter()
                .copied()
                .find(|f| f.is_srgb())
                .unwrap_or(caps.formats[0]);

        let config =
            wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: size.width,
                height: size.height,
                present_mode: caps.present_modes[0],
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            };

        surface.configure(
            &device,
            &config,
        );

        Self {
            surface,
            device,
            queue,
            adapter,
            config,
            size,
            surface_format,
        }
    }
}

impl Targets {
    fn new(gpu: &Gpu) -> Self {
        // Helper for single-sample textures (post-processing chain)
        let make_single_sample =
            |label: &str, fmt: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
                let t =
                    gpu.device
                        .create_texture(
                            &wgpu::TextureDescriptor {
                                label: Some(label),
                                size: wgpu::Extent3d {
                                    width: gpu.config.width,
                                    height: gpu.config.height,
                                    depth_or_array_layers: 1,
                                },
                                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                                format: fmt, usage, view_formats: &[],
                            }
                        );

                let v =
                    t.create_view(
                        &wgpu::TextureViewDescriptor::default(),
                    );

                (t, v)
            };

        // Helper for multi-sample textures (scene rendering)
        let make_multisampled =
            |label: &str, fmt: wgpu::TextureFormat| {
                let t =
                    gpu.device
                        .create_texture(
                            &wgpu::TextureDescriptor {
                                label: Some(label),
                                size: wgpu::Extent3d {
                                    width: gpu.config.width,
                                    height: gpu.config.height,
                                    depth_or_array_layers: 1,
                                },
                                mip_level_count: 1, sample_count: SAMPLE_COUNT, dimension: wgpu::TextureDimension::D2,
                                format: fmt,
                                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                                view_formats: &[],
                            }
                        );

                let v =
                    t.create_view(
                        &wgpu::TextureViewDescriptor::default(),
                    );

                (t, v)
            };

        // Multisampled scene targets
        let (scene_color, scene_color_view) = make_multisampled("SceneColor MS", gpu.surface_format);
        let (scene_depthlin, scene_depthlin_view) = make_multisampled("LinearDepth MS", wgpu::TextureFormat::Rgba16Float);

        // Resolved (single-sample) targets for post-processing to read from
        let (scene_color_resolved, scene_color_resolved_view) = make_single_sample(
            "SceneColor Resolved",
            gpu.surface_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let (scene_depthlin_resolved, scene_depthlin_resolved_view) = make_single_sample(
            "LinearDepth Resolved",
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        // Multisampled depth buffer
        let depth =
            gpu.device
                .create_texture(
                    &wgpu::TextureDescriptor {
                        label: Some("Depth MS"),
                        size: wgpu::Extent3d {
                            width: gpu.config.width,
                            height: gpu.config.height,
                            depth_or_array_layers: 1
                        },
                        mip_level_count: 1,
                        sample_count: SAMPLE_COUNT,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Depth32Float,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        view_formats: &[],
                    }
                );

        let depth_view =
            depth.create_view(
                &wgpu::TextureViewDescriptor::default(),
            );

        // Post-processing chain buffers (single-sample)
        let (post_edl, post_edl_view) = make_single_sample(
            "PostEDL",
            gpu.surface_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let (post_rgb, post_rgb_view) = make_single_sample(
            "PostRGB",
            gpu.surface_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let linear_samp =
            gpu.device
                .create_sampler(
                    &wgpu::SamplerDescriptor {
                        label: Some("Linear"),
                        mag_filter: wgpu::FilterMode::Linear,
                        min_filter: wgpu::FilterMode::Linear,
                        ..Default::default()
                    }
                );

        let nearest_samp =
            gpu.device
                .create_sampler(
                    &wgpu::SamplerDescriptor {
                        label: Some("Nearest"),
                        mag_filter: wgpu::FilterMode::Nearest,
                        min_filter: wgpu::FilterMode::Nearest,
                        ..Default::default()
                    }
                );

        Self {
            scene_color,
            scene_color_view,
            scene_color_resolved,
            scene_color_resolved_view,
            scene_depthlin,
            scene_depthlin_view,
            scene_depthlin_resolved,
            scene_depthlin_resolved_view,
            depth,
            depth_view,
            post_edl,
            post_edl_view,
            post_rgb,
            post_rgb_view,
            linear_samp,
            nearest_samp
        }
    }

    fn resize(&mut self, gpu: &Gpu) {
        *self = Self::new(gpu);
    }
}

impl App {
    async fn new(window: Arc<Window>) -> Self {
        let gpu =
            Gpu::new(
                window.clone(),
            )
            .await;

        // egui (used only for HUD lines/text; no windows)
        let egui_ctx = egui::Context::default();

        let egui_state =
            egui_winit::State::new(
                egui_ctx.clone(),
                egui_ctx.viewport_id(),
                &*window,
                None,
                None,
            );

        let egui_renderer =
            egui_wgpu::Renderer::new(
                &gpu.device,
                gpu.surface_format,
                None,
                1,
            );

        let tgt = Targets::new(&gpu);

        // pipelines
        let holo =
            HoloPipelines::new(
                &gpu.device,
                gpu.surface_format,
                wgpu::TextureFormat::Rgba16Float,
            );

        let mut edl =
            EdlPass::new(
                &gpu.device,
                gpu.surface_format,
            );

        edl.set_inputs(
            &gpu.device,
            &tgt.scene_color_resolved_view,  // Use resolved texture instead of multisampled
            &tgt.scene_depthlin_resolved_view,  // Use resolved texture instead of multisampled
            &tgt.nearest_samp,
        );

        let mut rgb =
            RgbShiftPass::new(
                &gpu.device,
                gpu.surface_format,
            );

        rgb.set_input(
            &gpu.device,
            &tgt.post_edl_view,
            &tgt.scene_depthlin_resolved_view,  // Use resolved texture
            &tgt.linear_samp,
        );

        let mut crt =
            CrtPass::new(
                &gpu.device,
                gpu.surface_format,
            );

        crt.set_inputs(
            &gpu.device,
            &tgt.post_rgb_view,
            &tgt.scene_depthlin_resolved_view,  // Use resolved texture
            &tgt.nearest_samp,
        );

        let grid =
            ground_grid::GroundGrid::new(
                &gpu.device,
                gpu.surface_format,
                wgpu::TextureFormat::Rgba16Float,
            );

        Self {
            gpu,
            window,
            egui_ctx,
            egui_state,
            egui_renderer,
            holo,
            edl,
            rgb,
            crt,
            grid,
            instances: None,
            instance_count: 0,
            cloud: None,
            sem_tex: None,
            sem_view: None,
            sem_samp: None,
            tgt,
            // camera (Z-up)
            cam_pos:    Vec3::new(0.0, -350.0, 220.0), // south & above
            cam_target: Vec3::new(0.0, 0.0, 0.0),
            cam_up:     Vec3::new(0.0, 0.0, 1.0),
            fov_y: 55.0_f32.to_radians(),
            near: 0.5,
            far: 8000.0,
            mouse_down: false,
            last_mouse: None,
            start: Instant::now(),
            grid_enabled: true,
        }
    }

    fn resolve_default_cloud() -> Option<String> {
        let candidates = [
            "../hypc/Tile-58-52-1-1.hypc",
            "hypc/Tile-58-52-1-1.hypc",
            "../hypc/Tile-58-52-1-1.ply",
            "hypc/Tile-58-52-1-1.ply",
        ];

        for p in candidates {
            if std::path::Path::new(p).exists() {
                return Some(p.to_string());
            }
        }

        None
    }

    fn load_cloud_into_gpu(&mut self, cloud: QuantizedPointCloud) -> Result<()> {
        use holographic_shaders::Instance;

        // instance buffer
        let instances: Vec<Instance> =
            cloud.positions
                .iter()
                .map(|p| Instance {
                    position: [p.x, p.y, p.z],
                })
                .collect();

        let vb =
            self.gpu.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Instances"),
                    contents: bytemuck::cast_slice(&instances),
                    usage: wgpu::BufferUsages::VERTEX,
                }
            );

        self.instances = Some(vb);
        self.instance_count = cloud.kept as u32;
        self.cloud = Some(cloud);

        // frame camera (Z-up) from decode bounds
        if let Some(pc) = &self.cloud {
            let size = pc.decode_max - pc.decode_min;

            let max_dim =
                size.x
                    .max(size.y)
                    .max(size.z);

            let dist = max_dim * 1.2;

            self.cam_up     =
                glam::Vec3::new(
                    0.0,
                    0.0,
                    1.0,
                );

            self.cam_target =
                (pc.decode_min + pc.decode_max)
                * 0.5;

            self.cam_pos =
                self.cam_target
                + glam::Vec3::new(
                    dist * 0.25,
                    -dist * 1.35,
                    dist * 0.9,
                );
        }

        // --- upload SMC1 if present ---
        // Extract semantic mask to avoid borrowing conflicts
        let semantic_mask_data = self.cloud.as_ref().and_then(|pc| pc.semantic_mask.clone());
        if let Some(semantic_mask) = semantic_mask_data.as_ref() {
            self.upload_semantic_mask(semantic_mask);
        }

        Ok(())
    }

    pub fn load_ply(&mut self, path: &str, point_budget: usize, target_extent: f32) -> Result<()> {
        self.load_cloud_into_gpu(
            point_cloud::QuantizedPointCloud::load_ply_quantized(
                path,
                point_budget,
                target_extent
            )?
        )
    }

    pub fn load_hypc(&mut self, path: &str) -> Result<()> {
        self.load_cloud_into_gpu(
            point_cloud::QuantizedPointCloud::load_hypc(
                path,
            )?
        )
    }

    fn update_uniforms(&mut self) {
        let view =
            Mat4::look_at_rh(
                self.cam_pos,
                self.cam_target,
                self.cam_up,
            );

        let proj_gl =
            Mat4::perspective_rh_gl(
                self.fov_y,
                self.gpu.config.width as f32 / self.gpu.config.height as f32,
                self.near,
                self.far,
            );

        let proj = OPENGL_TO_WGPU_MATRIX * proj_gl;

        let time =
            self.start
                .elapsed()
                .as_secs_f32();

        let (dmin, dmax) =
            if let Some(pc) = &self.cloud {
                (pc.decode_min, pc.decode_max)
            } else {
                (Vec3::ZERO, Vec3::ONE)
            };

        let u = HoloUniforms {
            view, proj,
            viewport: [
                self.gpu.config.width as f32,
                self.gpu.config.height as f32,
            ],
            base_size_px: self.holo.uniforms.base_size_px,
            size_atten: self.holo.uniforms.size_atten,
            time,
            near: self.near,
            far: self.far,
            _pad: 0.0,
            decode_min: dmin,
            decode_max: dmax,
            cyan: self.holo.uniforms.cyan,
            red:  self.holo.uniforms.red,
            ..Default::default()
        };

        self.holo.update_uniforms(
            &self.gpu.queue,
            u,
        );

        self.edl.update_uniforms(
            &self.gpu.queue,
            EdlUniforms {
                inv_size: [
                    1.0 / self.gpu.config.width as f32,
                    1.0 / self.gpu.config.height as f32,
                ],
                strength: 1.15,
                radius_px: 1.25,
            },
        );

        self.rgb.update_uniforms(
            &self.gpu.queue,
            RgbShiftUniforms {
                inv_size: [
                    1.0 / self.gpu.config.width as f32,
                    1.0 / self.gpu.config.height as f32,
                ],
                amount: 0.0018,
                angle: 0.0,
            },
        );

        self.crt.update_uniforms(
            &self.gpu.queue,
            CrtUniforms {
                inv_size: [
                    1.0 / self.gpu.config.width as f32,
                    1.0 / self.gpu.config.height as f32,
                ],
                time,
                intensity: self.crt.uniform.intensity,
                vignette: self.crt.uniform.vignette,
                _padding: 0.0,
            },
        );

        // Update grid uniforms
        let mut enable_grid = 0u32;
        let (lon_min, lat_min, lon_span, lat_span) = if let Some(pc) = &self.cloud {
            if let Some(bb) = &pc.geog_bbox_deg {
                enable_grid = if self.grid_enabled { 1 } else { 0 };
                (bb.lon_min as f32,
                 bb.lat_min as f32,
                 (bb.lon_max - bb.lon_min) as f32,
                 (bb.lat_max - bb.lat_min) as f32)
            } else { (0.0, 0.0, 0.0, 0.0) }
        } else { (0.0, 0.0, 0.0, 0.0) };

        // place the plane slightly below the data to avoid z-fighting
        let extent = (dmax - dmin).abs();
        let z_ofs = (extent.z.max(1e-6)) * 0.02;

        self.grid.update_uniforms(
            &self.gpu.queue,
            ground_grid::GridUniforms {
                view, proj,
                decode_min: dmin, _pad0: 0.0,
                decode_max: dmax, _pad1: 0.0,
                geot_lon_min: lon_min,
                geot_lat_min: lat_min,
                geot_lon_span: lon_span,
                geot_lat_span: lat_span,
                extent_mul: 12.0,          // larger => "more infinite"
                z_offset: z_ofs,
                opacity: 0.55,
                enabled: enable_grid,
                color_minor: glam::Vec3::new(0.90, 0.15, 0.15),
                color_major: glam::Vec3::new(1.00, 0.20, 0.20),
                _pad2: 0.0, _pad3: 0.0,
            }
        );
    }

    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }

        self.gpu.size = size;
        self.gpu.config.width = size.width;
        self.gpu.config.height = size.height;

        self.gpu.surface.configure(
            &self.gpu.device,
            &self.gpu.config,
        );

        self.tgt.resize(
            &self.gpu,
        );

        // rebind post inputs with resolved textures
        self.edl.set_inputs(
            &self.gpu.device,
            &self.tgt.scene_color_resolved_view,  // Use resolved texture
            &self.tgt.scene_depthlin_resolved_view,  // Use resolved texture
            &self.tgt.nearest_samp,
        );

        self.rgb.set_input(
            &self.gpu.device,
            &self.tgt.post_edl_view,
            &self.tgt.scene_depthlin_resolved_view,  // Use resolved texture
            &self.tgt.linear_samp,
        );

        self.crt.set_inputs(
            &self.gpu.device,
            &self.tgt.post_rgb_view,
            &self.tgt.scene_depthlin_resolved_view,  // Use resolved texture
            &self.tgt.nearest_samp,
        );
    }

    fn snap_north_up(&mut self) {
        self.cam_up =
            Vec3::new(
                0.0,
                0.0,
                1.0,
            );

        if let Some(pc) = &self.cloud {
            let size =
                pc.decode_max
                - pc.decode_min;

            let d =
                size.x
                    .max(size.y)
                    .max(size.z)
                * 1.2;

            self.cam_target =
                (
                    pc.decode_min
                    + pc.decode_max
                )
                * 0.5;

            self.cam_pos =
                self.cam_target
                + Vec3::new(
                    0.0,
                    -d * 1.35,
                    d * 0.9,
                );
        }
    }

    fn clamp_pitch(dir: Vec3) -> Vec3 {
        // avoid flipping over (keep ~[-85°, +85°] from horizon)
        let min_z = 0.05;

        if dir.z.abs() < min_z {
            let _s = (min_z / dir.z.abs()).max(1.0);

            return Vec3::new(
                dir.x,
                dir.y,
                dir.z.signum() * min_z,
            ).normalize();
        }

        dir
    }

    fn handle_mouse_button(&mut self, btn: MouseButton, state: ElementState) {
        if btn == MouseButton::Left {
            self.mouse_down =
                state == ElementState::Pressed;
        }
    }

    fn handle_cursor(&mut self, xy: (f64, f64)) {
        if let Some(last) = self.last_mouse {
            if self.mouse_down {
                let dx = (xy.0 - last.0) as f32 * 0.01;
                let dy = (xy.1 - last.1) as f32 * 0.01;

                let to_target =
                    (
                        self.cam_target
                        - self.cam_pos
                    ).normalize();

                let right =
                    to_target
                        .cross(
                            self.cam_up
                        )
                        .normalize();

                let yaw =
                    Mat4::from_axis_angle(
                        self.cam_up,
                        -dx,
                    );

                let mut dir =
                    yaw
                        .mul_vec4(
                            Vec4::from(
                                (to_target, 0.0)
                            )
                        )
                        .xyz();

                let pitch =
                    Mat4::from_axis_angle(
                        right,
                        -dy,
                    );

                dir =
                    pitch
                        .mul_vec4(
                            Vec4::from((dir, 0.0))
                        )
                        .xyz();

                dir =
                    Self::clamp_pitch(
                        dir,
                    );

                let dist =
                    (
                        self.cam_pos
                        - self.cam_target
                    ).length();

                self.cam_pos =
                    self.cam_target
                    - dir
                    * dist;
            }
        }

        self.last_mouse = Some(xy);
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.update_uniforms();

        let frame =
            self.gpu
                .surface
                .get_current_texture()?;

        let swap_view =
            frame
                .texture
                .create_view(
                    &wgpu::TextureViewDescriptor::default()
                );

        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("Frame"),
                    },
                );

        // PASS 1: points → scene targets (MSAA with resolve)
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Scene"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.tgt.scene_color_view,  // Draw to multisampled texture
                        resolve_target: Some(&self.tgt.scene_color_resolved_view),  // Resolve to single-sample texture
                        ops: wgpu::Operations {
                            load:
                                wgpu::LoadOp::Clear(
                                    wgpu::Color {
                                        r: 0.0196,
                                        g: 0.0196,
                                        b: 0.0275,
                                        a: 1.0,
                                    }
                                ),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.tgt.scene_depthlin_view,  // Draw to multisampled texture
                        resolve_target: Some(&self.tgt.scene_depthlin_resolved_view),  // Resolve to single-sample texture
                        ops: wgpu::Operations {
                            load:
                                wgpu::LoadOp::Clear(
                                    wgpu::Color::WHITE,
                                ),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(
                    wgpu::RenderPassDepthStencilAttachment {
                        view: &self.tgt.depth_view,
                        depth_ops: Some(
                            wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store
                            }
                        ),
                        stencil_ops: None,
                    }
                ),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // --- draw ground grid first ---
            rp.set_pipeline(&self.grid.pipeline);
            rp.set_bind_group(0, &self.grid.bind_group, &[]);
            rp.set_vertex_buffer(0, self.grid.quad_vb.slice(..));
            rp.draw(0..6, 0..1);

            // then the points (unchanged)
            if let (Some(inst), n) = (&self.instances, self.instance_count) {
                rp.set_pipeline(&self.holo.pipeline);
                rp.set_bind_group(0, &self.holo.bind_group, &[]);
                rp.set_vertex_buffer(0, self.holo.quad_vb.slice(..));
                rp.set_vertex_buffer(1, inst.slice(..));

                rp.draw(0..6, 0..n);
            }
        }

        // PASS 2: EDL(color, depthlin) → post_edl
        {
            let mut rp =
                encoder.begin_render_pass(
                    &wgpu::RenderPassDescriptor {
                        label: Some("EDL"),
                        color_attachments: &[
                            Some(wgpu::RenderPassColorAttachment {
                                view: &self.tgt.post_edl_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                        ],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    },
                );

            rp.set_pipeline(&self.edl.pipeline);
            rp.set_bind_group(0, &self.edl.bind_group, &[]);
            rp.set_vertex_buffer(0, self.edl.fsq_vb.slice(..));

            rp.draw(0..3, 0..1);
        }

        // PASS 3: RGB shift → post_rgb
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("RGBShift"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.tgt.post_rgb_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rp.set_pipeline(&self.rgb.pipeline);
            rp.set_bind_group(0, &self.rgb.bind_group, &[]);
            rp.set_vertex_buffer(0, self.rgb.fsq_vb.slice(..));

            rp.draw(0..3, 0..1);
        }

        // PASS 4: CRT / scanlines → swapchain
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("CRT"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &swap_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rp.set_pipeline(&self.crt.pipeline);
            rp.set_bind_group(0, &self.crt.bind_group, &[]);
            rp.set_vertex_buffer(0, self.crt.fsq_vb.slice(..));

            rp.draw(0..3, 0..1);
        }

        // HUD (no windows): status text + corner brackets + top-center dot
        self.draw_hud(
            &mut encoder,
            &swap_view,
        );

        self.gpu.queue.submit(
            std::iter::once(
                encoder.finish(),
            ),
        );

        frame.present();

        Ok(())
    }

    fn draw_hud(&mut self, encoder: &mut wgpu::CommandEncoder, swap_view: &wgpu::TextureView) {
        let eg_in =
            self.egui_state
                .take_egui_input(
                    &self.window,
                );

        self.egui_ctx
            .begin_frame(
                eg_in,
            );

        // Corner brackets & dot painter
        {
            let painter =
                self.egui_ctx.layer_painter(
                    egui::LayerId::new(
                        egui::Order::Foreground,
                        egui::Id::new("hud_lines"),
                    ),
                );

            let rect =
                self.egui_ctx
                    .screen_rect();

            let c =
                egui::Color32::from_rgba_unmultiplied(
                    45,
                    247,
                    255,
                    200,
                );

            let thick = 2.0;
            let m = 26.0;
            let len = 140.0;

            // TL
            painter.line_segment([egui::pos2(m, m), egui::pos2(m + len, m)], (thick, c));
            painter.line_segment([egui::pos2(m, m), egui::pos2(m, m + len)], (thick, c));

            // TR
            painter.line_segment([egui::pos2(rect.max.x - m - len, m), egui::pos2(rect.max.x - m, m)], (thick, c));
            painter.line_segment([egui::pos2(rect.max.x - m, m), egui::pos2(rect.max.x - m, m + len)], (thick, c));

            // BL
            painter.line_segment([egui::pos2(m, rect.max.y - m), egui::pos2(m + len, rect.max.y - m)], (thick, c));
            painter.line_segment([egui::pos2(m, rect.max.y - m - len), egui::pos2(m, rect.max.y - m)], (thick, c));

            // BR
            painter.line_segment([egui::pos2(rect.max.x - m - len, rect.max.y - m), egui::pos2(rect.max.x - m, rect.max.y - m)], (thick, c));
            painter.line_segment([egui::pos2(rect.max.x - m, rect.max.y - m - len), egui::pos2(rect.max.x - m, rect.max.y - m)], (thick, c));

            // top-center small dot
            painter.circle_filled(egui::pos2(rect.center().x, 16.0), 3.0, c);
        }

        // Top-left status text (monospace, cyan)
        {
            use egui::{Area, Frame, RichText};
            Area::new("hud_text".into())
                .interactable(false)
                .movable(false)
                .order(egui::Order::Foreground)
                .fixed_pos(egui::pos2(40.0, 42.0))
                .show(&self.egui_ctx, |ui| {
                    Frame::none().show(ui, |ui| {
                        ui.label(
                            RichText::new("HOLOGRAPHIC  SCAN  ACTIVE")
                                .monospace()
                                .color(egui::Color32::from_rgb(45,247,255))
                                .size(16.0)
                                .strong()
                        );

                        let pts = self.instance_count;
                        let alt = self.cam_pos.z.round() as i32;

                        ui.label(
                            RichText::new(format!("RESOLUTION: {:>11} POINTS", pts))
                                .monospace()
                                .color(egui::Color32::from_rgb(45,247,255))
                        );

                        ui.label(
                            RichText::new(format!("ALTITUDE: {}M", alt))
                                .monospace()
                                .color(egui::Color32::from_rgb(45,247,255))
                        );

                        ui.label(
                            RichText::new("STATUS:  SCAN  COMPLETE")
                                .monospace()
                                .color(egui::Color32::from_rgb(45,247,255))
                        );
                    });
                });
        }


        // draw egui to the swapchain
        let out =
            self.egui_ctx
                .end_frame();

        let shapes =
            self.egui_ctx
                .tessellate(
                    out.shapes,
                    self.egui_ctx.pixels_per_point(),
                );

        let screen =
            egui_wgpu::ScreenDescriptor {
                size_in_pixels: [
                    self.gpu.config.width,
                    self.gpu.config.height,
                ],
                pixels_per_point:
                    self.egui_state
                        .egui_ctx()
                        .pixels_per_point(),
            };

        for (id, delta) in &out.textures_delta.set {
            self.egui_renderer
                .update_texture(
                    &self.gpu.device,
                    &self.gpu.queue,
                    *id,
                    delta,
                );
        }

        self.egui_renderer
            .update_buffers(
                &self.gpu.device,
                &self.gpu.queue,
                encoder,
                &shapes,
                &screen,
            );

        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("HUD"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: swap_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            self.egui_renderer
                .render(
                    &mut rp,
                    &shapes,
                    &screen,
                );
        }

        for id in &out.textures_delta.free {
            self.egui_renderer
                .free_texture(
                    id,
                );
        }
    }

    fn upload_semantic_mask(&mut self, sm: &point_cloud::SemanticMask) {
        let w = sm.width as u32;
        let h = sm.height as u32;

        // Create R8Unorm texture for labels
        let tex = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SMC1 Labels"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Write with 256-byte row pitch
        let bytes_per_row = ((w + 255) / 256) * 256;
        let mut staging = vec![0u8; (bytes_per_row * h) as usize];
        for row in 0..h as usize {
            let src = &sm.data[row * w as usize .. (row + 1) * w as usize];
            let dst = &mut staging[row * bytes_per_row as usize .. row * bytes_per_row as usize + w as usize];
            dst.copy_from_slice(src);
        }

        self.gpu.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &staging,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );

        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let samp = self.gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SMC1 Nearest"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // Rebind the Holo pipeline with this texture/sampler
        self.holo.set_semantics_inputs(&self.gpu.device, &view, &samp);

        // Keep them alive
        self.sem_tex = Some(tex);
        self.sem_view = Some(view);
        self.sem_samp = Some(samp);
    }
}

fn main() -> Result<()> {
    env_logger::init();

    let event_loop = EventLoop::new()?;

    let window =
        Arc::new(
            WindowBuilder::new()
                .with_title("Holographic City Viewer — Rust")
                .with_inner_size(
                    LogicalSize::new(
                        1280,
                        720,
                    )
                )
                .build(&event_loop)?
        );

    let mut app =
        pollster::block_on(
            App::new(
                window.clone(),
            ),
        );

    // Auto-load the point cloud (no drag&drop needed)
    if let Some(p) = App::resolve_default_cloud() {
        match std::path::Path::new(&p).extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
            Some(ref ext) if ext == "hypc" => { let _ = app.load_hypc(&p); }
            Some(ref ext) if ext == "ply"  => { let _ = app.load_ply(&p, 1_200_000, 200.0); }
            _ => {}
        }
    }

    event_loop.run(
        move |event, elwt| {
            elwt.set_control_flow(
                winit::event_loop::ControlFlow::Poll
            );

            match event {
                Event::WindowEvent { window_id, event }
                    if window_id == window.id() => {
                        // let egui consume events first
                        let eg =
                            app
                                .egui_state
                                .on_window_event(
                                    &window,
                                    &event,
                                );

                        if eg.consumed {
                            return;
                        }

                        match event {
                            WindowEvent::CloseRequested => elwt.exit(),
                            WindowEvent::Resized(sz) => app.resize(sz),
                            WindowEvent::RedrawRequested => {
                                let _ =
                                    app
                                        .render()
                                        .map_err(|e|
                                            match e {
                                                wgpu::SurfaceError::Lost =>
                                                    app.resize(
                                                        app.gpu.size,
                                                    ),
                                                wgpu::SurfaceError::OutOfMemory =>
                                                    elwt.exit(),
                                                other =>
                                                    eprintln!("{other:?}"),
                                            }
                                        );
                            }
                            WindowEvent::KeyboardInput { event, .. } => {
                                if event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                                    elwt.exit();
                                }

                                if event.physical_key == PhysicalKey::Code(KeyCode::KeyN)
                                && event.state == ElementState::Pressed {
                                    app.snap_north_up();
                                }
                                // NEW: toggle grid
                                if event.physical_key == PhysicalKey::Code(KeyCode::KeyG)
                                && event.state == ElementState::Pressed {
                                    app.grid_enabled = !app.grid_enabled;
                                }
                            }
                            WindowEvent::MouseInput { button, state, .. } => {
                                app.handle_mouse_button(
                                    button,
                                    state,
                                );
                            }
                            WindowEvent::CursorMoved { position, .. } => {
                                app.handle_cursor(
                                    (
                                        position.x,
                                        position.y,
                                    ),
                                );
                            }
                            // drag&drop still works, but autoload already covers your case
                            WindowEvent::DroppedFile(path) => {
                                let ext =
                                    path.extension()
                                        .and_then(|e|
                                            e.to_str()
                                        )
                                        .map(|s|
                                            s.to_ascii_lowercase()
                                        );

                                match ext.as_deref() {
                                    Some("hypc") => {
                                        let _ =
                                            app.load_hypc(
                                                path.to_str()
                                                    .unwrap(),
                                            );
                                    }
                                    Some("ply")  => {
                                        let _ =
                                            app.load_ply(
                                                path.to_str().unwrap(),
                                                1_200_000,
                                                200.0,
                                            );
                                    }
                                    _ => {}
                                }
                            }
                            _ => {}
                        }
                    }
                Event::AboutToWait => {
                    window.request_redraw();
                }
                _ => {}
            }
        }
    )?;

    #[allow(unreachable_code)] Ok(())
}
