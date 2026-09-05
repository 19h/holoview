use crate::{
    camera::{Camera, CameraController},
    data::{streaming::{StreamScene, StreamConfig}, types::TileGpu},
    renderer::Renderer,
    ui,
};
use anyhow::Result;
use glam::Mat4;
use std::sync::Arc;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use winit::{event::WindowEvent, window::Window};

pub struct App {
    pub renderer: Renderer,
    pub camera: Camera,
    pub camera_controller: CameraController,
    pub egui_ctx: egui::Context,
    pub egui_state: egui_winit::State,
    pub scene: Option<StreamScene>,
    pub stream_config: StreamConfig,
    preparing: Option<Receiver<Result<PathBuf, String>>>,
    pub load_error: Option<String>,
    loading_path: Option<PathBuf>,
    frame_ms: f64,
    pub capture_next: Option<PathBuf>,
    pub record_metrics: bool,
    pub metrics: Vec<serde_json::Value>,
    last_frame: std::time::Instant,
}

impl App {
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let mut renderer = Renderer::new(window.clone()).await?;
        renderer.post_stack.params.rgb_on = false;
        renderer.post_stack.params.crt_on = false;
        renderer.post_stack.params.grid_on = false;
        let size = renderer.gfx.size;

        // WebGPU/wgpu uses 0..1 depth; glam::Mat4::perspective_rh is RH, depth in [0,1].
        let proj = Mat4::perspective_infinite_reverse_rh(
            60f32.to_radians(),
            size.width as f32 / size.height.max(1) as f32,
            0.1,
        );

        // Default camera, orbiting a point over Berlin at a 5km radius.
        let camera = Camera::new(52.52, 13.40, 5000.0, proj);
        let mut camera_controller = CameraController::new();
        camera_controller.set_viewport(size.width, size.height);

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui_ctx.viewport_id(),
            &*window,
            None,
            None,
        );

        Ok(Self {
            renderer,
            camera,
            camera_controller,
            egui_ctx,
            egui_state,
            scene: None,
            stream_config: StreamConfig::default(),
            preparing: None,
            load_error: None,
            loading_path: None,
            frame_ms: 16.67,
            capture_next: None,
            record_metrics: false,
            metrics: Vec::new(),
            last_frame: std::time::Instant::now(),
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.camera_controller.set_viewport(new_size.width, new_size.height);
            self.renderer.resize(new_size);
            self.camera.proj = Mat4::perspective_infinite_reverse_rh(
                // Field of view
                60f32.to_radians(),
                // Aspect ratio
                new_size.width as f32 / new_size.height as f32,
                // Near plane distance
                0.1,
            );
        }
    }

    pub fn handle_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        let response = self.egui_state.on_window_event(window, event);
        if matches!(event, WindowEvent::KeyboardInput { .. } | WindowEvent::MouseInput { .. } | WindowEvent::MouseWheel { .. } | WindowEvent::Focused(_)) {
            log::debug!("Navigation event {event:?}; UI consumed={}", response.consumed);
        }
        self.camera_controller.release_event(event);
        if response.consumed {
            return true;
        }

        self.camera_controller.handle_event(event, &mut self.camera);

        if let WindowEvent::Resized(physical_size) = event {
            self.resize(*physical_size);
        }

        false
    }

    /// Open a published hierarchy immediately, or prepare a raw folder in the
    /// background while the window remains responsive.
    pub fn load_dataset(&mut self, root: &Path) -> Result<()> {
        anyhow::ensure!(self.preparing.is_none(), "A dataset is already being prepared");
        let root = if root.is_file() && root.file_name().is_some_and(|s| s == "catalog.json") {
            root.parent().unwrap()
        } else { root };
        anyhow::ensure!(root.is_dir(), "Dataset directory does not exist: {}", root.display());
        self.load_error = None;
        if root.join("catalog.json").exists() { return self.activate_dataset(root); }
        let cache = root.join(".holoviewer-lod");
        if cache.join("catalog.json").exists() { return self.activate_dataset(&cache); }
        let (tx, rx) = mpsc::channel();
        let source = root.to_path_buf();
        self.loading_path = Some(cache.clone());
        self.preparing = Some(rx);
        std::thread::Builder::new().name("city-preparation".into()).spawn(move || {
            let result = crate::data::dataset::prepare_dataset(&source, &cache, 4)
                .map(|_| cache).map_err(|e| format!("{e:#}"));
            let _ = tx.send(result);
        })?;
        Ok(())
    }

    fn activate_dataset(&mut self, cache: &Path) -> Result<()> {
        let opened = std::time::Instant::now();
        let scene = StreamScene::open(cache, self.stream_config.clone(),
            &self.renderer.gfx.device, &self.renderer.holo.tile_layout, &self.camera,
            [self.renderer.gfx.size.width as f32, self.renderer.gfx.size.height as f32])?;
        log::info!("Opened {} source tiles / {} points / {} LOD nodes; pinned {} bytes",
            scene.dataset.source_tiles, scene.dataset.source_points, scene.dataset.nodes.len(), scene.stats.gpu_bytes);
        log::info!("Dataset opened in {:.3} s", opened.elapsed().as_secs_f64());
        self.scene = Some(scene);
        self.fit_dataset();
        self.loading_path = None;
        self.last_frame = std::time::Instant::now();
        Ok(())
    }

    pub fn fit_dataset(&mut self) {
        if let Some(scene) = &self.scene {
            let root = &scene.dataset.nodes[scene.dataset.root as usize];
            self.camera.azimuth_rad = std::f64::consts::PI;
            self.camera.elevation_rad = 70f64.to_radians();
            self.camera.set_target_and_radius(root.center().into(), 5000.0);
            let rotation = self.camera.view_ecef();
            let tx = (self.camera.proj.x_axis.x as f64).recip();
            let ty = (self.camera.proj.y_axis.y as f64).recip();
            let mut range = 100.0f64;
            // Fit actual source bounds, avoiding a loose ECEF root box wasting
            // most of the viewport on empty space.
            for n in &scene.dataset.nodes {
                if matches!(n.payload, crate::data::dataset::Payload::Source { .. }) {
                    let p = rotation.transform_vector3((n.center() - root.center()).as_vec3()).as_dvec3();
                    range = range.max(p.z + p.x.abs() / tx + n.radius() * (1.0 + tx.recip().powi(2)).sqrt());
                    range = range.max(p.z + p.y.abs() / ty + n.radius() * (1.0 + ty.recip().powi(2)).sqrt());
                }
            }
            self.camera.radius_m = (range * 1.12).clamp(100.0, 150_000.0); self.camera.update();
            self.renderer.grid.set_origin(root.center().into());
        }
    }

    fn poll_preparation(&mut self) {
        if let Some(receiver) = &self.preparing {
            match receiver.try_recv() {
                Ok(result) => {
                    self.preparing = None;
                    let result = result.map_err(anyhow::Error::msg).and_then(|path| self.activate_dataset(&path));
                    if let Err(error) = result { self.load_error = Some(format!("{error:#}")); }
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.preparing = None;
                    self.load_error = Some("Dataset preparation stopped unexpectedly".into());
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
    }

    fn capture_frame(&self, texture: &wgpu::Texture, path: &Path) -> Result<()> {
        anyhow::ensure!(self.renderer.gfx.config.usage.contains(wgpu::TextureUsages::COPY_SRC), "Surface does not support capture");
        let width = self.renderer.gfx.size.width;
        let height = self.renderer.gfx.size.height;
        let row = (width * 4).div_ceil(256) * 256;
        let device = &self.renderer.gfx.device;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Requested frame capture"), size: row as u64 * height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::ImageCopyBuffer { buffer: &buffer, layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(height) } },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 });
        self.renderer.gfx.queue.submit(Some(encoder.finish()));
        let (tx, rx) = mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        device.poll(wgpu::Maintain::Wait); rx.recv()??;
        let mapped = buffer.slice(..).get_mapped_range();
        let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
        let bgra = matches!(self.renderer.gfx.config.format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
        for scanline in mapped.chunks_exact(row as usize) {
            for p in scanline[..width as usize * 4].chunks_exact(4) {
                if bgra { rgba.extend_from_slice(&[p[2], p[1], p[0], p[3]]); }
                else { rgba.extend_from_slice(p); }
            }
        }
        drop(mapped); buffer.unmap();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) { std::fs::create_dir_all(parent)?; }
        let mut encoder = png::Encoder::new(std::fs::File::create(path)?, width, height);
        encoder.set_color(png::ColorType::Rgba); encoder.set_depth(png::BitDepth::Eight);
        encoder.write_header()?.write_image_data(&rgba)?;
        log::info!("Captured {}×{} frame to {}", width, height, path.display());
        Ok(())
    }

    pub fn render(&mut self, window: &Window) -> Result<(), wgpu::SurfaceError> {
        self.poll_preparation();
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f64();
        self.frame_ms = self.frame_ms * 0.9 + dt.min(0.1) * 100.0;
        self.camera_controller.update(&mut self.camera, dt);
        self.last_frame = now;
        let frame = self.renderer.gfx.surface.get_current_texture()?;
        let swap_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let viewport_size = [
            self.renderer.gfx.size.width as f32,
            self.renderer.gfx.size.height as f32,
        ];

        if let Some(scene) = &mut self.scene {
            scene.update(&self.camera, viewport_size, &self.renderer.gfx.device, &self.renderer.holo.tile_layout);
            let draws: Vec<_> = scene.draw_tiles().collect();
            self.renderer.render_streamed(&swap_view, &draws, &self.camera);
        } else {
            self.renderer.render_streamed(&swap_view, &[], &self.camera);
        }

        let egui_input = self.egui_state.take_egui_input(window);
        self.egui_ctx.begin_frame(egui_input);

        let action = ui::draw_map_ui(&self.egui_ctx, &mut self.camera,
            self.scene.as_mut(), self.frame_ms, self.preparing.is_some(), self.load_error.as_deref(),
            &mut self.renderer.post_stack.params);
        if action == ui::MapAction::Fit { self.fit_dataset(); }

        let egui_output = self.egui_ctx.end_frame();
        let shapes = self
            .egui_ctx
            .tessellate(egui_output.shapes, self.egui_ctx.pixels_per_point());

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [
                self.renderer.gfx.config.width,
                self.renderer.gfx.config.height,
            ],
            pixels_per_point: self.egui_ctx.pixels_per_point(),
        };

        let mut encoder = self
            .renderer
            .gfx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("UI Encoder"),
            });

        for (id, delta) in &egui_output.textures_delta.set {
            self.renderer.egui_renderer.update_texture(
                &self.renderer.gfx.device,
                &self.renderer.gfx.queue,
                *id,
                delta,
            );
        }

        self.renderer.egui_renderer.update_buffers(
            &self.renderer.gfx.device,
            &self.renderer.gfx.queue,
            &mut encoder,
            &shapes,
            &screen_descriptor,
        );

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("EGUI Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &swap_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            self.renderer
                .egui_renderer
                .render(&mut render_pass, &shapes, &screen_descriptor);
        }

        for id in &egui_output.textures_delta.free {
            self.renderer.egui_renderer.free_texture(id);
        }

        self.renderer
            .gfx
            .queue
            .submit(std::iter::once(encoder.finish()));
        if self.record_metrics {
            if let Some(scene) = &self.scene {
                self.metrics.push(serde_json::json!({
                    "frame":self.metrics.len(), "frame_ms":dt * 1000.0,
                    "camera_radius_m":self.camera.radius_m,
                    "camera_target_ecef_m":self.camera.target_ecef.to_array(),
                    "camera_azimuth_deg":self.camera.azimuth_rad.to_degrees(),
                    "unix_time_s":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs_f64(),
                    "camera_elevation_deg":self.camera.elevation_rad.to_degrees(),
                    "stream":scene.stats,
                }));
            }
        }
        if let Some(path) = self.capture_next.take() {
            if let Err(error) = self.capture_frame(&frame.texture, &path) {
                log::error!("Frame capture failed: {error:#}");
            }
        }
        frame.present();

        Ok(())
    }
}

impl TileGpu {
    pub fn make_uniform(
        &self,
        cam: &Camera,
        viewport_size: [f32; 2],
        point_size_px: f32,
    ) -> crate::data::types::TileUniformStd140 {
        cam.make_tile_uniform(
            self.anchor_units,
            self.units_per_meter,
            viewport_size,
            point_size_px,
        )
    }
}
