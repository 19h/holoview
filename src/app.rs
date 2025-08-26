// src/app.rs
use crate::{
    camera::{Camera, CameraController},
    data::{point_cloud, tile_aligner},
    renderer::{
        pipelines::{
            ground_grid::GridUniforms,
            hologram::HoloUniforms,
            postprocess::{CrtUniforms, EdlUniforms, RgbShiftUniforms},
        },
        Renderer, TileDraw,
    },
    ui,
};
use anyhow::Result;
use glam::{Mat4, Vec3};
use std::{sync::Arc, time::Instant};
use walkdir::WalkDir;
use wgpu::util::DeviceExt;
use winit::{
    event::{ElementState, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

/// WGPU (Vulkan/D3D) clip-space conversion for a GL-style projection
const OPENGL_TO_WGPU_MATRIX: Mat4 = Mat4::from_cols_array(&[
    1.0,  0.0,  0.0, 0.0,
    0.0, -1.0,  0.0, 0.0, // flip Y
    0.0,  0.0,  0.5, 0.0, // map z: [-1,1] -> [0,1]
    0.0,  0.0,  0.5, 1.0,
]);

pub struct App {
    pub window: Arc<Window>,
    pub renderer: Renderer,

    // UI
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,

    // Data
    tiles: Vec<TileDraw>,
    world_min: Vec3,
    world_max: Vec3,
    geot_union: Option<crate::data::GeoExtentDeg>,
    lon0: f64,
    lat0: f64,
    mdeg_lon_ref: f64,
    mdeg_lat_ref: f64,

    // Camera & Controls
    camera: Camera,
    camera_controller: CameraController,
    default_distance: f32,

    // State
    start: Instant,
    grid_enabled: bool,
}

impl App {
    pub async fn new(window: Arc<Window>) -> Self {
        let renderer = Renderer::new(window.clone()).await;

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui_ctx.viewport_id(),
            &*window,
            None,
            None,
        );

        let world_center = Vec3::ZERO;
        let default_distance = 500.0;

        let camera = Camera {
            position: world_center + Vec3::new(0.0, -350.0, 220.0),
            target: world_center,
            up: Vec3::Z,
            fov_y_rad: 55.0f32.to_radians(),
            near: 0.5,
            far: 8000.0,
        };

        let camera_controller = CameraController::new(default_distance);

        Self {
            window,
            renderer,
            egui_ctx,
            egui_state,
            tiles: Vec::new(),
            world_min: Vec3::ZERO,
            world_max: Vec3::ONE,
            geot_union: None,
            lon0: 0.0,
            lat0: 0.0,
            mdeg_lon_ref: 111_320.0,
            mdeg_lat_ref: 110_574.0,
            camera,
            camera_controller,
            default_distance,
            start: Instant::now(),
            grid_enabled: true,
        }
    }

    pub fn get_size(&self) -> winit::dpi::PhysicalSize<u32> {
        self.renderer.context.size
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.renderer.resize(new_size);
    }

    pub fn handle_event(&mut self, event: &WindowEvent) -> bool {
        // Give egui first dibs on the event
        let response = self.egui_state.on_window_event(&self.window, event);
        if response.consumed {
            return true;
        }

        // Handle camera controls
        self.camera_controller.handle_event(event, &mut self.camera);

        // Handle other window events
        match event {
            WindowEvent::Resized(size) => self.resize(*size),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::KeyN) => self.snap_north_up(),
                        PhysicalKey::Code(KeyCode::KeyG) => self.grid_enabled = !self.grid_enabled,
                        _ => {}
                    }
                }
            }
            _ => {}
        }

        false
    }

    fn snap_north_up(&mut self) {
        self.camera_controller.reset_zoom();

        if !self.tiles.is_empty() {
            let center = 0.5 * (self.world_min + self.world_max);

            self.camera.target = center;
            self.camera.position = center + Vec3::new(
                0.0,
                -self.default_distance * 1.35,
                self.default_distance * 0.9,
            );
            self.camera.up = Vec3::Z;
        }
    }

    fn update_uniforms(&mut self) {
        let view = self.camera.view_matrix();
        let aspect_ratio = self.renderer.context.config.width as f32
            / self.renderer.context.config.height as f32;
        let proj_gl = self.camera.projection_matrix_gl(aspect_ratio);
        let proj = OPENGL_TO_WGPU_MATRIX * proj_gl;

        let time = self.start.elapsed().as_secs_f32();
        let (dmin, dmax) = (self.world_min, self.world_max);

        let holo_uniforms = HoloUniforms {
            view,
            proj,
            viewport: [
                self.renderer.context.config.width as f32,
                self.renderer.context.config.height as f32,
            ],
            base_size_px: self.renderer.pipelines.hologram.uniforms.base_size_px,
            size_atten: self.renderer.pipelines.hologram.uniforms.size_atten,
            time,
            near: self.camera.near,
            far: self.camera.far,
            decode_min: dmin,
            decode_max: dmax,
            cyan: self.renderer.pipelines.hologram.uniforms.cyan,
            red: self.renderer.pipelines.hologram.uniforms.red,
            ..Default::default()
        };
        self.renderer
            .pipelines
            .hologram
            .update_uniforms(&self.renderer.context.queue, holo_uniforms);

        let inv_size = [
            1.0 / self.renderer.context.config.width as f32,
            1.0 / self.renderer.context.config.height as f32,
        ];

        // Update EDL post-processing uniforms
        self.renderer.pipelines.post_edl.update_uniforms(
            &self.renderer.context.queue,
            EdlUniforms {
                inv_size,
                strength: 1.15,
                radius_px: 1.25,
            },
        );

        // Update RGB shift post-processing uniforms
        self.renderer.pipelines.post_rgb.update_uniforms(
            &self.renderer.context.queue,
            RgbShiftUniforms {
                inv_size,
                amount: 0.0018,
                angle: 0.0,
            },
        );

        // Update CRT post-processing uniforms
        self.renderer.pipelines.post_crt.update_uniforms(
            &self.renderer.context.queue,
            CrtUniforms {
                inv_size,
                time,
                intensity: self.renderer.pipelines.post_crt.uniform.intensity,
                vignette: self.renderer.pipelines.post_crt.uniform.vignette,
                _padding: 0.0,
            },
        );

        // Compute grid uniform values
        let (lon_min, lat_min, lon_span, lat_span, enable_grid) = if let Some(bb) = &self.geot_union {
            (
                bb.lon_min as f32,
                bb.lat_min as f32,
                (bb.lon_max - bb.lon_min) as f32,
                (bb.lat_max - bb.lat_min) as f32,
                if self.grid_enabled { 1u32 } else { 0u32 },
            )
        } else {
            (0.0, 0.0, 0.0, 0.0, 0u32)
        };

        let extent = (dmax - dmin).abs();
        let z_offset = extent.z.max(1e-6) * 0.02;

        // Update ground grid uniforms
        self.renderer.pipelines.ground_grid.update_uniforms(
            &self.renderer.context.queue,
            GridUniforms {
                view,
                proj,
                decode_min: dmin,
                decode_max: dmax,
                geot_lon_min: lon_min,
                geot_lat_min: lat_min,
                geot_lon_span: lon_span,
                geot_lat_span: lat_span,
                extent_mul: 12.0,
                z_offset,
                opacity: 0.55,
                enabled: enable_grid,
                color_minor: glam::Vec3::new(0.90, 0.15, 0.15),
                color_major: glam::Vec3::new(1.00, 0.20, 0.20),
                ..Default::default()
            },
        );
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.update_uniforms();

        let frame = self.renderer.context.surface.get_current_texture()?;
        let swap_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.renderer.context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Frame Encoder"),
            },
        );

        self.renderer.render(&mut encoder, &swap_view, &self.tiles);

        let total_points: u32 = self.tiles.iter().map(|tile| tile.count).sum();
        let altitude = self.camera.position.z.round() as i32;

        ui::draw_hud(
            &mut self.egui_ctx,
            &mut self.egui_state,
            &mut self.renderer.egui_renderer,
            &self.window,
            &self.renderer.context,
            &mut encoder,
            &swap_view,
            total_points,
            altitude,
        );

        self.renderer.context.queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        Ok(())
    }

    fn scan_hypc_files(root: &str) -> Vec<String> {
        WalkDir::new(root)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
            .filter_map(|entry| {
                let path = entry.path();
                let extension = path.extension().and_then(|s| s.to_str());

                if extension.map_or(false, |ext| ext.eq_ignore_ascii_case("hypc")) {
                    Some(path.to_string_lossy().to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn build_all_tiles(&mut self, root: &str) -> Result<()> {
        use crate::data::{GeoExtentDeg, QuantizedPointCloud, point_cloud};
        use crate::renderer::pipelines::hologram::{Instance, TileUniforms};

        let paths = Self::scan_hypc_files(root);
        if paths.is_empty() {
            return Ok(());
        }

        struct Loaded {
            name: String,
            pc: QuantizedPointCloud,
        }

        let loaded: Vec<Loaded> = paths
            .into_iter()
            .filter_map(|p| match QuantizedPointCloud::load_hypc(&p) {
                Ok(pc) if pc.geog_bbox_deg.is_some() => {
                    let name = std::path::Path::new(&p)
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .to_string();
                    Some(Loaded { name, pc })
                }
                Ok(_) => {
                    log::warn!("Skipping {}: no GEOT footer", p);
                    None
                }
                Err(e) => {
                    log::warn!("Skipping {}: {}", p, e);
                    None
                }
            })
            .collect();

        if loaded.is_empty() {
            return Ok(());
        }

        let mut geot_union = GeoExtentDeg {
            lon_min: f64::INFINITY,
            lat_min: f64::INFINITY,
            lon_max: f64::NEG_INFINITY,
            lat_max: f64::NEG_INFINITY,
        };

        for l in &loaded {
            let bb = l.pc.geog_bbox_deg.as_ref().unwrap();
            geot_union.lon_min = geot_union.lon_min.min(bb.lon_min);
            geot_union.lon_max = geot_union.lon_max.max(bb.lon_max);
            geot_union.lat_min = geot_union.lat_min.min(bb.lat_min);
            geot_union.lat_max = geot_union.lat_max.max(bb.lat_max);
        }

        self.lon0 = 0.5 * (geot_union.lon_min + geot_union.lon_max);
        self.lat0 = 0.5 * (geot_union.lat_min + geot_union.lat_max);
        self.mdeg_lon_ref = point_cloud::meters_per_deg_lon(self.lat0);
        self.mdeg_lat_ref = point_cloud::meters_per_deg_lat(self.lat0);
        self.geot_union = Some(geot_union.clone());

        // Since obj2hypc centers each tile's decode space at origin,
        // we need to track the actual Z centers and ranges
        // The decode space is [-extent/2, extent/2] for each axis
        let mut z_centers = Vec::new();
        let mut z_ranges = Vec::new();
        for l in &loaded {
            let z_center = 0.5 * (l.pc.decode_min.z + l.pc.decode_max.z);
            let z_range = l.pc.decode_max.z - l.pc.decode_min.z;
            z_centers.push(z_center);
            z_ranges.push(z_range);
            log::info!("Tile {}: Z center={}, Z range={}", l.name, z_center, z_range);
        }
        
        // Use the average Z center as our global reference
        let z_center_global = if !z_centers.is_empty() {
            z_centers.iter().sum::<f32>() / z_centers.len() as f32
        } else {
            0.0
        };
        
        // Compute a global meters-per-decode scale using median of all tile scales
        let mut all_scales = Vec::new();
        for l in &loaded {
            let geot = l.pc.geog_bbox_deg.as_ref().unwrap();
            let decode_min = l.pc.decode_min;
            let decode_max = l.pc.decode_max;
            
            let lat_c = 0.5 * (geot.lat_min + geot.lat_max);
            let m_per_deg_lon = point_cloud::meters_per_deg_lon(lat_c);
            let m_per_deg_lat = point_cloud::meters_per_deg_lat(lat_c);
            
            let dx_dec = (decode_max.x - decode_min.x).abs().max(f32::EPSILON);
            let dy_dec = (decode_max.y - decode_min.y).abs().max(f32::EPSILON);
            let lon_span = (geot.lon_max - geot.lon_min).max(f64::EPSILON);
            let lat_span = (geot.lat_max - geot.lat_min).max(f64::EPSILON);
            
            let sx = (m_per_deg_lon as f32) * (lon_span as f32) / dx_dec;
            let sy = (m_per_deg_lat as f32) * (lat_span as f32) / dy_dec;
            let scale = 0.5 * (sx + sy);
            all_scales.push(scale);
        }
        
        // Use median scale for better robustness
        all_scales.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m_per_decode_global = if all_scales.len() % 2 == 0 {
            let mid = all_scales.len() / 2;
            0.5 * (all_scales[mid - 1] + all_scales[mid])
        } else {
            all_scales[all_scales.len() / 2]
        };
        
        log::info!("Global Z center: {}", z_center_global);
        log::info!("Global m_per_decode scale: {} (from {} tiles)", m_per_decode_global, all_scales.len());
        for (i, l) in loaded.iter().enumerate() {
            log::info!("Tile {}: decode_min.z = {}, decode_max.z = {}, tile_scale = {}", 
                i, l.pc.decode_min.z, l.pc.decode_max.z, all_scales[i]);
        }

        let mut tiles = Vec::<TileDraw>::new();
        let mut world_min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut world_max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

        for l in loaded {
            let pc = l.pc;
            let geot = pc.geog_bbox_deg.clone().unwrap();
            let (decode_min, decode_max) = (pc.decode_min, pc.decode_max);

            // Per-tile scale to convert decode units -> meters, derived from this tile's GEOT bbox
            let lat_c = 0.5 * (geot.lat_min + geot.lat_max);
            let m_per_deg_lon = point_cloud::meters_per_deg_lon(lat_c);
            let m_per_deg_lat = point_cloud::meters_per_deg_lat(lat_c);
            
            let dx_dec = (decode_max.x - decode_min.x).abs().max(f32::EPSILON);
            let dy_dec = (decode_max.y - decode_min.y).abs().max(f32::EPSILON);
            let lon_span = (geot.lon_max - geot.lon_min).max(f64::EPSILON);
            let lat_span = (geot.lat_max - geot.lat_min).max(f64::EPSILON);
            
            // meters per decode unit along each axis (affine assumption already made for XY)
            let sx = (m_per_deg_lon as f32) * (lon_span as f32) / dx_dec;
            let sy = (m_per_deg_lat as f32) * (lat_span as f32) / dy_dec;
            
            // If producer was isotropic in decode space (usual), use the mean:
            let m_per_decode = 0.5 * (sx + sy);
            
            log::info!("Tile {}: sx={}, sy={}, m_per_decode={}, dx_dec={}, dy_dec={}", 
                l.name, sx, sy, m_per_decode, dx_dec, dy_dec);
            log::info!("  decode Z range: {} to {}, using global scale: {}", 
                decode_min.z, decode_max.z, m_per_decode_global);
            
            let dx = dx_dec;
            let dy = dy_dec;

            let instance_positions: Vec<[f32; 3]> = pc
                .positions
                .iter()
                .map(|p| {
                    let lon = geot.lon_min + ((p.x - decode_min.x) as f64 / dx as f64) * lon_span;
                    let lat = geot.lat_min + ((p.y - decode_min.y) as f64 / dy as f64) * lat_span;
                    let x_world = ((lon - self.lon0) * self.mdeg_lon_ref) as f32;
                    let y_world = ((lat - self.lat0) * self.mdeg_lat_ref) as f32;
                    // Use GLOBAL scale and GLOBAL Z baseline for consistent Z across all tiles
                    let z_world = (p.z - z_center_global) * m_per_decode_global;

                    world_min = world_min.min(Vec3::new(x_world, y_world, z_world));
                    world_max = world_max.max(Vec3::new(x_world, y_world, z_world));

                    [x_world, y_world, z_world]
                })
                .collect();

            let dmin_world = Vec3::new(
                ((geot.lon_min - self.lon0) * self.mdeg_lon_ref) as f32,
                ((geot.lat_min - self.lat0) * self.mdeg_lat_ref) as f32,
                (decode_min.z - z_center_global) * m_per_decode_global,
            );

            let dmax_world = Vec3::new(
                ((geot.lon_max - self.lon0) * self.mdeg_lon_ref) as f32,
                ((geot.lat_max - self.lat0) * self.mdeg_lat_ref) as f32,
                (decode_max.z - z_center_global) * m_per_decode_global,
            );

            let edge_samples =
                tile_aligner::collect_edge_samples(&instance_positions, dmin_world, dmax_world);

            let instances: Vec<Instance> = instance_positions
                .iter()
                .map(|pos| Instance { position: *pos })
                .collect();

            let vb = self.renderer.context.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Instances({})", l.name)),
                    contents: bytemuck::cast_slice(&instances),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                },
            );

            let (sem_tex, sem_view, sem_samp) =
                self.renderer.upload_semantic_mask(pc.semantic_mask.as_ref());

            let holo_bg = self.renderer.pipelines.hologram.bind_group_for_semantics(
                &self.renderer.context.device,
                &sem_view,
                &sem_samp,
            );

            let tile_uniforms = TileUniforms {
                dmin_world: [dmin_world.x, dmin_world.y],
                dmax_world: [dmax_world.x, dmax_world.y],
                z_bias: 0.0,
                _pad_after_z: 0.0,
                xy_bias: [0.0, 0.0],
                _padding: [0u32; 22],
            };

            let tile_ubo = self.renderer.context.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Tile UBO ({})", l.name)),
                    contents: bytemuck::bytes_of(&tile_uniforms),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                },
            );

            let tile_bg = self.renderer.context.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("Holo BG (tile)"),
                    layout: &self.renderer.pipelines.hologram.bgl_tile,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: tile_ubo.as_entire_binding(),
                    }],
                },
            );

            tiles.push(TileDraw {
                name: l.name,
                vb,
                count: pc.kept as u32,
                sem_tex,
                sem_view,
                sem_samp,
                holo_bg,
                tile_ubo,
                tile_bg,
                dmin_world,
                dmax_world,
                geot,
                z_bias: 0.0,
                xy_bias: glam::Vec2::ZERO,
                edge_samples,
            });
        }

        let (bias_x, bias_y, bias_z) = tile_aligner::estimate_all_biases(&tiles);

        for (i, tile) in tiles.iter_mut().enumerate() {
            tile.xy_bias = glam::Vec2::new(
                *bias_x.get(i).unwrap_or(&0.0),
                *bias_y.get(i).unwrap_or(&0.0),
            );
            tile.z_bias = *bias_z.get(i).unwrap_or(&0.0);

            let dmin_biased = tile.dmin_world + glam::Vec3::new(tile.xy_bias.x, tile.xy_bias.y, 0.0);
            let dmax_biased = tile.dmax_world + glam::Vec3::new(tile.xy_bias.x, tile.xy_bias.y, 0.0);

            let tile_uniforms = TileUniforms {
                dmin_world: [dmin_biased.x, dmin_biased.y],
                dmax_world: [dmax_biased.x, dmax_biased.y],
                z_bias: tile.z_bias,
                _pad_after_z: 0.0,
                xy_bias: [tile.xy_bias.x, tile.xy_bias.y],
                _padding: [0u32; 22],
            };

            self.renderer.context.queue.write_buffer(
                &tile.tile_ubo,
                0,
                bytemuck::bytes_of(&tile_uniforms),
            );
        }

        let mut world_min_biased = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut world_max_biased = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

        for tile in &tiles {
            // Include XY biases in world bounds calculation for consistent post-processing
            let min_with_bias = tile.dmin_world + Vec3::new(tile.xy_bias.x, tile.xy_bias.y, tile.z_bias);
            let max_with_bias = tile.dmax_world + Vec3::new(tile.xy_bias.x, tile.xy_bias.y, tile.z_bias);
            world_min_biased = world_min_biased.min(min_with_bias);
            world_max_biased = world_max_biased.max(max_with_bias);
        }

        self.tiles = tiles;
        self.world_min = world_min_biased;
        self.world_max = world_max_biased;

        let extent = self.world_max - self.world_min;
        let max_dimension = extent.x.max(extent.y).max(extent.z);
        let distance = max_dimension * 1.2;

        self.default_distance = distance;
        self.camera_controller.set_default_distance(distance);

        let center = 0.5 * (self.world_min + self.world_max);
        self.camera.target = center;
        self.camera.position = center
            + Vec3::new(0.25 * distance, -1.35 * distance, 0.9 * distance);

        Ok(())
    }
}
