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
use point_cloud::{QuantizedPointCloud, GeoExtentDeg};
use post::{EdlPass, EdlUniforms, RgbShiftPass, RgbShiftUniforms, CrtPass, CrtUniforms};
use walkdir::WalkDir;

// Z-bias solver for cross-tile altitude consistency
struct MatchEdge { a: usize, b: usize, dz: f32, w: f32 }

fn solve_z_biases(n: usize, edges: &[MatchEdge]) -> Vec<f32> {
    if n == 0 || edges.is_empty() {
        return vec![0.0; n];
    }

    // Simple iterative solver for small systems
    // Initialize all biases to 0
    let mut biases = vec![0.0f32; n];

    // Anchor tile 0 to 0
    biases[0] = 0.0;

    // Iterative solver: update each bias based on constraints
    for _ in 0..20 {  // max iterations
        let mut new_biases = biases.clone();

        for edge in edges {
            if edge.a < n && edge.b < n {
                // Apply constraint: bias[b] - bias[a] ≈ edge.dz
                let current_diff = biases[edge.b] - biases[edge.a];
                let error = edge.dz - current_diff;
                let adjustment = error * edge.w * 0.1; // damping factor

                if edge.a != 0 { // don't adjust anchor
                    new_biases[edge.a] -= adjustment * 0.5;
                }
                if edge.b != 0 { // don't adjust anchor
                    new_biases[edge.b] += adjustment * 0.5;
                }
            }
        }

        biases = new_biases;
    }

    biases
}

fn compute_z_biases(tiles: &[TileDraw]) -> Vec<f32> {
    let n = tiles.len();
    if n <= 1 {
        return vec![0.0; n];
    }

    let mut edges = Vec::new();

    // Find neighboring tiles based on GEOT bounds proximity
    for i in 0..n {
        for j in (i + 1)..n {
            let tile_a = &tiles[i];
            let tile_b = &tiles[j];

            // Check if tiles are neighbors (bounding boxes touch within epsilon)
            let eps = 0.4; // meters

            let a_min_x = tile_a.dmin_world.x;
            let a_max_x = tile_a.dmax_world.x;
            let a_min_y = tile_a.dmin_world.y;
            let a_max_y = tile_a.dmax_world.y;

            let b_min_x = tile_b.dmin_world.x;
            let b_max_x = tile_b.dmax_world.x;
            let b_min_y = tile_b.dmin_world.y;
            let b_max_y = tile_b.dmax_world.y;

            // Check if tiles are adjacent
            let x_overlap = a_max_x >= (b_min_x - eps) && b_max_x >= (a_min_x - eps);
            let y_overlap = a_max_y >= (b_min_y - eps) && b_max_y >= (a_min_y - eps);

            if x_overlap && y_overlap {
                // Neighboring tiles - estimate height difference
                // For simplicity, use center Z difference as a proxy
                let a_center_z = (tile_a.dmin_world.z + tile_a.dmax_world.z) * 0.5;
                let b_center_z = (tile_b.dmin_world.z + tile_b.dmax_world.z) * 0.5;

                let dz = b_center_z - a_center_z;
                let weight = 1.0; // could be based on overlap area or confidence

                edges.push(MatchEdge { a: i, b: j, dz, w: weight });
            }
        }
    }

    solve_z_biases(n, &edges)
}

// WGPU (Vulkan/D3D) clip-space conversion for a GL-style projection
const OPENGL_TO_WGPU_MATRIX: Mat4 = Mat4::from_cols_array(&[
    1.0,  0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,  // flip Y
    0.0,  0.0, 0.5, 0.0,  // map z: [-1,1] -> [0,1]
    0.0,  0.0, 0.5, 1.0,
]);

// Multi-sampling sample count for anti-aliasing
const SAMPLE_COUNT: u32 = 4;

fn meters_per_deg_lat(lat_deg: f64) -> f64 {
    let φ = lat_deg.to_radians();
    111_132.92 - 559.82 * (2.0*φ).cos() + 1.175 * (4.0*φ).cos() - 0.0023 * (6.0*φ).cos()
}
fn meters_per_deg_lon(lat_deg: f64) -> f64 {
    let φ = lat_deg.to_radians();
    111_412.84 * φ.cos() - 93.5 * (3.0*φ).cos() + 0.118 * (5.0*φ).cos()
}

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

struct TileDraw {
    name: String,
    vb: wgpu::Buffer,
    count: u32,
    // semantics
    sem_tex: Option<wgpu::Texture>,
    sem_view: wgpu::TextureView,
    sem_samp: wgpu::Sampler,
    holo_bg: wgpu::BindGroup,
    // per-tile UBO and bind group for SMC1 UV bounds
    tile_ubo: wgpu::Buffer,
    tile_bg: wgpu::BindGroup,
    // per‑tile world decode bounds for SMC1 UV & height modulation
    dmin_world: Vec3,
    dmax_world: Vec3,
    geot: GeoExtentDeg,
    // Z-bias for consistent altitudes across tiles
    z_bias: f32,
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

    // Multi‑tile dataset
    tiles: Vec<TileDraw>,
    world_min: Vec3,
    world_max: Vec3,
    geot_union: Option<GeoExtentDeg>,
    // global geodetic anchor and scale
    lon0: f64,
    lat0: f64,
    mdeg_lon_ref: f64,
    mdeg_lat_ref: f64,
    // Fallback SMC1 (1×1 zero)
    sem_fallback_tex: wgpu::Texture,
    sem_fallback_view: wgpu::TextureView,
    sem_fallback_samp: wgpu::Sampler,

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

        // Fallback SMC1 resources
        let sem_fallback_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SMC1 Fallback 1x1"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &sem_fallback_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All
            },
            &[0u8], // label=0 → Unknown
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(1), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 }
        );
        let sem_fallback_view = sem_fallback_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sem_fallback_samp = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SMC1 Fallback Sampler"), mag_filter: wgpu::FilterMode::Nearest, min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest, address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge, address_mode_w: wgpu::AddressMode::ClampToEdge, ..Default::default()
        });

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
            tiles: Vec::new(),
            world_min: Vec3::ZERO,
            world_max: Vec3::ONE,
            geot_union: None,
            lon0: 0.0,
            lat0: 0.0,
            mdeg_lon_ref: 111_320.0,
            mdeg_lat_ref: 110_574.0,
            sem_fallback_tex,
            sem_fallback_view,
            sem_fallback_samp,
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

    fn upload_semantic_mask_to_texture(&self, sm: &point_cloud::SemanticMask) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
        let w = sm.width as u32;
        let h = sm.height as u32;
        let tex = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SMC1 Labels"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // 256‑byte aligned rows
        let bpr = ((w + 255) / 256) * 256;
        let mut staging = vec![0u8; (bpr * h) as usize];
        for row in 0..h as usize {
            let src = &sm.data[row * w as usize .. (row + 1) * w as usize];
            let dst = &mut staging[row * bpr as usize .. row * bpr as usize + w as usize];
            dst.copy_from_slice(src);
        }

        self.gpu.queue.write_texture(
            wgpu::ImageCopyTexture { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &staging,
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );

        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let samp = self.gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SMC1 Nearest"),
            mag_filter: wgpu::FilterMode::Nearest, min_filter: wgpu::FilterMode::Nearest, mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge, address_mode_v: wgpu::AddressMode::ClampToEdge, address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        (tex, view, samp)
    }

    fn scan_hypc_files(root: &str) -> Vec<String> {
        let mut out = Vec::new();
        for e in WalkDir::new(root).into_iter().filter_map(Result::ok) {
            if e.file_type().is_file() {
                if let Some(ext) = e.path().extension().and_then(|s| s.to_str()) {
                    if ext.eq_ignore_ascii_case("hypc") {
                        out.push(e.path().to_string_lossy().to_string());
                    }
                }
            }
        }
        out
    }

    fn build_all_tiles(&mut self, root: &str) -> anyhow::Result<()> {
        let paths = Self::scan_hypc_files(root);
        if paths.is_empty() { return Ok(()); }

        // 1) Load all tiles with GEOT
        struct Loaded { name: String, pc: QuantizedPointCloud }
        let mut loaded = Vec::<Loaded>::new();
        for p in paths {
            match QuantizedPointCloud::load_hypc(&p) {
                Ok(pc) => {
                    if pc.geog_bbox_deg.is_some() {
                        loaded.push(Loaded { name: std::path::Path::new(&p).file_name().unwrap().to_string_lossy().to_string(), pc });
                    } else {
                        log::warn!("Skipping {}: no GEOT footer", p);
                    }
                }
                Err(e) => log::warn!("Skipping {}: {}", p, e),
            }
        }
        if loaded.is_empty() { return Ok(()); }

        // 2) Union GEOT and anchor
        let mut g = GeoExtentDeg { lon_min: f64::INFINITY, lat_min: f64::INFINITY, lon_max: f64::NEG_INFINITY, lat_max: f64::NEG_INFINITY };
        for l in &loaded {
            let bb = l.pc.geog_bbox_deg.as_ref().unwrap();
            g.lon_min = g.lon_min.min(bb.lon_min);
            g.lon_max = g.lon_max.max(bb.lon_max);
            g.lat_min = g.lat_min.min(bb.lat_min);
            g.lat_max = g.lat_max.max(bb.lat_max);
        }
        self.lon0 = 0.5 * (g.lon_min + g.lon_max);
        self.lat0 = 0.5 * (g.lat_min + g.lat_max);
        self.mdeg_lon_ref = meters_per_deg_lon(self.lat0);
        self.mdeg_lat_ref = meters_per_deg_lat(self.lat0);
        self.geot_union = Some(g);

        // 3) Per‑tile transform to world (meters), build GPU buffers and per‑tile bind groups
        let mut tiles = Vec::<TileDraw>::new();
        let mut wmin = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut wmax = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

        use holographic_shaders::Instance;
        for l in loaded {
            let pc = l.pc;
            let geot = pc.geog_bbox_deg.clone().unwrap();
            let dmin = pc.decode_min;
            let dmax = pc.decode_max;

            let lat_c = 0.5 * (geot.lat_min + geot.lat_max);
            let mdeg_lon_tile = meters_per_deg_lon(lat_c);
            let mdeg_lat_tile = meters_per_deg_lat(lat_c);

            let dx = (dmax.x - dmin.x).max(f32::EPSILON);
            let dy = (dmax.y - dmin.y).max(f32::EPSILON);

            let lon_span = (geot.lon_max - geot.lon_min).max(f64::EPSILON);
            let lat_span = (geot.lat_max - geot.lat_min).max(f64::EPSILON);

            let sx = (lon_span * mdeg_lon_tile) as f32 / dx;
            let sy = (lat_span * mdeg_lat_tile) as f32 / dy;
            let sz = 0.5 * (sx + sy); // isotropic vertical per A4

            let mut inst = Vec::<Instance>::with_capacity(pc.positions.len());
            let mut instance_positions = Vec::<[f32; 3]>::with_capacity(pc.positions.len());

            for p in &pc.positions {
                // decode‑XY -> lon/lat
                let lon = geot.lon_min + ((p.x - dmin.x) as f64 / dx as f64) * lon_span;
                let lat = geot.lat_min + ((p.y - dmin.y) as f64 / dy as f64) * lat_span;
                // lon/lat -> world meters about (lon0,lat0)
                let x_world = ((lon - self.lon0) * self.mdeg_lon_ref) as f32;
                let y_world = ((lat - self.lat0) * self.mdeg_lat_ref) as f32;
                let z_world = (p.z - dmin.z) * sz;

                // Store position for later Z-bias application
                instance_positions.push([x_world, y_world, z_world]);

                wmin.x = wmin.x.min(x_world); wmin.y = wmin.y.min(y_world); wmin.z = wmin.z.min(z_world);
                wmax.x = wmax.x.max(x_world); wmax.y = wmax.y.max(y_world); wmax.z = wmax.z.max(z_world);
            }

            // Create placeholder vertex buffer - will be updated with Z-bias later
            let inst: Vec<Instance> = instance_positions.iter()
                .map(|pos| Instance { position: *pos })
                .collect();

            let vb = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Instances({})", l.name)),
                contents: bytemuck::cast_slice(&inst),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

            // per‑tile world decode bounds (for SMC1 UV + height)
            let dmin_world = Vec3::new(
                ((geot.lon_min - self.lon0) * self.mdeg_lon_ref) as f32,
                ((geot.lat_min - self.lat0) * self.mdeg_lat_ref) as f32,
                0.0
            );
            let dmax_world = Vec3::new(
                ((geot.lon_max - self.lon0) * self.mdeg_lon_ref) as f32,
                ((geot.lat_max - self.lat0) * self.mdeg_lat_ref) as f32,
                (dmax.z - dmin.z) * sz
            );

            // semantics
            let (sem_tex, sem_view, sem_samp) = if let Some(sm) = pc.semantic_mask.as_ref() {
                let (t, v, s) = self.upload_semantic_mask_to_texture(sm);
                (Some(t), v, s)
            } else {
                // Create new references to the fallback texture and sampler
                let fallback_view = self.sem_fallback_tex.create_view(&wgpu::TextureViewDescriptor::default());
                let fallback_samp = self.gpu.device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("SMC1 Fallback Sampler (copy)"),
                    mag_filter: wgpu::FilterMode::Nearest,
                    min_filter: wgpu::FilterMode::Nearest,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    ..Default::default()
                });
                (None, fallback_view, fallback_samp)
            };

            // per‑tile holo bind group
            let holo_bg = self.holo.bind_group_for_semantics(&self.gpu.device, &sem_view, &sem_samp);

            // create per-tile UBO with world bounds for SMC1 UVs
            let tile_u = holographic_shaders::TileUniforms {
                dmin_world: [dmin_world.x, dmin_world.y],
                dmax_world: [dmax_world.x, dmax_world.y],
            };
            let tile_ubo = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Tile UBO ({})", l.name)),
                contents: bytemuck::bytes_of(&tile_u),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let tile_bg = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Holo BG (tile)"),
                layout: &self.holo.bgl_tile,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tile_ubo.as_entire_binding(),
                }],
            });

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
                z_bias: 0.0, // will be computed below
            });
        }

        // Compute Z-biases for consistent cross-tile altitudes
        let z_biases = compute_z_biases(&tiles);

        // Apply Z-biases and recompute world bounds
        for (i, tile) in tiles.iter_mut().enumerate() {
            if i < z_biases.len() {
                tile.z_bias = z_biases[i];

                // Update vertex buffer with Z-bias applied
                if tile.z_bias != 0.0 {
                    // Need to reload positions and apply bias
                    // For now, we'll skip this complex step and just store the bias
                    // The bias could be applied in the shader or via a transform
                    log::info!("Applied Z-bias {} to tile {}", tile.z_bias, tile.name);
                }
            }
        }

        // Recompute world bounds with biases applied
        let mut wmin_biased = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut wmax_biased = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

        for tile in &tiles {
            let min_z = tile.dmin_world.z + tile.z_bias;
            let max_z = tile.dmax_world.z + tile.z_bias;

            wmin_biased.x = wmin_biased.x.min(tile.dmin_world.x);
            wmin_biased.y = wmin_biased.y.min(tile.dmin_world.y);
            wmin_biased.z = wmin_biased.z.min(min_z);

            wmax_biased.x = wmax_biased.x.max(tile.dmax_world.x);
            wmax_biased.y = wmax_biased.y.max(tile.dmax_world.y);
            wmax_biased.z = wmax_biased.z.max(max_z);
        }

        self.tiles = tiles;
        self.world_min = wmin_biased;
        self.world_max = wmax_biased;

        // Frame camera based on union world bounds (Z‑up)
        let size = self.world_max - self.world_min;
        let max_dim = size.x.max(size.y).max(size.z);
        let dist = max_dim * 1.2;
        let center = 0.5 * (self.world_min + self.world_max);
        self.cam_target = center;
        self.cam_up = Vec3::new(0.0, 0.0, 1.0);
        self.cam_pos = center + Vec3::new(0.25*dist, -1.35*dist, 0.9*dist);

        Ok(())
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

        let (dmin, dmax) = (self.world_min, self.world_max);

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

        // Grid: use union GEOT + union world bounds
        let (lon_min, lat_min, lon_span, lat_span, enable_grid) = if let Some(bb) = &self.geot_union {
            (bb.lon_min as f32,
             bb.lat_min as f32,
             (bb.lon_max - bb.lon_min) as f32,
             (bb.lat_max - bb.lat_min) as f32,
             if self.grid_enabled { 1u32 } else { 0u32 })
        } else { (0.0, 0.0, 0.0, 0.0, 0u32) };

        // place plane just below global min z
        let extent = (dmax - dmin).abs();
        let z_ofs = extent.z.max(1e-6) * 0.02;

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
                extent_mul: 12.0,
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
        self.cam_up = Vec3::new(0.0, 0.0, 1.0);

        if !self.tiles.is_empty() {
            let size = self.world_max - self.world_min;
            let max_dim = size.x.max(size.y).max(size.z);
            let dist = max_dim * 1.2;
            let center = 0.5 * (self.world_min + self.world_max);
            self.cam_target = center;
            self.cam_pos = center + Vec3::new(0.0, -dist * 1.35, dist * 0.9);
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

            // --- draw all tiles ---
            rp.set_pipeline(&self.holo.pipeline);
            rp.set_vertex_buffer(0, self.holo.quad_vb.slice(..));

            for t in &self.tiles {
                // Use pre-built bind groups that already contain the per-tile uniform data
                rp.set_bind_group(0, &t.holo_bg, &[]);   // global UBO + per-tile SMC1 tex/sampler
                rp.set_bind_group(1, &t.tile_bg, &[]);   // per-tile UV bounds
                rp.set_vertex_buffer(1, t.vb.slice(..));
                rp.draw(0..6, 0..t.count);
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

                        let pts: u32 = self.tiles.iter().map(|t| t.count).sum();
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

    // Load all .hypc tiles found under ./hypc (or ../hypc)
    if std::path::Path::new("hypc").exists() {
        let _ = app.build_all_tiles("hypc");
    } else if std::path::Path::new("../hypc").exists() {
        let _ = app.build_all_tiles("../hypc");
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
                            WindowEvent::DroppedFile(_path) => {
                                // This functionality is now handled by build_all_tiles on startup.
                                // Could be re-wired to rebuild the dataset.
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
