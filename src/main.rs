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
use std::cmp::Ordering;

// -------- robust helpers --------

fn percentile_in_place(v: &mut [f32], q: f32) -> f32 {
    if v.is_empty() { return f32::NAN; }
    let q = q.clamp(0.0, 1.0);
    let k = ((v.len() - 1) as f32 * q).round() as usize;
    let (_, nth, _) = v.select_nth_unstable_by(k, |a, b| {
        a.partial_cmp(b).unwrap_or(Ordering::Equal)
    });
    *nth
}

fn median_in_place(v: &mut [f32]) -> f32 {
    percentile_in_place(v, 0.5)
}

fn median_and_mad(mut xs: Vec<f32>) -> (f32, f32) {
    if xs.is_empty() { return (0.0, f32::INFINITY); }
    let m = median_in_place(&mut xs);
    let mut dev: Vec<f32> = xs.into_iter().map(|x| (x - m).abs()).collect();
    let mad = median_in_place(&mut dev);
    (m, mad)
}

// -------- robust edge dz estimation along a shared boundary --------

struct TempTile<'a> {
    idx: usize,
    dmin: glam::Vec3,
    dmax: glam::Vec3,
    // Only points near the border (world meters)
    border_pts: &'a [[f32; 3]],
}

// Estimate dz between two neighboring tiles by comparing a thin strip along
// their touching edge. We bin along the edge direction and use the 20th
// percentile per bin (≈ ground) to be robust against trees/buildings.
// Returns (dz_b_minus_a, weight).
fn estimate_edge_dz(a: &TempTile, b: &TempTile) -> Option<(f32, f32)> {
    // Are they neighbors along X (east/west) or Y (north/south)?
    let eps = 0.50; // meters, bbox touch tolerance
    let border_w = 8.0; // meters, strip thickness inside each tile
    let bin = 2.0;      // meters, bin length along the edge
    let min_pts_per_bin = 8usize;

    // Convenience
    let a_min = a.dmin; let a_max = a.dmax;
    let b_min = b.dmin; let b_max = b.dmax;

    // Helper to compute per‑bin 20th percentile array for points inside a strip
    let mut compute_bins = |pts: &[[f32; 3]], axis_min: f32, axis_max: f32, axis_bin: f32,
                            // selector returns true if the point is inside the strip
                            in_strip: &dyn Fn(&[f32;3])->bool,
                            coord_along: &dyn Fn(&[f32;3])->f32| -> Vec<Vec<f32>> {
        let span = (axis_max - axis_min).max(1e-3);
        let nb = ((span / axis_bin).ceil() as usize).max(1);
        let mut bins: Vec<Vec<f32>> = vec![Vec::new(); nb];
        for p in pts {
            if !in_strip(p) { continue; }
            let t = coord_along(p).clamp(axis_min, axis_max - 1e-6);
            let i = (((t - axis_min) / axis_bin) as usize).min(nb - 1);
            bins[i].push(p[2]); // store z
        }
        bins
    };

    // X‑touching edge (east/west)
    if (a_max.x - b_min.x).abs() < eps && a_max.y.max(b_max.y) > a_min.y.min(b_min.y) {
        let y0 = a_min.y.max(b_min.y);
        let y1 = a_max.y.min(b_max.y);
        if y1 <= y0 { return None; }

        // A's east strip, B's west strip
        let bins_a = compute_bins(
            a.border_pts,
            y0, y1, bin,
            &|p| p[0] > a_max.x - border_w && p[1] >= y0 && p[1] <= y1,
            &|p| p[1],
        );
        let bins_b = compute_bins(
            b.border_pts,
            y0, y1, bin,
            &|p| p[0] < b_min.x + border_w && p[1] >= y0 && p[1] <= y1,
            &|p| p[1],
        );

        let mut dzs = Vec::<f32>::new();
        for (ba, bb) in bins_a.into_iter().zip(bins_b.into_iter()) {
            if ba.len() >= min_pts_per_bin && bb.len() >= min_pts_per_bin {
                let mut va = ba; let mut vb = bb;
                let qa = percentile_in_place(&mut va, 0.20);
                let qb = percentile_in_place(&mut vb, 0.20);
                if qa.is_finite() && qb.is_finite() { dzs.push(qb - qa); }
            }
        }
        if dzs.len() < 6 { return None; }
        let dzs_len = dzs.len() as f32;
        let (med, mad) = median_and_mad(dzs);
        // Weight: more bins → higher weight; big scatter → lower weight.
        let w = dzs_len / (1.0 + (mad / 1.5).powi(2)); // 1.5 m scale
        return Some((med, w.max(1e-3)));
    }

    // Y‑touching edge (north/south)
    if (a_max.y - b_min.y).abs() < eps && a_max.x.max(b_max.x) > a_min.x.min(b_min.x) {
        let x0 = a_min.x.max(b_min.x);
        let x1 = a_max.x.min(b_max.x);
        if x1 <= x0 { return None; }

        // A's north strip, B's south strip
        let bins_a = compute_bins(
            a.border_pts,
            x0, x1, bin,
            &|p| p[1] > a_max.y - border_w && p[0] >= x0 && p[0] <= x1,
            &|p| p[0],
        );
        let bins_b = compute_bins(
            b.border_pts,
            x0, x1, bin,
            &|p| p[1] < b_min.y + border_w && p[0] >= x0 && p[0] <= x1,
            &|p| p[0],
        );

        let mut dzs = Vec::<f32>::new();
        for (ba, bb) in bins_a.into_iter().zip(bins_b.into_iter()) {
            if ba.len() >= min_pts_per_bin && bb.len() >= min_pts_per_bin {
                let mut va = ba; let mut vb = bb;
                let qa = percentile_in_place(&mut va, 0.20);
                let qb = percentile_in_place(&mut vb, 0.20);
                if qa.is_finite() && qb.is_finite() { dzs.push(qb - qa); }
            }
        }
        if dzs.len() < 6 { return None; }
        let dzs_len = dzs.len() as f32;
        let (med, mad) = median_and_mad(dzs);
        let w = dzs_len / (1.0 + (mad / 1.5).powi(2));
        return Some((med, w.max(1e-3)));
    }

    None
}

// Z-bias solver for cross-tile altitude consistency
struct MatchEdge { a: usize, b: usize, dz: f32, w: f32 }

// Weighted least squares on a graph Laplacian: minimize ∑ w_e (b_j - b_i - dz_e)^2,
// with b_0 anchored to 0.
fn solve_biases_wls(n: usize, edges: &[MatchEdge]) -> Vec<f32> {
    if n == 0 { return vec![]; }
    if n == 1 { return vec![0.0]; }

    // Build l * b = rhs
    let mut l = vec![vec![0.0f32; n]; n];
    let mut rhs = vec![0.0f32; n];

    for e in edges {
        let i = e.a; let j = e.b; let w = e.w.max(1e-6);
        l[i][i] += w; l[j][j] += w;
        l[i][j] -= w; l[j][i] -= w;
        rhs[i] -= w * e.dz;
        rhs[j] += w * e.dz;
    }

    // Anchor tile 0
    for k in 0..n { l[0][k] = 0.0; }
    l[0][0] = 1.0; rhs[0] = 0.0;

    // Dense Gaussian elimination (n is usually small)
    let mut a = l; let mut b = rhs;
    for p in 0..n {
        let pivot = a[p][p].abs().max(1e-6);
        // normalize row p
        let inv = 1.0 / pivot;
        for c in p..n { a[p][c] *= inv; }
        b[p] *= inv;
        // eliminate rows below
        for r in (p+1)..n {
            let f = a[r][p];
            if f.abs() < 1e-9 { continue; }
            for c in p..n { a[r][c] -= f * a[p][c]; }
            b[r] -= f * b[p];
        }
    }
    // back substitution
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i+1)..n { s -= a[i][j] * x[j]; }
        x[i] = s / a[i][i].max(1e-6);
    }
    x
}

#[derive(Clone, Copy)]
struct OffsetEdge { a: usize, b: usize, dx: f32, dy: f32, dz: f32, w: f32 }

// small grid index for fast NN search
struct Grid {
    cell: f32,
    map: std::collections::HashMap<(i32,i32), Vec<usize>>,
}
impl Grid {
    fn build(samples: &[[f32;3]], cell: f32) -> Self {
        let mut map = std::collections::HashMap::new();
        for (i, s) in samples.iter().enumerate() {
            let ix = (s[0] / cell).floor() as i32;
            let iy = (s[1] / cell).floor() as i32;
            map.entry((ix,iy)).or_insert_with(Vec::new).push(i);
        }
        Self { cell, map }
    }
    fn neighbors(&self, x: f32, y: f32, r: f32) -> impl Iterator<Item=&usize> + '_ {
        let rx = (r / self.cell).ceil() as i32;
        let ix = (x / self.cell).floor() as i32;
        let iy = (y / self.cell).floor() as i32;
        (-rx..=rx).flat_map(move |dx| {
            (-rx..=rx).filter_map(move |dy| self.map.get(&(ix+dx, iy+dy))).flatten()
        })
    }
}

fn median(mut v: Vec<f32>) -> Option<f32> {
    if v.is_empty() { return None; }
    let mid = v.len()/2;
    v.select_nth_unstable_by(mid, |a,b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(v[mid])
}

// robustly estimate (Δx,Δy,Δz) between two tiles from edge samples
fn estimate_pair_offset(a: &TileDraw, b: &TileDraw) -> Option<(f32, f32, f32, f32)> {
    if a.edge_samples.is_empty() || b.edge_samples.is_empty() {
        return None;
    }

    // PATCH 2: Loosen the matcher gates so edges are actually found
    let r = 6.0_f32;         // was 2.0
    let cell = 3.0_f32;      // was 1.5
    let idx_b = Grid::build(&b.edge_samples, cell);

    let mut dxs = Vec::new();
    let mut dys = Vec::new();
    let mut dzs = Vec::new();

    for sa in &a.edge_samples {
        let (ax, ay, az) = (sa[0], sa[1], sa[2]);

        let mut best = None::<(f32, &[f32; 3])>; // (d2, sampleB)

        for &j in idx_b.neighbors(ax, ay, r) {
            let sb = &b.edge_samples[j];
            let d2 = (sb[0] - ax).powi(2) + (sb[1] - ay).powi(2);

            if d2 <= r * r {
                if best.map(|(d, _)| d2 < d).unwrap_or(true) {
                    best = Some((d2, sb));
                }
            }
        }

        if let Some((_, sb)) = best {
            dxs.push(sb[0] - ax);
            dys.push(sb[1] - ay);
            dzs.push(sb[2] - az);
        }
    }

    dbg!(&dxs, &dys, &dzs);

    // Keep only if we have enough correspondences
    if dzs.len() < 60 {  // was 200
        return None;
    }

    // Median to reject outliers, then a light 20% trim around the median
    let mdx = median(dxs.clone())?;
    let mdy = median(dys.clone())?;
    let mdz = median(dzs.clone())?;

    let trim = 0.20;

    let dx2 = dxs
        .into_iter()
        .filter(|d| (d - mdx).abs() < 3.0)
        .collect::<Vec<_>>();

    let dy2 = dys
        .into_iter()
        .filter(|d| (d - mdy).abs() < 3.0)
        .collect::<Vec<_>>();

    let dz2 = dzs
        .into_iter()
        .filter(|d| (d - mdz).abs() < 5.0)
        .collect::<Vec<_>>();

    let mut dx2_sorted = dx2.clone();
    let mut dy2_sorted = dy2.clone();
    let mut dz2_sorted = dz2.clone();

    dx2_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    dy2_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    dz2_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let cutx = (dx2.len() as f32 * trim * 0.5) as usize;
    let cuty = (dy2.len() as f32 * trim * 0.5) as usize;
    let cutz = (dz2.len() as f32 * trim * 0.5) as usize;

    let dx2_trimmed =
        if cutx < dx2.len() && 2 * cutx < dx2.len() {
            dx2_sorted[cutx..dx2.len() - cutx].to_vec()
        } else {
            dx2
        };

    let dy2_trimmed =
        if cuty < dy2.len() && 2 * cuty < dy2.len() {
            dy2_sorted[cuty..dy2.len() - cuty].to_vec()
        } else {
            dy2
        };

    let dz2_trimmed =
        if cutz < dz2.len() && 2 * cutz < dz2.len() {
            dz2_sorted[cutz..dz2.len() - cutz].to_vec()
        } else {
            dz2
        };

    let dz2_len = dz2_trimmed.len();

    let mx = median(dx2_trimmed).unwrap_or(mdx);
    let my = median(dy2_trimmed).unwrap_or(mdy);
    let mz = median(dz2_trimmed).unwrap_or(mdz);

    let w = dz2_len as f32; // weight by usable matches

    Some((mx, my, mz, w.max(1.0)))
}

// Generic small iterative solver: bias[b] - bias[a] ≈ d (weighted)
fn solve_biases(n: usize, edges: &[(usize, usize, f32, f32)]) -> Vec<f32> {
    let mut bias = vec![0.0f32; n];

    for _ in 0..40 {
        let mut next_bias = bias.clone();

        for &(a, b, target_diff, weight) in edges {
            if a >= n || b >= n {
                continue;
            }

            let current_diff = bias[b] - bias[a];
            let error = target_diff - current_diff;
            let adjustment = 0.12 * weight * error; // damping factor

            if a != 0 {
                next_bias[a] -= 0.5 * adjustment;
            }

            if b != 0 {
                next_bias[b] += 0.5 * adjustment;
            }
        }

        bias = next_bias;
    }

    bias
}

// FULL estimator: returns (bias_x, bias_y, bias_z)
fn estimate_all_biases(tiles: &[TileDraw]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = tiles.len();
    let mut ex = Vec::<(usize, usize, f32, f32)>::new();
    let mut ey = Vec::<(usize, usize, f32, f32)>::new();
    let mut ez = Vec::<(usize, usize, f32, f32)>::new();

    // Neighbor detection via AABBs with tolerance
    const EPS: f32 = 2.5; // Allow small XY gaps
    for a in 0..n {
        for b in (a + 1)..n {
            let ta = &tiles[a];
            let tb = &tiles[b];

            let ax0 = ta.dmin_world.x;
            let ax1 = ta.dmax_world.x;
            let ay0 = ta.dmin_world.y;
            let ay1 = ta.dmax_world.y;

            let bx0 = tb.dmin_world.x;
            let bx1 = tb.dmax_world.x;
            let by0 = tb.dmin_world.y;
            let by1 = tb.dmax_world.y;

            let x_overlap = ax1 >= bx0 - EPS && bx1 >= ax0 - EPS;
            let y_overlap = ay1 >= by0 - EPS && by1 >= ay0 - EPS;

            if !(x_overlap && y_overlap) {
                continue;
            }

            if let Some((dx, dy, dz, w)) = estimate_pair_offset(ta, tb) {
                ex.push((a, b, dx, w));
                ey.push((a, b, dy, w));
                ez.push((a, b, dz, w));
            }
        }
    }

    // Fallback if no robust edges found: keep zeros
    if ex.is_empty() && ey.is_empty() && ez.is_empty() {
        return (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    }

    (
        solve_biases(n, &ex),
        solve_biases(n, &ey),
        solve_biases(n, &ez),
    )
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
    let phi = lat_deg.to_radians();

    111_132.92 - 559.82 * (2.0 * phi).cos() + 1.175 * (4.0 * phi).cos() - 0.0023 * (6.0 * phi).cos()
}

fn meters_per_deg_lon(lat_deg: f64) -> f64 {
    let phi = lat_deg.to_radians();

    111_412.84 * phi.cos() - 93.5 * (3.0 * phi).cos() + 0.118 * (5.0 * phi).cos()
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
    // Biases for consistent alignment across tiles
    z_bias: f32,
    xy_bias: glam::Vec2,          // NEW: XY alignment bias
    edge_samples: Vec<[f32; 3]>,  // NEW: (x,y,z) near the tile border
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

    // Zoom
    zoom_factor: f32,
    default_distance: f32,

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
            zoom_factor: 1.0,
            default_distance: 500.0, // reasonable default distance
        }
    }

    fn upload_semantic_mask_to_texture(
        &self,
        sm: &point_cloud::SemanticMask,
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
        let width = sm.width as u32;
        let height = sm.height as u32;

        // Create texture
        let texture = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SMC1 Labels"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Create 256-byte aligned staging buffer
        let bytes_per_row = ((width + 255) / 256) * 256;
        let mut staging_data = vec![0u8; (bytes_per_row * height) as usize];

        // Copy row data into staging buffer
        for row in 0..height as usize {
            let source_row = &sm.data[row * width as usize..(row + 1) * width as usize];
            let destination_row = &mut staging_data[row * bytes_per_row as usize..row * bytes_per_row as usize + width as usize];
            destination_row.copy_from_slice(source_row);
        }

        // Upload texture data
        self.gpu.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &staging_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        // Create view and sampler
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = self.gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SMC1 Nearest"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        (texture, view, sampler)
    }

    fn scan_hypc_files(root: &str) -> Vec<String> {
        let mut out = Vec::new();
        for entry in WalkDir::new(root).into_iter().filter_map(Result::ok) {
            if entry.file_type().is_file() {
                if let Some(extension) = entry.path().extension().and_then(|s| s.to_str()) {
                    if extension.eq_ignore_ascii_case("hypc") {
                        out.push(entry.path().to_string_lossy().to_string());
                    }
                }
            }
        }
        out
    }

    fn build_all_tiles(&mut self, root: &str) -> anyhow::Result<()> {
        let paths = Self::scan_hypc_files(root);
        if paths.is_empty() {
            return Ok(());
        }

        // 1) Load all tiles with GEOT
        struct Loaded {
            name: String,
            pc: QuantizedPointCloud,
        }

        let mut loaded = Vec::<Loaded>::new();
        for p in paths {
            match QuantizedPointCloud::load_hypc(&p) {
                Ok(pc) => {
                    if pc.geog_bbox_deg.is_some() {
                        let file_name = std::path::Path::new(&p)
                            .file_name()
                            .unwrap()
                            .to_string_lossy()
                            .to_string();
                        loaded.push(Loaded { name: file_name, pc });
                    } else {
                        log::warn!("Skipping {}: no GEOT footer", p);
                    }
                }
                Err(e) => log::warn!("Skipping {}: {}", p, e),
            }
        }

        if loaded.is_empty() {
            return Ok(());
        }

        // 2) Union GEOT and anchor
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

        self.mdeg_lon_ref = meters_per_deg_lon(self.lat0);
        self.mdeg_lat_ref = meters_per_deg_lat(self.lat0);

        self.geot_union = Some(geot_union.clone());

        // 3) Per-tile transform to world (meters), build GPU buffers and per-tile bind groups
        let mut tiles = Vec::<TileDraw>::new();
        let mut world_min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut world_max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

        use holographic_shaders::Instance;

        for l in loaded {
            let pc = l.pc;
            let geot = pc.geog_bbox_deg.clone().unwrap();
            let decode_min = pc.decode_min;
            let decode_max = pc.decode_max;

            // Compute a consistent Z scale across all tiles
            let lon_span_union = (geot_union.lon_max - geot_union.lon_min) as f32;
            let lat_span_union = (geot_union.lat_max - geot_union.lat_min) as f32;

            let decode_span_x = (decode_max.x - decode_min.x).abs();
            let decode_span_y = (decode_max.y - decode_min.y).abs();

            let sz_ref = (self.mdeg_lon_ref as f32 * lon_span_union + self.mdeg_lat_ref as f32 * lat_span_union)
                / (decode_span_x + decode_span_y).max(1e-6) * 0.5;

            let mut instance_positions = Vec::<[f32; 3]>::with_capacity(pc.positions.len());

            let dx = (decode_max.x - decode_min.x).max(f32::EPSILON);
            let dy = (decode_max.y - decode_min.y).max(f32::EPSILON);

            let lon_span = (geot.lon_max - geot.lon_min).max(f64::EPSILON);
            let lat_span = (geot.lat_max - geot.lat_min).max(f64::EPSILON);

            for p in &pc.positions {
                // decode-XY -> lon/lat
                let lon = geot.lon_min + ((p.x - decode_min.x) as f64 / dx as f64) * lon_span;
                let lat = geot.lat_min + ((p.y - decode_min.y) as f64 / dy as f64) * lat_span;

                // lon/lat -> world meters about (lon0, lat0)
                let x_world = ((lon - self.lon0) * self.mdeg_lon_ref) as f32;
                let y_world = ((lat - self.lat0) * self.mdeg_lat_ref) as f32;
                let z_world = (p.z - decode_min.z) * sz_ref;

                // Store position for later bias application
                instance_positions.push([x_world, y_world, z_world]);

                world_min.x = world_min.x.min(x_world);
                world_min.y = world_min.y.min(y_world);
                world_min.z = world_min.z.min(z_world);

                world_max.x = world_max.x.max(x_world);
                world_max.y = world_max.y.max(y_world);
                world_max.z = world_max.z.max(z_world);
            }

            // Per-tile world bounding box for SMC1 UV + height modulation
            let dmin_world = Vec3::new(
                ((geot.lon_min - self.lon0) * self.mdeg_lon_ref) as f32,
                ((geot.lat_min - self.lat0) * self.mdeg_lat_ref) as f32,
                0.0,
            );

            let dmax_world = Vec3::new(
                ((geot.lon_max - self.lon0) * self.mdeg_lon_ref) as f32,
                ((geot.lat_max - self.lat0) * self.mdeg_lat_ref) as f32,
                (decode_max.z - decode_min.z) * sz_ref,
            );

            // Collect edge samples for robust alignment
            let belt_width = 6.0_f32;       // meters from the border
            let sample_stride = 8;           // decimate to keep it light
            let mut edge_samples = Vec::<[f32; 3]>::new();

            for (idx, p) in instance_positions.iter().enumerate() {
                let x = p[0];
                let y = p[1];
                let z = p[2];

                let dx = (x - dmin_world.x).abs().min((dmax_world.x - x).abs());
                let dy = (y - dmin_world.y).abs().min((dmax_world.y - y).abs());

                if dx < belt_width || dy < belt_width {
                    if (idx % sample_stride) == 0 {
                        edge_samples.push([x, y, z]);
                    }
                }
            }

            // Create vertex buffer with instance positions
            let instances: Vec<Instance> = instance_positions
                .iter()
                .map(|pos| Instance {
                    position: *pos,
                })
                .collect();

            let vb = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Instances({})", l.name)),
                contents: bytemuck::cast_slice(&instances),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

            // Semantics: upload mask or use fallback
            let (sem_tex, sem_view, sem_samp) = if let Some(sm) = pc.semantic_mask.as_ref() {
                let (t, v, s) = self.upload_semantic_mask_to_texture(sm);
                (Some(t), v, s)
            } else {
                let fallback_view = self
                    .sem_fallback_tex
                    .create_view(&wgpu::TextureViewDescriptor::default());

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

            // Per-tile bind group for holographic shader
            let holo_bg = self
                .holo
                .bind_group_for_semantics(&self.gpu.device, &sem_view, &sem_samp);

            // Create per-tile UBO for world bounds (used in SMC1 UV mapping)
            let tile_uniforms = holographic_shaders::TileUniforms {
                dmin_world: [dmin_world.x, dmin_world.y],
                dmax_world: [dmax_world.x, dmax_world.y],
                z_bias: 0.0,
                _pad_after_z: 0.0,
                xy_bias: [0.0, 0.0],
                _padding: [0u32; 22],
            };

            let tile_ubo = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Tile UBO ({})", l.name)),
                contents: bytemuck::bytes_of(&tile_uniforms),
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
                z_bias: 0.0,
                xy_bias: glam::Vec2::ZERO,
                edge_samples,
            });
        }

        // Compute robust (Δx, Δy, Δz) alignment biases from border matches
        let (bias_x, bias_y, bias_z) = estimate_all_biases(&tiles);

        // Apply biases and update UBOs for shader consumption
        for (i, tile) in tiles.iter_mut().enumerate() {
            tile.xy_bias = glam::Vec2::new(
                *bias_x.get(i).unwrap_or(&0.0),
                *bias_y.get(i).unwrap_or(&0.0),
            );

            tile.z_bias = *bias_z.get(i).unwrap_or(&0.0);

            // Shift the UV mapping frame to match the new XY bias
            let dmin_biased = tile.dmin_world + glam::Vec3::new(tile.xy_bias.x, tile.xy_bias.y, 0.0);
            let dmax_biased = tile.dmax_world + glam::Vec3::new(tile.xy_bias.x, tile.xy_bias.y, 0.0);

            let tile_uniforms = holographic_shaders::TileUniforms {
                dmin_world: [dmin_biased.x, dmin_biased.y],
                dmax_world: [dmax_biased.x, dmax_biased.y],
                z_bias: tile.z_bias,
                _pad_after_z: 0.0,
                xy_bias: [tile.xy_bias.x, tile.xy_bias.y],
                _padding: [0u32; 22],
            };

            self.gpu
                .queue
                .write_buffer(&tile.tile_ubo, 0, bytemuck::bytes_of(&tile_uniforms));
        }

        // Recompute global world bounds after applying tile biases
        let mut world_min_biased = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut world_max_biased = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

        for tile in &tiles {
            let min_z = tile.dmin_world.z + tile.z_bias;
            let max_z = tile.dmax_world.z + tile.z_bias;

            world_min_biased.x = world_min_biased.x.min(tile.dmin_world.x);
            world_min_biased.y = world_min_biased.y.min(tile.dmin_world.y);
            world_min_biased.z = world_min_biased.z.min(min_z);

            world_max_biased.x = world_max_biased.x.max(tile.dmax_world.x);
            world_max_biased.y = world_max_biased.y.max(tile.dmax_world.y);
            world_max_biased.z = world_max_biased.z.max(max_z);
        }

        self.tiles = tiles;
        self.world_min = world_min_biased;
        self.world_max = world_max_biased;

        // Position the camera to frame the complete dataset
        let extent = self.world_max - self.world_min;
        let max_dimension = extent.x.max(extent.y).max(extent.z);
        let distance = max_dimension * 1.2;

        self.default_distance = distance;

        let center = 0.5 * (self.world_min + self.world_max);

        self.cam_target = center;
        self.cam_up = Vec3::new(0.0, 0.0, 1.0);
        self.cam_pos = center + Vec3::new(0.25 * distance, -1.35 * distance, 0.9 * distance);

        Ok(())
    }

    fn update_uniforms(&mut self) {
        let view = Mat4::look_at_rh(self.cam_pos, self.cam_target, self.cam_up);

        let aspect_ratio = self.gpu.config.width as f32 / self.gpu.config.height as f32;
        let proj_gl = Mat4::perspective_rh_gl(self.fov_y, aspect_ratio, self.near, self.far);
        let proj = OPENGL_TO_WGPU_MATRIX * proj_gl;

        let time = self.start.elapsed().as_secs_f32();
        let (dmin, dmax) = (self.world_min, self.world_max);

        let holo_uniforms = HoloUniforms {
            view,
            proj,
            viewport: [self.gpu.config.width as f32, self.gpu.config.height as f32],
            base_size_px: self.holo.uniforms.base_size_px,
            size_atten: self.holo.uniforms.size_atten,
            time,
            near: self.near,
            far: self.far,
            _pad: 0.0,
            decode_min: dmin,
            decode_max: dmax,
            cyan: self.holo.uniforms.cyan,
            red: self.holo.uniforms.red,
            ..Default::default()
        };

        self.holo.update_uniforms(&self.gpu.queue, holo_uniforms);

        let inv_size = [
            1.0 / self.gpu.config.width as f32,
            1.0 / self.gpu.config.height as f32,
        ];

        self.edl.update_uniforms(
            &self.gpu.queue,
            EdlUniforms {
                inv_size,
                strength: 1.15,
                radius_px: 1.25,
            },
        );

        self.rgb.update_uniforms(
            &self.gpu.queue,
            RgbShiftUniforms {
                inv_size,
                amount: 0.0018,
                angle: 0.0,
            },
        );

        self.crt.update_uniforms(
            &self.gpu.queue,
            CrtUniforms {
                inv_size,
                time,
                intensity: self.crt.uniform.intensity,
                vignette: self.crt.uniform.vignette,
                _padding: 0.0,
            },
        );

        // Grid: use union GEOT + union world bounds
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

        // Place plane just below global min z
        let extent = (dmax - dmin).abs();
        let z_offset = extent.z.max(1e-6) * 0.02;

        self.grid.update_uniforms(
            &self.gpu.queue,
            ground_grid::GridUniforms {
                view,
                proj,
                decode_min: dmin,
                _pad0: 0.0,
                decode_max: dmax,
                _pad1: 0.0,
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
                _pad2: 0.0,
                _pad3: 0.0,
            },
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
        self.zoom_factor = 1.0; // Reset zoom

        if !self.tiles.is_empty() {
            let dist = self.default_distance;
            let center = 0.5 * (self.world_min + self.world_max);
            self.cam_target = center;
            self.cam_pos = center + Vec3::new(0.0, -dist * 1.35, dist * 0.9);
        }
    }

    fn clamp_pitch(dir: Vec3) -> Vec3 {
        // Avoid flipping over (keep ~[-85°, +85°] from horizon)
        const MIN_Z: f32 = 0.05;

        if dir.z.abs() < MIN_Z {
            let _s = (MIN_Z / dir.z.abs()).max(1.0);

            return Vec3::new(
                dir.x,
                dir.y,
                dir.z.signum() * MIN_Z,
            )
            .normalize();
        }

        dir
    }

    fn handle_mouse_button(&mut self, btn: MouseButton, state: ElementState) {
        if btn == MouseButton::Left {
            self.mouse_down =
                state == ElementState::Pressed;
        }
    }

    fn handle_scroll(&mut self, delta: f32) {
        // Exponential dolly zoom: positive delta -> zoom in, negative -> zoom out.
        // Winit docs: positive scroll on Pixel/Line delta means "content moves right/down"
        // (we invert with the minus sign so positive delta zooms IN).
        // https://docs.rs/winit/latest/winit/event/enum.MouseScrollDelta.html
        const ZOOM_STEP: f32 = 1.20; // ~20% per notch
        let scale = ZOOM_STEP.powf(-delta); // >1 => zoom out, <1 => zoom in

        // Work in offset space to avoid normalizing near zero distance.
        let mut offset = self.cam_pos - self.cam_target; // world units (meters)
        let old_len = offset.length();

        // If degenerate, pick a stable fallback direction (-Y) with a reasonable radius.
        if old_len < 1e-6 {
            let fallback = self.default_distance.max(self.near * 2.0);
            offset = glam::Vec3::new(0.0, -1.0, 0.0) * fallback;
        }

        // Apply zoom scale.
        offset *= scale;

        // Clamp to the frustum range to avoid crossing the near/far planes.
        let new_len = offset.length();
        let min_distance = (self.near * 2.0).max(0.10);   // stay away from near plane
        let max_distance = (self.far * 0.90).max(min_distance);  // keep some headroom to far plane

        if new_len < min_distance {
            let direction = offset / new_len.max(1e-6);
            offset = direction * min_distance;
        } else if new_len > max_distance {
            offset = offset / new_len * max_distance;
        }

        // Update camera position and UI zoom factor.
        self.cam_pos = self.cam_target + offset;
        if self.default_distance > 0.0 {
            self.zoom_factor = self.default_distance / offset.length();
        }
    }

    fn handle_cursor(&mut self, xy: (f64, f64)) {
        if let Some(last) = self.last_mouse {
            if self.mouse_down {
                let dx = (xy.0 - last.0) as f32 * 0.01;
                let dy = (xy.1 - last.1) as f32 * 0.01;

                let to_target = (self.cam_target - self.cam_pos).normalize();
                let right = to_target.cross(self.cam_up).normalize();

                let yaw = Mat4::from_axis_angle(self.cam_up, -dx);
                let mut dir = yaw.mul_vec4(Vec4::from((to_target, 0.0))).xyz();

                let pitch = Mat4::from_axis_angle(right, -dy);
                dir = pitch.mul_vec4(Vec4::from((dir, 0.0))).xyz();

                dir = Self::clamp_pitch(dir);

                let current_dist = (self.cam_pos - self.cam_target).length();
                self.cam_pos = self.cam_target - dir * current_dist;
            }
        }

        self.last_mouse = Some(xy);
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.update_uniforms();

        let frame = self.gpu.surface.get_current_texture()?;
        let swap_view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Frame"),
        });

        // PASS 1: Points -> scene targets (MSAA with resolve)
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Scene"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.tgt.scene_color_view,
                        resolve_target: Some(&self.tgt.scene_color_resolved_view),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0196,
                                g: 0.0196,
                                b: 0.0275,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.tgt.scene_depthlin_view,
                        resolve_target: Some(&self.tgt.scene_depthlin_resolved_view),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.tgt.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Draw ground grid first
            rp.set_pipeline(&self.grid.pipeline);
            rp.set_bind_group(0, &self.grid.bind_group, &[]);
            rp.set_vertex_buffer(0, self.grid.quad_vb.slice(..));
            rp.draw(0..6, 0..1);

            // Draw all tiles
            rp.set_pipeline(&self.holo.pipeline);
            rp.set_vertex_buffer(0, self.holo.quad_vb.slice(..));

            for t in &self.tiles {
                // Use pre-built bind groups that already contain the per-tile uniform data
                rp.set_bind_group(0, &t.holo_bg, &[]);  // Global UBO + per-tile SMC1 texture/sampler
                rp.set_bind_group(1, &t.tile_bg, &[]);  // Per-tile UV bounds
                rp.set_vertex_buffer(1, t.vb.slice(..));
                rp.draw(0..6, 0..t.count);
            }
        }

        // PASS 2: EDL(color, depthlin) -> post_edl
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("EDL"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.tgt.post_edl_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rp.set_pipeline(&self.edl.pipeline);
            rp.set_bind_group(0, &self.edl.bind_group, &[]);
            rp.set_vertex_buffer(0, self.edl.fsq_vb.slice(..));
            rp.draw(0..3, 0..1);
        }

        // PASS 3: RGB shift -> post_rgb
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("RGBShift"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.tgt.post_rgb_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rp.set_pipeline(&self.rgb.pipeline);
            rp.set_bind_group(0, &self.rgb.bind_group, &[]);
            rp.set_vertex_buffer(0, self.rgb.fsq_vb.slice(..));
            rp.draw(0..3, 0..1);
        }

        // PASS 4: CRT / scanlines -> swapchain
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("CRT"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &swap_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
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
        self.draw_hud(&mut encoder, &swap_view);

        self.gpu.queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        Ok(())
    }

    fn draw_hud(&mut self, encoder: &mut wgpu::CommandEncoder, swap_view: &wgpu::TextureView) {
        let egui_input = self.egui_state.take_egui_input(&self.window);
        self.egui_ctx.begin_frame(egui_input);

        // Corner brackets & dot painter
        {
            let painter = self.egui_ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("hud_lines"),
            ));

            let rect = self.egui_ctx.screen_rect();
            let color = egui::Color32::from_rgba_unmultiplied(45, 247, 255, 200);

            let thickness = 2.0;
            let margin = 26.0;
            let length = 140.0;

            // Top-left bracket
            painter.line_segment(
                [egui::pos2(margin, margin), egui::pos2(margin + length, margin)],
                (thickness, color),
            );
            painter.line_segment(
                [egui::pos2(margin, margin), egui::pos2(margin, margin + length)],
                (thickness, color),
            );

            // Top-right bracket
            painter.line_segment(
                [
                    egui::pos2(rect.max.x - margin - length, margin),
                    egui::pos2(rect.max.x - margin, margin),
                ],
                (thickness, color),
            );
            painter.line_segment(
                [
                    egui::pos2(rect.max.x - margin, margin),
                    egui::pos2(rect.max.x - margin, margin + length),
                ],
                (thickness, color),
            );

            // Bottom-left bracket
            painter.line_segment(
                [
                    egui::pos2(margin, rect.max.y - margin),
                    egui::pos2(margin + length, rect.max.y - margin),
                ],
                (thickness, color),
            );
            painter.line_segment(
                [
                    egui::pos2(margin, rect.max.y - margin - length),
                    egui::pos2(margin, rect.max.y - margin),
                ],
                (thickness, color),
            );

            // Bottom-right bracket
            painter.line_segment(
                [
                    egui::pos2(rect.max.x - margin - length, rect.max.y - margin),
                    egui::pos2(rect.max.x - margin, rect.max.y - margin),
                ],
                (thickness, color),
            );
            painter.line_segment(
                [
                    egui::pos2(rect.max.x - margin, rect.max.y - margin - length),
                    egui::pos2(rect.max.x - margin, rect.max.y - margin),
                ],
                (thickness, color),
            );

            // Top-center dot
            painter.circle_filled(egui::pos2(rect.center().x, 16.0), 3.0, color);
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
                                .color(egui::Color32::from_rgb(45, 247, 255))
                                .size(16.0)
                                .strong(),
                        );

                        let total_points: u32 = self.tiles.iter().map(|tile| tile.count).sum();
                        let altitude = self.cam_pos.z.round() as i32;

                        ui.label(
                            RichText::new(format!("RESOLUTION: {:>11} POINTS", total_points))
                                .monospace()
                                .color(egui::Color32::from_rgb(45, 247, 255)),
                        );

                        ui.label(
                            RichText::new(format!("ALTITUDE: {}M", altitude))
                                .monospace()
                                .color(egui::Color32::from_rgb(45, 247, 255)),
                        );

                        ui.label(
                            RichText::new("STATUS:  SCAN  COMPLETE")
                                .monospace()
                                .color(egui::Color32::from_rgb(45, 247, 255)),
                        );
                    });
                });
        }

        // Render egui to the swapchain
        let egui_output = self.egui_ctx.end_frame();
        let shapes = self.egui_ctx.tessellate(
            egui_output.shapes,
            self.egui_ctx.pixels_per_point(),
        );

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [
                self.gpu.config.width,
                self.gpu.config.height,
            ],
            pixels_per_point: self.egui_state.egui_ctx().pixels_per_point(),
        };

        for (id, delta) in &egui_output.textures_delta.set {
            self.egui_renderer.update_texture(
                &self.gpu.device,
                &self.gpu.queue,
                *id,
                delta,
            );
        }

        self.egui_renderer.update_buffers(
            &self.gpu.device,
            &self.gpu.queue,
            encoder,
            &shapes,
            &screen_descriptor,
        );

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("HUD"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: swap_view,
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

            self.egui_renderer.render(
                &mut render_pass,
                &shapes,
                &screen_descriptor,
            );
        }

        for id in &egui_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();

    let event_loop = EventLoop::new()?;

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Holographic City Viewer — Rust")
            .with_inner_size(LogicalSize::new(1280, 720))
            .build(&event_loop)?
    );

    let mut app = pollster::block_on(App::new(window.clone()));

    // Load all .hypc tiles found under ./hypc (or ../hypc)
    if std::path::Path::new("hypc").exists() {
        let _ = app.build_all_tiles("hypc");
    } else if std::path::Path::new("../hypc").exists() {
        let _ = app.build_all_tiles("../hypc");
    }

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(winit::event_loop::ControlFlow::Poll);

        match event {
            Event::WindowEvent { window_id, event } if window_id == window.id() => {
                // Let egui consume events first
                let egui_response = app.egui_state.on_window_event(&window, &event);
                if egui_response.consumed {
                    return;
                }

                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(size) => app.resize(size),
                    WindowEvent::RedrawRequested => {
                        if let Err(e) = app.render() {
                            match e {
                                wgpu::SurfaceError::Lost => app.resize(app.gpu.size),
                                wgpu::SurfaceError::OutOfMemory => elwt.exit(),
                                _ => eprintln!("{e:?}"),
                            }
                        }
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                            elwt.exit();
                        }

                        if event.physical_key == PhysicalKey::Code(KeyCode::KeyN)
                            && event.state == ElementState::Pressed
                        {
                            app.snap_north_up();
                        }

                        // Toggle grid visibility
                        if event.physical_key == PhysicalKey::Code(KeyCode::KeyG)
                            && event.state == ElementState::Pressed
                        {
                            app.grid_enabled = !app.grid_enabled;
                        }
                    }
                    WindowEvent::MouseInput { button, state, .. } => {
                        app.handle_mouse_button(button, state);
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        app.handle_cursor((position.x, position.y));
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        let scroll_delta = match delta {
                            winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                            // Approximate: 120 px per wheel "notch" on Windows; device-dependent elsewhere
                            winit::event::MouseScrollDelta::PixelDelta(pos) => (pos.y as f32) / 120.0,
                        };
                        app.handle_scroll(scroll_delta);
                    }
                    WindowEvent::DroppedFile(_) => {
                        // This functionality is now handled by build_all_tiles on startup
                        // Could be re-wired to rebuild the dataset dynamically
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    })?;

    #[allow(unreachable_code)]
    Ok(())
}
