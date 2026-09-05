//! Screen-space hierarchy traversal with a complete resident cut, asynchronous
//! bounded I/O, persistent coarse coverage and a byte-budgeted GPU LRU cache.
use super::{dataset::{Dataset, Node, Payload}, point_cloud::{PreparedTile, upload_tile}, types::TileGpu};
use crate::camera::Camera;
use anyhow::{ensure, Result};
use std::{cmp::Ordering, collections::{BinaryHeap, HashMap, HashSet, VecDeque}, path::{Path, PathBuf}, sync::{mpsc::{self, Receiver, SyncSender}, Arc, Mutex, RwLock}, time::Instant};

#[derive(Clone, Debug)]
pub struct StreamConfig {
    pub gpu_bytes: u64,
    pub io_bytes: u64,
    pub upload_bytes_per_frame: u64,
    pub point_budget: u64,
    pub draw_budget: usize,
    pub target_spacing_px: f64,
    pub workers: usize,
}
impl Default for StreamConfig {
    fn default() -> Self {
        Self { gpu_bytes: 512 * 1024 * 1024, io_bytes: 96 * 1024 * 1024,
            upload_bytes_per_frame: 16 * 1024 * 1024, point_budget: 3_000_000,
            draw_budget: 1500, target_spacing_px: 2.0, workers: 4 }
    }
}

#[derive(Default, Clone, Debug, serde::Serialize)]
pub struct StreamStats {
    pub effective_point_budget: u64,
    pub source_tiles: usize,
    pub source_points: u64,
    pub visible_points: u64,
    pub draw_calls: usize,
    pub resident_nodes: usize,
    pub gpu_bytes: u64,
    pub pending_nodes: usize,
    pub pending_bytes: u64,
    pub uploads: u64,
    pub evictions: u64,
    pub cancelled: u64,
    pub failures: usize,
    pub refinement_pending: usize,
    pub selection_ms: f64,
    pub update_ms: f64,
    pub uploaded_bytes: u64,
    pub loaded_bytes: u64,
    pub max_io_ms: f64,
}

#[derive(Copy, Clone, PartialEq)]
struct Priority { score: f64, id: u32 }
impl Eq for Priority {}
impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering { self.score.total_cmp(&other.score).then(self.id.cmp(&other.id)) }
}
impl PartialOrd for Priority { fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) } }

/// Pure CPU selection: no disk reads, GPU calls, or dependencies on residency.
#[derive(Default, Debug)]
pub struct Selection {
    pub leaves: HashSet<u32>,
    pub refined: HashSet<u32>,
    pub points: u64,
}

struct View {
    eye: glam::DVec3,
    rotation: glam::Mat4,
    focal_px: f64,
    tan_x: f64,
    tan_y: f64,
    margin: f64,
}
impl View {
    fn new(camera: &Camera, viewport: [f32; 2], margin: f64) -> Self {
        Self { eye: camera.ecef_m().into(), rotation: camera.view_ecef(),
            focal_px: viewport[1] as f64 * camera.proj.y_axis.y as f64 * 0.5,
            tan_x: 1.0 / camera.proj.x_axis.x as f64,
            tan_y: 1.0 / camera.proj.y_axis.y as f64, margin }
    }
    fn visible(&self, node: &Node) -> bool {
        let p = self.rotation.transform_vector3((node.center() - self.eye).as_vec3()).as_dvec3();
        let r = node.radius();
        let depth = -p.z;
        (node.center() - self.eye).length() - r < 300_000.0
            && depth + r > 0.1
            && p.x.abs() <= depth * self.tan_x * self.margin + r * (1.0 + (self.tan_x * self.margin).powi(2)).sqrt()
            && p.y.abs() <= depth * self.tan_y * self.margin + r * (1.0 + (self.tan_y * self.margin).powi(2)).sqrt()
    }
    fn score(&self, node: &Node) -> f64 {
        let distance = ((node.center() - self.eye).length() - node.radius()).max(1.0);
        node.spacing_m as f64 * self.focal_px / distance
    }
}

pub fn select(dataset: &Dataset, camera: &Camera, viewport: [f32; 2], config: &StreamConfig) -> Selection {
    select_view(dataset, &View::new(camera, viewport, 1.0), config, None)
}

fn select_view(dataset: &Dataset, view: &View, config: &StreamConfig, previous: Option<&HashSet<u32>>) -> Selection {
    let mut selected = Selection::default();
    let root = &dataset.nodes[dataset.root as usize];
    if !view.visible(root) { return selected; }
    selected.leaves.insert(dataset.root);
    selected.points = root.points as u64;
    let mut heap = BinaryHeap::new();
    heap.push(Priority { score: view.score(root), id: dataset.root });
    while let Some(Priority { score, id }) = heap.pop() {
        let n = &dataset.nodes[id as usize];
        let threshold = config.target_spacing_px * match previous {
            Some(refined) if refined.contains(&id) => 0.8,
            Some(_) => 1.1,
            None => 1.0,
        };
        if score <= threshold || n.children.is_empty() { continue; }
        let children: Vec<_> = n.children.iter().copied().filter(|&c| view.visible(&dataset.nodes[c as usize])).collect();
        let child_points: u64 = children.iter().map(|&c| dataset.nodes[c as usize].points as u64).sum();
        let candidate_points = selected.points - n.points as u64 + child_points;
        if candidate_points > config.point_budget || selected.leaves.len() - 1 + children.len() > config.draw_budget { continue; }
        selected.points = candidate_points;
        selected.leaves.remove(&id);
        selected.refined.insert(id);
        for child in children {
            selected.leaves.insert(child);
            heap.push(Priority { score: view.score(&dataset.nodes[child as usize]), id: child });
        }
    }
    selected
}

/// Select a complete cut of the visible tree: a parent is replaced only when
/// every visible immediate child is resident. Missing grandchildren retain their
/// own parent. Thus asynchronous completion cannot leave a missing tile region.
pub fn resident_cut(dataset: &Dataset, selection: &Selection, resident: &HashSet<u32>, camera: &Camera, viewport: [f32; 2]) -> (Vec<u32>, Vec<u32>) {
    let view = View::new(camera, viewport, 1.0);
    let mut draws = Vec::new();
    let mut needed = Vec::new();
    let mut stack = vec![dataset.root];
    while let Some(id) = stack.pop() {
        let n = &dataset.nodes[id as usize];
        if !view.visible(n) { continue; }
        if !resident.contains(&id) { needed.push(id); continue; }
        if selection.refined.contains(&id) {
            let children: Vec<_> = n.children.iter().copied().filter(|&c| view.visible(&dataset.nodes[c as usize])).collect();
            if children.iter().all(|c| resident.contains(c)) {
                stack.extend(children.into_iter().rev());
                continue;
            }
            needed.extend(children.into_iter().filter(|c| !resident.contains(c)));
        }
        draws.push(id);
    }
    (draws, needed)
}

/// Culling can leave a fallback parent slightly larger than its visible child
/// subset. Replan that transient cut until the actual draw count fits the cap.
fn bounded_resident_cut(dataset: &Dataset, selection: &Selection, resident: &HashSet<u32>, camera: &Camera, viewport: [f32; 2], config: &StreamConfig) -> (Vec<u32>, Vec<u32>) {
    let mut cut = resident_cut(dataset, selection, resident, camera, viewport);
    let mut reduced = config.clone();
    for _ in 0..8 {
        let actual: u64 = cut.0.iter().map(|&id| dataset.nodes[id as usize].points as u64).sum();
        if actual <= config.point_budget && cut.0.len() <= config.draw_budget { return cut; }
        reduced.point_budget = reduced.point_budget.saturating_sub(actual.saturating_sub(config.point_budget) + config.point_budget / 50)
            .max(dataset.nodes[dataset.root as usize].points as u64);
        let adjusted = select(dataset, camera, viewport, &reduced);
        cut = resident_cut(dataset, &adjusted, resident, camera, viewport);
    }
    // Root is pinned and validated to fit. This branch is only a last resort for
    // pathological custom hierarchies; retain coverage instead of dropping draws.
    (vec![dataset.root], cut.1)
}

struct Resident { tile: TileGpu, last_used: u64, pinned: bool }
struct Completion { id: u32, result: Result<Option<PreparedTile>>, ms: f64 }

pub struct StreamScene {
    pub dataset: Arc<Dataset>,
    pub config: StreamConfig,
    pub stats: StreamStats,
    resident: HashMap<u32, Resident>,
    resident_ids: HashSet<u32>,
    request: Option<SyncSender<u32>>,
    completed: Receiver<Completion>,
    needed: Arc<RwLock<HashSet<u32>>>,
    inflight: HashSet<u32>,
    ready: VecDeque<(u32, PreparedTile)>,
    failures: HashMap<u32, String>,
    pub draw_ids: Vec<u32>,
    frame: u64,
    last_target: glam::DVec3,
    velocity: glam::DVec3,
    last_update: Instant,
    previous_refined: HashSet<u32>,
    last_radius: f64,
    zoom_velocity: f64,
    frame_budget: super::frame_budget::FrameBudget,
}

impl StreamScene {
    pub fn open(cache: &Path, config: StreamConfig, device: &wgpu::Device, layout: &wgpu::BindGroupLayout, camera: &Camera, viewport: [f32; 2]) -> Result<Self> {
        ensure!((1..=8).contains(&config.workers), "Invalid I/O worker count");
        ensure!(config.gpu_bytes >= 64 * 1024 * 1024 && config.io_bytes >= 32 * 1024 * 1024, "Streaming budgets too small");
        let dataset = Arc::new(Dataset::open(cache)?);
        ensure!(config.point_budget >= dataset.nodes[dataset.root as usize].points as u64 && config.draw_budget > 0, "Render budget cannot hold root coverage");
        ensure!(dataset.nodes.iter().all(|n| n.bytes() <= config.io_bytes && n.bytes() <= config.gpu_bytes / 4), "Node cannot fit cache budget");
        let (send, jobs) = mpsc::sync_channel::<u32>(config.workers * 2);
        let jobs = Arc::new(Mutex::new(jobs));
        let (results, completed) = mpsc::sync_channel(config.workers * 2);
        let needed: Arc<RwLock<HashSet<u32>>> = Arc::new(RwLock::new(HashSet::new()));
        for worker in 0..config.workers {
            let (jobs, results, dataset, needed, cache) = (jobs.clone(), results.clone(), dataset.clone(), needed.clone(), PathBuf::from(cache));
            std::thread::Builder::new().name(format!("city-io-{worker}")).spawn(move || {
                loop {
                    let job = jobs.lock().unwrap().recv();
                    let Ok(id) = job else { break; };
                    let start = Instant::now();
                    let result = if needed.read().unwrap().contains(&id) {
                        dataset.read_node(&cache, id).map(Some)
                    } else { Ok(None) };
                    if results.send(Completion { id, result, ms: start.elapsed().as_secs_f64() * 1000.0 }).is_err() { break; }
                }
            })?;
        }
        let frame_budget = super::frame_budget::FrameBudget::new(config.point_budget);
        let mut scene = Self { dataset, config, stats: StreamStats::default(), resident: HashMap::new(), resident_ids: HashSet::new(),
            request: Some(send), completed, needed, inflight: HashSet::new(), ready: VecDeque::new(), failures: HashMap::new(), draw_ids: vec![], frame: 0,
            last_target: camera.target_ecef, velocity: glam::DVec3::ZERO, last_update: Instant::now(), previous_refined: HashSet::new(), last_radius: camera.radius_m, zoom_velocity: 0.0, frame_budget };
        // A small breadth-first overview is permanently resident. Startup never
        // touches finest-level HYPC and its byte cost is independent of city size.
        let mut queue = VecDeque::from([scene.dataset.root]);
        let mut pinned_bytes = 0;
        while let Some(id) = queue.pop_front() {
            let n = &scene.dataset.nodes[id as usize];
            if !matches!(n.payload, Payload::Packed { .. }) { continue; }
            if pinned_bytes + n.bytes() > 8 * 1024 * 1024 { break; }
            let prepared = scene.dataset.read_node(cache, id)?;
            let tile = upload_tile(device, layout, camera, &prepared, viewport);
            pinned_bytes += n.bytes();
            scene.resident.insert(id, Resident { tile, last_used: 0, pinned: true });
            scene.resident_ids.insert(id);
            queue.extend(n.children.iter().copied());
        }
        ensure!(scene.resident.contains_key(&scene.dataset.root), "Root coverage unavailable");
        scene.stats.gpu_bytes = pinned_bytes;
        scene.stats.source_tiles = scene.dataset.source_tiles;
        scene.stats.source_points = scene.dataset.source_points;
        Ok(scene)
    }

    fn evict_for(&mut self, bytes: u64, protected: &HashSet<u32>) -> bool {
        while self.stats.gpu_bytes + bytes > self.config.gpu_bytes {
            let victim = self.resident.iter().filter(|(id, r)| !r.pinned && !protected.contains(id)).min_by_key(|(id, r)| (r.last_used, **id)).map(|(&id, _)| id);
            let Some(id) = victim else { return false; };
            self.resident.remove(&id);
            self.resident_ids.remove(&id);
            self.stats.gpu_bytes -= self.dataset.nodes[id as usize].bytes();
            self.stats.evictions += 1;
        }
        true
    }

    pub fn update(&mut self, camera: &Camera, viewport: [f32; 2], device: &wgpu::Device, layout: &wgpu::BindGroupLayout) {
        let start = Instant::now();
        self.frame += 1;
        let mut effective = self.config.clone();
        effective.point_budget = self.frame_budget.current().min(self.config.point_budget)
            .max(self.dataset.nodes[self.dataset.root as usize].points as u64);
        self.stats.effective_point_budget = effective.point_budget;
        let selected = select_view(&self.dataset, &View::new(camera, viewport, 1.0), &effective, Some(&self.previous_refined));
        self.previous_refined.clone_from(&selected.refined);
        self.stats.selection_ms = start.elapsed().as_secs_f64() * 1000.0;
        let dt = start.duration_since(self.last_update).as_secs_f64().clamp(0.001, 0.1);
        self.last_update = start;
        let displacement = camera.target_ecef - self.last_target;
        self.last_target = camera.target_ecef;
        self.velocity = self.velocity * 0.75 + displacement / dt * 0.25;
        let mut wanted: HashSet<_> = selected.leaves.union(&selected.refined).copied().collect();
        // Keep a wider frustum warm; translate it toward predicted movement.
        // Its smaller point budget prevents prefetch from competing with detail.
        let rate = (camera.radius_m / self.last_radius).ln() / dt;
        self.zoom_velocity = self.zoom_velocity * 0.65 + rate.clamp(-8.0, 8.0) * 0.35;
        self.last_radius = camera.radius_m;
        let mut predicted = camera.clone();
        predicted.radius_m = (camera.radius_m * (self.zoom_velocity * 0.3).exp().clamp(0.3, 1.5)).clamp(5.0, 150_000.0);
        predicted.update();
        let lookahead = self.velocity * 0.3;
        if lookahead.length() < camera.radius_m * 2.0 { predicted.translate_surface(lookahead); }
        let mut prefetch_config = effective.clone();
        prefetch_config.point_budget = if self.zoom_velocity < -0.2 { effective.point_budget } else { (effective.point_budget / 3).max(100_000) };
        prefetch_config.target_spacing_px *= if self.zoom_velocity < -0.2 { 1.0 } else { 1.6 };
        let prefetch = select_view(&self.dataset, &View::new(&predicted, viewport, 1.35), &prefetch_config, None);
        wanted.extend(prefetch.leaves.union(&prefetch.refined).copied());
        *self.needed.write().unwrap() = wanted.clone();
        while let Ok(done) = self.completed.try_recv() {
            self.stats.max_io_ms = self.stats.max_io_ms.max(done.ms);
            match done.result {
                Ok(Some(tile)) if wanted.contains(&done.id) => {
                    self.stats.loaded_bytes += self.dataset.nodes[done.id as usize].bytes();
                    self.ready.push_back((done.id, tile));
                }
                Ok(_) => { self.inflight.remove(&done.id); self.stats.cancelled += 1; }
                Err(error) => {
                    self.inflight.remove(&done.id);
                    log::error!("LOD node {} failed; retaining ancestor coverage: {error:#}", done.id);
                    self.failures.insert(done.id, format!("{error:#}"));
                }
            }
        }
        let (old_cut, _) = resident_cut(&self.dataset, &selected, &self.resident_ids, camera, viewport);
        let protected: HashSet<_> = old_cut.into_iter().collect();
        // Prioritize current-view completions ahead of predictive work.
        self.ready.make_contiguous().sort_by_key(|(id, _)| !selected.leaves.contains(id) && !selected.refined.contains(id));
        let mut uploaded = 0u64;
        let upload_start = Instant::now();
        while let Some((id, _)) = self.ready.front() {
            let id = *id;
            let bytes = self.dataset.nodes[id as usize].bytes();
            if !wanted.contains(&id) || self.resident.contains_key(&id) {
                self.ready.pop_front(); self.inflight.remove(&id); self.stats.cancelled += 1; continue;
            }
            if uploaded > 0 && (uploaded + bytes > self.config.upload_bytes_per_frame || upload_start.elapsed().as_secs_f64() > 0.002) { break; }
            if !self.evict_for(bytes, &protected) { break; }
            let (_, prepared) = self.ready.pop_front().unwrap();
            let tile = upload_tile(device, layout, camera, &prepared, viewport);
            self.resident.insert(id, Resident { tile, last_used: self.frame, pinned: false });
            self.resident_ids.insert(id);
            self.inflight.remove(&id);
            self.stats.gpu_bytes += bytes;
            self.stats.uploads += 1;
            uploaded += bytes;
        }
        self.stats.uploaded_bytes = uploaded;
        let (draws, mut requests) = bounded_resident_cut(&self.dataset, &selected, &self.resident_ids, camera, viewport, &effective);
        self.stats.refinement_pending = requests.len();
        let (_, prefetch_requests) = resident_cut(&self.dataset, &prefetch, &self.resident_ids, &predicted, viewport);
        let view = View::new(camera, viewport, 1.0);
        requests.sort_by(|&a, &b| view.score(&self.dataset.nodes[b as usize]).total_cmp(&view.score(&self.dataset.nodes[a as usize])));
        requests.extend(prefetch_requests);
        let mut pending_bytes: u64 = self.inflight.iter().map(|&id| self.dataset.nodes[id as usize].bytes()).sum();
        for id in requests {
            if self.inflight.contains(&id) || self.resident.contains_key(&id) || self.failures.contains_key(&id) { continue; }
            let bytes = self.dataset.nodes[id as usize].bytes();
            if pending_bytes + bytes > self.config.io_bytes || self.inflight.len() >= self.config.workers * 2 { continue; }
            if self.request.as_ref().unwrap().try_send(id).is_ok() {
                self.inflight.insert(id); pending_bytes += bytes;
            }
        }
        for id in &draws { self.resident.get_mut(id).unwrap().last_used = self.frame; }
        self.draw_ids = draws;
        self.stats.visible_points = self.draw_ids.iter().map(|&id| self.dataset.nodes[id as usize].points as u64).sum();
        self.stats.draw_calls = self.draw_ids.len();
        self.stats.resident_nodes = self.resident.len();
        self.stats.pending_nodes = self.inflight.len();
        self.stats.pending_bytes = pending_bytes;
        self.stats.failures = self.failures.len();
        self.stats.update_ms = start.elapsed().as_secs_f64() * 1000.0;
    }

    pub fn frame_feedback(&mut self, frame_ms: f64, foreground: bool) {
        self.frame_budget.observe(frame_ms, foreground, self.config.point_budget);
    }

    /// Navigation uses only the resident height grid. Ground support is
    /// approximate; it shares the source's unmodified vertical reference.
    pub fn constrain_camera(&self, camera: &mut Camera, follow_ground: bool, dt: f64) {
        let Some(terrain) = &self.dataset.terrain else { return; };
        let up = glam::DVec3::from(terrain.up);
        if let Some(ground) = terrain.ground(camera.target_ecef) {
            let offset = (ground + up * 0.75 - camera.target_ecef).dot(up);
            let correction = if offset > 0.0 { offset } else if follow_ground { offset * (1.0 - (-dt.clamp(0.0, 0.1) / 0.12).exp()) } else { 0.0 };
            if correction.abs() > 1e-5 { camera.target_ecef += up * correction; camera.update(); }
        }
        let eye = glam::DVec3::from(camera.ecef_m());
        if let Some(ground) = terrain.ground(eye) {
            let lift = (ground + up * 1.5 - eye).dot(up);
            if lift > 0.0 { camera.target_ecef += up * lift; camera.update(); }
        }
    }

    pub fn draw_tiles(&self) -> impl Iterator<Item = (&TileGpu, f32)> {
        self.draw_ids.iter().map(|id| (&self.resident[id].tile, self.dataset.nodes[*id as usize].spacing_m))
    }

    pub fn retry_failed(&mut self) { self.failures.clear(); }
    pub fn errors(&self) -> impl Iterator<Item = (u32, &str)> { self.failures.iter().map(|(&id, e)| (id, e.as_str())) }
}

impl Drop for StreamScene {
    fn drop(&mut self) {
        self.needed.write().unwrap().clear();
        self.request.take(); // close the queue; workers finish bounded in-flight I/O
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::dataset::SourceFile;
    fn fixture() -> (Dataset, Camera) {
        let center = glam::DVec3::from(hypc::geodetic_to_ecef(52.52, 13.4, 40.0));
        let mut camera = Camera::new(52.52, 13.4, 1000.0,
            glam::Mat4::perspective_infinite_reverse_rh(60f32.to_radians(), 1.0, 0.1));
        camera.set_target_and_radius(center.into(), 1000.0);
        camera.elevation_rad = 70f64.to_radians(); camera.update();
        let mut nodes = vec![];
        for id in 0..7 {
            let children = match id { 2 => vec![0, 1], 5 => vec![3, 4], 6 => vec![2, 5], _ => vec![] };
            let leaf = children.is_empty();
            nodes.push(Node {
                anchor_units: center.to_array().map(|v| (v * 2000.0).round() as i64), units_per_meter: 2000,
                min: (center - glam::DVec3::splat(10.0)).into(), max: (center + glam::DVec3::splat(10.0)).into(),
                spacing_m: if leaf { 0.5 } else { 16.0 }, error_bound_m: 0.0,
                points: if leaf { 1000 } else { 100 }, children,
                payload: if leaf { Payload::Source { source: SourceFile { path: "unused.hypc".into(), bytes: 0, modified_ns: 0 }, crc32: 0 } }
                    else { Payload::Packed { offset: 8, crc32: 0 } },
            });
        }
        (Dataset { version: 1, source_root: PathBuf::new(), source_tiles: 4, source_points: 4000, packed_bytes: 8, root: 6, nodes, terrain: None }, camera)
    }
    fn source_descendants(dataset: &Dataset, id: u32) -> Vec<u32> {
        let n = &dataset.nodes[id as usize];
        if n.children.is_empty() { vec![id] }
        else { n.children.iter().flat_map(|&c| source_descendants(dataset, c)).collect() }
    }
    #[test]
    fn every_residency_order_preserves_complete_nonoverlapping_coverage() {
        let (data, camera) = fixture();
        let config = StreamConfig { target_spacing_px: 0.01, ..Default::default() };
        let selection = select(&data, &camera, [1024.0; 2], &config);
        assert_eq!(selection.leaves, HashSet::from([0, 1, 3, 4]));
        // Exhaust all 64 partial residency states, including grandchildren arriving
        // before their parent and one sibling arriving much later than the other.
        for mask in 0..64 {
            let mut resident = HashSet::from([6]);
            for id in 0..6 { if mask & (1 << id) != 0 { resident.insert(id); } }
            let (draws, requests) = resident_cut(&data, &selection, &resident, &camera, [1024.0; 2]);
            let mut covered: Vec<_> = draws.iter().flat_map(|&id| source_descendants(&data, id)).collect();
            covered.sort_unstable();
            assert_eq!(covered, vec![0, 1, 3, 4], "Coverage changed with residency mask {mask}");
            assert!(draws.iter().all(|id| resident.contains(id)));
            assert!(requests.iter().all(|id| !resident.contains(id)));
        }
    }
    #[test]
    fn selection_obeys_point_and_draw_limits() {
        let (data, camera) = fixture();
        for points in [100, 200, 1100, 2100, 4000] {
            for draws in [1, 2, 3, 4] {
                let config = StreamConfig { target_spacing_px: 0.01, point_budget: points, draw_budget: draws, ..Default::default() };
                let s = select(&data, &camera, [1024.0; 2], &config);
                assert!(s.points <= points && s.leaves.len() <= draws);
                assert_eq!(s.points, s.leaves.iter().map(|&id| data.nodes[id as usize].points as u64).sum::<u64>());
            }
        }
    }
    #[test]
    fn looking_away_culls_the_city_before_requesting_detail() {
        let (data, _) = fixture();
        let camera = Camera::new(-52.52, -166.6, 1000.0,
            glam::Mat4::perspective_infinite_reverse_rh(60f32.to_radians(), 1.0, 0.1));
        let selected = select(&data, &camera, [1024.0; 2], &StreamConfig::default());
        // Opposite-side data lies beyond the target but can remain in an infinite
        // projection; the runtime must explicitly reject beyond-city distances.
        // This probe is completed by the visibility distance bound below.
        assert!(selected.leaves.is_empty());
    }
    #[test]
    fn hysteresis_keeps_a_stable_lod_inside_the_dead_band() {
        let (data, camera) = fixture();
        let view = View::new(&camera, [1024.0; 2], 1.0);
        let config = StreamConfig { target_spacing_px: view.score(&data.nodes[6]), ..Default::default() };
        let coarse = select_view(&data, &view, &config, Some(&HashSet::new()));
        let refined = select_view(&data, &view, &config, Some(&HashSet::from([6])));
        assert!(coarse.leaves.contains(&6));
        assert!(!refined.leaves.contains(&6));
    }

    #[test]
    fn actual_fallback_draws_obey_the_point_cap() {
        let (mut data, camera) = fixture();
        data.nodes[0].points = 50; data.nodes[1].points = 50;
        data.nodes[2].points = 1400; data.nodes[3].points = 1000; data.nodes[4].points = 100;
        let config = StreamConfig { point_budget: 1500, target_spacing_px: 0.01, ..Default::default() };
        let selection = select(&data, &camera, [1024.0; 2], &config);
        let resident = HashSet::from([2,3,4,5,6]);
        let (draws, _) = bounded_resident_cut(&data, &selection, &resident, &camera, [1024.0; 2], &config);
        assert!(draws.iter().map(|&id| data.nodes[id as usize].points as u64).sum::<u64>() <= 1500);
        let mut covered: Vec<_> = draws.iter().flat_map(|&id| source_descendants(&data,id)).collect();
        covered.sort_unstable(); assert_eq!(covered, vec![0,1,3,4]);
    }

}
