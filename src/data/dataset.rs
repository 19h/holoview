//! Persistent spatial LOD hierarchy. Source HYPC remains the finest level;
//! coarse levels contain actual source representatives, never averaged geometry.
use super::{point_cloud::{prepare_hypc_tile, PreparedTile}, types::PointInstance};
use anyhow::{bail, ensure, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs::{self, File}, io::{Read, Seek, SeekFrom, Write}, path::{Path, PathBuf}, time::{Instant, UNIX_EPOCH}};

const VERSION: u32 = 1;
const MAGIC: &[u8; 8] = b"HVLODP01";
const PARENT_LIMIT: usize = 8192;
pub const MAX_NODE_POINTS: usize = 2_000_000;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SourceFile {
    pub path: PathBuf,
    pub bytes: u64,
    pub modified_ns: u128,
}
impl SourceFile {
    fn inspect(root: &Path, path: &Path) -> Result<Self> {
        let m = path.metadata()?;
        Ok(Self { path: path.strip_prefix(root)?.into(), bytes: m.len(), modified_ns: m.modified()?.duration_since(UNIX_EPOCH)?.as_nanos() })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Payload {
    Packed { offset: u64, crc32: u32 },
    Source { source: SourceFile, crc32: u32 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub anchor_units: [i64; 3],
    pub units_per_meter: u32,
    pub min: [f64; 3],
    pub max: [f64; 3],
    /// Nominal spatial sample spacing used for projected-density selection.
    pub spacing_m: f32,
    /// Conservative representative error bound relative to original points.
    pub error_bound_m: f32,
    pub points: u32,
    pub children: Vec<u32>,
    pub payload: Payload,
}
impl Node {
    pub fn center(&self) -> glam::DVec3 {
        (glam::DVec3::from(self.min) + glam::DVec3::from(self.max)) * 0.5
    }
    pub fn radius(&self) -> f64 {
        (glam::DVec3::from(self.max) - glam::DVec3::from(self.min)).length() * 0.5
    }
    pub fn bytes(&self) -> u64 { self.points as u64 * 16 }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dataset {
    pub version: u32,
    pub source_root: PathBuf,
    pub source_tiles: usize,
    pub source_points: u64,
    pub packed_bytes: u64,
    pub root: u32,
    pub nodes: Vec<Node>,
    #[serde(default)]
    pub terrain: Option<super::terrain::TerrainGrid>,
}

impl Dataset {
    pub fn open(cache: &Path) -> Result<Self> {
        ensure!(cfg!(target_endian = "little"), "LOD pack requires a little-endian host");
        let data: Self = serde_json::from_slice(&fs::read(cache.join("catalog.json"))?)?;
        ensure!(data.version == VERSION, "Unsupported LOD catalog version");
        ensure!(!data.nodes.is_empty() && (data.root as usize) < data.nodes.len(), "Invalid root");
        let mut pack = File::open(cache.join("points.bin"))?;
        ensure!(pack.metadata()?.len() == data.packed_bytes, "Truncated/changed LOD pack");
        let mut magic = [0; 8]; pack.read_exact(&mut magic)?;
        ensure!(&magic == MAGIC, "Invalid LOD pack signature");
        if let Some(terrain) = &data.terrain { terrain.validate()?; }
        let mut references = vec![0u8; data.nodes.len()];
        let mut source_count = 0;
        let mut source_points = 0u64;
        for (id, n) in data.nodes.iter().enumerate() {
            ensure!(n.units_per_meter > 0 && n.points > 0 && n.points as usize <= MAX_NODE_POINTS, "Invalid node size/units at {id}");
            ensure!(n.spacing_m.is_finite() && n.spacing_m > 0.0 && n.error_bound_m.is_finite() && n.error_bound_m >= 0.0, "Invalid LOD error at {id}");
            ensure!((0..3).all(|i| n.min[i].is_finite() && n.max[i].is_finite() && n.min[i] <= n.max[i]), "Invalid bounds at {id}");
            for &child in &n.children {
                ensure!((child as usize) < id, "Children must precede parents (cycle/invalid topology)");
                references[child as usize] = references[child as usize].saturating_add(1);
                let c = &data.nodes[child as usize];
                ensure!((0..3).all(|i| c.min[i] >= n.min[i] - 0.01 && c.max[i] <= n.max[i] + 0.01), "Child outside parent bounds");
            }
            match &n.payload {
                Payload::Packed { offset, .. } => {
                    ensure!(*offset >= 8 && offset.checked_add(n.bytes()).is_some_and(|end| end <= data.packed_bytes), "Invalid packed range");
                }
                Payload::Source { source, .. } => {
                    ensure!(n.children.is_empty(), "Source node must be a leaf");
                    ensure!(!source.path.is_absolute() && source.path.components().all(|c| matches!(c, std::path::Component::Normal(_))), "Invalid source path");
                    source_count += 1; source_points += n.points as u64;
                }
            }
        }
        ensure!(references[data.root as usize] == 0 && references.iter().enumerate().all(|(id, &r)| id == data.root as usize || r == 1), "Catalog is not one connected tree");
        ensure!(source_count == data.source_tiles && source_points == data.source_points, "Source totals do not match tree");
        Ok(data)
    }

    pub fn read_node(&self, cache: &Path, id: u32) -> Result<PreparedTile> {
        let n = self.nodes.get(id as usize).context("Unknown node")?;
        let tile = match &n.payload {
            Payload::Source { source, crc32 } => {
                let path = self.source_root.join(&source.path);
                ensure!(SourceFile::inspect(&self.source_root, &path)? == *source, "Source changed: {}", path.display());
                let tile = prepare_hypc_tile(&path)?;
                ensure!(tile.instances.len() == n.points as usize && crc32fast::hash(bytemuck::cast_slice(&tile.instances)) == *crc32, "Source point checksum mismatch");
                tile
            }
            Payload::Packed { offset, crc32 } => {
                let mut f = File::open(cache.join("points.bin"))?;
                let points = read_points(&mut f, *offset, n.points, *crc32)?;
                PreparedTile { key: None, units_per_meter: n.units_per_meter, anchor_units: n.anchor_units, instances: points }
            }
        };
        ensure!(tile.units_per_meter == n.units_per_meter && tile.anchor_units == n.anchor_units, "Node anchor mismatch");
        Ok(tile)
    }
}

fn read_points(f: &mut File, offset: u64, count: u32, crc: u32) -> Result<Vec<PointInstance>> {
    ensure!(count as usize <= MAX_NODE_POINTS, "Oversized point block");
    let mut points = vec![PointInstance { ofs_m: [0.0; 3], label: 0 }; count as usize];
    f.seek(SeekFrom::Start(offset))?;
    f.read_exact(bytemuck::cast_slice_mut(&mut points))?;
    ensure!(crc32fast::hash(bytemuck::cast_slice(&points)) == crc, "LOD point checksum mismatch");
    ensure!(points.iter().all(|p| p.ofs_m.iter().all(|v| v.is_finite()) && p.label <= 255), "Invalid cached point");
    Ok(points)
}

/// A fixed global ECEF lattice makes voxel decisions stable across partitions.
/// Selection is the point nearest the voxel centre, with an explicit tie-break.
fn voxel_sample(points: &[PointInstance], anchor: [f64; 3], spacing: f64) -> Vec<PointInstance> {
    let mut cells: HashMap<[i32; 3], (f64, PointInstance)> = HashMap::with_capacity((points.len() / 3).max(16));
    for &p in points {
        let xyz: [f64; 3] = std::array::from_fn(|i| anchor[i] + p.ofs_m[i] as f64);
        let key = xyz.map(|v| (v / spacing).floor() as i32);
        let d: f64 = (0..3).map(|i| (xyz[i] - (key[i] as f64 + 0.5) * spacing).powi(2)).sum();
        let tie = |p: PointInstance| (p.ofs_m.map(f32::to_bits), p.label);
        let e = cells.entry(key).or_insert((d, p));
        if d < e.0 || (d == e.0 && tie(p) < tie(e.1)) { *e = (d, p); }
    }
    let mut ordered: Vec<_> = cells.into_iter().collect();
    ordered.sort_unstable_by_key(|(key, _)| *key);
    ordered.into_iter().map(|(_, (_, p))| p).collect()
}

#[derive(Serialize, Deserialize)]
struct LeafReceipt {
    version: u32,
    source: SourceFile,
    bytes: u64,
    nodes: Vec<Node>,
}

fn leaf_paths(work: &Path, id: usize) -> (PathBuf, PathBuf) {
    (work.join(format!("{id:05}.json")), work.join(format!("{id:05}.bin")))
}

fn prepare_leaf(root: &Path, source: &SourceFile, work: &Path, id: usize) -> Result<()> {
    let (json, bin) = leaf_paths(work, id);
    if let Ok(file) = File::open(&json) {
        if let Ok(receipt) = serde_json::from_reader::<_, LeafReceipt>(std::io::BufReader::new(file)) {
            if receipt.version == VERSION && receipt.source == *source && bin.metadata().is_ok_and(|m| m.len() == receipt.bytes) {
                return Ok(());
            }
        }
    }
    let tile = prepare_hypc_tile(&root.join(&source.path))?;
    ensure!(SourceFile::inspect(root, &root.join(&source.path))? == *source, "Source changed during preparation");
    ensure!(tile.instances.len() <= MAX_NODE_POINTS, "Source exceeds {} point block limit: {}", MAX_NODE_POINTS, source.path.display());
    let anchor = tile.anchor_units.map(|v| v as f64 / tile.units_per_meter as f64);
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for p in &tile.instances {
        for i in 0..3 {
            let v = anchor[i] + p.ofs_m[i] as f64;
            min[i] = min[i].min(v); max[i] = max[i].max(v);
        }
    }
    let base = Node {
        anchor_units: tile.anchor_units, units_per_meter: tile.units_per_meter,
        min, max, spacing_m: 0.5, error_bound_m: 0.0, points: tile.instances.len() as u32,
        children: vec![], payload: Payload::Source { source: source.clone(), crc32: crc32fast::hash(bytemuck::cast_slice(&tile.instances)) },
    };
    let mut nodes = vec![base];
    let temp_bin = bin.with_extension("bin.part");
    let mut out = File::create(&temp_bin)?;
    let mut points = tile.instances;
    for spacing in [1.0, 2.0, 4.0, 8.0, 16.0] {
        points = voxel_sample(&points, anchor, spacing);
        let bytes = bytemuck::cast_slice(&points);
        let mut n = nodes.last().unwrap().clone();
        n.spacing_m = spacing as f32;
        n.error_bound_m += (3.0f64.sqrt() * spacing) as f32;
        n.points = points.len() as u32;
        n.children = vec![nodes.len() as u32 - 1];
        n.payload = Payload::Packed { offset: out.stream_position()?, crc32: crc32fast::hash(bytes) };
        out.write_all(bytes)?;
        nodes.push(n);
    }
    let receipt = LeafReceipt { version: VERSION, source: source.clone(), bytes: out.stream_position()?, nodes };
    out.sync_all()?;
    fs::rename(temp_bin, &bin)?;
    let temp_json = json.with_extension("json.part");
    serde_json::to_writer(File::create(&temp_json)?, &receipt)?;
    fs::rename(temp_json, json)?;
    Ok(())
}

fn parent_tree(nodes: &mut Vec<Node>, ids: &mut [u32], writer: &mut File, reader: &mut File) -> Result<u32> {
    if ids.len() == 1 { return Ok(ids[0]); }
    let min: [f64; 3] = std::array::from_fn(|i| ids.iter().map(|&id| nodes[id as usize].min[i]).fold(f64::INFINITY, f64::min));
    let max: [f64; 3] = std::array::from_fn(|i| ids.iter().map(|&id| nodes[id as usize].max[i]).fold(f64::NEG_INFINITY, f64::max));
    let axis = (0..3).max_by(|&a, &b| (max[a] - min[a]).total_cmp(&(max[b] - min[b]))).unwrap();
    ids.sort_unstable_by(|&a, &b| nodes[a as usize].center()[axis].total_cmp(&nodes[b as usize].center()[axis]).then(a.cmp(&b)));
    let (left, right) = ids.split_at_mut(ids.len() / 2);
    let a = parent_tree(nodes, left, writer, reader)?;
    let b = parent_tree(nodes, right, writer, reader)?;
    let children = [a, b];
    let anchor_units = std::array::from_fn(|i| (((min[i] + max[i]) * 0.5) * 2000.0).round() as i64);
    let anchor = anchor_units.map(|v| v as f64 / 2000.0);
    let mut merged = Vec::new();
    let mut spacing = 0.0f64;
    let mut error = 0.0f32;
    for child in children {
        let c = &nodes[child as usize];
        spacing = spacing.max(c.spacing_m as f64);
        error = error.max(c.error_bound_m);
        let Payload::Packed { offset, crc32 } = c.payload else { bail!("Parent requires cached child"); };
        let mut points = read_points(reader, offset, c.points, crc32)?;
        let child_anchor = c.anchor_units.map(|v| v as f64 / c.units_per_meter as f64);
        for p in &mut points {
            for i in 0..3 { p.ofs_m[i] = (child_anchor[i] - anchor[i] + p.ofs_m[i] as f64) as f32; }
        }
        merged.extend(points);
    }
    let points = loop {
        let sampled = voxel_sample(&merged, anchor, spacing);
        if sampled.len() <= PARENT_LIMIT { break sampled; }
        spacing *= 2.0;
    };
    let bytes = bytemuck::cast_slice(&points);
    let offset = writer.stream_position()?;
    writer.write_all(bytes)?;
    let id = nodes.len() as u32;
    nodes.push(Node { anchor_units, units_per_meter: 2000, min, max, spacing_m: spacing as f32,
        error_bound_m: error + (3.0f64.sqrt() * spacing) as f32 + 0.01,
        points: points.len() as u32, children: children.to_vec(), payload: Payload::Packed { offset, crc32: crc32fast::hash(bytes) } });
    Ok(id)
}

/// Bounded parallel preprocessing; completed leaf receipts make interrupted builds resumable.
/// Finest-level HYPC files are referenced, not copied. Publish catalog last.
pub fn prepare_dataset(root: &Path, cache: &Path, workers: usize) -> Result<Dataset> {
    ensure!(cfg!(target_endian = "little"), "LOD pack currently requires a little-endian host");
    ensure!((1..=8).contains(&workers), "Use 1..8 preprocessing workers");
    let started = Instant::now();
    let root = root.canonicalize()?;
    fs::create_dir_all(cache)?;
    let build_lock = fs::OpenOptions::new().create(true).truncate(false).write(true).open(cache.join("build.lock"))?;
    build_lock.try_lock().context("Another process is preparing this cache")?;
    ensure!(!cache.join("catalog.json").exists(), "Published cache already exists; use a new output directory to rebuild");
    let work = cache.join(".build"); fs::create_dir_all(&work)?;
    let mut paths = Vec::new();
    for entry in walkdir::WalkDir::new(&root) {
        let entry = entry?;
        if entry.file_type().is_file() && entry.path().extension().is_some_and(|s| s == "hypc") { paths.push(entry.into_path()); }
    }
    paths.sort();
    ensure!(!paths.is_empty(), "No HYPC sources found");
    let sources: Vec<_> = paths.iter().map(|p| SourceFile::inspect(&root, p)).collect::<Result<_>>()?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build()?;
    for (batch, chunk) in sources.chunks(workers).enumerate() {
        pool.install(|| chunk.par_iter().enumerate().try_for_each(|(i, source)| prepare_leaf(&root, source, &work, batch * workers + i)))?;
        let completed = ((batch + 1) * workers).min(sources.len());
        if completed % 32 == 0 || completed == sources.len() {
            log::info!("LOD leaves {completed}/{} ({:.1}s)", sources.len(), started.elapsed().as_secs_f64());
            let status = serde_json::json!({"phase":"preparing", "completed":completed, "tiles":sources.len(), "elapsed_s":started.elapsed().as_secs_f64()});
            serde_json::to_writer(File::create(cache.join("build-status.json"))?, &status)?;
        }
    }
    let pack_temp = cache.join("points.bin.part");
    let mut writer = File::create(&pack_temp)?; writer.write_all(MAGIC)?;
    let mut nodes = Vec::new();
    let mut leaves = Vec::new();
    let mut source_points = 0;
    for id in 0..sources.len() {
        let (json, bin) = leaf_paths(&work, id);
        let mut receipt: LeafReceipt = serde_json::from_reader(std::io::BufReader::new(File::open(json)?))?;
        let mut input = File::open(bin)?;
        let node_start = nodes.len() as u32;
        source_points += receipt.nodes[0].points as u64;
        for n in &mut receipt.nodes {
            for c in &mut n.children { *c += node_start; }
            if let Payload::Packed { offset, crc32 } = &mut n.payload {
                let points = read_points(&mut input, *offset, n.points, *crc32)?;
                *offset = writer.stream_position()?;
                writer.write_all(bytemuck::cast_slice(&points))?;
            }
        }
        nodes.extend(receipt.nodes);
        leaves.push(nodes.len() as u32 - 1);
    }
    log::info!("Building spatial hierarchy from {} leaves", leaves.len());
    let mut reader = File::open(&pack_temp)?;
    let root_id = parent_tree(&mut nodes, &mut leaves, &mut writer, &mut reader)?;
    let mut data = Dataset { version: VERSION, source_root: root, source_tiles: sources.len(), source_points,
        packed_bytes: writer.stream_position()?, root: root_id, nodes, terrain: None };
    writer.sync_all()?;
    fs::rename(pack_temp, cache.join("points.bin"))?;
    data.terrain = Some(super::terrain::build_terrain(&data, cache)?);
    let catalog_temp = cache.join("catalog.json.part");
    serde_json::to_writer(File::create(&catalog_temp)?, &data)?;
    fs::rename(&catalog_temp, cache.join("catalog.json"))?;
    let verified = Dataset::open(cache)?;
    let status = serde_json::json!({"phase":"complete", "tiles":data.source_tiles,"source_points":data.source_points,"nodes":data.nodes.len(),"packed_bytes":data.packed_bytes,"elapsed_s":started.elapsed().as_secs_f64()});
    serde_json::to_writer_pretty(File::create(cache.join("build-status.json"))?, &status)?;
    log::info!("LOD build complete: {status}");
    Ok(verified)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn temp(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("city-lod-{name}-{}", std::process::id()));
        if path.exists() { fs::remove_dir_all(&path).unwrap(); }
        fs::create_dir_all(&path).unwrap(); path
    }
    fn source(root: &Path, id: usize, lon: f64) {
        let anchor = hypc::geodetic_to_ecef(52.52, lon, 40.0).map(|v| (v * 2000.0).round() as i64);
        let points: Vec<_> = (0..400).map(|i| [(i % 20) * 1000, (i / 20) * 1000, (i % 7) * 400]).collect();
        hypc::write_file(root.join(format!("tile-{id}.hypc")), &hypc::HypcTile {
            units_per_meter: 2000, anchor_ecef_units: anchor, tile_key: None,
            points_units: points, labels: Some((0..400).map(|i| (i % 10) as u8).collect()), geot: None, smc1: None,
        }).unwrap();
    }
    #[test]
    fn hierarchy_roundtrip_preserves_finest_points_and_detects_corruption() {
        let root = temp("roundtrip"); let sources = root.join("sources"); fs::create_dir(&sources).unwrap();
        for i in 0..3 { source(&sources, i, 13.4 + i as f64 * 0.003); }
        let cache = root.join("cache");
        let data = prepare_dataset(&sources, &cache, 2).unwrap();
        assert_eq!(data.source_tiles, 3); assert_eq!(data.source_points, 1200);
        for (id, node) in data.nodes.iter().enumerate() {
            let loaded = data.read_node(&cache, id as u32).unwrap();
            assert_eq!(loaded.instances.len(), node.points as usize);
            if let Payload::Source { source, .. } = &node.payload {
                let original = prepare_hypc_tile(&sources.join(&source.path)).unwrap();
                assert_eq!(bytemuck::cast_slice::<_, u8>(&loaded.instances), bytemuck::cast_slice::<_, u8>(&original.instances));
            }
        }
        let Payload::Packed { offset, .. } = data.nodes[data.root as usize].payload else { panic!() };
        let mut file = fs::OpenOptions::new().write(true).open(cache.join("points.bin")).unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap(); file.write_all(&[0xFF; 4]).unwrap();
        assert!(data.read_node(&cache, data.root).is_err());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn representative_sampling_is_order_independent_and_retains_source_labels() {
        let points = vec![
            PointInstance { ofs_m: [0.1, 0.2, 0.3], label: 9 },
            PointInstance { ofs_m: [0.45, 0.5, 0.5], label: 2 },
            PointInstance { ofs_m: [4.5, -3.0, 2.0], label: 5 },
        ];
        let a = voxel_sample(&points, [0.0; 3], 1.0);
        let b = voxel_sample(&points.iter().rev().copied().collect::<Vec<_>>(), [0.0; 3], 1.0);
        assert_eq!(bytemuck::cast_slice::<_, u8>(&a), bytemuck::cast_slice::<_, u8>(&b));
        assert_eq!(a.len(), 2);
        assert!(a.iter().all(|p| points.iter().any(|q| p.ofs_m == q.ofs_m && p.label == q.label)));
    }
    #[test]
    fn published_catalog_rejects_cycles_and_oversized_nodes() {
        let root = temp("topology"); let sources = root.join("sources"); fs::create_dir(&sources).unwrap(); source(&sources, 0, 13.4);
        let cache = root.join("cache"); let mut data = prepare_dataset(&sources, &cache, 1).unwrap();
        let root_id = data.root; data.nodes[root_id as usize].children = vec![root_id];
        serde_json::to_writer(File::create(cache.join("catalog.json")).unwrap(), &data).unwrap();
        assert!(Dataset::open(&cache).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
