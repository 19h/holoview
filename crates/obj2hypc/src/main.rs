use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use log::{info, warn};
use rayon::prelude::*;
use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File},
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use walkdir::WalkDir;

// OSM / geometry utilities
use osmpbf::{Element, ElementReader, Way};
use rstar::{RTree, RTreeObject, AABB};
use smallvec::SmallVec;

// HYPC writer + math
use hypc::{
    geodetic_to_ecef, quantize_units, smc1_encode_rle, GeoExtentQ7, HypcTile, Smc1Chunk,
    Smc1CoordSpace, Smc1Encoding,
};

/// How to interpret incoming OBJ vertex triples.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum InputCs {
    /// Try to decide automatically from ranges.
    Auto,
    /// OBJ is [lon, lat, h_m].
    Geodetic,
    /// OBJ is ECEF meters [X, Y, Z].
    Ecef,
    /// OBJ is local meters [x, y, z] in an arbitrary local frame.
    LocalM,
}
impl std::fmt::Display for InputCs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            InputCs::Auto => "auto",
            InputCs::Geodetic => "geodetic",
            InputCs::Ecef => "ecef",
            InputCs::LocalM => "local_m",
        })
    }
}

#[derive(Parser, Debug, Clone)]
#[command(name = "obj2hypc", version)]
struct Args {
    #[arg(long, default_value = "tiles")]
    input_dir: String,

    #[arg(long, default_value = "tiles_bin")]
    output_dir: String,

    /// Units per meter for HYPC integer lattice (1000 = millimetres)
    #[arg(long, default_value_t = 1000)]
    units_per_meter: u32,

    /// Either auto-detect or force the input coordinate system of OBJ vertices.
    #[arg(long, value_enum, default_value_t = InputCs::Auto)]
    input_cs: InputCs,

    #[arg(long, default_value_t = false)]
    overwrite: bool,

    /// Optional path to a FeatureCollection (GeoJSON-like) file for semantic mask generation.
    #[arg(long)]
    feature_index: Option<String>,

    /// If multiple matches exist, prefer .zip over .obj
    #[arg(long, default_value_t = true)]
    prefer_zip: bool,

    /// Write the optional GEOT footer with bbox in CRS:84 (deg, 1e-7 ticks)
    #[arg(long, default_value_t = true)]
    write_geot: bool,

    // === SMC1 additions ===
    /// Optional OSM .pbf path for semantic overlays (roads/buildings/water/parks/etc.)
    #[arg(long)]
    osm_pbf: Option<String>,

    /// Semantic mask grid size (width=height); 512 is a good default
    #[arg(long, default_value_t = 512)]
    sem_grid: u16,

    /// Write SMC1 semantic mask chunk (requires --osm-pbf and tile bbox from --feature-index)
    #[arg(long, default_value_t = true)]
    write_smc1: bool,

    /// Compress SMC1 with internal RLE (no external deps); if false -> raw bytes
    #[arg(long, default_value_t = true)]
    smc1_compress: bool,

    /// Expand each tile bbox by this margin when retaining nodes (meters).
    #[arg(long, default_value_t = 50.0)]
    osm_margin_m: f64,

    /// Log a progress line every N elements (nodes/ways). Default: 2,000,000
    #[arg(long, default_value_t = 2_000_000)]
    osm_log_every: usize,

    /// Try to run 'osmium extract' + 'osmium tags-filter' to shrink the PBF first.
    #[arg(long, default_value_t = false)]
    osm_prefilter: bool,
}

#[derive(Debug, Clone)]
struct WorkItem {
    prefix: String,
    bbox: Option<GeoBboxDeg>,
}

#[derive(Debug, serde::Deserialize)]
struct GeoJsonRoot {
    features: Vec<Feature>,
}

#[derive(Debug, serde::Deserialize)]
struct Feature {
    geometry: Geometry,
    properties: Properties,
}

#[derive(Debug, serde::Deserialize)]
struct Geometry {
    coordinates: Vec<Vec<[f64; 2]>>,
}

#[derive(Debug, serde::Deserialize)]
struct Properties {
    url: String,
}

#[derive(Clone, Copy, Debug)]
struct GeoBboxDeg {
    lon_min: f64,
    lat_min: f64,
    lon_max: f64,
    lat_max: f64,
}

fn bbox_from_polygon_deg(poly: &Geometry) -> GeoBboxDeg {
    let ring = &poly.coordinates[0];
    let (mut xmin, mut ymin, mut xmax, mut ymax) =
        (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);

    for &[lon, lat] in ring {
        if lon.is_finite() && lat.is_finite() {
            if lon < xmin {
                xmin = lon
            }
            if lon > xmax {
                xmax = lon
            }
            if lat < ymin {
                ymin = lat
            }
            if lat > ymax {
                ymax = lat
            }
        }
    }

    GeoBboxDeg {
        lon_min: xmin,
        lat_min: ymin,
        lon_max: xmax,
        lat_max: ymax,
    }
}

fn load_feature_index(path: &str) -> anyhow::Result<Vec<WorkItem>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let root: GeoJsonRoot = serde_json::from_reader(reader)?;

    let mut items = Vec::with_capacity(root.features.len());
    for feature in root.features {
        let url = feature.properties.url;
        let prefix = std::path::Path::new(&url)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let bbox = Some(bbox_from_polygon_deg(&feature.geometry));
        items.push(WorkItem { prefix, bbox });
    }

    Ok(items)
}

#[derive(Default)]
struct LocalIndex {
    exact: HashMap<String, PathBuf>,
    names: BTreeMap<String, Vec<PathBuf>>,
}

fn build_local_index(input_dir: &str) -> LocalIndex {
    let mut idx = LocalIndex::default();

    for entry in WalkDir::new(input_dir)
        .follow_links(true)
        .into_iter()
        .filter_map(Result::ok)
    {
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.into_path();
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();

        if ext != "zip" && ext != "obj" {
            continue;
        }

        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();

        idx.names.entry(stem.clone()).or_default().push(path.clone());

        idx.exact
            .entry(stem.clone())
            .and_modify(|existing| {
                let existing_ext = existing.extension().and_then(|s| s.to_str()).unwrap_or("");
                if existing_ext.eq_ignore_ascii_case("obj") && ext == "zip" {
                    *existing = path.clone();
                }
            })
            .or_insert(path.clone());
    }

    idx
}

fn resolve_by_prefix(idx: &LocalIndex, prefix: &str, prefer_zip: bool) -> Option<PathBuf> {
    if let Some(p) = idx.exact.get(prefix) {
        return Some(p.clone());
    }

    for (name, paths) in idx.names.range(prefix.to_string()..) {
        if !name.starts_with(prefix) {
            break;
        }

        if paths.is_empty() {
            continue;
        }

        if !prefer_zip {
            return Some(paths[0].clone());
        }

        if let Some(zip_path) = paths.iter().find(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .map(|s| s.eq_ignore_ascii_case("zip"))
                .unwrap_or(false)
        }) {
            return Some(zip_path.clone());
        }

        return Some(paths[0].clone());
    }

    None
}

fn tilekey_from_prefix(prefix: &str) -> [u8; 32] {
    let mut key = [0u8; 32];
    let bytes = prefix.as_bytes();
    let len = bytes.len().min(32);
    key[..len].copy_from_slice(&bytes[..len]);
    key
}

/// Read raw vertex triples as they appear in the OBJ (no interpretation here).
fn parse_obj_vertices<R: Read>(r: R) -> Result<Vec<[f64; 3]>> {
    let reader = BufReader::new(r);
    let mut vertices = Vec::<[f64; 3]>::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if !trimmed.starts_with("v ") {
            continue;
        }

        let mut parts = trimmed.split_whitespace();
        parts.next(); // "v"

        let x: f64 = parts.next().context("Missing x")?.parse()?;
        let y: f64 = parts.next().context("Missing y")?.parse()?;
        let z: f64 = parts.next().context("Missing z")?.parse()?;

        if x.is_finite() && y.is_finite() && z.is_finite() {
            vertices.push([x, y, z]);
        }
    }

    Ok(vertices)
}

// ==============================
// === SMC1: semantics & PBF  ===
// ==============================

#[repr(u8)]
#[derive(Clone, Copy)]
enum SemClass {
    Unknown = 0,
    Building = 1,
    RoadMajor = 2,
    RoadMinor = 3,
    Path = 4,
    Water = 5,
    Park = 6,
    Woodland = 7,
    Railway = 8,
    Parking = 9,
}

#[inline(always)]
fn class_precedence(c: u8) -> u8 {
    match c {
        5 | 1 => 200, // Water, Building
        8 => 160,     // Railway
        2 => 150,     // RoadMajor
        3 => 140,     // RoadMinor
        4 => 130,     // Path
        6 => 100,     // Park
        7 => 90,      // Woodland
        9 => 80,      // Parking
        _ => 0,       // Unknown or unhandled
    }
}

#[derive(Clone)]
struct Polyline {
    class: u8,
    width_m: f32,
    pts: Arc<Vec<(f64, f64)>>,
}
#[derive(Clone)]
struct Polygon {
    class: u8,
    ring: Arc<Vec<(f64, f64)>>,
}
#[derive(Default, Clone)]
struct SemOverlayPerTile {
    roads: Vec<Polyline>,
    areas: Vec<Polygon>,
}
type OverlayMap = HashMap<String, SemOverlayPerTile>;

#[derive(Clone)]
struct NodeRec {
    lon: f64,
    lat: f64,
    tiles: SmallVec<[u32; 4]>,
}

struct Tick(Instant, Instant, usize);
impl Tick {
    fn new(every: usize) -> Self {
        Self(Instant::now(), Instant::now(), every.max(1))
    }
    fn should(&mut self, count: usize) -> bool {
        count % self.2 == 0 && self.1.elapsed() >= std::time::Duration::from_millis(200)
    }
    fn bump(&mut self) {
        self.1 = Instant::now();
    }
    fn rate_mps(&self, count: usize) -> f64 {
        (count as f64) / 1.0e6 / self.0.elapsed().as_secs_f64().max(1e-9)
    }
}

#[inline]
fn pad_degrees_for(lat_deg: f64, pad_m: f64) -> (f64, f64) {
    let m_per_deg_lat = 110_574.0;
    let m_per_deg_lon = 111_320.0 * lat_deg.to_radians().cos().abs().max(1e-6);
    (pad_m / m_per_deg_lat, pad_m / m_per_deg_lon)
}

#[inline]
fn default_highway_width_m(kind: &str, lanes: Option<u32>, width: Option<f32>) -> (u8, f32) {
    if let Some(w) = width {
        let class = match kind {
            "motorway" | "trunk" | "primary" => SemClass::RoadMajor as u8,
            "secondary" | "tertiary" | "residential" | "service" => SemClass::RoadMinor as u8,
            _ => SemClass::Path as u8,
        };
        return (class, w.max(1.0));
    }

    let lanes_f = lanes.unwrap_or(0) as f32;
    let base_width = match kind {
        "motorway" => 12.0,
        "trunk" => 10.0,
        "primary" => 8.0,
        "secondary" => 7.0,
        "tertiary" => 6.0,
        "residential" | "service" => 5.0,
        _ => 2.0,
    };

    let width_m = if lanes_f >= 2.0 {
        (lanes_f * 3.2).max(base_width)
    } else {
        base_width
    };

    let class = match kind {
        "motorway" | "trunk" | "primary" => SemClass::RoadMajor as u8,
        "secondary" | "tertiary" | "residential" | "service" => SemClass::RoadMinor as u8,
        _ => SemClass::Path as u8,
    };

    (class, width_m)
}

#[inline]
fn parse_width_m(s: &str) -> Option<f32> {
    let t = s.trim().to_ascii_lowercase();
    if let Some(v) = t.strip_suffix('m') {
        return v.trim().parse::<f32>().ok();
    }
    if let Some(v) = t.strip_suffix("ft") {
        return v.trim().parse::<f32>().ok().map(|x| x * 0.3048);
    }
    t.parse::<f32>().ok()
}

fn classify_way(w: &Way) -> Option<(u8, f32, bool)> {
    let tags: Vec<(&str, &str)> = w.tags().collect();
    let find = |k: &str| tags.iter().find_map(|(kk, vv)| if *kk == k { Some(*vv) } else { None });

    if let Some(b) = find("building") {
        if !b.is_empty() || b == "yes" {
            return Some((SemClass::Building as u8, 0.0, true));
        }
    }
    if let Some(h) = find("highway") {
        let lanes = find("lanes").and_then(|v| v.parse().ok());
        let width = find("width").and_then(parse_width_m);
        let (class, width_m) = default_highway_width_m(h, lanes, width);
        return Some((class, width_m, false));
    }
    if find("natural") == Some("water") {
        return Some((SemClass::Water as u8, 0.0, true));
    }
    if find("waterway") == Some("riverbank") {
        return Some((SemClass::Water as u8, 0.0, true));
    }
    if let Some(l) = find("landuse") {
        if matches!(l, "forest" | "grass" | "meadow" | "reservoir") {
            let class = if l == "forest" {
                SemClass::Woodland as u8
            } else {
                SemClass::Park as u8
            };
            return Some((class, 0.0, true));
        }
    }
    if let Some(l) = find("leisure") {
        if matches!(l, "park" | "pitch") {
            return Some((SemClass::Park as u8, 0.0, true));
        }
    }
    if let Some(r) = find("railway") {
        if !r.is_empty() {
            return Some((SemClass::Railway as u8, 4.0, false));
        }
    }
    if find("amenity") == Some("parking") {
        return Some((SemClass::Parking as u8, 0.0, true));
    }

    None
}

#[derive(Clone)]
struct TileBox {
    idx: u32,
    env: AABB<[f64; 2]>,
}
impl RTreeObject for TileBox {
    type Envelope = AABB<[f64; 2]>;
    #[inline]
    fn envelope(&self) -> Self::Envelope {
        self.env
    }
}

fn build_osm_overlays(
    pbf_path: &str,
    tiles: &[WorkItem],
    margin_m: f64,
    log_every: usize,
    prefilter: bool,
) -> Result<OverlayMap> {
    for t in tiles {
        if t.bbox.is_none() {
            anyhow::bail!("--osm-pbf requires --feature-index with bbox per tile");
        }
    }

    let boxes: Vec<TileBox> = tiles
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let bb = t.bbox.unwrap();
            let (pad_lat, pad_lon) = pad_degrees_for(0.5 * (bb.lat_min + bb.lat_max), margin_m);
            TileBox {
                idx: i as u32,
                env: AABB::from_corners(
                    [bb.lon_min - pad_lon, bb.lat_min - pad_lat],
                    [bb.lon_max + pad_lon, bb.lat_max + pad_lat],
                ),
            }
        })
        .collect();

    let tree = RTree::bulk_load(boxes);

    let pbf_source = if prefilter {
        prefilter_with_osmium(pbf_path, tiles, margin_m).unwrap_or_else(|| pbf_path.to_string())
    } else {
        pbf_path.to_string()
    };

    let mut node_map: hashbrown::HashMap<i64, NodeRec, nohash_hasher::BuildNoHashHasher<i64>> =
        hashbrown::HashMap::with_hasher(nohash_hasher::BuildNoHashHasher::default());

    let mut total_nodes = 0usize;
    let mut tick = Tick::new(log_every);

    ElementReader::from_path(&pbf_source)?.for_each(|elem| {
        let (id, lon, lat) = match elem {
            Element::Node(n) => (n.id(), n.lon(), n.lat()),
            Element::DenseNode(dn) => (dn.id(), dn.lon(), dn.lat()),
            _ => return,
        };

        total_nodes += 1;

        let mut tiles_touch = SmallVec::<[u32; 4]>::new();
        for tb in tree.locate_in_envelope_intersecting(&AABB::from_point([lon, lat])) {
            tiles_touch.push(tb.idx);
        }

        if !tiles_touch.is_empty() {
            node_map.insert(id, NodeRec { lon, lat, tiles: tiles_touch });
        }

        if tick.should(total_nodes) {
            info!(
                "Pass A: nodes seen {:>11}, kept {:>11}, rate {:5.2} M/s",
                total_nodes,
                node_map.len(),
                tick.rate_mps(total_nodes)
            );
            tick.bump();
        }
    })?;

    let mut overlays: OverlayMap = HashMap::new();
    let mut total_ways = 0usize;
    tick = Tick::new(log_every);

    ElementReader::from_path(&pbf_source)?.for_each(|elem| {
        if let Element::Way(w) = elem {
            total_ways += 1;

            if let Some((class_id, width_m, is_area)) = classify_way(&w) {
                let mut coords = Vec::with_capacity(w.refs().len());
                let mut tiles_hit = SmallVec::<[u32; 8]>::new();

                for nr in w.refs() {
                    if let Some(ni) = node_map.get(&nr) {
                        coords.push((ni.lon, ni.lat));
                        for &ti in &ni.tiles {
                            if !tiles_hit.contains(&ti) {
                                tiles_hit.push(ti);
                            }
                        }
                    }
                }

                if coords.len() >= if is_area { 3 } else { 2 } && !tiles_hit.is_empty() {
                    let coords_arc = Arc::new(coords);
                    for ti in tiles_hit {
                        let t = &tiles[ti as usize];
                        let entry = overlays.entry(t.prefix.clone()).or_default();
                        if is_area {
                            entry.areas.push(Polygon { class: class_id, ring: coords_arc.clone() });
                        } else {
                            entry.roads.push(Polyline { class: class_id, width_m, pts: coords_arc.clone() });
                        }
                    }
                }
            }

            if tick.should(total_ways) {
                info!(
                    "Pass B: ways seen {:>11}, rate {:5.2} M/s",
                    total_ways,
                    tick.rate_mps(total_ways)
                );
                tick.bump();
            }
        }
    })?;

    Ok(overlays)
}

fn prefilter_with_osmium(pbf_in: &str, tiles: &[WorkItem], margin_m: f64) -> Option<String> {
    use std::process::Command;

    if Command::new("osmium").arg("--version").output().is_err() {
        warn!("'osmium' not found; skipping prefilter.");
        return None;
    }

    let mut lon_min = f64::INFINITY;
    let mut lat_min = f64::INFINITY;
    let mut lon_max = f64::NEG_INFINITY;
    let mut lat_max = f64::NEG_INFINITY;

    for t in tiles {
        if let Some(bb) = t.bbox {
            let (pad_lat, pad_lon) = pad_degrees_for(0.5 * (bb.lat_min + bb.lat_max), margin_m);
            lon_min = lon_min.min(bb.lon_min - pad_lon);
            lon_max = lon_max.max(bb.lon_max + pad_lon);
            lat_min = lat_min.min(bb.lat_min - pad_lat);
            lat_max = lat_max.max(bb.lat_max + pad_lat);
        }
    }

    if !lon_min.is_finite() {
        return None;
    }

    let bbox = format!("{},{},{},{}", lon_min, lat_min, lon_max, lat_max);
    let tmp_extract = format!("{}.extract.pbf", pbf_in);
    let tmp_filtered = format!("{}.filtered.pbf", pbf_in);

    let extract_status = Command::new("osmium")
        .args([
            "extract",
            "-b",
            &bbox,
            "--overwrite",
            "-o",
            &tmp_extract,
            pbf_in,
        ])
        .status()
        .ok()?;

    if !extract_status.success() {
        return None;
    }

    let filter = "nwr/building nwr/highway nwr/landuse=forest,grass,meadow,reservoir nwr/leisure=park,pitch nwr/natural=water nwr/waterway=riverbank nwr/railway nwr/amenity=parking";

    let filter_status = Command::new("osmium")
        .args([
            "tags-filter",
            "--overwrite",
            "-o",
            &tmp_filtered,
            &tmp_extract,
        ])
        .args(filter.split_whitespace())
        .status()
        .ok()?;

    if !filter_status.success() {
        return Some(tmp_extract);
    }

    Some(tmp_filtered)
}

// ---------- SMC1 raster (as before) ----------

struct SemMask {
    w: u16,
    h: u16,
    data: Vec<u8>,
}

#[inline]
fn clamp_i(v: i32, lo: i32, hi: i32) -> i32 {
    v.max(lo).min(hi)
}
#[inline]
fn uv_to_pixel(u: f32, v: f32, w: u16, h: u16) -> (i32, i32) {
    ((u * w as f32).round() as i32, (v * h as f32).round() as i32)
}
fn paint_pixel(mask: &mut SemMask, x: i32, y: i32, class: u8) {
    if x < 0 || y < 0 || x >= mask.w as i32 || y >= mask.h as i32 {
        return;
    }
    let idx = y as usize * mask.w as usize + x as usize;
    if class_precedence(class) >= class_precedence(mask.data[idx]) {
        mask.data[idx] = class;
    }
}
fn rasterize_polygon(mask: &mut SemMask, poly: &[(i32, i32)], class: u8) {
    if poly.len() < 3 {
        return;
    }
    let (mut xmin, mut ymin, mut xmax, mut ymax) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for &(x, y) in poly {
        xmin = xmin.min(x);
        xmax = xmax.max(x);
        ymin = ymin.min(y);
        ymax = ymax.max(y);
    }
    xmin = clamp_i(xmin, 0, mask.w as i32 - 1);
    xmax = clamp_i(xmax, 0, mask.w as i32 - 1);
    ymin = clamp_i(ymin, 0, mask.h as i32 - 1);
    ymax = clamp_i(ymax, 0, mask.h as i32 - 1);

    for y in ymin..=ymax {
        for x in xmin..=xmax {
            let mut inside = false;
            let mut j = poly.len() - 1;
            for i in 0..poly.len() {
                let (xi, yi) = poly[i];
                let (xj, yj) = poly[j];
                if (yi > y) != (yj > y) {
                    let x_inter = (xj - xi) as f32 * ((y - yi) as f32 / ((yj - yi) as f32 + 1e-20))
                        + xi as f32;
                    if (x as f32) < x_inter {
                        inside = !inside;
                    }
                }
                j = i;
            }
            if inside {
                paint_pixel(mask, x, y, class);
            }
        }
    }
}
#[inline]
fn sqr(x: f32) -> f32 {
    x * x
}
fn rasterize_polyline(mask: &mut SemMask, line: &[(i32, i32)], radius_px: f32, class: u8) {
    if line.len() < 2 {
        return;
    }
    let r = radius_px.max(0.5);
    let r2 = r * r;
    for seg in line.windows(2) {
        let (x0, y0) = (seg[0].0 as f32, seg[0].1 as f32);
        let (x1, y1) = (seg[1].0 as f32, seg[1].1 as f32);
        let minx = clamp_i((x0.min(x1) - r).floor() as i32, 0, mask.w as i32 - 1);
        let maxx = clamp_i((x0.max(x1) + r).ceil() as i32, 0, mask.w as i32 - 1);
        let miny = clamp_i((y0.min(y1) - r).floor() as i32, 0, mask.h as i32 - 1);
        let maxy = clamp_i((y0.max(y1) + r).ceil() as i32, 0, mask.h as i32 - 1);
        let vx = x1 - x0;
        let vy = y1 - y0;
        let denom = vx * vx + vy * vy + 1e-12;

        for y in miny..=maxy {
            for x in minx..=maxx {
                let px = x as f32 + 0.5;
                let py = y as f32 + 0.5;
                let t = (((px - x0) * vx + (py - y0) * vy) / denom).clamp(0.0, 1.0);
                let qx = x0 + t * vx;
                let qy = y0 + t * vy;
                if sqr(px - qx) + sqr(py - qy) <= r2 {
                    paint_pixel(mask, x, y, class);
                }
            }
        }
    }
}

fn build_smc1_mask(overlay: &SemOverlayPerTile, tile_bbox_deg: GeoBboxDeg, grid: u16) -> SemMask {
    let mut mask = SemMask {
        w: grid,
        h: grid,
        data: vec![0; grid as usize * grid as usize],
    };

    let (lon_w, lat_h) = (
        (tile_bbox_deg.lon_max - tile_bbox_deg.lon_min).max(1e-12),
        (tile_bbox_deg.lat_max - tile_bbox_deg.lat_min).max(1e-12),
    );

    let lon_to_u = |lon: f64| ((lon - tile_bbox_deg.lon_min) / lon_w) as f32;
    let lat_to_v = |lat: f64| ((lat - tile_bbox_deg.lat_min) / lat_h) as f32;

    // Areas
    for a in &overlay.areas {
        let ring_px: Vec<_> = a
            .ring
            .iter()
            .map(|&(lon, lat)| uv_to_pixel(lon_to_u(lon), lat_to_v(lat), grid, grid))
            .collect();
        rasterize_polygon(&mut mask, &ring_px, a.class);
    }

    // Polylines (width in meters to pixels)
    let lat0 = 0.5 * (tile_bbox_deg.lat_min + tile_bbox_deg.lat_max);
    let px_m = 0.5_f64
        * (((lon_w * 111_320.0 * lat0.to_radians().cos().abs().max(1e-6)) / grid as f64)
            + ((lat_h * 110_574.0) / grid as f64));

    for r in &overlay.roads {
        let radius_px = (r.width_m as f64 * 0.5 / px_m) as f32;
        let line_px: Vec<_> = r
            .pts
            .iter()
            .map(|&(lon, lat)| uv_to_pixel(lon_to_u(lon), lat_to_v(lat), grid, grid))
            .collect();
        rasterize_polyline(&mut mask, &line_px, radius_px, r.class);
    }

    mask
}

// ---------- Input CS detection and safe quantization ----------

/// Cheap heuristics to decide how OBJ coordinates should be interpreted.
fn detect_input_cs(sample: &[[f64; 3]]) -> InputCs {
    let n = sample.len().max(1) as f64;
    let mut likely_geo = 0usize;

    for p in sample {
        if p[0].abs() <= 180.0 && p[1].abs() <= 90.0 {
            likely_geo += 1;
        }
    }

    // If >= 90% look like lon/lat, call it geodetic
    if (likely_geo as f64) / n >= 0.9 {
        return InputCs::Geodetic;
    }

    // Check ECEF radius band ~ [5e6, 8e6] meters
    let mut rsum = 0.0;
    let take = sample.len().min(4096);
    for p in &sample[..take] {
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        rsum += r;
    }
    let rmean = rsum / take.max(1) as f64;
    if rmean.is_finite() && (5_000_000.0..=8_000_000.0).contains(&rmean) {
        return InputCs::Ecef;
    }

    InputCs::LocalM
}

struct Quantized {
    anchor_units: [i64; 3],
    points_units: Vec<[i32; 3]>,
    used_upm: u32,
}

/// Quantize to integer lattice with an anchor, automatically downscaling UPM to fit i32 if needed.
fn quantize_with_anchor(points_m: &[[f64; 3]], requested_upm: u32) -> Quantized {
    debug_assert!(!points_m.is_empty());

    // Anchor at mean (in meters)
    let (sx, sy, sz) = points_m.iter().fold((0.0, 0.0, 0.0), |(ax, ay, az), p| {
        (ax + p[0], ay + p[1], az + p[2])
    });
    let inv_n = 1.0f64 / (points_m.len() as f64);
    let anchor_m = [sx * inv_n, sy * inv_n, sz * inv_n];

    // Max absolute offset (meters) across axes
    let mut max_off_m = 0.0f64;
    for p in points_m {
        max_off_m = max_off_m
            .max((p[0] - anchor_m[0]).abs())
            .max((p[1] - anchor_m[1]).abs())
            .max((p[2] - anchor_m[2]).abs());
    }

    // If all points are identical, just keep the requested scale.
    let mut upm = if max_off_m <= 1e-12 {
        requested_upm
    } else {
        // Fit in i32 with a little headroom (5%)
        let max_upm_fit =
            ((i32::MAX as f64) / (max_off_m * 1.05)).floor().clamp(1.0, requested_upm as f64) as u32;
        if max_upm_fit < requested_upm {
            warn!(
                "units_per_meter={} too high for this tile span (~{:.3} m max offset). \
                 Using {} u/m instead.",
                requested_upm, max_off_m, max_upm_fit
            );
        }
        max_upm_fit.max(1)
    };

    // Compute anchor in integer units at chosen upm
    let mut anchor_units = [
        quantize_units(anchor_m[0], upm),
        quantize_units(anchor_m[1], upm),
        quantize_units(anchor_m[2], upm),
    ];

    // Now quantize each point and compute i32 offsets. If (rarely) a rounding spike still
    // overflows, reduce UPM further and retry once.
    let mut make_points = |upm_now: u32, anchor_now: [i64; 3]| -> Option<Vec<[i32; 3]>> {
        let mut pts = Vec::with_capacity(points_m.len());
        for p in points_m {
            let ux = quantize_units(p[0], upm_now) - anchor_now[0];
            let uy = quantize_units(p[1], upm_now) - anchor_now[1];
            let uz = quantize_units(p[2], upm_now) - anchor_now[2];
            if ux < i32::MIN as i64
                || ux > i32::MAX as i64
                || uy < i32::MIN as i64
                || uy > i32::MAX as i64
                || uz < i32::MIN as i64
                || uz > i32::MAX as i64
            {
                return None;
            }
            pts.push([ux as i32, uy as i32, uz as i32]);
        }
        Some(pts)
    };

    let mut points_units = match make_points(upm, anchor_units) {
        Some(v) => v,
        None => {
            // One more downscale with a larger safety margin
            upm = (upm as f64 * 0.5).floor().max(1.0) as u32;
            warn!("Further reduced units_per_meter to {} for safety.", upm);
            anchor_units = [
                quantize_units(anchor_m[0], upm),
                quantize_units(anchor_m[1], upm),
                quantize_units(anchor_m[2], upm),
            ];
            make_points(upm, anchor_units)
                .expect("Internal: downscale should guarantee i32 range")
        }
    };

    Quantized {
        anchor_units,
        points_units,
        used_upm: upm,
    }
}

fn process_one_mesh(
    path: &Path,
    args: &Args,
    prefix: &str,
    bbox: Option<GeoBboxDeg>,
    overlays: Option<&SemOverlayPerTile>,
) -> Result<()> {
    let out_path = Path::new(&args.output_dir).join(format!(
        "{}.hypc",
        Path::new(prefix).file_stem().unwrap().to_str().unwrap()
    ));

    if out_path.exists() && !args.overwrite {
        return Ok(());
    }

    info!("Processing {} -> {}", path.display(), out_path.display());

    let raw_xyz: Vec<[f64; 3]> = if path.extension().and_then(|s| s.to_str()) == Some("zip") {
        let file = File::open(path)?;
        let mut archive = zip::ZipArchive::new(file)?;
        let obj_filename = archive
            .file_names()
            .find(|n| n.to_ascii_lowercase().ends_with(".obj"))
            .context("No .obj file found in zip archive")?
            .to_string();
        let mut obj_file = archive.by_name(&obj_filename)?;
        parse_obj_vertices(&mut obj_file)?
    } else {
        parse_obj_vertices(File::open(path)?)?
    };

    if raw_xyz.is_empty() {
        warn!("{}: no vertices", path.display());
        return Ok(());
    }

    // Decide input coordinate system
    let cs = match args.input_cs {
        InputCs::Auto => {
            let take = raw_xyz.len().min(4096);
            let guess = detect_input_cs(&raw_xyz[..take]);
            info!("Input CS (auto-detected): {}", guess);
            guess
        }
        c => {
            info!("Input CS (forced): {}", c);
            c
        }
    };

    // Convert to meters in some (possibly local) 3D frame for quantization.
    // Also gather lon/lat bounds only when truly geodetic.
    let mut points_m: Vec<[f64; 3]> = Vec::with_capacity(raw_xyz.len());
    let mut lon_min = f64::INFINITY;
    let mut lon_max = f64::NEG_INFINITY;
    let mut lat_min = f64::INFINITY;
    let mut lat_max = f64::NEG_INFINITY;

    match cs {
        InputCs::Geodetic => {
            for &[x, y, z] in &raw_xyz {
                // x=lon, y=lat, z=h_m
                lon_min = lon_min.min(x);
                lon_max = lon_max.max(x);
                lat_min = lat_min.min(y);
                lat_max = lat_max.max(y);
                points_m.push(geodetic_to_ecef(y, x, z));
            }
        }
        InputCs::Ecef | InputCs::LocalM => {
            // Already meters; keep as-is.
            points_m.extend(raw_xyz.iter().map(|&p| p));
        }
        InputCs::Auto => unreachable!(),
    }

    // Quantize with safe scaling if necessary
    let q = quantize_with_anchor(&points_m, args.units_per_meter);

    // SMC1 (uses bbox+OSM only; independent of geometry CS)
    let smc1_opt = if args.write_smc1 {
        if let (Some(bb), Some(ov)) = (bbox, overlays) {
            let mask = build_smc1_mask(ov, bb, args.sem_grid);
            let (encoding, data) = if args.smc1_compress {
                (Smc1Encoding::Rle, smc1_encode_rle(&mask.data))
            } else {
                (Smc1Encoding::Raw, mask.data)
            };

            Some(Smc1Chunk {
                width: args.sem_grid,
                height: args.sem_grid,
                coord_space: Smc1CoordSpace::Crs84BboxNorm,
                encoding,
                data,
                palette: (0u8..=9u8).map(|i| (i, class_precedence(i))).collect(),
            })
        } else {
            None
        }
    } else {
        None
    };

    // GEOT: prefer feature-index bbox if present; otherwise, if the OBJ itself was geodetic,
    // compute from vertex lon/lat. (When the OBJ is local/ECEF and no bbox is provided,
    // we skip GEOT.)
    let geot = if args.write_geot {
        if let Some(bb) = bbox {
            Some(GeoExtentQ7::from_deg(bb.lon_min, bb.lon_max, bb.lat_min, bb.lat_max))
        } else if matches!(cs, InputCs::Geodetic)
            && lon_min.is_finite()
            && lon_max.is_finite()
            && lat_min.is_finite()
            && lat_max.is_finite()
        {
            Some(GeoExtentQ7::from_deg(lon_min, lon_max, lat_min, lat_max))
        } else {
            None
        }
    } else {
        None
    };

    let tile = HypcTile {
        units_per_meter: q.used_upm,
        anchor_ecef_units: q.anchor_units,
        tile_key: Some(tilekey_from_prefix(prefix)),
        points_units: q.points_units,
        labels: None,
        geot,
        smc1: smc1_opt,
    };

    hypc::write_file(&out_path, &tile)?;

    info!(
        "OK {} -> {} ({} pts, {} u/m)",
        path.display(),
        out_path.display(),
        tile.points_units.len(),
        tile.units_per_meter
    );

    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    fs::create_dir_all(&args.output_dir)?;

    let lidx = build_local_index(&args.input_dir);

    let work: Vec<WorkItem> = if let Some(fi) = &args.feature_index {
        let mut items = load_feature_index(fi)?;
        items.retain(|it| resolve_by_prefix(&lidx, &it.prefix, args.prefer_zip).is_some());
        items
    } else {
        lidx.names
            .keys()
            .map(|k| WorkItem {
                prefix: k.clone(),
                bbox: None,
            })
            .collect()
    };

    #[derive(Clone)]
    struct Resolved {
        item: WorkItem,
        path: PathBuf,
    }

    let resolved: Vec<_> = work
        .iter()
        .filter_map(|w| {
            resolve_by_prefix(&lidx, &w.prefix, args.prefer_zip)
                .map(|p| Resolved {
                    item: w.clone(),
                    path: p,
                })
        })
        .collect();

    let overlays_map = if let Some(pbf) = &args.osm_pbf {
        let work_items: Vec<_> = resolved.iter().map(|r| r.item.clone()).collect();
        Some(Arc::new(build_osm_overlays(
            pbf,
            &work_items,
            args.osm_margin_m,
            args.osm_log_every,
            args.osm_prefilter,
        )?))
    } else {
        None
    };

    info!("Processing {} items...", resolved.len());
    resolved.par_iter().for_each(|r| {
        let ov = overlays_map.as_ref().and_then(|m| m.get(&r.item.prefix));
        if let Err(e) = process_one_mesh(&r.path, &args, &r.item.prefix, r.item.bbox, ov) {
            warn!("Error processing {}: {:#}", r.path.display(), e);
        }
    });

    Ok(())
}
