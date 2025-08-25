use anyhow::{Context, Result};
use byteorder::{LittleEndian as LE, WriteBytesExt};
use clap::Parser;
use log::{debug, error, info, warn};
use rayon::prelude::*;
use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File},
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use walkdir::WalkDir;

use regex::Regex;

// Fast hash map for integer keys (node ids)
use hashbrown::HashMap as FastHashMap;
use nohash_hasher::BuildNoHashHasher;

// === SMC1 additions ===
use miniz_oxide::deflate::compress_to_vec_zlib;
use osmpbf::{Element, ElementReader, Way};
use rstar::{RTree, RTreeObject, AABB};
use smallvec::SmallVec;

/// The program performs the following high‑level steps:
///
/// 1. Initialise logging and parse command‑line arguments.
/// 2. Ensure the output directory exists.
/// 3. Scan the input directory for `.zip` and `.obj` files and build a
///    `LocalIndex` for fast lookup by filename prefix.
/// 4. Construct the list of work items either from a supplied GeoJSON feature
///    index (providing a bounding box per tile) or, when no index is given,
///    from the discovered filenames alone.
/// 5. Optionally build per‑tile OSM semantic overlays if an OSM PBF file is
///    supplied; the result is placed in an `Arc<OverlayMap>` so it can be shared
///    across threads.
/// 6. Process each work item in parallel using Rayon:
///    * Resolve the mesh file path via the local index.
///    * Retrieve the corresponding overlay, if any.
///    * Call `process_one_mesh` which writes the `.hypc` file, optionally
///      appending a semantic mask (SMC1) and a GEOT footer.
/// 7. Report overall timing information.
///

//
// ---------------- CLI ----------------
//

#[derive(Parser, Debug, Clone)]
#[command(name = "obj2hypc", version)]
struct Args {
    #[arg(long, default_value = "tiles")]
    input_dir: String,
    #[arg(long, default_value = "tiles_bin")]
    output_dir: String,
    #[arg(long, default_value_t = 1_200_000)]
    point_budget: usize,
    #[arg(long, default_value_t = 200.0)]
    target_extent: f32,
    #[arg(long, default_value_t = false)]
    overwrite: bool,

    /// Optional path to a FeatureCollection (GeoJSON-like) file
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

    /// Compress SMC1 with DEFLATE (zlib); if false → raw
    #[arg(long, default_value_t = true)]
    smc1_compress: bool,

    /// Expand each tile bbox by this margin when retaining nodes (meters).
    #[arg(long, default_value_t = 50.0)]
    osm_margin_m: f64,

    /// Log a progress line every N elements (nodes/ways). Default: 2,000,000
    #[arg(long, default_value_t = 2_000_000)]
    osm_log_every: usize,

    /// Try to run 'osmium extract' + 'osmium tags-filter' to shrink the PBF first.
    /// If not available, falls back to direct streaming parse.
    #[arg(long, default_value_t = false)]
    osm_prefilter: bool,
}

//
// ---------------- GeoJSON types ----------------
//

#[derive(Debug, Clone)]
struct WorkItem {
    prefix: String,               // basename without extension, e.g. "Tile-53-25-1-1"
    bbox: Option<GeoBboxDeg>,     // Some if feature index is given
}

#[derive(Debug, serde::Deserialize)]
struct GeoJsonRoot {
    #[serde(rename="type")]
    r#type: String,
    name: String,
    features: Vec<Feature>,
}

#[derive(Debug, serde::Deserialize)]
struct Feature {
    #[serde(rename="type")]
    r#type: String,
    geometry: Geometry,
    properties: Properties,
}

#[derive(Debug, serde::Deserialize)]
struct Geometry {
    #[serde(rename="type")]
    r#type: String,
    // GeoJSON Polygon: coordinates[0] is an outer ring: [[lon,lat],...]
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
    for [lon, lat] in ring {
        if *lon < xmin { xmin = *lon; }
        if *lon > xmax { xmax = *lon; }
        if *lat < ymin { ymin = *lat; }
        if *lat > ymax { ymax = *lat; }
    }
    GeoBboxDeg { lon_min: xmin, lat_min: ymin, lon_max: xmax, lat_max: ymax }
}

fn load_feature_index(path: &str) -> anyhow::Result<Vec<WorkItem>> {
    // Process‑info: start timer
    let start = std::time::Instant::now();

    // Open the GeoJSON feature index file
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // Parse the entire file into the GeoJsonRoot structure
    let root: GeoJsonRoot = serde_json::from_reader(reader)?;

    // Allocate space for the resulting work items
    let mut items = Vec::with_capacity(root.features.len());

    // Transform each GeoJSON feature into a WorkItem
    for feature in root.features {
        // The URL field contains a filename; we keep its stem as the prefix
        let url = feature.properties.url;
        let prefix = Path::new(&url)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        // Compute the bounding box from the polygon geometry
        let bbox = Some(bbox_from_polygon_deg(&feature.geometry));

        items.push(WorkItem { prefix, bbox });
    }

    // Process‑info: finish timer and log summary
    let elapsed = start.elapsed();
    info!(
        "Loaded {} feature(s) from '{}' in {:.2?}",
        items.len(),
        path,
        elapsed
    );

    Ok(items)
}

//
// ---------------- File enumeration ----------------
//

#[derive(Default)]
struct LocalIndex {
    exact: HashMap<String, PathBuf>,          // exact basename -> best path
    names: BTreeMap<String, Vec<PathBuf>>,    // for starts_with scans
}

fn build_local_index(input_dir: &str) -> LocalIndex {
    let mut idx = LocalIndex::default();

    // Process‑info: start timer and counters
    let start = std::time::Instant::now();
    let mut total_files = 0usize;
    let mut matched_files = 0usize;

    let walker = WalkDir::new(input_dir)
        .follow_links(true)
        .into_iter()
        .filter_map(Result::ok);

    for entry in walker {
        // Only process regular files
        if !entry.file_type().is_file() {
            continue;
        }
        total_files += 1;

        let path = entry.into_path();

        // Normalise the file extension to lower‑case
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();

        // We're only interested in .zip and .obj files
        if ext != "zip" && ext != "obj" {
            continue;
        }
        matched_files += 1;

        // Use the stem (filename without extension) as the key
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();

        // Record every occurrence for prefix scans
        idx.names
            .entry(stem.clone())
            .or_default()
            .push(path.clone());

        // Keep the “best” exact match: prefer a .zip over a .obj
        idx.exact
            .entry(stem.clone())
            .and_modify(|existing| {
                let existing_ext = existing
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("");
                let prefer_new = existing_ext.eq_ignore_ascii_case("obj") && ext == "zip";
                if prefer_new {
                    *existing = path.clone();
                }
            })
            .or_insert_with(|| path.clone());
    }

    // Process‑info: summary
    let elapsed = start.elapsed();
    info!(
        "Local index built: scanned {} files, found {} .zip/.obj files in {:.2?}",
        total_files, matched_files, elapsed
    );

    idx
}

fn resolve_by_prefix(idx: &LocalIndex, prefix: &str, prefer_zip: bool) -> Option<PathBuf> {
    if let Some(p) = idx.exact.get(prefix) { return Some(p.clone()); }
    for (name, paths) in idx.names.range(prefix.to_string()..) {
        if !name.starts_with(prefix) { break; }
        if paths.is_empty() { continue; }
        if !prefer_zip { return Some(paths[0].clone()); }
        if let Some(zip_path) = paths.iter().find(|p| {
            p.extension().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("zip")).unwrap_or(false)
        }) {
            return Some(zip_path.clone());
        }
        return Some(paths[0].clone());
    }
    None
}

//
// ---------------- TileKey ----------------
//

const FLAG_TILEKEY_PRESENT: u32 = 1 << 0;

#[repr(u8)]
enum KeyType { XY = 0, NameHash64 = 4 }

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
    h
}

/// Parse `Tile-{x}-{y}-{l1}-{l2}` -> (x, y, zoom)
fn parse_xy_from_prefix(prefix: &str) -> Option<(u32, u32, u8)> {
    static RE: once_cell::sync::Lazy<Regex> =
        once_cell::sync::Lazy::new(|| Regex::new(r"^Tile-(\d+)-(\d+)-(\d+)-(\d+)").unwrap());
    RE.captures(prefix).and_then(|cap| {
        let x = cap.get(1)?.as_str().parse::<u32>().ok()?;
        let y = cap.get(2)?.as_str().parse::<u32>().ok()?;
        let l1 = cap.get(3)?.as_str().parse::<u32>().ok()?;
        Some((x, y, u8::try_from(l1).unwrap_or(u8::MAX)))
    })
}

fn pack_tilekey_reserved(prefix: &str) -> (u32, [u8; 11]) {
    if let Some((x, y, zoom)) = parse_xy_from_prefix(prefix) {
        let mut reserved = [0u8; 11];
        reserved[0] = KeyType::XY as u8;
        reserved[1] = zoom;
        reserved[2..6].copy_from_slice(&x.to_le_bytes());
        reserved[6..10].copy_from_slice(&y.to_le_bytes());
        reserved[10] = 0;  // dataset-defined scheme
        (FLAG_TILEKEY_PRESENT, reserved)
    } else {
        let mut reserved = [0u8; 11];
        reserved[0] = KeyType::NameHash64 as u8;
        let hash = fnv1a64(prefix.as_bytes());
        reserved[1..9].copy_from_slice(&hash.to_le_bytes());
        (FLAG_TILEKEY_PRESENT, reserved)
    }
}

//
// ---------------- GEOT footer ----------------
//

fn write_geot_footer_deg_q7(mut out: &File, bbox: GeoBboxDeg) -> anyhow::Result<()> {
    const Q: f64 = 1e7;
    let lon_min_q7 = (bbox.lon_min * Q).round().clamp(i32::MIN as f64, i32::MAX as f64) as i32;
    let lat_min_q7 = (bbox.lat_min * Q).round().clamp(i32::MIN as f64, i32::MAX as f64) as i32;
    let dlon_q7 = ((bbox.lon_max - bbox.lon_min) * Q).round().clamp(0.0, u32::MAX as f64) as u32;
    let dlat_q7 = ((bbox.lat_max - bbox.lat_min) * Q).round().clamp(0.0, u32::MAX as f64) as u32;

    out.write_all(b"GEOT")?;
    out.write_u8(1)?; // version
    out.write_u8(1)?; // crs_id = 1 (CRS:84 lon/lat degrees)
    out.write_u8(0)?; // mode = 0 (BBOX_DEG_Q7)
    out.write_u8(0)?; // reserved
    out.write_i32::<LE>(lon_min_q7)?;
    out.write_i32::<LE>(lat_min_q7)?;
    out.write_u32::<LE>(dlon_q7)?;
    out.write_u32::<LE>(dlat_q7)?;
    Ok(())
}

//
// ---------------- Quantization (unchanged) ----------------
//

fn quantize_positions(
    positions: &[f32],
    point_budget: usize,
    target_extent: f32,
) -> (Vec<u16>, usize, [f32; 3], [f32; 3]) {
    let n = positions.len() / 3;
    let (mut minx, mut miny, mut minz) = (positions[0], positions[1], positions[2]);
    let (mut maxx, mut maxy, mut maxz) = (minx, miny, minz);
    let mut i = 3;
    while i < positions.len() {
        let x = positions[i]; let y = positions[i+1]; let z = positions[i+2];
        if x < minx { minx = x; } if y < miny { miny = y; } if z < minz { minz = z; }
        if x > maxx { maxx = x; } if y > maxy { maxy = y; } if z > maxz { maxz = z; }
        i += 3;
    }
    let sx=(maxx-minx).max(0.0); let sy=(maxy-miny).max(0.0); let sz=(maxz-minz).max(0.0);
    let eps=1e-20f32;
    let scale = target_extent / sx.max(sy).max(sz).max(eps);
    let dx=0.5*sx*scale; let dy=0.5*sy*scale; let dz=0.5*sz*scale;
    let decode_min=[-dx,-dy,-dz]; let decode_max=[dx,dy,dz];

    let stride = ((n + point_budget - 1) / point_budget).max(1);
    let m = n / stride; let step = stride*3;

    let u=65535.0f32;
    let qx=u/sx.max(eps); let qy=u/sy.max(eps); let qz=u/sz.max(eps);

    let mut q=Vec::<u16>::with_capacity(m*3);
    let mut src=0usize;
    for _ in 0..m {
        let qxv = (((positions[src]   - minx) * qx + 0.5).floor().clamp(0.0,u)) as u16;
        let qyv = (((positions[src+1] - miny) * qy + 0.5).floor().clamp(0.0,u)) as u16;
        let qzv = (((positions[src+2] - minz) * qz + 0.5).floor().clamp(0.0,u)) as u16;
        q.extend_from_slice(&[qxv,qyv,qzv]);
        src += step;
    }
    (q, m, decode_min, decode_max)
}

//
// ---------------- OBJ parsing (unchanged) ----------------
//

fn parse_obj_vertices<R: Read>(r: R) -> Result<Vec<f32>> {
    let mut rd = BufReader::new(r);
    let mut line = String::with_capacity(256);
    let mut out = Vec::<f32>::new();
    loop {
        line.clear();
        let n = rd.read_line(&mut line)?;
        if n == 0 { break; }
        let bytes = line.as_bytes();
        if bytes.len() < 2 || bytes[0] != b'v' || bytes[1] != b' ' { continue; }
        let mut it = line.split_whitespace();
        it.next();
        let x = match it.next().and_then(|s| s.parse::<f32>().ok()) { Some(v) => v, None => continue };
        let y = match it.next().and_then(|s| s.parse::<f32>().ok()) { Some(v) => v, None => continue };
        let z = match it.next().and_then(|s| s.parse::<f32>().ok()) { Some(v) => v, None => continue };
        out.extend_from_slice(&[x,y,z]);
    }
    Ok(out)
}

//
// ==============================
// === SMC1: semantics & PBF  ===
// ==============================

// ---- Class IDs & precedence ----
#[repr(u8)]
#[derive(Clone, Copy)]
enum SemClass {
    Unknown   = 0,
    Building  = 1,
    RoadMajor = 2,
    RoadMinor = 3,
    Path      = 4,
    Water     = 5,
    Park      = 6,
    Woodland  = 7,
    Railway   = 8,
    Parking   = 9,
}

#[inline(always)]
fn class_precedence(c: u8) -> u8 {
    match c {
        5 | 1 => 200, // Water, Building paint last
        8 => 160,     // Railway
        2 => 150,     // RoadMajor
        3 => 140,     // RoadMinor
        4 => 130,     // Path
        6 => 100,     // Park
        7 => 90,      // Woodland
        9 => 80,      // Parking
        _ => 0,
    }
}

// ---- OSM overlay structures (per tile) ----

#[derive(Clone)]
struct Polyline { class: u8, width_m: f32, pts: Arc<Vec<(f64,f64)>> }

#[derive(Clone)]
struct Polygon  { class: u8, ring: Arc<Vec<(f64,f64)>> }

#[derive(Default, Clone)]
struct SemOverlayPerTile {
    roads: Vec<Polyline>,
    areas: Vec<Polygon>,
}

type OverlayMap = HashMap<String, SemOverlayPerTile>;

// ---- Tile index for fast point queries ----
#[derive(Clone)]
struct TileBox {
    idx: u32,
    prefix: String,
    bbox: GeoBboxDeg,      // original bbox
    env: AABB<[f64; 2]>,   // expanded envelope for node hit-tests
}
impl RTreeObject for TileBox {
    type Envelope = AABB<[f64; 2]>;
    #[inline] fn envelope(&self) -> Self::Envelope { self.env }
}
#[inline]
fn make_env(bb: &GeoBboxDeg, pad_lat_deg: f64, pad_lon_deg: f64) -> AABB<[f64; 2]> {
    AABB::from_corners([bb.lon_min - pad_lon_deg, bb.lat_min - pad_lat_deg],
                       [bb.lon_max + pad_lon_deg, bb.lat_max + pad_lat_deg])
}

// ---- Node store limited to nodes near our tiles ----
#[derive(Clone)]
struct NodeRec {
    lon: f64, lat: f64,
    tiles: SmallVec<[u32; 4]>, // tile indices this node touches
}

struct Tick {
    start: Instant,
    last: Instant,
    every: usize,
}
impl Tick {
    fn new(every: usize) -> Self { Self { start: Instant::now(), last: Instant::now(), every } }
    fn should(&mut self, count: usize) -> bool {
        count % self.every == 0 && self.last.elapsed() >= Duration::from_millis(200)
    }
    fn bump(&mut self) { self.last = Instant::now(); }
    fn rate_mps(&self, count: usize) -> f64 {
        let secs = self.start.elapsed().as_secs_f64().max(1e-9);
        (count as f64) / 1.0e6 / secs
    }
}

// Compute pad in degrees for ~50 m
#[inline]
fn pad_degrees_for(lat_deg: f64, pad_m: f64) -> (f64, f64) {
    // meters per deg
    let lat_rad = lat_deg.to_radians();
    let m_per_deg_lat = 110_574.0_f64;
    let m_per_deg_lon = 111_320.0_f64 * lat_rad.cos().abs().max(1e-6);
    (pad_m / m_per_deg_lat, pad_m / m_per_deg_lon)
}

// Width defaults (meters) for highways (very lightweight model)
#[inline]
fn default_highway_width_m(
    kind: &str,
    lanes: Option<u32>,
    width: Option<f32>,
) -> (u8, f32) {
    // If an explicit width is provided, use it (minimum 1.0 m) and map the road kind to a class.
    if let Some(w) = width {
        let class = match kind {
            "motorway" | "trunk" | "primary" => SemClass::RoadMajor as u8,
            "secondary" | "tertiary" | "residential" | "service" => SemClass::RoadMinor as u8,
            _ => SemClass::Path as u8,
        };
        return (class, w.max(1.0));
    }

    // No explicit width – compute from lane count or fall back to defaults.
    let lanes_f = lanes.unwrap_or(0) as f32;
    let lane_width = 3.2_f32;

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
        (lanes_f * lane_width).max(base_width)
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

fn classify_way(
    w: &Way,
) -> Option<(u8 /* class */, f32 /* width_m */, bool /* is_area */)> {
    // Collect all tags of the way into a vector for easy lookup.
    let tags: Vec<(&str, &str)> = w.tags().collect();

    // Helper closure to retrieve the value of a tag by its key.
    let find_tag = |key: &str| -> Option<&str> {
        tags.iter()
            .find_map(|(k, v)| if *k == key { Some(*v) } else { None })
    };

    // Buildings – always an area.
    if let Some(b) = find_tag("building") {
        if !b.is_empty() || b == "yes" {
            return Some((SemClass::Building as u8, 0.0, true));
        }
    }

    // Highways – linear features with a width.
    if let Some(highway) = find_tag("highway") {
        let lanes = find_tag("lanes")
            .and_then(|v| v.parse::<u32>().ok());

        let width = find_tag("width")
            .and_then(|v| parse_width_m(v));

        let (class_id, width_m) = default_highway_width_m(highway, lanes, width);
        return Some((class_id, width_m, false));
    }

    // Natural water bodies.
    if let Some(natural) = find_tag("natural") {
        if natural == "water" {
            return Some((SemClass::Water as u8, 0.0, true));
        }
    }

    // Waterways – we only care about polygons (riverbanks).
    if let Some(waterway) = find_tag("waterway") {
        if waterway == "riverbank" {
            return Some((SemClass::Water as u8, 0.0, true));
        }
        // Other waterway types (e.g., streams) are ignored here.
    }

    // Land‑use categories.
    if let Some(landuse) = find_tag("landuse") {
        if matches!(landuse, "forest" | "grass" | "meadow" | "reservoir") {
            let class_id = match landuse {
                "forest" => SemClass::Woodland as u8,
                _ => SemClass::Park as u8,
            };
            return Some((class_id, 0.0, true));
        }
    }

    // Leisure areas.
    if let Some(leisure) = find_tag("leisure") {
        if matches!(leisure, "park" | "pitch") {
            return Some((SemClass::Park as u8, 0.0, true));
        }
    }

    // Railway lines – treated as linear features with a default width.
    if let Some(railway) = find_tag("railway") {
        if !railway.is_empty() {
            // Width 4 m is a reasonable default for visualisation.
            return Some((SemClass::Railway as u8, 4.0, false));
        }
    }

    // Parking areas.
    if let Some(amenity) = find_tag("amenity") {
        if amenity == "parking" {
            return Some((SemClass::Parking as u8, 0.0, true));
        }
    }

    // No known classification.
    None
}

fn union_bbox(tiles: &[WorkItem], pad_m: f64) -> Option<GeoBboxDeg> {
    // Start with an "empty" bbox using infinities.
    let mut bb = GeoBboxDeg {
        lon_min: f64::INFINITY,
        lat_min: f64::INFINITY,
        lon_max: f64::NEG_INFINITY,
        lat_max: f64::NEG_INFINITY,
    };

    for tile in tiles {
        if let Some(b) = tile.bbox {
            // Pad the tile's bbox, converting the padding from metres to degrees.
            let lat_center = 0.5 * (b.lat_min + b.lat_max);
            let (pad_lat, pad_lon) = pad_degrees_for(lat_center, pad_m);

            bb.lon_min = bb.lon_min.min(b.lon_min - pad_lon);
            bb.lat_min = bb.lat_min.min(b.lat_min - pad_lat);
            bb.lon_max = bb.lon_max.max(b.lon_max + pad_lon);
            bb.lat_max = bb.lat_max.max(b.lat_max + pad_lat);
        }
    }

    if bb.lon_min.is_finite() {
        Some(bb)
    } else {
        None
    }
}

/// Build per-tile OSM overlays by streaming the PBF in two passes.
/// Pass A: collect nodes within expanded tile boxes; Pass B: collect ways touching those nodes.
fn build_osm_overlays(
    pbf_path: &str,
    tiles: &[WorkItem],
    margin_m: f64,
    log_every: usize,
    prefilter: bool,
) -> Result<OverlayMap> {
    // Require bbox per tile
    for t in tiles {
        if t.bbox.is_none() {
            anyhow::bail!("--osm-pbf requires --feature-index with bbox per tile");
        }
    }

    // Compute and log union bbox of tiles (diagnostic)
    let mut u = GeoBboxDeg { lon_min: f64::INFINITY, lat_min: f64::INFINITY,
                              lon_max: f64::NEG_INFINITY, lat_max: f64::NEG_INFINITY };

    for t in tiles {
        let bb = t.bbox.unwrap();
        u.lon_min = u.lon_min.min(bb.lon_min);
        u.lat_min = u.lat_min.min(bb.lat_min);
        u.lon_max = u.lon_max.max(bb.lon_max);
        u.lat_max = u.lat_max.max(bb.lat_max);
    }

    info!(
        "Tiles union bbox: lon[{:.6},{:.6}] lat[{:.6},{:.6}] ({} tiles)",
        u.lon_min, u.lon_max, u.lat_min, u.lat_max, tiles.len()
    );

    // Build expanded tile envelopes
    #[derive(Clone)]
    struct TileBox { idx: u32, prefix: String, bbox: GeoBboxDeg, env: AABB<[f64; 2]> }

    impl RTreeObject for TileBox {
        type Envelope = AABB<[f64; 2]>;
        #[inline] fn envelope(&self) -> Self::Envelope { self.env }
    }

    let mut boxes = Vec::<TileBox>::with_capacity(tiles.len());

    for (i, t) in tiles.iter().enumerate() {
        let bb = t.bbox.unwrap();
        let lat0 = 0.5 * (bb.lat_min + bb.lat_max);
        let (pad_lat, pad_lon) = pad_degrees_for(lat0, margin_m);
        boxes.push(TileBox {
            idx: i as u32,
            prefix: t.prefix.clone(),
            bbox: bb,
            env: AABB::from_corners([bb.lon_min - pad_lon, bb.lat_min - pad_lat],
                                    [bb.lon_max + pad_lon, bb.lat_max + pad_lat]),
        });
    }

    let tree = RTree::bulk_load(boxes);

    if let Some(ubb) = union_bbox(tiles, margin_m) {
        info!(
            "Tiles union bbox (deg): lon [{:.6},{:.6}] lat [{:.6},{:.6}]",
            ubb.lon_min, ubb.lon_max, ubb.lat_min, ubb.lat_max
        );
    }

    // External prefilter (optional)
    let pbf_source = if prefilter {
        prefilter_with_osmium(pbf_path, tiles, margin_m).unwrap_or_else(|| pbf_path.to_string())
    } else {
        pbf_path.to_string()
    };

    let mut tile_node_counts: Vec<u32> = vec![0; tiles.len()];

    info!("OSM pass A: collecting nodes near tiles …");

    let mut node_map: FastHashMap<i64, NodeRec, BuildNoHashHasher<i64>> =
        FastHashMap::with_hasher(BuildNoHashHasher::default());

    let mut total_nodes: usize = 0;
    let mut kept_nodes:  usize = 0;
    let mut seen_dense:  usize = 0;
    let mut seen_plain:  usize = 0;

    let mut tick = Tick::new(log_every.max(1));

    let reader = ElementReader::from_path(&pbf_source)?;
    reader.for_each(|elem| {
        match elem {
            Element::Node(n) => {
                seen_plain += 1;
                total_nodes += 1;

                let lon = n.lon();
                let lat = n.lat();
                let pt_env = AABB::from_point([lon, lat]);

                let mut tiles_touch = SmallVec::<[u32; 4]>::new();

                for tb in tree.locate_in_envelope_intersecting(&pt_env) {
                    tiles_touch.push(tb.idx);
                }

                if !tiles_touch.is_empty() {
                    // Count per-tile hits *before* moving tiles_touch into NodeRec
                    for &ti in tiles_touch.iter() {
                        tile_node_counts[ti as usize] += 1;
                    }
                    node_map.insert(n.id(), NodeRec { lon, lat, tiles: tiles_touch });
                    kept_nodes += 1;
                }
            }
            Element::DenseNode(dn) => {
                // DenseNode is a single node in a compressed array; handle exactly like Node
                seen_dense += 1;
                total_nodes += 1;

                let lon = dn.lon();
                let lat = dn.lat();
                let pt_env = AABB::from_point([lon, lat]);

                let mut tiles_touch = SmallVec::<[u32; 4]>::new();

                for tb in tree.locate_in_envelope_intersecting(&pt_env) {
                    tiles_touch.push(tb.idx);
                }

                if !tiles_touch.is_empty() {
                    // Count per-tile hits *before* moving tiles_touch into NodeRec
                    for &ti in tiles_touch.iter() {
                        tile_node_counts[ti as usize] += 1;
                    }

                    node_map.insert(dn.id(), NodeRec { lon, lat, tiles: tiles_touch });

                    kept_nodes += 1;
                }
            }
            _ => {}
        }

        if tick.should(total_nodes) {
            info!(
                "Pass A: nodes seen {:>11} (dense {:>11} / plain {:>11}), kept {:>11} ({:5.1}%), rate {:>5.2} M/s",
                total_nodes, seen_dense, seen_plain, kept_nodes,
                100.0 * kept_nodes as f64 / (total_nodes as f64).max(1.0),
                tick.rate_mps(total_nodes)
            );
            tick.bump();
        }
    })?;

    info!(
        "OSM pass A: processed {:>11} nodes (dense {:>11} / plain {:>11}), kept {:>11} ({} tiles touched)",
        total_nodes, seen_dense, seen_plain, node_map.len(), tiles.len()
    );

    // Top-10 tiles by retained nodes (debug)
    if let Some(mut pairs) = {
        let mut v: Vec<(usize,u32)> = tile_node_counts.iter().enumerate().map(|(i,c)| (i,*c)).collect();
        v.sort_by_key(|&(_,c)| std::cmp::Reverse(c));
        if !v.is_empty() { Some(v) } else { None }
    } {
        let top = pairs.iter().take(10).map(|&(i,c)| (&tiles[i].prefix, c)).collect::<Vec<_>>();
        debug!("Top tiles by retained nodes (prefix → nodes): {:?}", top);
    }

    info!("OSM pass B: collecting ways …");

    let mut overlays: OverlayMap = HashMap::new();
    let mut total_ways: usize = 0;
    let mut used_ways: usize = 0;
    let mut tick2 = Tick::new(log_every.max(1));

    let reader2 = ElementReader::from_path(&pbf_source)?;
    reader2.for_each(|elem| {
        if let Element::Way(w) = elem {
            total_ways += 1;

            if let Some((class_id, width_m, is_area)) = classify_way(&w) {
                let mut coords: Vec<(f64, f64)> = Vec::with_capacity(w.refs().len().min(2048));
                let mut tiles_hit: SmallVec<[u32; 8]> = SmallVec::new();

                for nr in w.refs() {
                    if let Some(nrinfo) = node_map.get(&nr) {
                        coords.push((nrinfo.lon, nrinfo.lat));
                        for &ti in nrinfo.tiles.iter() {
                            if !tiles_hit.contains(&ti) { tiles_hit.push(ti); }
                        }
                    }
                }

                let need_n = if is_area { 3 } else { 2 };

                if coords.len() >= need_n && !tiles_hit.is_empty() {
                    used_ways += 1;

                    let coords_arc = Arc::new(coords);

                    for ti in tiles_hit {
                        let t = &tiles[ti as usize];
                        let entry = overlays
                            .entry(t.prefix.clone())
                            .or_insert_with(SemOverlayPerTile::default);

                        if is_area {
                            entry.areas.push(Polygon { class: class_id, ring: coords_arc.clone() });
                        } else {
                            entry.roads.push(Polyline { class: class_id, width_m, pts: coords_arc.clone() });
                        }
                    }
                }
            }

            if tick2.should(total_ways) {
                info!(
                    "Pass B: ways seen {:>11}, kept {:>11} ({:5.1}%), rate {:>5.2} M/s",
                    total_ways,
                    used_ways,
                    100.0 * used_ways as f64 / (total_ways as f64).max(1.0),
                    tick2.rate_mps(total_ways)
                );
                tick2.bump();
            }
        }
    })?;

    // Coverage summary
    let tiles_with_overlays = overlays.len();
    let (mut sum_roads, mut sum_areas) = (0usize, 0usize);
    for v in overlays.values() {
        sum_roads += v.roads.len();
        sum_areas += v.areas.len();
    }
    info!(
        "OSM pass B: processed {} ways; overlays for {} tile(s) (roads {}, areas {})",
        total_ways, tiles_with_overlays, sum_roads, sum_areas
    );

    // If zero overlays, print three sample tiles with bbox for debugging.
    if tiles_with_overlays == 0 {
        for (i, t) in tiles.iter().take(3).enumerate() {
            let bb = t.bbox.unwrap();
            warn!(
                "No overlays; sample tile[{}] {} bbox lon[{:.6},{:.6}] lat[{:.6},{:.6}]",
                i, t.prefix, bb.lon_min, bb.lon_max, bb.lat_min, bb.lat_max
            );
        }
    }

    Ok(overlays)
}

fn prefilter_with_osmium(pbf_in: &str, tiles: &[WorkItem], margin_m: f64) -> Option<String> {
    use std::process::Command;
    // Try to detect 'osmium'
    let Ok(ver) = Command::new("osmium").arg("--version").output() else {
        warn!("'osmium' not found; skipping external prefilter.");
        return None;
    };
    info!("Using external prefilter via {}", String::from_utf8_lossy(&ver.stdout).trim());

    // Union bbox of all tiles (expanded by margin)
    let mut lon_min = f64::INFINITY;
    let mut lat_min = f64::INFINITY;
    let mut lon_max = f64::NEG_INFINITY;
    let mut lat_max = f64::NEG_INFINITY;
    for t in tiles {
        if let Some(bb) = t.bbox {
            let lat0 = 0.5 * (bb.lat_min + bb.lat_max);
            let (pad_lat, pad_lon) = pad_degrees_for(lat0, margin_m.max(0.0));
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

    // Temp paths
    let tmp_extract = format!("{}.extract.pbf", pbf_in);
    let tmp_filtered = format!("{}.filtered.pbf", pbf_in);

    // Extract by bbox
    let st = Command::new("osmium")
        .args(["extract", "-b", &bbox, "--overwrite", "-o", &tmp_extract, pbf_in])
        .status()
        .ok()?;
    if !st.success() {
        warn!("osmium extract failed; skipping prefilter.");
        return None;
    }

    // Filter to relevant tags (kept minimal, extend as needed)
    let filter = "nwr/building nwr/highway nwr/landuse=forest,grass,meadow,reservoir nwr/leisure=park,pitch nwr/natural=water nwr/waterway=riverbank nwr/railway nwr/amenity=parking";
    let st2 = Command::new("osmium")
        .args(["tags-filter", "--overwrite", "-o", &tmp_filtered, &tmp_extract])
        .args(filter.split_whitespace())
        .status()
        .ok()?;
    if !st2.success() {
        warn!("osmium tags-filter failed; using bbox extract only.");
        return Some(tmp_extract);
    }
    Some(tmp_filtered)
}

// ---- Rasterization into SMC1 (decode space) ----

struct SemMask {
    w: u16, h: u16,
    data: Vec<u8>, // row-major, 1 byte per pixel
}

#[inline]
fn lonlat_to_decode_xy(
    lon: f64,
    lat: f64,
    bb: GeoBboxDeg,
    dmin: [f32; 3],
    dmax: [f32; 3],
) -> (f32, f32) {
    let sx = (dmax[0] - dmin[0]) as f64;
    let sy = (dmax[1] - dmin[1]) as f64;

    let x = dmin[0] as f64
        + (lon - bb.lon_min) / (bb.lon_max - bb.lon_min + f64::EPSILON) * sx;
    let y = dmin[1] as f64
        + (lat - bb.lat_min) / (bb.lat_max - bb.lat_min + f64::EPSILON) * sy;

    (x as f32, y as f32)
}

#[inline]
fn decode_xy_to_pixel(
    x: f32,
    y: f32,
    dmin: [f32; 3],
    dmax: [f32; 3],
    w: u16,
    h: u16,
) -> (i32, i32) {
    let wx = (x - dmin[0]) * (w as f32) / (dmax[0] - dmin[0]).max(f32::EPSILON);
    let wy = (y - dmin[1]) * (h as f32) / (dmax[1] - dmin[1]).max(f32::EPSILON);

    (wx.floor() as i32, wy.floor() as i32)
}

#[inline]
fn clamp_i(v: i32, lo: i32, hi: i32) -> i32 {
    v.max(lo).min(hi)
}

fn paint_pixel(mask: &mut SemMask, x: i32, y: i32, class: u8) {
    // Discard coordinates that lie outside the mask bounds.
    if x < 0
        || y < 0
        || x >= mask.w as i32
        || y >= mask.h as i32
    {
        return;
    }

    // Linear index into the mask's class buffer.
    let idx = (y as usize) * (mask.w as usize) + (x as usize);
    let previous_class = mask.data[idx];

    // Overwrite only if the new class has equal or higher precedence.
    if class_precedence(class) >= class_precedence(previous_class) {
        mask.data[idx] = class;
    }
}

fn rasterize_polygon(mask: &mut SemMask, poly: &[(i32,i32)], class: u8) {
    if poly.len() < 3 { return; }
    // bounding box
    let (mut xmin, mut ymin, mut xmax, mut ymax) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for &(x,y) in poly { xmin = xmin.min(x); xmax = xmax.max(x); ymin = ymin.min(y); ymax = ymax.max(y); }
    xmin = clamp_i(xmin, 0, mask.w as i32 - 1);
    xmax = clamp_i(xmax, 0, mask.w as i32 - 1);
    ymin = clamp_i(ymin, 0, mask.h as i32 - 1);
    ymax = clamp_i(ymax, 0, mask.h as i32 - 1);

    // Even-odd fill by point-in-polygon test per pixel center
    for y in ymin..=ymax {
        for x in xmin..=xmax {
            let mut c = false;
            let mut j = poly.len() - 1;
            for i in 0..poly.len() {
                let (xi, yi) = poly[i]; let (xj, yj) = poly[j];
                let yi_cmp = (yi > y) != (yj > y);
                if yi_cmp {
                    let x_inter = (xj - xi) as f32 * ( (y - yi) as f32 / ((yj - yi) as f32 + 1e-20) ) + xi as f32;
                    if (x as f32) < x_inter { c = !c; }
                }
                j = i;
            }
            if c { paint_pixel(mask, x, y, class); }
        }
    }
}

#[inline]
fn sqr(x: f32) -> f32 { x * x }

/// Rasterizes a polyline (road, path, etc.) onto a semantic mask.
///
/// * `mask` – The mutable semantic mask that will receive the rasterised pixels.
/// * `line` – A slice of pixel coordinates `(i32, i32)` representing the polyline
///   in image space (already transformed from geographic coordinates). The slice
///   must contain at least two vertices; otherwise the function returns early.
/// * `radius_px` – Desired half‑width of the line in **pixel** units. The function
///   clamps this value to a minimum of `0.5` px to guarantee that a line is at
///   least one pixel wide and to avoid division‑by‑zero issues.
/// * `class` – The semantic class identifier (e.g. `SemClass::RoadMajor as u8`).
///   Pixels are painted using `paint_pixel`, which respects class precedence.
///
/// The algorithm walks each consecutive pair of vertices (`seg`) and draws a
/// thick line segment with radius `r`. For each segment it computes an axis‑aligned
/// bounding box expanded by the radius, clamps the box to the mask extents, and
/// then iterates over every pixel in that box. For each pixel the shortest
/// distance to the line segment is calculated; if the distance is within the
/// radius the pixel is painted with the supplied `class`.
///
fn rasterize_polyline(mask: &mut SemMask, line: &[(i32,i32)], radius_px: f32, class: u8) {
    // Need at least two points to form a line segment.
    if line.len() < 2 { return; }

    // Clamp radius to a sensible minimum and pre‑compute its square for distance tests.
    let r = radius_px.max(0.5);
    let r2 = r * r;

    // Process each consecutive pair of vertices as a line segment.
    for seg in line.windows(2) {
        // Segment endpoints as floating‑point coordinates (pixel centre is at +0.5).
        let (x0, y0) = (seg[0].0 as f32, seg[0].1 as f32);
        let (x1, y1) = (seg[1].0 as f32, seg[1].1 as f32);

        // Compute an axis‑aligned bounding box expanded by the radius.
        let (minx, maxx) = ((x0.min(x1) - r).floor() as i32, (x0.max(x1) + r).ceil() as i32);
        let (miny, maxy) = ((y0.min(y1) - r).floor() as i32, (y0.max(y1) + r).ceil() as i32);

        // Clamp the bounding box to the mask dimensions.
        let (minx, maxx) = (clamp_i(minx, 0, mask.w as i32 - 1), clamp_i(maxx, 0, mask.w as i32 - 1));
        let (miny, maxy) = (clamp_i(miny, 0, mask.h as i32 - 1), clamp_i(maxy, 0, mask.h as i32 - 1));

        // Vector of the segment and a small epsilon to avoid division by zero.
        let vx = x1 - x0;
        let vy = y1 - y0;
        let denom = vx * vx + vy * vy + 1e-12;

        // Test every pixel inside the clamped bounding box.
        for y in miny..=maxy {
            for x in minx..=maxx {
                // Pixel centre in floating‑point coordinates.
                let px = x as f32 + 0.5;
                let py = y as f32 + 0.5;

                // Project the pixel centre onto the line, clamped to the segment.
                let t = ((px - x0) * vx + (py - y0) * vy) / denom;
                let t = t.clamp(0.0, 1.0);
                let qx = x0 + t * vx;
                let qy = y0 + t * vy;

                // Squared distance from pixel centre to the nearest point on the segment.
                let d2 = sqr(px - qx) + sqr(py - qy);

                // Paint the pixel if it lies within the radius.
                if d2 <= r2 {
                    paint_pixel(mask, x, y, class);
                }
            }
        }
    }
}

/// Build a semantic mask (SMC1) for a single tile.
fn build_smc1_mask(
    overlay: &SemOverlayPerTile,
    tile_bbox_deg: GeoBboxDeg,
    decode_min: [f32; 3],
    decode_max: [f32; 3],
    grid: u16,
) -> SemMask {
    // Initialise the mask with the "unknown" class.
    let mut mask = SemMask {
        w: grid,
        h: grid,
        data: vec![SemClass::Unknown as u8; (grid as usize) * (grid as usize)],
    };

    // ------------------------------------------------------------------------
    // 1. Rasterise area polygons (lowest precedence, later draws overwrite higher).
    // ------------------------------------------------------------------------
    for a in &overlay.areas {
        // Convert the geographic ring to decode‑space, then to pixel coordinates.
        let mut ring_px: Vec<(i32, i32)> = Vec::with_capacity(a.ring.len());

        for &(lon, lat) in a.ring.as_ref().iter() {
            let (dx, dy) = lonlat_to_decode_xy(lon, lat, tile_bbox_deg, decode_min, decode_max);
            let (ix, iy) = decode_xy_to_pixel(dx, dy, decode_min, decode_max, grid, grid);
            ring_px.push((ix, iy));
        }

        rasterize_polygon(&mut mask, &ring_px, a.class);
    }

    // ------------------------------------------------------------------------
    // 2. Rasterise road polylines (higher precedence than areas).
    //    Compute an isotropic pixel size (meters per pixel) at the tile centre.
    // ------------------------------------------------------------------------
    let lat0 = 0.5 * (tile_bbox_deg.lat_min + tile_bbox_deg.lat_max);
    let (m_per_deg_lat, m_per_deg_lon) = (
        110_574.0_f64,
        111_320.0_f64 * lat0.to_radians().cos().abs().max(1e-6),
    );
    let deg_w = tile_bbox_deg.lon_max - tile_bbox_deg.lon_min;
    let deg_h = tile_bbox_deg.lat_max - tile_bbox_deg.lat_min;
    let px_m_x = (deg_w * m_per_deg_lon) / (grid as f64);
    let px_m_y = (deg_h * m_per_deg_lat) / (grid as f64);
    // Approximate isotropic meters‑per‑pixel by averaging the X/Y values.
    let px_m = ((px_m_x + px_m_y) * 0.5) as f32;

    for r in &overlay.roads {
        // Convert roadway width (meters) to a pixel radius.
        let radius_px = r.width_m * 0.5 / px_m;

        // Transform each geographic point of the polyline to pixel coordinates.
        let mut line_px: Vec<(i32, i32)> = Vec::with_capacity(r.pts.len());

        for &(lon, lat) in r.pts.as_ref().iter() {
            let (dx, dy) = lonlat_to_decode_xy(lon, lat, tile_bbox_deg, decode_min, decode_max);
            let (ix, iy) = decode_xy_to_pixel(dx, dy, decode_min, decode_max, grid, grid);
            line_px.push((ix, iy));
        }

        rasterize_polyline(&mut mask, &line_px, radius_px, r.class);
    }

    mask
}

fn print_semantic_mask_ascii(mask: &SemMask) {
    info!("Semantic Mask ASCII Preview ({}x{}):", mask.w, mask.h);

    for y in 0..mask.h {
        let row_start = (y as usize) * (mask.w as usize);
        let row_end = row_start + (mask.w as usize);
        let row_data = &mask.data[row_start..row_end];

        let line: String = row_data
            .iter()
            .map(|&class| {
                match class {
                    0 => '.',   // Unknown
                    1 => 'B',   // Building
                    2 => '#',   // RoadMajor
                    3 => '+',   // RoadMinor
                    4 => '-',   // Path
                    5 => '~',   // Water
                    6 => 'P',   // Park
                    7 => 'W',   // Woodland
                    8 => 'R',   // Railway
                    9 => 'p',   // Parking
                    _ => '?',
                }
            })
            .collect();

        println!("{}", line); // or use `info!("{}", line);` if you want it in logs
    }
}

// ---- SMC1 chunk writer ----
fn write_smc1_chunk(out: &mut File, mask: &SemMask, compress: bool) -> Result<()> {
    print_semantic_mask_ascii(mask);

    // Header fields
    let version: u8 = 1;
    let encoding: u8 = if compress { 1 } else { 0 };

    // Class table: (class_id, precedence)
    let class_table: &[(u8, u8)] = &[
        (SemClass::Unknown as u8,   class_precedence(SemClass::Unknown as u8)),
        (SemClass::Building as u8,  class_precedence(SemClass::Building as u8)),
        (SemClass::RoadMajor as u8, class_precedence(SemClass::RoadMajor as u8)),
        (SemClass::RoadMinor as u8, class_precedence(SemClass::RoadMinor as u8)),
        (SemClass::Path as u8,      class_precedence(SemClass::Path as u8)),
        (SemClass::Water as u8,     class_precedence(SemClass::Water as u8)),
        (SemClass::Park as u8,      class_precedence(SemClass::Park as u8)),
        (SemClass::Woodland as u8,  class_precedence(SemClass::Woodland as u8)),
        (SemClass::Railway as u8,   class_precedence(SemClass::Railway as u8)),
        (SemClass::Parking as u8,   class_precedence(SemClass::Parking as u8)),
    ];

    // Optional compression of the mask payload
    let payload: Vec<u8> = if compress {
        compress_to_vec_zlib(&mask.data, 6)
    } else {
        mask.data.clone()
    };
    let data_len = payload.len();

    // Build the chunk payload
    let mut buf = Vec::new();
    buf.write_u8(version)?;                     // version
    buf.write_u8(encoding)?;                    // encoding (0 = raw, 1 = zlib)
    buf.write_u16::<LE>(mask.w)?;               // width
    buf.write_u16::<LE>(mask.h)?;               // height
    buf.write_u8(0)?;                           // coord_space = 0 (decode XY)
    buf.write_u8(class_table.len() as u8)?;     // number of classes
    buf.write_u16::<LE>(0)?;                    // reserved

    // Write each class entry: id, precedence, reserved (0)
    for (cid, prec) in class_table.iter() {
        buf.write_u8(*cid)?;
        buf.write_u8(*prec)?;
        buf.write_u16::<LE>(0)?; // reserved per-entry
    }

    // Length of the (compressed) mask data followed by the data itself
    buf.write_u32::<LE>(data_len as u32)?;
    buf.extend_from_slice(&payload);

    // Final chunk layout: "SMC1" + length + payload
    out.write_all(b"SMC1")?;
    out.write_u32::<LE>(buf.len() as u32)?;
    out.write_all(&buf)?;
    Ok(())
}

//
// ---------------- Mesh processing ----------------
//

fn process_one_mesh(
    path: &Path,
    args: &Args,
    prefix: &str,
    bbox: Option<GeoBboxDeg>,
    overlays: Option<&SemOverlayPerTile>, // === SMC1 additions ===
) -> Result<()> {
    // --------------------------------------------------------------------
    // Determine output filename.
    // --------------------------------------------------------------------
    let stem = Path::new(prefix)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("tile");
    let out_path = Path::new(&args.output_dir).join(format!("{stem}.hypc"));

    // Skip existing files unless overwriting is requested.
    if out_path.exists() && !args.overwrite {
        info!("SKIP {} (exists)", out_path.display());
        return Ok(());
    }

    info!("Processing {} → {}", path.display(), out_path.display());

    // --------------------------------------------------------------------
    // Load vertex positions from either a .zip or a plain .obj file.
    // --------------------------------------------------------------------
    let positions: Vec<f32> = match path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
    {
        Some(ext) if ext == "zip" => {
            // Open the zip archive and locate the largest .obj entry.
            let file = File::open(path)
                .with_context(|| format!("open {}", path.display()))?;
            let mut archive = match zip::ZipArchive::new(file) {
                Ok(a) => a,
                Err(e) => {
                    warn!("Skipping unreadable zip {}: {e}", path.display());
                    return Ok(());
                }
            };

            let mut best_idx: Option<usize> = None;
            let mut best_size: u64 = 0;

            for i in 0..archive.len() {
                let entry = archive.by_index(i)?;
                if entry.is_file()
                    && entry
                        .name()
                        .to_ascii_lowercase()
                        .ends_with(".obj")
                {
                    let sz = entry.size();
                    if sz > best_size {
                        best_size = sz;
                        best_idx = Some(i);
                    }
                }
            }

            let idx = match best_idx {
                Some(i) => i,
                None => {
                    warn!("No .obj in {} — skipping", path.display());
                    return Ok(());
                }
            };

            let mut obj_file = archive.by_index(idx)?;
            parse_obj_vertices(&mut obj_file)?
        }
        Some(ext) if ext == "obj" => {
            // Plain .obj file.
            let file = File::open(path)
                .with_context(|| format!("open {}", path.display()))?;
            parse_obj_vertices(file)?
        }
        _ => {
            warn!("Unsupported extension for {}", path.display());
            return Ok(());
        }
    };

    if positions.is_empty() {
        warn!("{}: no vertices", path.display());
        return Ok(());
    }

    // --------------------------------------------------------------------
    // Quantize positions (unchanged logic elsewhere).
    // --------------------------------------------------------------------
    let (q, m, decode_min, decode_max) =
        quantize_positions(&positions, args.point_budget, args.target_extent);

    // --------------------------------------------------------------------
    // Write the HPC1 header + quantized payload.
    // --------------------------------------------------------------------
    let mut out = File::create(&out_path)
        .with_context(|| format!("create {}", out_path.display()))?;
    out.write_all(b"HPC1")?;

    let (flags, reserved) = pack_tilekey_reserved(prefix);
    out.write_u32::<LE>(1)?; // version
    out.write_u32::<LE>(flags)?; // flags
    out.write_u32::<LE>(m as u32)?; // point count
    out.write_u8(16)?; // quantisation bits
    out.write_all(&reserved)?; // reserved bytes

    for &v in &decode_min {
        out.write_f32::<LE>(v)?;
    }
    for &v in &decode_max {
        out.write_f32::<LE>(v)?;
    }
    for v in q {
        out.write_u16::<LE>(v)?;
    }
    out.flush()?;

    // --------------------------------------------------------------------
    // Optional SMC1 semantic mask.
    // --------------------------------------------------------------------
    if args.write_smc1 {
        match (bbox, overlays) {
            (Some(bb), Some(ov)) => {
                debug!(
                    "SMC1 for '{}' → roads {}  areas {}  (grid {}x{}, {})",
                    prefix,
                    ov.roads.len(),
                    ov.areas.len(),
                    args.sem_grid, args.sem_grid,
                    if args.smc1_compress { "zlib" } else { "raw" }
                );

                let mask = build_smc1_mask(ov, bb, decode_min, decode_max, args.sem_grid);
                write_smc1_chunk(&mut out, &mask, args.smc1_compress)?;
            }
            (Some(_), None) if args.osm_pbf.is_some() => {
                debug!("SMC1 omitted for '{}' (overlay map has no entry for this prefix)", prefix);
            }
            (None, _) if args.osm_pbf.is_some() => {
                debug!("SMC1 omitted for '{}' (no bbox available from feature index)", prefix);
            }
            _ => {}
        }
    }

    // --------------------------------------------------------------------
    // Optional GEOT footer (legacy).
    // --------------------------------------------------------------------
    if args.write_geot {
        if let Some(bb) = bbox {
            write_geot_footer_deg_q7(&out, bb)?;
        } else {
            debug!("No bbox for '{}' → GEOT footer omitted", prefix);
        }
    }

    // --------------------------------------------------------------------
    // Estimate output size for logging.
    // --------------------------------------------------------------------
    let mut tail = 52usize + m * 6; // base size
    if args.write_smc1 && overlays.is_some() {
        // Rough size: mask header + payload (compression varies)
        tail += (args.sem_grid as usize) * (args.sem_grid as usize) + 64;
    }
    if args.write_geot && bbox.is_some() {
        tail += 24; // GEOT footer size
    }

    info!(
        "OK {} → {} ({} pts → {} pts; ~{:.2} MiB)",
        path.file_name()
            .unwrap_or_default()
            .to_string_lossy(),
        out_path.display(),
        positions.len() / 3,
        m,
        (tail as f64) / (1024.0 * 1024.0)
    );

    Ok(())
}

//
// ---------------- main ----------------
//

fn main() -> Result<()> {
    // Initialise the logger (environment‑controlled) and parse CLI arguments.
    env_logger::init();
    let args = Args::parse();

    // Create the output directory hierarchy, propagating any I/O errors.
    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("creating output dir {}", args.output_dir))?;

    // Discover local .zip/.obj files and index them by basename.
    let lidx = build_local_index(&args.input_dir);

    // Build the list of work items.
    // When a feature index is supplied we also obtain per‑tile bounding boxes;
    // otherwise only the prefix is known and SMC1 cannot be generated.
    let work: Vec<WorkItem> = if let Some(fi) = &args.feature_index {
        info!("Loading feature index: {}", fi);

        let mut items = load_feature_index(fi)?;

        // keep only those that resolve
        items.retain(|it| resolve_by_prefix(&lidx, &it.prefix, args.prefer_zip).is_some());

        info!("After pruning to existing meshes: {} work items", items.len());

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

    // --------------------------------------------------------------------
    // Resolve feature‑index tiles to on‑disk files once.
    // --------------------------------------------------------------------
    #[derive(Clone)]
    struct Resolved {
        item: WorkItem,
        path: PathBuf,
    }

    let mut resolved: Vec<Resolved> = Vec::with_capacity(work.len());
    let mut missing: Vec<String> = Vec::new();

    for w in &work {
        if let Some(p) = resolve_by_prefix(&lidx, &w.prefix, args.prefer_zip) {
            resolved.push(Resolved {
                item: w.clone(),
                path: p,
            });
        } else {
            missing.push(w.prefix.clone());
        }
    }

    info!(
        "Matched {} / {} feature(s) to local files ({} missing).",
        resolved.len(),
        work.len(),
        missing.len()
    );
    if !missing.is_empty() {
        debug!(
            "First 20 missing prefixes: {:?}",
            &missing.iter().take(20).collect::<Vec<_>>()
        );
    }

    // --- build overlays for exactly these tiles ---
    let overlays_map: Option<Arc<OverlayMap>> = if let Some(pbf) = &args.osm_pbf {
        info!("OSM PBF: {}", pbf);
        // pass only the resolved tiles into OSM overlay builder
        let tiles_for_overlays: Vec<WorkItem> = resolved.iter().map(|r| r.item.clone()).collect();
        let overlays = build_osm_overlays(
            pbf,
            &tiles_for_overlays,
            args.osm_margin_m,
            args.osm_log_every,
            args.osm_prefilter,
        )?;
        Some(Arc::new(overlays))
    } else {
        None
    };

    info!("Work items: {}", resolved.len());

    // --------------------------------------------------------------------
    // Parallel processing of all resolved tiles.
    // --------------------------------------------------------------------
    let start_time = Instant::now();
    let total_items = resolved.len();
    info!(
        "Starting processing of {} items using {} Rayon threads",
        total_items,
        rayon::current_num_threads()
    );

    // Clone the Arc for thread‑safe sharing.
    let overlays_map = overlays_map.clone();

    resolved.par_iter().enumerate().for_each(|(idx, r)| {
        info!(
            "Processing {}/{}: {}",
            idx + 1,
            total_items,
            r.item.prefix
        );

        // Look up the optional overlay for this tile.
        let ov = overlays_map.as_ref().and_then(|m| m.get(&r.item.prefix));

        // Process the mesh; any error is logged but does not abort the whole run.
        if let Err(e) = process_one_mesh(&r.path, &args, &r.item.prefix, r.item.bbox, ov) {
            error!("{}: {e:#}", r.path.display());
        }
    });

    // Report overall elapsed time.
    let elapsed = start_time.elapsed();
    info!(
        "Finished processing {} items in {:.2?}",
        total_items, elapsed
    );

    Ok(())
}
