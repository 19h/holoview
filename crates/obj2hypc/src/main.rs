use anyhow::{Context, Result};
use clap::Parser;
use log::{debug, error, info, warn};
use rayon::prelude::*;
use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File},
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use walkdir::WalkDir;

use regex::Regex;

// Fast hash map for integer keys (node ids)
use hashbrown::HashMap as FastHashMap;
use nohash_hasher::BuildNoHashHasher;

// OSM / geometry utilities
use osmpbf::{Element, ElementReader, Way};
use rstar::{RTree, RTreeObject, AABB};
use smallvec::SmallVec;

use hypc::{self, GeoCrs, GeoExtentDeg, HypcWrite, SemanticMask as Smc1, Smc1Encoding, TileKey};

/// The program performs the following high‑level steps:
/// 1. Initialise logging and parse command‑line arguments.
/// 2. Ensure the output directory exists.
/// 3. Scan input for `.zip` and `.obj`.
/// 4. Build work list (optionally from feature index with per‑tile bbox).
/// 5. Optionally build OSM overlays (per‑tile).
/// 6. For each tile, quantize and write `.hypc` using the `hypc` crate.
/// 7. Report timing.
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
    let start = std::time::Instant::now();

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

    info!(
        "Loaded {} feature(s) from '{}' in {:.2?}",
        items.len(),
        path,
        start.elapsed()
    );

    Ok(items)
}

#[derive(Default)]
struct LocalIndex {
    exact: HashMap<String, PathBuf>,          // exact basename -> best path
    names: BTreeMap<String, Vec<PathBuf>>,    // for starts_with scans
}

fn build_local_index(input_dir: &str) -> LocalIndex {
    let mut idx = LocalIndex::default();

    let start = std::time::Instant::now();
    let mut total_files = 0usize;
    let mut matched_files = 0usize;

    let walker = WalkDir::new(input_dir)
        .follow_links(true)
        .into_iter()
        .filter_map(Result::ok);

    for entry in walker {
        if !entry.file_type().is_file() { continue; }
        total_files += 1;

        let path = entry.into_path();
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();

        if ext != "zip" && ext != "obj" {
            continue;
        }
        matched_files += 1;

        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();

        idx.names
            .entry(stem.clone())
            .or_default()
            .push(path.clone());

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

    info!(
        "Local index built: scanned {} files, found {} .zip/.obj files in {:.2?}",
        total_files, matched_files, start.elapsed()
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

// ---------------- TileKey helpers ----------------

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

fn tilekey_from_prefix(prefix: &str) -> TileKey {
    if let Some((x, y, zoom)) = parse_xy_from_prefix(prefix) {
        TileKey::XY {
            zoom,
            x,
            y,
            scheme: 0, // dataset-defined scheme=0
        }
    } else {
        let hash = fnv1a64(prefix.as_bytes());
        TileKey::NameHash64 { hash }
    }
}

// ---------------- GEOT footer ----------------

fn to_hypc_geot(bb: GeoBboxDeg) -> GeoExtentDeg {
    GeoExtentDeg {
        lon_min: bb.lon_min,
        lat_min: bb.lat_min,
        lon_max: bb.lon_max,
        lat_max: bb.lat_max,
    }
}

// ---------------- Quantization (unchanged) ----------------

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

// ---------------- OBJ parsing (unchanged) ----------------

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

// TileBox/RTree and overlay builder identical to previous version omitted for brevity in review.
// (Full code remains exactly as in your provided file; only writer call changes.)

// ---- Utils for meters->degrees padding ----
#[inline]
fn pad_degrees_for(lat_deg: f64, pad_m: f64) -> (f64, f64) {
    let lat_rad = lat_deg.to_radians();
    let m_per_deg_lat = 110_574.0_f64;
    let m_per_deg_lon = 111_320.0_f64 * lat_rad.cos().abs().max(1e-6);
    (pad_m / m_per_deg_lat, pad_m / m_per_deg_lon)
}

// ... [OSM overlay builder functions from your original code remain unchanged]
// ... build_osm_overlays(..) and prefilter_with_osmium(..) are identical to the version you supplied.
// (They are long and were not modified; keep them verbatim.)
// For completeness in your repo, retain the exact implementations you posted.

/// Rasterization storage used locally to build SMC1.
struct SemMask {
    w: u16, h: u16,
    data: Vec<u8>, // row-major, 1 byte per pixel
}

// decode-space transforms and rasterizers identical to your original
// (lonlat_to_decode_xy, decode_xy_to_pixel, rasterize_polygon, rasterize_polyline, etc.)
// Keep them verbatim from your provided file.

fn build_smc1_mask(
    overlay: &SemOverlayPerTile,
    tile_bbox_deg: GeoBboxDeg,
    decode_min: [f32; 3],
    decode_max: [f32; 3],
    grid: u16,
) -> SemMask {
    let mut mask = SemMask {
        w: grid,
        h: grid,
        data: vec![SemClass::Unknown as u8; (grid as usize) * (grid as usize)],
    };

    // (1) polygons
    for a in &overlay.areas {
        let mut ring_px: Vec<(i32, i32)> = Vec::with_capacity(a.ring.len());
        for &(lon, lat) in a.ring.as_ref().iter() {
            let (dx, dy) = lonlat_to_decode_xy(lon, lat, tile_bbox_deg, decode_min, decode_max);
            let (ix, iy) = decode_xy_to_pixel(dx, dy, decode_min, decode_max, grid, grid);
            ring_px.push((ix, iy));
        }
        rasterize_polygon(&mut mask, &ring_px, a.class);
    }

    // (2) polylines with isotropic px size
    let lat0 = 0.5 * (tile_bbox_deg.lat_min + tile_bbox_deg.lat_max);
    let (m_per_deg_lat, m_per_deg_lon) = (
        110_574.0_f64,
        111_320.0_f64 * lat0.to_radians().cos().abs().max(1e-6),
    );
    let deg_w = tile_bbox_deg.lon_max - tile_bbox_deg.lon_min;
    let deg_h = tile_bbox_deg.lat_max - tile_bbox_deg.lat_min;
    let px_m_x = (deg_w * m_per_deg_lon) / (grid as f64);
    let px_m_y = (deg_h * m_per_deg_lat) / (grid as f64);
    let px_m = ((px_m_x + px_m_y) * 0.5) as f32;

    for r in &overlay.roads {
        let radius_px = r.width_m * 0.5 / px_m;
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
                    0 => '.', 1 => 'B', 2 => '#', 3 => '+', 4 => '-',
                    5 => '~', 6 => 'P', 7 => 'W', 8 => 'R', 9 => 'p', _ => '?',
                }
            })
            .collect();
        println!("{}", line);
    }
}

// ---------------- Mesh processing ----------------

fn process_one_mesh(
    path: &Path,
    args: &Args,
    prefix: &str,
    bbox: Option<GeoBboxDeg>,
    overlays: Option<&SemOverlayPerTile>,
) -> Result<()> {
    // Output file
    let stem = Path::new(prefix)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("tile");
    let out_path = Path::new(&args.output_dir).join(format!("{stem}.hypc"));

    if out_path.exists() && !args.overwrite {
        info!("SKIP {} (exists)", out_path.display());
        return Ok(());
    }

    info!("Processing {} → {}", path.display(), out_path.display());

    // Load vertex positions
    let positions: Vec<f32> = match path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
    {
        Some(ext) if ext == "zip" => {
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
                if entry.is_file() && entry.name().to_ascii_lowercase().ends_with(".obj") {
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

    // Quantize
    let (q, m, decode_min, decode_max) =
        quantize_positions(&positions, args.point_budget, args.target_extent);

    // Build optional SMC1 (data is decompressed in-memory; hypc handles compression)
    let smc1_opt: Option<Smc1> = if args.write_smc1 {
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
                print_semantic_mask_ascii(&mask);

                // Build palette identical to reader/writer contract
                let palette: Vec<(u8, u8)> = vec![
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

                Some(Smc1 {
                    width: mask.w,
                    height: mask.h,
                    data: mask.data,
                    palette,
                    coord_space: 0,
                    encoding: if args.smc1_compress { Smc1Encoding::Zlib } else { Smc1Encoding::Raw },
                })
            }
            (Some(_), None) if args.osm_pbf.is_some() => {
                debug!("SMC1 omitted for '{}' (overlay map has no entry for this prefix)", prefix);
                None
            }
            (None, _) if args.osm_pbf.is_some() => {
                debug!("SMC1 omitted for '{}' (no bbox available from feature index)", prefix);
                None
            }
            _ => None,
        }
    } else {
        None
    };

    // Optional GEOT
    let (geog_crs, geog_bbox) = if args.write_geot {
        if let Some(bb) = bbox {
            (Some(GeoCrs::Crs84), Some(to_hypc_geot(bb)))
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    // Tile key
    let tile_key = Some(tilekey_from_prefix(prefix));

    // Assemble writer params and emit
    let writer = HypcWrite {
        quant_bits: 16,
        quantized_positions: &q,
        decode_min,
        decode_max,
        tile_key,
        geog_crs,
        geog_bbox_deg: geog_bbox,
        smc1: smc1_opt.as_ref(),
    };

    hypc::write_file(&out_path.to_string_lossy(), &writer)
        .with_context(|| format!("write {}", out_path.display()))?;

    // Estimate output size (rough)
    let mut tail = 52usize + m * 6; // base size
    if writer.smc1.is_some() {
        tail += (args.sem_grid as usize) * (args.sem_grid as usize) + 64;
    }
    if writer.geog_crs.is_some() && writer.geog_bbox_deg.is_some() {
        tail += 24;
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

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("creating output dir {}", args.output_dir))?;

    let lidx = build_local_index(&args.input_dir);

    // Build work list
    let work: Vec<WorkItem> = if let Some(fi) = &args.feature_index {
        info!("Loading feature index: {}", fi);
        let mut items = load_feature_index(fi)?;
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

    // Resolve tiles to files
    #[derive(Clone)]
    struct Resolved { item: WorkItem, path: PathBuf }
    let mut resolved: Vec<Resolved> = Vec::with_capacity(work.len());
    let mut missing: Vec<String> = Vec::new();
    for w in &work {
        if let Some(p) = resolve_by_prefix(&lidx, &w.prefix, args.prefer_zip) {
            resolved.push(Resolved { item: w.clone(), path: p });
        } else {
            missing.push(w.prefix.clone());
        }
    }
    info!(
        "Matched {} / {} feature(s) to local files ({} missing).",
        resolved.len(), work.len(), missing.len()
    );
    if !missing.is_empty() {
        debug!("First 20 missing prefixes: {:?}", &missing.iter().take(20).collect::<Vec<_>>());
    }

    // Overlays (optional)
    let overlays_map: Option<Arc<OverlayMap>> = if let Some(pbf) = &args.osm_pbf {
        info!("OSM PBF: {}", pbf);
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

    // Parallel processing
    let start_time = Instant::now();
    let total_items = resolved.len();
    info!(
        "Starting processing of {} items using {} Rayon threads",
        total_items,
        rayon::current_num_threads()
    );
    let overlays_map = overlays_map.clone();
    resolved.par_iter().enumerate().for_each(|(idx, r)| {
        info!("Processing {}/{}: {}", idx + 1, total_items, r.item.prefix);
        let ov = overlays_map.as_ref().and_then(|m| m.get(&r.item.prefix));
        if let Err(e) = process_one_mesh(&r.path, &args, &r.item.prefix, r.item.bbox, ov) {
            error!("{}: {e:#}", r.path.display());
        }
    });
    info!("Finished processing {} items in {:.2?}", total_items, start_time.elapsed());
    Ok(())
}
