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
    /// OBJ is `[lon, lat, h_m]`.
    Geodetic,
    /// OBJ is ECEF meters `[X, Y, Z]`.
    Ecef,
    /// OBJ is local meters `[x, y, z]` in an arbitrary local frame.
    LocalM,
}

impl std::fmt::Display for InputCs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            InputCs::Auto => "auto",
            InputCs::Geodetic => "geodetic",
            InputCs::Ecef => "ecef",
            InputCs::LocalM => "local_m",
        };

        f.write_str(s)
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
    // The first ring is the outer boundary of the polygon.
    let ring = &poly.coordinates[0];

    // Initialise the extents.
    let (mut min_lon, mut min_lat) = (f64::INFINITY, f64::INFINITY);
    let (mut max_lon, mut max_lat) = (f64::NEG_INFINITY, f64::NEG_INFINITY);

    // Walk all vertices, updating the bounds for finite coordinates.
    for &[lon, lat] in ring {
        if lon.is_finite() && lat.is_finite() {
            if lon < min_lon {
                min_lon = lon;
            }
            if lon > max_lon {
                max_lon = lon;
            }
            if lat < min_lat {
                min_lat = lat;
            }
            if lat > max_lat {
                max_lat = lat;
            }
        }
    }

    GeoBboxDeg {
        lon_min: min_lon,
        lat_min: min_lat,
        lon_max: max_lon,
        lat_max: max_lat,
    }
}

fn load_feature_index(path: &str) -> anyhow::Result<Vec<WorkItem>> {
    // Open and deserialize the GeoJSON feature collection.
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let root: GeoJsonRoot = serde_json::from_reader(reader)?;

    // Transform each feature into a `WorkItem`.
    let items = root
        .features
        .into_iter()
        .map(|feature| {
            // Derive the prefix from the feature's URL (file stem without extension).
            let prefix = std::path::Path::new(&feature.properties.url)
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();

            // Compute the bounding box from the geometry.
            let bbox = Some(bbox_from_polygon_deg(&feature.geometry));

            WorkItem { prefix, bbox }
        })
        .collect();

    Ok(items)
}

#[derive(Default)]
struct LocalIndex {
    exact: HashMap<String, PathBuf>,
    names: BTreeMap<String, Vec<PathBuf>>,
}

fn build_local_index(input_dir: &str) -> LocalIndex {
    let mut index = LocalIndex::default();

    // Walk the directory tree, following symlinks, and collect only regular files.
    for entry in WalkDir::new(input_dir)
        .follow_links(true)
        .into_iter()
        .filter_map(Result::ok)
    {
        // Skip non‑files (e.g. directories).
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.into_path();

        // Determine the lower‑cased file extension.
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();

        // We're only interested in OBJ and ZIP files.
        if ext != "obj" && ext != "zip" {
            continue;
        }

        // Use the file stem (name without extension) as the lookup key.
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_owned();

        // Register the path for prefix‑based searches.
        index
            .names
            .entry(stem.clone())
            .or_default()
            .push(path.clone());

        // Maintain an exact‑match map, preferring `.zip` over `.obj`.
        index
            .exact
            .entry(stem.clone())
            .and_modify(|existing| {
                let existing_ext = existing
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("");
                // Replace a stored `.obj` with a `.zip` if we encounter one.
                if existing_ext.eq_ignore_ascii_case("obj") && ext == "zip" {
                    *existing = path.clone();
                }
            })
            .or_insert_with(|| path.clone());
    }

    index
}

fn resolve_by_prefix(idx: &LocalIndex, prefix: &str, prefer_zip: bool) -> Option<PathBuf> {
    // 1️⃣  Exact match – the fast path.
    if let Some(path) = idx.exact.get(prefix) {
        return Some(path.clone());
    }

    // 2️⃣  Prefix‑based lookup.  `BTreeMap::range` gives us all keys that are
    //     lexicographically ≥ the given start string.
    let start = prefix.to_string();
    for (name, paths) in idx.names.range(start..) {
        // Once we encounter a key that does not start with the prefix we can stop.
        if !name.starts_with(prefix) {
            break;
        }

        // Choose a suitable path from the collected candidates.
        let candidate = if !prefer_zip {
            // Caller prefers the first entry regardless of extension.
            paths.first().cloned()
        } else {
            // Look for a `.zip` entry first; fall back to the first entry.
            paths
                .iter()
                .find(|p| {
                    p.extension()
                        .and_then(|s| s.to_str())
                        .map(|s| s.eq_ignore_ascii_case("zip"))
                        .unwrap_or(false)
                })
                .cloned()
                .or_else(|| paths.first().cloned())
        };

        if candidate.is_some() {
            return candidate;
        }
    }

    // No match found.
    None
}

/// Produce a fixed‑size 32‑byte tile key from a textual prefix.
/// The key is zero‑padded or truncated to exactly 32 bytes.
fn tilekey_from_prefix(prefix: &str) -> [u8; 32] {
    let mut key = [0u8; 32];
    let bytes = prefix.as_bytes();
    let len = bytes.len().min(key.len());
    key[..len].copy_from_slice(&bytes[..len]);
    key
}

/// Read the raw vertex triples from an OBJ file (or any `Read` source).
fn parse_obj_vertices<R: Read>(reader: R) -> Result<Vec<[f64; 3]>> {
    let mut vertices = Vec::new();

    for line_result in BufReader::new(reader).lines() {
        let line = line_result?;
        let trimmed = line.trim();

        // OBJ vertex records begin with "v ".
        if !trimmed.starts_with("v ") {
            continue;
        }

        // Split the line into its whitespace‑separated components.
        let mut parts = trimmed.split_whitespace();

        parts.next(); // Skip the leading "v"

        // Parse the three coordinate values, providing a clear error if missing.
        let x: f64 = parts
            .next()
            .context("Missing x coordinate")?
            .parse()?;

        let y: f64 = parts
            .next()
            .context("Missing y coordinate")?
            .parse()?;

        let z: f64 = parts
            .next()
            .context("Missing z coordinate")?
            .parse()?;

        // Store only finite triples.
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

/// Helper that periodically logs progress.
#[derive(Debug, Clone, Copy)]
struct Tick {
    start: Instant,
    last: Instant,
    every: usize,
}

impl Tick {
    /// Create a new `Tick` that will trigger at most once per `every` items.
    #[inline]
    fn new(every: usize) -> Self {
        let now = Instant::now();
        Self {
            start: now,
            last: now,
            every: every.max(1),
        }
    }

    /// Returns `true` when the supplied `count` is a multiple of `every` **and**
    /// at least 200 ms have elapsed since the previous log.
    #[inline]
    fn should(&mut self, count: usize) -> bool {
        const MIN_INTERVAL: std::time::Duration = std::time::Duration::from_millis(200);
        count % self.every == 0 && self.last.elapsed() >= MIN_INTERVAL
    }

    /// Record the current instant as the time of the latest log.
    #[inline]
    fn bump(&mut self) {
        self.last = Instant::now();
    }

    /// Compute the processing rate in million items per second.
    #[inline]
    fn rate_mps(&self, count: usize) -> f64 {
        const ONE_MILLION: f64 = 1_000_000.0;

        let elapsed = self.start.elapsed().as_secs_f64().max(1e-9);

        (count as f64) / ONE_MILLION / elapsed
    }
}

/// Convert padding (metres) to degree offsets for latitude and longitude.
#[inline]
fn pad_degrees_for(latitude_deg: f64, pad_meters: f64) -> (f64, f64) {
    const METERS_PER_DEG_LAT: f64 = 110_574.0;
    const METERS_PER_DEG_LON_EQUATOR: f64 = 111_320.0;

    // Scale longitude based on latitude.
    let meters_per_deg_lon = METERS_PER_DEG_LON_EQUATOR
        * latitude_deg.to_radians().cos().abs()
        .max(1e-6);

    (pad_meters / METERS_PER_DEG_LAT, pad_meters / meters_per_deg_lon)
}

#[inline]
fn default_highway_width_m(kind: &str, lanes: Option<u32>, width: Option<f32>) -> (u8, f32) {
    // If an explicit width is supplied, use it (minimum 1 m) and map the class.
    if let Some(w) = width {
        let class = match kind {
            "motorway" | "trunk" | "primary" => SemClass::RoadMajor as u8,
            "secondary" | "tertiary" | "residential" | "service" => SemClass::RoadMinor as u8,
            _ => SemClass::Path as u8,
        };
        return (class, w.max(1.0));
    }

    // Base width for known highway types.
    let base_width = match kind {
        "motorway" => 12.0,
        "trunk" => 10.0,
        "primary" => 8.0,
        "secondary" => 7.0,
        "tertiary" => 6.0,
        "residential" | "service" => 5.0,
        _ => 2.0,
    };

    // Adjust width based on lane count, if available.
    let lanes_f = lanes.unwrap_or(0) as f32;
    let width_m = if lanes_f >= 2.0 {
        (lanes_f * 3.2).max(base_width)
    } else {
        base_width
    };

    // Determine the semantic class for the highway kind.
    let class = match kind {
        "motorway" | "trunk" | "primary" => SemClass::RoadMajor as u8,
        "secondary" | "tertiary" | "residential" | "service" => SemClass::RoadMinor as u8,
        _ => SemClass::Path as u8,
    };

    (class, width_m)
}

#[inline]
fn parse_width_m(s: &str) -> Option<f32> {
    // Normalise the input string.
    let s = s.trim().to_ascii_lowercase();

    // Strip known unit suffixes.
    if let Some(num) = s.strip_suffix('m') {
        return num.trim().parse::<f32>().ok();
    }
    if let Some(num) = s.strip_suffix("ft") {
        return num.trim().parse::<f32>().ok().map(|v| v * 0.3048);
    }

    // Fallback: plain number interpreted as metres.
    s.parse::<f32>().ok()
}

fn classify_way(w: &Way) -> Option<(u8, f32, bool)> {
    // Collect all tags for repeated lookup.
    let tags: Vec<(&str, &str)> = w.tags().collect();

    // Helper that returns the first value associated with a given key.
    let get = |key: &str| tags.iter().find_map(|(k, v)| if *k == key { Some(*v) } else { None });

    // ----- Buildings -------------------------------------------------------
    if let Some(building) = get("building") {
        if !building.is_empty() || building == "yes" {
            return Some((SemClass::Building as u8, 0.0, true));
        }
    }

    // ----- Highways --------------------------------------------------------
    if let Some(highway) = get("highway") {
        let lanes = get("lanes").and_then(|v| v.parse::<u32>().ok());
        let width = get("width").and_then(parse_width_m);
        let (class, width_m) = default_highway_width_m(highway, lanes, width);
        return Some((class, width_m, false));
    }

    // ----- Water -----------------------------------------------------------
    if let Some(natural) = get("natural") {
        if natural == "water" {
            return Some((SemClass::Water as u8, 0.0, true));
        }
    }

    if let Some(waterway) = get("waterway") {
        if waterway == "riverbank" {
            return Some((SemClass::Water as u8, 0.0, true));
        }
    }

    // ----- Land‑use (forest, grass, meadow, reservoir) --------------------
    if let Some(landuse) = get("landuse") {
        let class = match landuse {
            "forest" => SemClass::Woodland as u8,
            "grass" | "meadow" | "reservoir" => SemClass::Park as u8,
            _ => 0,
        };
        if class != 0 {
            return Some((class, 0.0, true));
        }
    }

    // ----- Leisure (park, pitch) -------------------------------------------
    if let Some(leisure) = get("leisure") {
        if matches!(leisure, "park" | "pitch") {
            return Some((SemClass::Park as u8, 0.0, true));
        }
    }

    // ----- Railway ---------------------------------------------------------
    if let Some(rail) = get("railway") {
        if !rail.is_empty() {
            return Some((SemClass::Railway as u8, 4.0, false));
        }
    }

    // ----- Parking amenity -------------------------------------------------
    if let Some(amenity) = get("amenity") {
        if amenity == "parking" {
            return Some((SemClass::Parking as u8, 0.0, true));
        }
    }

    // No matching classification.
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
    // --------------------------------------------------------------------
    // Ensure every tile provides a bounding box – required for the OSM overlay.
    // --------------------------------------------------------------------
    for tile in tiles {
        if tile.bbox.is_none() {
            anyhow::bail!("--osm-pbf requires --feature-index with bbox per tile");
        }
    }

    // --------------------------------------------------------------------
    // Build an R‑tree index of all (padded) tile bounding boxes.
    // --------------------------------------------------------------------
    let tile_boxes: Vec<TileBox> = tiles
        .iter()
        .enumerate()
        .map(|(idx, tile)| {
            let bbox = tile.bbox.unwrap(); // safe – validated above
            let (pad_lat, pad_lon) = pad_degrees_for(
                0.5 * (bbox.lat_min + bbox.lat_max),
                margin_m,
            );
            TileBox {
                idx: idx as u32,
                env: AABB::from_corners(
                    [bbox.lon_min - pad_lon, bbox.lat_min - pad_lat],
                    [bbox.lon_max + pad_lon, bbox.lat_max + pad_lat],
                ),
            }
        })
        .collect();

    let tile_tree = RTree::bulk_load(tile_boxes);

    // --------------------------------------------------------------------
    // Possibly pre‑filter the PBF with osmium, falling back to the original.
    // --------------------------------------------------------------------
    let pbf_source = if prefilter {
        prefilter_with_osmium(pbf_path, tiles, margin_m).unwrap_or_else(|| pbf_path.to_string())
    } else {
        pbf_path.to_string()
    };

    // --------------------------------------------------------------------
    // First pass: read all nodes, keep those that intersect any tile.
    // --------------------------------------------------------------------
    let mut node_map: hashbrown::HashMap<i64, NodeRec, nohash_hasher::BuildNoHashHasher<i64>> =
        hashbrown::HashMap::with_hasher(nohash_hasher::BuildNoHashHasher::default());

    let mut seen_nodes = 0usize;
    let mut tick = Tick::new(log_every);

    ElementReader::from_path(&pbf_source)?.for_each(|elem| {
        // Extract node data; ignore everything else.
        let (id, lon, lat) = match elem {
            Element::Node(node) => (node.id(), node.lon(), node.lat()),
            Element::DenseNode(dn) => (dn.id(), dn.lon(), dn.lat()),
            _ => return,
        };

        seen_nodes += 1;

        // Determine which tiles contain this node.
        let mut touching_tiles = SmallVec::<[u32; 4]>::new();
        for tb in tile_tree.locate_in_envelope_intersecting(&AABB::from_point([lon, lat])) {
            touching_tiles.push(tb.idx);
        }

        // Keep the node only if it belongs to at least one tile.
        if !touching_tiles.is_empty() {
            node_map.insert(
                id,
                NodeRec {
                    lon,
                    lat,
                    tiles: touching_tiles,
                },
            );
        }

        // Periodic progress report.
        if tick.should(seen_nodes) {
            info!(
                "Pass A: nodes seen {:>11}, kept {:>11}, rate {:5.2} M/s",
                seen_nodes,
                node_map.len(),
                tick.rate_mps(seen_nodes)
            );
            tick.bump();
        }
    })?;

    // --------------------------------------------------------------------
    // Second pass: read ways and build per‑tile semantic overlays.
    // --------------------------------------------------------------------
    let mut overlays: OverlayMap = HashMap::new();
    let mut seen_ways = 0usize;
    tick = Tick::new(log_every);

    ElementReader::from_path(&pbf_source)?.for_each(|elem| {
        if let Element::Way(way) = elem {
            seen_ways += 1;

            // Classify the way and obtain its rendering parameters.
            if let Some((class_id, width_m, is_area)) = classify_way(&way) {
                // Gather coordinates for all referenced nodes that are present in
                // our node_map, and collect the set of tiles the way touches.
                let mut coords = Vec::with_capacity(way.refs().len());
                let mut touched_tiles = SmallVec::<[u32; 8]>::new();

                for node_ref in way.refs() {
                    if let Some(node) = node_map.get(&node_ref) {
                        coords.push((node.lon, node.lat));
                        for &ti in &node.tiles {
                            if !touched_tiles.contains(&ti) {
                                touched_tiles.push(ti);
                            }
                        }
                    }
                }

                // We need at least two points for a line or three for a polygon.
                let enough_coords = if is_area { coords.len() >= 3 } else { coords.len() >= 2 };
                if enough_coords && !touched_tiles.is_empty() {
                    let coords_arc = Arc::new(coords);
                    for tile_idx in touched_tiles {
                        let tile = &tiles[tile_idx as usize];
                        let entry = overlays.entry(tile.prefix.clone()).or_default();
                        if is_area {
                            entry
                                .areas
                                .push(Polygon { class: class_id, ring: coords_arc.clone() });
                        } else {
                            entry.roads.push(Polyline {
                                class: class_id,
                                width_m,
                                pts: coords_arc.clone(),
                            });
                        }
                    }
                }
            }

            // Periodic progress report.
            if tick.should(seen_ways) {
                info!(
                    "Pass B: ways seen {:>11}, rate {:5.2} M/s",
                    seen_ways,
                    tick.rate_mps(seen_ways)
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

/// Convert normalized UV coordinates (0.0 to 1.0) to pixel coordinates.
#[inline]
fn uv_to_pixel(u: f32, v: f32, w: u16, h: u16) -> (i32, i32) {
    let u_clamped = u.clamp(0.0, 1.0);
    let v_clamped = v.clamp(0.0, 1.0);

    let x = (u_clamped * (w as f32 - 1.0)).round() as i32;
    let y = (v_clamped * (h as f32 - 1.0)).round() as i32;

    (x, y)
}

fn paint_pixel(mask: &mut SemMask, x: i32, y: i32, class: u8) {
    // Check bounds
    if x < 0 || y < 0 || x >= mask.w as i32 || y >= mask.h as i32 {
        return;
    }

    // Compute index and update if new class has higher precedence
    let idx = y as usize * mask.w as usize + x as usize;
    if class_precedence(class) >= class_precedence(mask.data[idx]) {
        mask.data[idx] = class;
    }
}

fn rasterize_polygon(mask: &mut SemMask, poly: &[(i32, i32)], class: u8) {
    // A polygon needs at least three vertices.
    if poly.len() < 3 {
        return;
    }

    // ------- Compute the axis‑aligned bounding box of the polygon ------------
    let (mut xmin, mut ymin, mut xmax, mut ymax) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for &(x, y) in poly {
        xmin = xmin.min(x);
        xmax = xmax.max(x);
        ymin = ymin.min(y);
        ymax = ymax.max(y);
    }

    // ------- Clamp the bbox to the mask extents ------------------------------
    xmin = clamp_i(xmin, 0, mask.w as i32 - 1);
    xmax = clamp_i(xmax, 0, mask.w as i32 - 1);
    ymin = clamp_i(ymin, 0, mask.h as i32 - 1);
    ymax = clamp_i(ymax, 0, mask.h as i32 - 1);

    // ------- Scan the bounding rectangle and apply the even‑odd rule ---------
    let n = poly.len();
    for y in ymin..=ymax {
        for x in xmin..=xmax {
            let mut inside = false;
            let mut j = n - 1; // Index of the previous vertex

            for i in 0..n {
                let (xi, yi) = poly[i];
                let (xj, yj) = poly[j];

                // Edge crosses the horizontal line at y?
                if (yi > y) != (yj > y) {
                    // Compute the x‑coordinate of the intersection.
                    let x_inter = (xj - xi) as f32
                        * ((y - yi) as f32 / ((yj - yi) as f32 + 1e-20))
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

/// Rasterises a polyline onto the semantic mask, expanding it by a
/// radius (in pixels) and writing the given class to any covered
/// pixels.
fn rasterize_polyline(
    mask: &mut SemMask,
    line: &[(i32, i32)],
    radius_px: f32,
    class: u8,
) {
    // Need at least a start and end point to form a segment.
    if line.len() < 2 {
        return;
    }

    // Ensure a sensible minimum radius (half‑pixel) and pre‑compute its square.
    let radius = radius_px.max(0.5);
    let radius_sq = radius * radius;

    // Process each consecutive pair of vertices.
    for segment in line.windows(2) {
        // Convert the integer coordinates to floating point for distance math.
        let (x0, y0) = (segment[0].0 as f32, segment[0].1 as f32);
        let (x1, y1) = (segment[1].0 as f32, segment[1].1 as f32);

        // Determine an axis‑aligned bounding box for the segment, expanded
        // by the radius, and clamp it to the mask extents.
        let min_x = clamp_i((x0.min(x1) - radius).floor() as i32, 0, mask.w as i32 - 1);
        let max_x = clamp_i((x0.max(x1) + radius).ceil() as i32, 0, mask.w as i32 - 1);
        let min_y = clamp_i((y0.min(y1) - radius).floor() as i32, 0, mask.h as i32 - 1);
        let max_y = clamp_i((y0.max(y1) + radius).ceil() as i32, 0, mask.h as i32 - 1);

        // Vector from the first to the second endpoint.
        let dx = x1 - x0;
        let dy = y1 - y0;
        // Length‑squared of the segment (add epsilon to avoid division by zero).
        let denom = dx * dx + dy * dy + 1e-12_f32;

        // Scan the bounded pixel region.
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                // Coordinates of the pixel centre.
                let px = x as f32 + 0.5;
                let py = y as f32 + 0.5;

                // Projection of the pixel centre onto the segment (clamped to [0,1]).
                let t = ((px - x0) * dx + (py - y0) * dy) / denom;
                let t = t.clamp(0.0, 1.0);

                // Closest point on the segment to the pixel centre.
                let cx = x0 + t * dx;
                let cy = y0 + t * dy;

                // If the pixel centre lies within the radius, paint it.
                if sqr(px - cx) + sqr(py - cy) <= radius_sq {
                    paint_pixel(mask, x, y, class);
                }
            }
        }
    }
}

fn build_smc1_mask(
    overlay: &SemOverlayPerTile,
    tile_bbox_deg: GeoBboxDeg,
    grid: u16,
) -> SemMask {
    // --------------------------------------------------------------------
    // Initialise an empty mask – one-byte per pixel, initially all zero.
    // --------------------------------------------------------------------
    let mut mask = SemMask {
        w: grid,
        h: grid,
        data: vec![0u8; (grid as usize).pow(2)],
    };

    // --------------------------------------------------------------------
    // Helpers for converting geographic coordinates to normalised UV space.
    // --------------------------------------------------------------------
    const EPS: f64 = 1e-12; // guard against degenerate zero‑size tiles

    let lon_range = (tile_bbox_deg.lon_max - tile_bbox_deg.lon_min).max(EPS);
    let lat_range = (tile_bbox_deg.lat_max - tile_bbox_deg.lat_min).max(EPS);

    let lon_to_u = |lon: f64| ((lon - tile_bbox_deg.lon_min) / lon_range) as f32;
    let lat_to_v = |lat: f64| ((lat - tile_bbox_deg.lat_min) / lat_range) as f32;

    // --------------------------------------------------------------------
    // Rasterise polygonal areas (e.g. buildings, water, parks).
    // --------------------------------------------------------------------
    for area in &overlay.areas {
        let ring_px: Vec<(i32, i32)> = area
            .ring
            .iter()
            .map(|&(lon, lat)| uv_to_pixel(lon_to_u(lon), lat_to_v(lat), grid, grid))
            .collect();

        rasterize_polygon(&mut mask, &ring_px, area.class);
    }

    // --------------------------------------------------------------------
    // Determine an approximate metres‑per‑pixel scale.
    // --------------------------------------------------------------------
    let mid_lat = 0.5 * (tile_bbox_deg.lat_min + tile_bbox_deg.lat_max);
    let metres_per_lon_deg = 111_320.0 * mid_lat.to_radians().cos().abs().max(1e-6);
    let metres_per_lat_deg = 110_574.0;

    let metres_per_px_lon = (lon_range * metres_per_lon_deg) / grid as f64;
    let metres_per_px_lat = (lat_range * metres_per_lat_deg) / grid as f64;
    let avg_metres_per_px = 0.5 * (metres_per_px_lon + metres_per_px_lat);

    // --------------------------------------------------------------------
    // Rasterise road polylines, expanding each by half its width (in metres).
    // --------------------------------------------------------------------
    for road in &overlay.roads {
        // Convert half‑width from metres to pixel radius.
        let radius_px = (road.width_m as f64 * 0.5 / avg_metres_per_px) as f32;

        let line_px: Vec<(i32, i32)> = road
            .pts
            .iter()
            .map(|&(lon, lat)| uv_to_pixel(lon_to_u(lon), lat_to_v(lat), grid, grid))
            .collect();

        rasterize_polyline(&mut mask, &line_px, radius_px, road.class);
    }

    mask
}

// ---------- Input CS detection and safe quantization ----------

/// Heuristic to decide how OBJ vertex coordinates should be interpreted.
fn detect_input_cs(sample: &[[f64; 3]]) -> InputCs {
    // --------------------------------------------------------------------
    // 1  Count how many vertices appear to be geographic (lon/lat) values.
    // --------------------------------------------------------------------
    let sample_len = sample.len().max(1);
    let sample_len_f64 = sample_len as f64;

    let geo_like = sample
        .iter()
        .filter(|p| p[0].abs() <= 180.0 && p[1].abs() <= 90.0)
        .count();

    // If at least 90% of the vertices are within geographic bounds, treat as
    // geodetic coordinates.
    if (geo_like as f64) / sample_len_f64 >= 0.90 {
        return InputCs::Geodetic;
    }

    // --------------------------------------------------------------------
    // 2   Try to recognise Earth‑Centered‑Earth‑Fixed (ECEF) coordinates.
    //     We look at the mean distance from the origin and compare it with the
    //     typical Earth radius (+‑ a generous margin for height / noise).
    // --------------------------------------------------------------------
    const ECEF_MIN: f64 = 6_200_000.0; // metres
    const ECEF_MAX: f64 = 6_500_000.0; // metres
    const MAX_SAMPLES: usize = 4_096;

    let take = sample_len.min(MAX_SAMPLES);
    let radius_sum: f64 = sample
        .iter()
        .take(take)
        .map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt())
        .sum();

    let radius_mean = radius_sum / take as f64;
    if radius_mean.is_finite() && (ECEF_MIN..=ECEF_MAX).contains(&radius_mean) {
        return InputCs::Ecef;
    }

    // --------------------------------------------------------------------
    // 3   Default to a generic local meter‑based coordinate system.
    // --------------------------------------------------------------------
    InputCs::LocalM
}

#[derive(Debug, Clone)]
struct Quantized {
    /// Anchor point expressed in integer units.
    anchor_units: [i64; 3],
    /// Vertices quantised to signed 32‑bit integers relative to the anchor.
    points_units: Vec<[i32; 3]>,
    /// The units‑per‑meter value actually used after any down‑scaling.
    used_upm: u32,
}

/// Quantize to integer lattice with an anchor, automatically down‑scaling
/// `units_per_meter` (UPM) to fit into an `i32` if necessary.
fn quantize_with_anchor(points_m: &[[f64; 3]], requested_upm: u32) -> Quantized {
    debug_assert!(!points_m.is_empty());

    // ------------------------------------------------------------------------
    // 1  Compute the centroid (anchor) in metres.
    // ------------------------------------------------------------------------
    let (sum_x, sum_y, sum_z) = points_m.iter().fold((0.0_f64, 0.0_f64, 0.0_f64), |(ax, ay, az), p| {
        (ax + p[0], ay + p[1], az + p[2])
    });
    let inv_n = 1.0_f64 / points_m.len() as f64;
    let anchor_m = [sum_x * inv_n, sum_y * inv_n, sum_z * inv_n];

    // ------------------------------------------------------------------------
    // 2  Determine the maximum absolute offset from the anchor (in metres).
    // ------------------------------------------------------------------------
    const EPS: f64 = 1e-12;
    let max_off_m = points_m
        .iter()
        .map(|p| {
            (p[0] - anchor_m[0])
                .abs()
                .max((p[1] - anchor_m[1]).abs())
                .max((p[2] - anchor_m[2]).abs())
        })
        .fold(0.0_f64, f64::max);

    // ------------------------------------------------------------------------
    // 3  Choose a usable UPM that fits all offsets into a signed 32‑bit int.
    // ------------------------------------------------------------------------
    // If the geometry collapses to a point we can keep the caller's request.
    let mut upm = if max_off_m <= EPS {
        requested_upm
    } else {
        // Aim to keep a 5% head‑room before hitting i32::MAX.
        let max_upm_fit = ((i32::MAX as f64) / (max_off_m * 1.05))
            .floor()
            .clamp(1.0, requested_upm as f64) as u32;

        if max_upm_fit < requested_upm {
            warn!(
                "units_per_meter={} too high for this tile span (~{:.3} m max offset). \
                 Using {} u/m instead.",
                requested_upm, max_off_m, max_upm_fit
            );
        }
        max_upm_fit.max(1)
    };

    // ------------------------------------------------------------------------
    // 4  Helper: try to quantise all points with the current UPM.
    // ------------------------------------------------------------------------
    fn try_quantize(
        points_m: &[[f64; 3]],
        upm: u32,
        anchor_units: [i64; 3],
    ) -> Option<Vec<[i32; 3]>> {
        let mut out = Vec::with_capacity(points_m.len());

        for p in points_m {
            let ux = quantize_units(p[0], upm) - anchor_units[0];
            let uy = quantize_units(p[1], upm) - anchor_units[1];
            let uz = quantize_units(p[2], upm) - anchor_units[2];

            // Guard against overflow of the signed 32‑bit range.
            if ux < i32::MIN as i64
                || ux > i32::MAX as i64
                || uy < i32::MIN as i64
                || uy > i32::MAX as i64
                || uz < i32::MIN as i64
                || uz > i32::MAX as i64
            {
                return None;
            }

            out.push([ux as i32, uy as i32, uz as i32]);
        }

        Some(out)
    }

    // ------------------------------------------------------------------------
    // 5  Compute the anchor in integer units for the current UPM.
    // ------------------------------------------------------------------------
    let mut anchor_units = [
        quantize_units(anchor_m[0], upm),
        quantize_units(anchor_m[1], upm),
        quantize_units(anchor_m[2], upm),
    ];

    // ------------------------------------------------------------------------
    // 6  Attempt quantisation; on failure, keep halving UPM until it succeeds.
    // ------------------------------------------------------------------------
    let points_units = loop {
        match try_quantize(points_m, upm, anchor_units) {
            Some(v) => break v,
            None => {
                // Reduce UPM (add safety margin) and recompute anchor units.
                upm = ((upm as f64) * 0.5).floor().max(1.0) as u32;
                warn!("Further reduced units_per_meter to {} for safety.", upm);
                anchor_units = [
                    quantize_units(anchor_m[0], upm),
                    quantize_units(anchor_m[1], upm),
                    quantize_units(anchor_m[2], upm),
                ];
                // Loop will retry with the new values.
            }
        }
    };

    // ------------------------------------------------------------------------
    // 7  Assemble the result.
    // ------------------------------------------------------------------------
    Quantized {
        anchor_units,
        points_units,
        used_upm: upm,
    }
}

/// Compute East‑North‑Up orthonormal basis vectors for a given latitude/longitude.
#[inline]
fn enu_basis(lat_deg: f64, lon_deg: f64) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();

    let (sl, cl) = lat.sin_cos(); // sin(lat), cos(lat)
    let (so, co) = lon.sin_cos(); // sin(lon), cos(lon)

    // East vector
    let e = [-so, co, 0.0];
    // North vector
    let n = [-sl * co, -sl * so, cl];
    // Up vector
    let u = [cl * co, cl * so, sl];

    (e, n, u)
}

fn process_one_mesh(
    path: &Path,
    args: &Args,
    prefix: &str,
    bbox: Option<GeoBboxDeg>,
    overlays: Option<&SemOverlayPerTile>,
) -> Result<()> {
    use log::debug;

    // ---------------------------------------------------------------------
    // Output path handling
    // ---------------------------------------------------------------------
    let out_path = Path::new(&args.output_dir).join(format!(
        "{}.hypc",
        Path::new(prefix)
            .file_stem()
            .expect("prefix must have a stem")
            .to_string_lossy()
    ));

    if out_path.exists() && !args.overwrite {
        debug!("Skipping existing file: {}", out_path.display());
        return Ok(());
    }

    info!("Processing {} -> {}", path.display(), out_path.display());

    // ---------------------------------------------------------------------
    // Load raw OBJ vertices (supports plain .obj or .zip containing a single .obj)
    // ---------------------------------------------------------------------
    debug!("Loading vertices from {}", path.display());
    let raw_xyz: Vec<[f64; 3]> = if path.extension().and_then(|s| s.to_str()) == Some("zip") {
        debug!("Opening ZIP archive");
        let file = File::open(path)?;

        let mut archive = zip::ZipArchive::new(file)?;

        let obj_name = archive
            .file_names()
            .find(|n| n.to_ascii_lowercase().ends_with(".obj"))
            .context("No .obj file found in zip archive")?.to_owned();

        debug!("Found OBJ file in ZIP: {}", obj_name);
        let mut obj_file = archive.by_name(&obj_name)?;

        parse_obj_vertices(&mut obj_file)?
    } else {
        debug!("Opening OBJ file directly");
        parse_obj_vertices(File::open(path)?)?
    };

    if raw_xyz.is_empty() {
        warn!("{}: no vertices", path.display());
        return Ok(());
    }

    debug!("Loaded {} raw vertices", raw_xyz.len());

    // ---------------------------------------------------------------------
    // Determine coordinate system (auto‑detect if requested)
    // ---------------------------------------------------------------------
    let cs = match args.input_cs {
        InputCs::Auto => {
            debug!("Auto-detecting coordinate system from {} sample vertices", raw_xyz.len().min(4096));
            let sample_len = raw_xyz.len().min(4096);
            let guess = detect_input_cs(&raw_xyz[..sample_len]);

            info!("Input CS (auto‑detected): {guess}");
            guess
        }
        forced => {
            debug!("Using forced coordinate system: {forced}");
            info!("Input CS (forced): {forced}");
            forced
        }
    };

    // ---------------------------------------------------------------------
    // Convert vertices to ECEF metres and optionally track lon/lat bounds
    // ---------------------------------------------------------------------
    debug!("Converting {} vertices from {:?} to ECEF", raw_xyz.len(), cs);
    let mut points_m = Vec::with_capacity(raw_xyz.len());
    let mut lon_min = f64::INFINITY;
    let mut lon_max = f64::NEG_INFINITY;
    let mut lat_min = f64::INFINITY;
    let mut lat_max = f64::NEG_INFINITY;

    match cs {
        InputCs::Geodetic => {
            debug!("Processing {} geodetic coordinates (lon, lat, height)", raw_xyz.len());
            for &[lon, lat, h_m] in &raw_xyz {
                lon_min = lon_min.min(lon);
                lon_max = lon_max.max(lon);
                lat_min = lat_min.min(lat);
                lat_max = lat_max.max(lat);
                points_m.push(geodetic_to_ecef(lat, lon, h_m));
            }
            debug!("Geodetic bounds: lon=[{:.6}, {:.6}], lat=[{:.6}, {:.6}]", lon_min, lon_max, lat_min, lat_max);
            debug!("Height range: [{:.3}, {:.3}]m",
                   raw_xyz.iter().map(|p| p[2]).fold(f64::INFINITY, f64::min),
                   raw_xyz.iter().map(|p| p[2]).fold(f64::NEG_INFINITY, f64::max));
        }
        InputCs::Ecef => {
            debug!("Using {} ECEF coordinates directly", raw_xyz.len());
            points_m.extend(raw_xyz.iter().copied());

            // Calculate some basic statistics for debugging
            let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
            let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
            let (mut min_z, mut max_z) = (f64::INFINITY, f64::NEG_INFINITY);
            for &[x, y, z] in &raw_xyz {
                min_x = min_x.min(x); max_x = max_x.max(x);
                min_y = min_y.min(y); max_y = max_y.max(y);
                min_z = min_z.min(z); max_z = max_z.max(z);
            }
            debug!("ECEF bounds: X=[{:.1}, {:.1}], Y=[{:.1}, {:.1}], Z=[{:.1}, {:.1}]m",
                   min_x, max_x, min_y, max_y, min_z, max_z);
        }
        InputCs::LocalM => {
            debug!("Converting {} local ENU coordinates to ECEF", raw_xyz.len());

            // Local ENU coordinates – requires a geographic bounding box.
            let bbox =
                bbox.context("LocalM coordinate system needs a bbox (provide --feature-index)")?;

            debug!("Using tile bbox: lon=[{:.6}, {:.6}], lat=[{:.6}, {:.6}]",
                   bbox.lon_min, bbox.lon_max, bbox.lat_min, bbox.lat_max);

            // Geographic centre of the tile.
            let lat_c = (bbox.lat_min + bbox.lat_max) * 0.5;
            let lon_c = (bbox.lon_min + bbox.lon_max) * 0.5;

            // Calculate local coordinate bounds for debugging
            let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
            let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
            let (mut min_z, mut max_z) = (f64::INFINITY, f64::NEG_INFINITY);
            for &[x, y, z] in &raw_xyz {
                min_x = min_x.min(x); max_x = max_x.max(x);
                min_y = min_y.min(y); max_y = max_y.max(y);
                min_z = min_z.min(z); max_z = max_z.max(z);
            }

            // Robust anchor height: median Z value of the vertices.
            let mut heights: Vec<f64> = raw_xyz.iter().map(|p| p[2]).collect();
            let median_idx = heights.len() / 2;

            heights.select_nth_unstable_by(median_idx, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let anchor_h = heights[median_idx];

            // Derive a planar origin so large UTM-like values become small offsets
            let e0 = 0.5 * (min_x + max_x);
            let n0 = 0.5 * (min_y + max_y);

            debug!(
                "Local origin (E0,N0,U0)=({:.2}, {:.2}, {:.2}) m",
                e0, n0, anchor_h
            );

            // ECEF centre of the tile at the median height.
            let ecef_center = geodetic_to_ecef(lat_c, lon_c, anchor_h);
            debug!("ECEF anchor point: [{:.1}, {:.1}, {:.1}]", ecef_center[0], ecef_center[1], ecef_center[2]);

            // ENU basis vectors at the centre point.
            let (enu_e, enu_n, enu_u) = enu_basis(lat_c, lon_c);
            debug!("ENU basis vectors computed for lat={:.6}, lon={:.6}", lat_c, lon_c);

            // Transform *offset* ENU coords to ECEF
            for &[x_e, y_n, z_u] in &raw_xyz {
                let xe = x_e - e0;         // subtract planar origin
                let yn = y_n - n0;
                let zu = z_u - anchor_h;   // remove absolute height baseline

                let x = ecef_center[0] + xe * enu_e[0] + yn * enu_n[0] + zu * enu_u[0];
                let y = ecef_center[1] + xe * enu_e[1] + yn * enu_n[1] + zu * enu_u[1];
                let z = ecef_center[2] + xe * enu_e[2] + yn * enu_n[2] + zu * enu_u[2];

                points_m.push([x, y, z]);
            }

            debug!("Successfully transformed {} ENU coordinates to ECEF", raw_xyz.len());
        }
        InputCs::Auto => unreachable!(),
    }

    // ---------------------------------------------------------------------
    // Quantize coordinates with a safe units‑per‑meter value.
    // ---------------------------------------------------------------------
    debug!("Quantizing with requested units_per_meter: {}", args.units_per_meter);
    let q = quantize_with_anchor(&points_m, args.units_per_meter);

    if q.used_upm != args.units_per_meter {
        debug!("Quantization used reduced units_per_meter: {} -> {}", args.units_per_meter, q.used_upm);
    }
    debug!("Quantized {} points with anchor: [{}, {}, {}]",
           q.points_units.len(), q.anchor_units[0], q.anchor_units[1], q.anchor_units[2]);

    // ---------------------------------------------------------------------
    // Optional SMC1 semantic mask
    // ---------------------------------------------------------------------
    let smc1_opt = if args.write_smc1 {
        if let (Some(bb), Some(ov)) = (bbox, overlays) {
            debug!("Building SMC1 semantic mask {}x{} with {} roads, {} areas",
                   args.sem_grid, args.sem_grid, ov.roads.len(), ov.areas.len());
            let mask = build_smc1_mask(ov, bb, args.sem_grid);
            let (encoding, data) = if args.smc1_compress {
                let compressed = smc1_encode_rle(&mask.data);
                debug!("SMC1 RLE compression: {} -> {} bytes ({:.1}%)",
                       mask.data.len(), compressed.len(),
                       (compressed.len() as f64 / mask.data.len() as f64) * 100.0);
                (Smc1Encoding::Rle, compressed)
            } else {
                debug!("SMC1 using raw encoding: {} bytes", mask.data.len());
                (Smc1Encoding::Raw, mask.data)
            };

            Some(Smc1Chunk {
                width: args.sem_grid,
                height: args.sem_grid,
                coord_space: Smc1CoordSpace::Crs84BboxNorm,
                encoding,
                data,
                palette: (0u8..=9u8)
                    .map(|i| (i, class_precedence(i)))
                    .collect(),
            })
        } else {
            debug!("SMC1 requested but no bbox or overlays available");
            None
        }
    } else {
        debug!("SMC1 semantic mask generation disabled");
        None
    };

    // ---------------------------------------------------------------------
    // Optional GEOT (geographic extent) information
    // ---------------------------------------------------------------------
    let geot = if args.write_geot {
        if let Some(bb) = bbox {
            debug!("Using GEOT from bbox: lon=[{:.6}, {:.6}], lat=[{:.6}, {:.6}]",
                   bb.lon_min, bb.lon_max, bb.lat_min, bb.lat_max);
            Some(GeoExtentQ7::from_deg(
                bb.lon_min,
                bb.lon_max,
                bb.lat_min,
                bb.lat_max,
            ))
        } else if matches!(cs, InputCs::Geodetic)
            && lon_min.is_finite()
            && lon_max.is_finite()
            && lat_min.is_finite()
            && lat_max.is_finite()
        {
            debug!("Using GEOT from computed bounds: lon=[{:.6}, {:.6}], lat=[{:.6}, {:.6}]",
                   lon_min, lon_max, lat_min, lat_max);
            Some(GeoExtentQ7::from_deg(lon_min, lon_max, lat_min, lat_max))
        } else {
            debug!("No GEOT information available");
            None
        }
    } else {
        debug!("GEOT generation disabled");
        None
    };

    // ---------------------------------------------------------------------
    // Assemble the HYPC tile and write it to disk
    // ---------------------------------------------------------------------
    let tile = HypcTile {
        units_per_meter: q.used_upm,
        anchor_ecef_units: q.anchor_units,
        tile_key: Some(tilekey_from_prefix(prefix)),
        points_units: q.points_units,
        labels: None,
        geot,
        smc1: smc1_opt,
    };

    debug!("Writing HYPC tile to {}", out_path.display());
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

    // Parse arguments and prepare output directory.
    let args = Args::parse();
    fs::create_dir_all(&args.output_dir)?;

    // Index all OBJ/ZIP files in the input directory.
    let local_index = build_local_index(&args.input_dir);

    // Determine work items, optionally filtering with a feature index.
    let work_items: Vec<WorkItem> = match &args.feature_index {
        Some(feature_path) => {
            let mut items = load_feature_index(feature_path)?;
            items.retain(|item| {
                resolve_by_prefix(&local_index, &item.prefix, args.prefer_zip).is_some()
            });
            items
        }
        None => local_index
            .names
            .keys()
            .map(|k| WorkItem {
                prefix: k.clone(),
                bbox: None,
            })
            .collect(),
    };

    // Helper struct that couples a work item with its resolved file path.
    #[derive(Clone)]
    struct ResolvedWorkItem {
        item: WorkItem,
        path: PathBuf,
    }

    // Resolve each work item to an actual file on disk.
    let resolved_items: Vec<ResolvedWorkItem> = work_items
        .iter()
        .filter_map(|work_item| {
            resolve_by_prefix(&local_index, &work_item.prefix, args.prefer_zip).map(|path| {
                ResolvedWorkItem {
                    item: work_item.clone(),
                    path,
                }
            })
        })
        .collect();

    // Build semantic overlays once if an OSM PBF file was supplied.
    let overlays_map = if let Some(pbf_path) = &args.osm_pbf {
        let overlay_items: Vec<WorkItem> = resolved_items
            .iter()
            .map(|ri| ri.item.clone())
            .collect();

        Some(Arc::new(build_osm_overlays(
            pbf_path,
            &overlay_items,
            args.osm_margin_m,
            args.osm_log_every,
            args.osm_prefilter,
        )?))
    } else {
        None
    };

    info!("Processing {} items...", resolved_items.len());

    // Process meshes in parallel, reporting any errors.
    resolved_items.par_iter().for_each(|resolved_item| {
        let overlay = overlays_map
            .as_ref()
            .and_then(|map| map.get(&resolved_item.item.prefix));

        if let Err(err) = process_one_mesh(
            &resolved_item.path,
            &args,
            &resolved_item.item.prefix,
            resolved_item.item.bbox,
            overlay,
        ) {
            warn!(
                "Error processing {}: {:#}",
                resolved_item.path.display(),
                err
            );
        }
    });

    Ok(())
}
