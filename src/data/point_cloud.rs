// src/data/point_cloud.rs

// NOTE: This file now delegates HYPC reading and SMC1 parsing to the `hypc` crate.
//       Two public helpers for meters/degree remain available.

use anyhow::Result;
use glam::Vec3;
use ply_rs::{parser, ply};
use std::fs::File;
use std::io::BufReader;

use hypc::{self, GeoCrs as HypcGeoCrs, HypcPointCloud as HypcPC, SemanticMask as HypcSM};

/// Returns the number of meters per degree of latitude at a given latitude.
pub fn meters_per_deg_lat(lat_deg: f64) -> f64 {
    let phi = lat_deg.to_radians();
    111_132.92 - 559.82 * (2.0 * phi).cos() + 1.175 * (4.0 * phi).cos() - 0.0023 * (6.0 * phi).cos()
}

/// Returns the number of meters per degree of longitude at a given latitude.
pub fn meters_per_deg_lon(lat_deg: f64) -> f64 {
    let phi = lat_deg.to_radians();
    111_412.84 * phi.cos() - 93.5 * (3.0 * phi).cos() + 0.118 * (5.0 * phi).cos()
}

/// Represents a tile key for point cloud data.
#[derive(Debug, Clone)]
pub enum TileKey {
    /// A tile key based on x, y coordinates and zoom level.
    XY { zoom: u8, x: u32, y: u32, scheme: u8 },
    /// A tile key based on a 64-bit hash of a name.
    NameHash64 { hash: u64 },
}

/// Represents a geographic coordinate reference system.
#[derive(Debug, Clone)]
pub enum GeoCrs {
    /// WGS 84 / CRS84 - longitude/latitude in degrees.
    Crs84,
}

/// Represents a geographic bounding box in degrees.
#[derive(Debug, Clone)]
pub struct GeoExtentDeg {
    /// Minimum longitude in degrees.
    pub lon_min: f64,
    /// Minimum latitude in degrees.
    pub lat_min: f64,
    /// Maximum longitude in degrees.
    pub lon_max: f64,
    /// Maximum latitude in degrees.
    pub lat_max: f64,
}

/// Represents a semantic mask for point cloud data.
#[derive(Debug, Clone)]
pub struct SemanticMask {
    /// Width of the semantic mask in pixels.
    pub width: u16,
    /// Height of the semantic mask in pixels.
    pub height: u16,
    /// Decompressed label data in row-major order.
    pub data: Vec<u8>,
    /// Label palette mapping class IDs to precedence values.
    pub palette: Vec<(u8 /* class */, u8 /* precedence */ )>,
    /// Coordinate space of the semantic mask.
    /// 0 = decode XY space.
    pub coord_space: u8,
    /// Encoding format of the semantic mask data.
    /// 0 = raw, 1 = zlib compressed.
    pub encoding: u8,
}

/// A point cloud with quantized positions and optional geographic and semantic metadata.
#[derive(Debug, Clone)]
pub struct QuantizedPointCloud {
    /// Point positions in centered and scaled local space.
    pub positions: Vec<glam::Vec3>,
    /// Minimum coordinates in decode space.
    pub decode_min: glam::Vec3,
    /// Maximum coordinates in decode space.
    pub decode_max: glam::Vec3,
    /// Number of points kept after quantization.
    pub kept: usize,
    /// Optional tile key associated with this point cloud.
    pub tile_key: Option<TileKey>,
    /// Optional geographic coordinate reference system.
    pub geog_crs: Option<GeoCrs>,
    /// Optional geographic bounding box in degrees.
    pub geog_bbox_deg: Option<GeoExtentDeg>,
    /// Optional semantic mask data.
    pub semantic_mask: Option<SemanticMask>,
}

impl QuantizedPointCloud {
    /// Mirrors JS: load → AABB → analytic center/scale → stride sample → quantize → decode
    pub fn load_ply_quantized(path: &str, point_budget: usize, target_extent: f32) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let vertex_parser =
            parser::Parser::<linked_hash_map::LinkedHashMap<String, ply::Property>>::new();
        let ply = vertex_parser.read_ply(&mut reader)?;
        let verts = ply
            .payload
            .get("vertex")
            .ok_or_else(|| anyhow::anyhow!("No 'vertex' element"))?;

        let mut positions: Vec<f32> = Vec::with_capacity(verts.len() * 3);
        for record in verts {
            let x = match record.get("x") {
                Some(ply::Property::Float(value)) => *value,
                Some(ply::Property::Double(value)) => *value as f32,
                _ => 0.0,
            };
            let y = match record.get("y") {
                Some(ply::Property::Float(value)) => *value,
                Some(ply::Property::Double(value)) => *value as f32,
                _ => 0.0,
            };
            let z = match record.get("z") {
                Some(ply::Property::Float(value)) => *value,
                Some(ply::Property::Double(value)) => *value as f32,
                _ => 0.0,
            };
            positions.extend([x, y, z]);
        }

        if positions.len() < 3 {
            return Ok(Self {
                positions: vec![],
                decode_min: Vec3::ZERO,
                decode_max: Vec3::ZERO,
                kept: 0,
                tile_key: None,
                geog_crs: None,
                geog_bbox_deg: None,
                semantic_mask: None,
            });
        }

        let num_points = positions.len() / 3;

        // Compute axis-aligned bounding box
        let mut min_x = positions[0];
        let mut min_y = positions[1];
        let mut min_z = positions[2];
        let mut max_x = min_x;
        let mut max_y = min_y;
        let mut max_z = min_z;

        let mut index = 3usize;
        while index < positions.len() {
            let x = positions[index];
            let y = positions[index + 1];
            let z = positions[index + 2];
            min_x = f32::min(min_x, x);
            min_y = f32::min(min_y, y);
            min_z = f32::min(min_z, z);
            max_x = f32::max(max_x, x);
            max_y = f32::max(max_y, y);
            max_z = f32::max(max_z, z);
            index += 3;
        }

        let size_x = max_x - min_x;
        let size_y = max_y - min_y;
        let size_z = max_z - min_z;

        let epsilon = 1e-20f32;
        let scale = target_extent / size_x.max(size_y).max(size_z).max(epsilon);

        let half_size_x = 0.5 * size_x * scale;
        let half_size_y = 0.5 * size_y * scale;
        let half_size_z = 0.5 * size_z * scale;

        let decode_min = Vec3::new(-half_size_x, -half_size_y, -half_size_z);
        let decode_max = Vec3::new(half_size_x, half_size_y, half_size_z);

        // Stride sampling
        let stride = ((num_points + point_budget - 1) / point_budget).max(1);
        let kept_points = num_points / stride;
        let step = stride * 3;

        // Quantization parameters
        let max_quantized_value = 65535.0f32;
        let quantization_scale_x = max_quantized_value / size_x.max(epsilon);
        let quantization_scale_y = max_quantized_value / size_y.max(epsilon);
        let quantization_scale_z = max_quantized_value / size_z.max(epsilon);

        let step_x = (size_x * scale) / max_quantized_value;
        let step_y = (size_y * scale) / max_quantized_value;
        let step_z = (size_z * scale) / max_quantized_value;

        let mut output_positions = Vec::with_capacity(kept_points);
        let mut source_index = 0usize;

        for _ in 0..kept_points {
            let quantized_x = ((positions[source_index] - min_x) * quantization_scale_x + 0.5)
                .floor() as u32;
            let quantized_y = ((positions[source_index + 1] - min_y) * quantization_scale_y + 0.5)
                .floor() as u32;
            let quantized_z = ((positions[source_index + 2] - min_z) * quantization_scale_z + 0.5)
                .floor() as u32;

            let x = -half_size_x + (quantized_x as f32) * step_x;
            let y = -half_size_y + (quantized_y as f32) * step_y;
            let z = -half_size_z + (quantized_z as f32) * step_z;

            output_positions.push(Vec3::new(x, y, z));
            source_index += step;
        }

        Ok(Self {
            positions: output_positions,
            decode_min,
            decode_max,
            kept: kept_points,
            tile_key: None,
            geog_crs: None,
            geog_bbox_deg: None,
            semantic_mask: None,
        })
    }

    /// New: Read `.hypc` via the `hypc` crate.
    pub fn load_hypc(path: &str) -> Result<Self> {
        let point_cloud: HypcPC = hypc::read_file(path)?;

        let positions = point_cloud
            .positions
            .into_iter()
            .map(|position| Vec3::new(position[0], position[1], position[2]))
            .collect::<Vec<_>>();

        let decode_min = Vec3::new(
            point_cloud.decode_min[0],
            point_cloud.decode_min[1],
            point_cloud.decode_min[2],
        );
        let decode_max = Vec3::new(
            point_cloud.decode_max[0],
            point_cloud.decode_max[1],
            point_cloud.decode_max[2],
        );

        // Map tile key + GEOT + SMC1 to local types
        let tile_key = point_cloud.tile_key.map(|tile_key| match tile_key {
            hypc::TileKey::XY {
                zoom,
                x,
                y,
                scheme,
            } => TileKey::XY { zoom, x, y, scheme },
            hypc::TileKey::NameHash64 { hash } => TileKey::NameHash64 { hash },
        });

        let geog_crs = point_cloud.geog_crs.map(|crs| match crs {
            HypcGeoCrs::Crs84 => GeoCrs::Crs84,
        });

        let geog_bbox_deg = point_cloud.geog_bbox_deg.map(|bbox| GeoExtentDeg {
            lon_min: bbox.lon_min,
            lat_min: bbox.lat_min,
            lon_max: bbox.lon_max,
            lat_max: bbox.lat_max,
        });

        let semantic_mask = point_cloud.semantic_mask.map(|mask: HypcSM| SemanticMask {
            width: mask.width,
            height: mask.height,
            data: mask.data,
            palette: mask.palette,
            coord_space: mask.coord_space,
            encoding: u8::from(mask.encoding),
        });

        let kept = positions.len();

        Ok(Self {
            positions,
            decode_min,
            decode_max,
            kept,
            tile_key,
            geog_crs,
            geog_bbox_deg,
            semantic_mask,
        })
    }

    /// O(1) class lookup from a position in decode space (x,y).
    pub fn class_of_xy(&self, x: f32, y: f32) -> u8 {
        let Some(ref semantic_mask) = self.semantic_mask else {
            return 0;
        };

        if semantic_mask.coord_space != 0 {
            return 0;
        }

        let width = semantic_mask.width as f32;
        let height = semantic_mask.height as f32;

        let x_range = self.decode_max.x - self.decode_min.x;
        let y_range = self.decode_max.y - self.decode_min.y;

        let index_x = (((x - self.decode_min.x) * width) / x_range.max(f32::EPSILON)).floor()
            as i32;
        let index_y = (((y - self.decode_min.y) * height) / y_range.max(f32::EPSILON)).floor()
            as i32;

        if index_x < 0
            || index_y < 0
            || index_x >= semantic_mask.width as i32
            || index_y >= semantic_mask.height as i32
        {
            return 0;
        }

        semantic_mask.data[(index_y as usize) * semantic_mask.width as usize + index_x as usize]
    }

    /// Convert lon/lat (deg) to decode XY using GEOT v1 bbox.
    pub fn lonlat_to_decode_xy(&self, longitude: f64, latitude: f64) -> Option<(f32, f32)> {
        let bbox = self.geog_bbox_deg.as_ref()?;
        let x_size = (self.decode_max.x - self.decode_min.x) as f64;
        let y_size = (self.decode_max.y - self.decode_min.y) as f64;

        let x = self.decode_min.x as f64
            + (longitude - bbox.lon_min) / (bbox.lon_max - bbox.lon_min + f64::EPSILON) * x_size;
        let y = self.decode_min.y as f64
            + (latitude - bbox.lat_min) / (bbox.lat_max - bbox.lat_min + f64::EPSILON) * y_size;

        Some((x as f32, y as f32))
    }

    /// Inverse of `lonlat_to_decode_xy`.
    pub fn decode_xy_to_lonlat(&self, x: f32, y: f32) -> Option<(f64, f64)> {
        let bbox = self.geog_bbox_deg.as_ref()?;
        let x_size = (self.decode_max.x - self.decode_min.x) as f64;
        let y_size = (self.decode_max.y - self.decode_min.y) as f64;

        let longitude = bbox.lon_min
            + ((x as f64 - self.decode_min.x as f64) / x_size) * (bbox.lon_max - bbox.lon_min);
        let latitude = bbox.lat_min
            + ((y as f64 - self.decode_min.y as f64) / y_size) * (bbox.lat_max - bbox.lat_min);

        Some((longitude, latitude))
    }

    /// Convenience flags
    pub fn has_semantics(&self) -> bool {
        self.semantic_mask.is_some()
    }

    pub fn has_geot(&self) -> bool {
        self.geog_bbox_deg.is_some()
    }
}
