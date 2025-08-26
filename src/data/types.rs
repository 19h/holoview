// src/data/types.rs
//! Core data types for the holographic viewer.
//!
//! This module defines the fundamental data structures used throughout
//! the application for representing geographic data, tile metadata,
//! and semantic information.

/// Represents a tile key for point cloud data.
#[derive(Debug, Clone)]
pub enum TileKey {
    /// A tile key based on x, y coordinates and zoom level.
    XY { 
        zoom: u8, 
        x: u32, 
        y: u32, 
        scheme: u8 
    },
    /// A tile key based on a 64-bit hash of a name.
    NameHash64 { 
        hash: u64 
    },
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
///
/// Semantic masks provide class labels for regions of the point cloud,
/// typically aligned to the decode XY space of the quantized data.
#[derive(Debug, Clone)]
pub struct SemanticMask {
    /// Width of the semantic mask in pixels.
    pub width: u16,
    /// Height of the semantic mask in pixels.
    pub height: u16,
    /// Decompressed label data in row-major order.
    /// Each byte represents a class ID.
    pub data: Vec<u8>,
    /// Label palette mapping class IDs to precedence values.
    /// Format: (class_id, precedence)
    pub palette: Vec<(u8, u8)>,
    /// Coordinate space of the semantic mask.
    /// - 0 = decode XY space (most common)
    /// - Other values reserved for future use
    pub coord_space: u8,
    /// Encoding format of the semantic mask data.
    /// - 0 = raw (uncompressed)
    /// - 1 = zlib compressed
    pub encoding: u8,
}
