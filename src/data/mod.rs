// src/data/mod.rs
//! Data handling modules for the holographic viewer.
//!
//! This module provides functionality for:
//! - Point cloud loading and quantization
//! - Data types and structures
//! - Tile alignment algorithms

pub mod point_cloud;
pub mod tile_aligner;
pub mod types;

// Re-export commonly used types
pub use types::{TileKey, GeoCrs, GeoExtentDeg, SemanticMask};
pub use point_cloud::QuantizedPointCloud;
