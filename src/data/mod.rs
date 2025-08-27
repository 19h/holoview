// src/data/mod.rs
//! Data handling modules for the holographic viewer.
//!
//! This module provides functionality for:
//! - Loading HYPC point clouds and preparing them for the GPU.
//! - Defining the data structures for GPU buffers.

pub mod point_cloud;
pub mod types;

// Re-export commonly used types for convenience.
pub use self::types::{PointInstance, TileGpu, TileKey32, TileUniformStd140};
