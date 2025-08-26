// src/math/mod.rs
//! Mathematical utilities for the holographic viewer.
//!
//! This module provides reusable mathematical components including:
//! - Robust statistics (median, percentiles)
//! - Spatial indexing (uniform grid)

pub mod grid;
pub mod robust;

pub use self::grid::Grid;
pub use self::robust::{median, median_in_place, percentile_in_place};
