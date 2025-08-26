// src/data/tile_aligner.rs
//! This module contains all the pure data processing logic for calculating
//! alignment biases between adjacent point cloud tiles. This logic is
//! computationally intensive and completely decoupled from rendering.
//!
//! # Algorithm Overview
//! 1. Collect edge samples from tile boundaries (belt region)
//! 2. Find nearest neighbor correspondences between adjacent tiles
//! 3. Compute robust median offsets for X, Y, Z dimensions
//! 4. Solve global alignment using iterative least-squares

use crate::math::{grid::Grid, robust::median};
use crate::renderer::TileDraw;
use glam::Vec3;

/// Configuration parameters for tile alignment.
#[derive(Debug, Clone)]
pub struct AlignmentConfig {
    /// Width of the edge belt region in meters.
    /// Points within this distance from tile boundaries are used for alignment.
    pub belt_width: f32,
    
    /// Stride for sampling edge points.
    /// Only every nth point is kept to reduce computation.
    pub sample_stride: usize,
    
    /// Maximum search radius for nearest neighbor matching in meters.
    pub neighbor_radius: f32,
    
    /// Grid cell size for spatial indexing in meters.
    pub grid_cell_size: f32,
    
    /// Minimum number of point pairs required for reliable alignment.
    pub min_pairs: usize,
    
    /// Number of iterations for the global bias solver.
    pub solver_iterations: usize,
    
    /// Damping factor for the iterative solver (0.0 to 1.0).
    pub solver_damping: f32,
    
    /// XY tolerance for detecting adjacent tiles in meters.
    pub adjacency_tolerance: f32,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            belt_width: 6.0,
            sample_stride: 8,
            neighbor_radius: 6.0,
            grid_cell_size: 3.0,
            min_pairs: 60,
            solver_iterations: 40,
            solver_damping: 0.12,
            adjacency_tolerance: 2.5,
        }
    }
}

// -------- Edge Sample Collection --------

/// Collects sample points from the edge regions of a tile.
///
/// # Arguments
/// * `instance_positions` - All point positions in the tile
/// * `dmin_world` - Minimum bounds of the tile in world space
/// * `dmax_world` - Maximum bounds of the tile in world space
/// * `config` - Alignment configuration parameters
///
/// # Returns
/// A vector of 3D points near the tile boundaries.
///
/// # Complexity
/// O(n) where n is the number of instance positions.
pub fn collect_edge_samples_with_config(
    instance_positions: &[[f32; 3]],
    dmin_world: Vec3,
    dmax_world: Vec3,
    config: &AlignmentConfig,
) -> Vec<[f32; 3]> {
    let mut edge_samples = Vec::<[f32; 3]>::new();

    for (idx, p) in instance_positions.iter().enumerate() {
        let x = p[0];
        let y = p[1];

        let dx = (x - dmin_world.x).abs().min((dmax_world.x - x).abs());
        let dy = (y - dmin_world.y).abs().min((dmax_world.y - y).abs());

        if dx < config.belt_width || dy < config.belt_width {
            if (idx % config.sample_stride) == 0 {
                edge_samples.push(*p);
            }
        }
    }
    edge_samples
}

/// Legacy function for backward compatibility.
pub fn collect_edge_samples(
    instance_positions: &[[f32; 3]],
    dmin_world: Vec3,
    dmax_world: Vec3,
) -> Vec<[f32; 3]> {
    collect_edge_samples_with_config(
        instance_positions,
        dmin_world,
        dmax_world,
        &AlignmentConfig::default(),
    )
}

// -------- Pairwise Offset Estimation --------

/// Robustly estimates the (Δx, Δy, Δz) offset between two tiles from edge samples.
///
/// # Arguments
/// * `a` - First tile
/// * `b` - Second tile
/// * `config` - Alignment configuration parameters
///
/// # Returns
/// Optional tuple of (dx, dy, dz, weight) where weight indicates confidence.
///
/// # Algorithm
/// 1. Build spatial index for tile B's edge samples
/// 2. For each sample in A, find nearest neighbor in B
/// 3. Compute median offsets with outlier rejection
/// 4. Return refined medians with confidence weight
///
/// # Complexity
/// O(M) for building the grid + O(N * K) for neighbor searches,
/// where M = |b.edge_samples|, N = |a.edge_samples|, K = average neighbors per query.
fn estimate_pair_offset_with_config(
    a: &TileDraw,
    b: &TileDraw,
    config: &AlignmentConfig,
) -> Option<(f32, f32, f32, f32)> {
    if a.edge_samples.is_empty() || b.edge_samples.is_empty() {
        return None;
    }

    // Build spatial index for fast neighbor search
    let idx_b = Grid::build(&b.edge_samples, config.grid_cell_size);

    let mut dxs = Vec::new();
    let mut dys = Vec::new();
    let mut dzs = Vec::new();

    // Find nearest neighbors and compute offsets
    for sa in &a.edge_samples {
        let (ax, ay, az) = (sa[0], sa[1], sa[2]);
        let mut best = None::<(f32, &[f32; 3])>; // (d2, sampleB)

        for &j in idx_b.neighbors(ax, ay, config.neighbor_radius) {
            let sb = &b.edge_samples[j];
            let d2 = (sb[0] - ax).powi(2) + (sb[1] - ay).powi(2);
            let r2 = config.neighbor_radius * config.neighbor_radius;
            if d2 <= r2 && best.map(|(d, _)| d2 < d).unwrap_or(true) {
                best = Some((d2, sb));
            }
        }

        if let Some((_, sb)) = best {
            dxs.push(sb[0] - ax);
            dys.push(sb[1] - ay);
            dzs.push(sb[2] - az);
        }
    }

    if dzs.len() < config.min_pairs {
        return None;
    }

    // First pass: compute initial medians
    let mdx = median(dxs.clone())?;
    let mdy = median(dys.clone())?;
    let mdz = median(dzs.clone())?;

    // Second pass: filter outliers and recompute medians
    // Use fixed thresholds for outlier rejection (could be made configurable)
    let dx2: Vec<_> = dxs.into_iter().filter(|d| (d - mdx).abs() < 3.0).collect();
    let dy2: Vec<_> = dys.into_iter().filter(|d| (d - mdy).abs() < 3.0).collect();
    let dz2: Vec<_> = dzs.into_iter().filter(|d| (d - mdz).abs() < 5.0).collect();

    let mx = median(dx2).unwrap_or(mdx);
    let my = median(dy2).unwrap_or(mdy);
    let mz = median(dz2.clone()).unwrap_or(mdz);
    let w = dz2.len() as f32;

    Some((mx, my, mz, w.max(1.0)))
}

/// Legacy function for backward compatibility.
fn estimate_pair_offset(a: &TileDraw, b: &TileDraw) -> Option<(f32, f32, f32, f32)> {
    estimate_pair_offset_with_config(a, b, &AlignmentConfig::default())
}

// -------- Global Bias Solver --------

/// Solves for global biases using iterative weighted least squares.
///
/// Tries to satisfy constraints: bias[b] - bias[a] ≈ target_diff
///
/// # Arguments
/// * `n` - Number of tiles
/// * `edges` - List of (tile_a, tile_b, target_diff, weight) constraints
/// * `config` - Alignment configuration
///
/// # Returns
/// Vector of bias values, one per tile.
///
/// # Algorithm
/// Iterative Jacobi-like relaxation with damping.
/// Tile 0 is fixed as reference (bias = 0).
///
/// # Complexity
/// O(iterations * edges) where iterations is config.solver_iterations.
fn solve_biases_with_config(
    n: usize,
    edges: &[(usize, usize, f32, f32)],
    config: &AlignmentConfig,
) -> Vec<f32> {
    if n == 0 {
        return vec![];
    }
    
    let mut bias = vec![0.0f32; n];

    for _ in 0..config.solver_iterations {
        let mut next_bias = bias.clone();
        
        for &(a, b, target_diff, weight) in edges {
            if a >= n || b >= n {
                continue;
            }
            
            let current_diff = bias[b] - bias[a];
            let error = target_diff - current_diff;
            let adjustment = config.solver_damping * weight.sqrt() * error;
            
            // Keep tile 0 fixed as reference
            if a != 0 {
                next_bias[a] -= 0.5 * adjustment;
            }
            if b != 0 {
                next_bias[b] += 0.5 * adjustment;
            }
        }
        
        bias = next_bias;
    }
    
    bias
}

/// Legacy function for backward compatibility.
fn solve_biases(n: usize, edges: &[(usize, usize, f32, f32)]) -> Vec<f32> {
    solve_biases_with_config(n, edges, &AlignmentConfig::default())
}

/// Estimates and solves for (bias_x, bias_y, bias_z) for all tiles.
///
/// # Arguments
/// * `tiles` - Array of tiles to align
/// * `config` - Alignment configuration parameters
///
/// # Returns
/// Tuple of (x_biases, y_biases, z_biases) vectors.
///
/// # Algorithm
/// 1. Detect adjacent tile pairs based on bounding box overlap
/// 2. Estimate pairwise offsets using edge sample matching
/// 3. Solve global optimization to find consistent biases
///
/// # Complexity
/// O(n²) for pair detection + O(n * m) for offset estimation,
/// where n = number of tiles, m = average edge samples per tile.
pub fn estimate_all_biases_with_config(
    tiles: &[TileDraw],
    config: &AlignmentConfig,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = tiles.len();
    if n < 2 {
        return (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    }

    let mut ex = Vec::<(usize, usize, f32, f32)>::new();
    let mut ey = Vec::<(usize, usize, f32, f32)>::new();
    let mut ez = Vec::<(usize, usize, f32, f32)>::new();

    let eps = config.adjacency_tolerance;
    // Find all adjacent tile pairs and estimate offsets
    for a in 0..n {
        for b in (a + 1)..n {
            let ta = &tiles[a];
            let tb = &tiles[b];

            let ax0 = ta.dmin_world.x;
            let ax1 = ta.dmax_world.x;
            let ay0 = ta.dmin_world.y;
            let ay1 = ta.dmax_world.y;
            let bx0 = tb.dmin_world.x;
            let bx1 = tb.dmax_world.x;
            let by0 = tb.dmin_world.y;
            let by1 = tb.dmax_world.y;

            // Check if tiles are adjacent (with tolerance)
            let x_overlap = ax1 >= bx0 - eps && bx1 >= ax0 - eps;
            let y_overlap = ay1 >= by0 - eps && by1 >= ay0 - eps;

            if x_overlap && y_overlap {
                if let Some((dx, dy, dz, w)) = estimate_pair_offset_with_config(ta, tb, config) {
                    ex.push((a, b, dx, w));
                    ey.push((a, b, dy, w));
                    ez.push((a, b, dz, w));
                }
            }
        }
    }

    if ex.is_empty() && ey.is_empty() && ez.is_empty() {
        return (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    }

    // Solve for global biases
    (
        solve_biases_with_config(n, &ex, config),
        solve_biases_with_config(n, &ey, config),
        solve_biases_with_config(n, &ez, config),
    )
}

/// Legacy function for backward compatibility.
pub fn estimate_all_biases(tiles: &[TileDraw]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    estimate_all_biases_with_config(tiles, &AlignmentConfig::default())
}
