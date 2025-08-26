// src/data/tile_aligner.rs
//! This module contains all the pure data processing logic for calculating
//! alignment biases between adjacent point cloud tiles. This logic is
//! computationally intensive and completely decoupled from rendering.

use crate::renderer::TileDraw;
use glam::Vec3;
use std::cmp::Ordering;

// -------- Robust Statistics Helpers --------

fn percentile_in_place(v: &mut [f32], q: f32) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    let q = q.clamp(0.0, 1.0);
    let k = ((v.len() - 1) as f32 * q).round() as usize;
    let (_, nth, _) = v.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    *nth
}

fn median_in_place(v: &mut [f32]) -> f32 {
    percentile_in_place(v, 0.5)
}

fn median(mut v: Vec<f32>) -> Option<f32> {
    if v.is_empty() {
        return None;
    }
    let mid = v.len() / 2;
    v.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    Some(v[mid])
}

// -------- Edge Sample Collection --------

pub fn collect_edge_samples(
    instance_positions: &[[f32; 3]],
    dmin_world: Vec3,
    dmax_world: Vec3,
) -> Vec<[f32; 3]> {
    let belt_width = 6.0_f32; // meters from the border
    let sample_stride = 8; // decimate to keep it light
    let mut edge_samples = Vec::<[f32; 3]>::new();

    for (idx, p) in instance_positions.iter().enumerate() {
        let x = p[0];
        let y = p[1];

        let dx = (x - dmin_world.x).abs().min((dmax_world.x - x).abs());
        let dy = (y - dmin_world.y).abs().min((dmax_world.y - y).abs());

        if dx < belt_width || dy < belt_width {
            if (idx % sample_stride) == 0 {
                edge_samples.push(*p);
            }
        }
    }
    edge_samples
}

// -------- Pairwise Offset Estimation --------

// Small grid index for fast NN search
struct Grid {
    cell: f32,
    map: std::collections::HashMap<(i32, i32), Vec<usize>>,
}
impl Grid {
    fn build(samples: &[[f32; 3]], cell: f32) -> Self {
        let mut map = std::collections::HashMap::new();
        for (i, s) in samples.iter().enumerate() {
            let ix = (s[0] / cell).floor() as i32;
            let iy = (s[1] / cell).floor() as i32;
            map.entry((ix, iy)).or_insert_with(Vec::new).push(i);
        }
        Self { cell, map }
    }
    fn neighbors(&self, x: f32, y: f32, r: f32) -> impl Iterator<Item = &usize> + '_ {
        let rx = (r / self.cell).ceil() as i32;
        let ix = (x / self.cell).floor() as i32;
        let iy = (y / self.cell).floor() as i32;
        (-rx..=rx)
            .flat_map(move |dx| {
                (-rx..=rx)
                    .filter_map(move |dy| self.map.get(&(ix + dx, iy + dy)))
                    .flatten()
            })
    }
}

// Robustly estimate (Δx,Δy,Δz) between two tiles from edge samples
fn estimate_pair_offset(a: &TileDraw, b: &TileDraw) -> Option<(f32, f32, f32, f32)> {
    if a.edge_samples.is_empty() || b.edge_samples.is_empty() {
        return None;
    }

    let r = 6.0_f32;
    let cell = 3.0_f32;
    let idx_b = Grid::build(&b.edge_samples, cell);

    let mut dxs = Vec::new();
    let mut dys = Vec::new();
    let mut dzs = Vec::new();

    for sa in &a.edge_samples {
        let (ax, ay, az) = (sa[0], sa[1], sa[2]);
        let mut best = None::<(f32, &[f32; 3])>; // (d2, sampleB)

        for &j in idx_b.neighbors(ax, ay, r) {
            let sb = &b.edge_samples[j];
            let d2 = (sb[0] - ax).powi(2) + (sb[1] - ay).powi(2);
            if d2 <= r * r && best.map(|(d, _)| d2 < d).unwrap_or(true) {
                best = Some((d2, sb));
            }
        }

        if let Some((_, sb)) = best {
            dxs.push(sb[0] - ax);
            dys.push(sb[1] - ay);
            dzs.push(sb[2] - az);
        }
    }

    if dzs.len() < 60 {
        return None;
    }

    let mdx = median(dxs.clone())?;
    let mdy = median(dys.clone())?;
    let mdz = median(dzs.clone())?;

    let dx2: Vec<_> = dxs.into_iter().filter(|d| (d - mdx).abs() < 3.0).collect();
    let dy2: Vec<_> = dys.into_iter().filter(|d| (d - mdy).abs() < 3.0).collect();
    let dz2: Vec<_> = dzs.into_iter().filter(|d| (d - mdz).abs() < 5.0).collect();

    let mx = median(dx2).unwrap_or(mdx);
    let my = median(dy2).unwrap_or(mdy);
    let mz = median(dz2.clone()).unwrap_or(mdz);
    let w = dz2.len() as f32;

    Some((mx, my, mz, w.max(1.0)))
}

// -------- Global Bias Solver --------

// Generic small iterative solver: bias[b] - bias[a] ≈ d (weighted)
fn solve_biases(n: usize, edges: &[(usize, usize, f32, f32)]) -> Vec<f32> {
    if n == 0 { return vec![]; }
    let mut bias = vec![0.0f32; n];

    for _ in 0..40 {
        let mut next_bias = bias.clone();
        for &(a, b, target_diff, weight) in edges {
            if a >= n || b >= n { continue; }
            let current_diff = bias[b] - bias[a];
            let error = target_diff - current_diff;
            let adjustment = 0.12 * weight.sqrt() * error; // Damping factor, sqrt weight
            if a != 0 { next_bias[a] -= 0.5 * adjustment; }
            if b != 0 { next_bias[b] += 0.5 * adjustment; }
        }
        bias = next_bias;
    }
    bias
}

// Main entry point for this module.
// Estimates and solves for (bias_x, bias_y, bias_z) for all tiles.
pub fn estimate_all_biases(tiles: &[TileDraw]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = tiles.len();
    if n < 2 {
        return (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    }

    let mut ex = Vec::<(usize, usize, f32, f32)>::new();
    let mut ey = Vec::<(usize, usize, f32, f32)>::new();
    let mut ez = Vec::<(usize, usize, f32, f32)>::new();

    const EPS: f32 = 2.5; // Allow small XY gaps for neighbor detection
    for a in 0..n {
        for b in (a + 1)..n {
            let ta = &tiles[a];
            let tb = &tiles[b];

            let ax0 = ta.dmin_world.x; let ax1 = ta.dmax_world.x;
            let ay0 = ta.dmin_world.y; let ay1 = ta.dmax_world.y;
            let bx0 = tb.dmin_world.x; let bx1 = tb.dmax_world.x;
            let by0 = tb.dmin_world.y; let by1 = tb.dmax_world.y;

            let x_overlap = ax1 >= bx0 - EPS && bx1 >= ax0 - EPS;
            let y_overlap = ay1 >= by0 - EPS && by1 >= ay0 - EPS;

            if x_overlap && y_overlap {
                if let Some((dx, dy, dz, w)) = estimate_pair_offset(ta, tb) {
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

    (solve_biases(n, &ex), solve_biases(n, &ey), solve_biases(n, &ez))
}
