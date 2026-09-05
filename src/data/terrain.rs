//! Compact navigation surface derived from the lowest non-building LOD samples.
//! This is an approximate camera support surface, not a surveyed terrain model.
use super::{dataset::Dataset, point_cloud::PreparedTile};
use anyhow::{ensure, Result};
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TerrainGrid {
    pub origin: [f64; 3],
    pub east: [f64; 3],
    pub north: [f64; 3],
    pub up: [f64; 3],
    pub west_m: f64,
    pub south_m: f64,
    pub cell_m: f64,
    pub width: usize,
    pub height: usize,
    /// Local-up metres × 10. i16::MIN denotes no navigation sample.
    pub heights_q10: Vec<i16>,
    #[serde(default)]
    pub coverage: Vec<u8>,
}
impl TerrainGrid {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.width > 0 && self.height > 0 && self.width <= 4096 && self.height <= 4096
            && self.width.checked_mul(self.height) == Some(self.heights_q10.len()), "Invalid terrain dimensions");
        ensure!(self.cell_m.is_finite() && self.cell_m > 0.0 && self.west_m.is_finite() && self.south_m.is_finite(), "Invalid terrain scale");
        for axis in [self.east, self.north, self.up] { ensure!(axis.iter().all(|v| v.is_finite()) && (DVec3::from(axis).length() - 1.0).abs() < 1e-6, "Invalid terrain basis"); }
        ensure!(self.origin.iter().all(|v| v.is_finite()), "Invalid terrain origin");
        ensure!(self.coverage.is_empty() || self.coverage.len() == self.heights_q10.len(), "Invalid terrain coverage");
        let (e,n,u) = (DVec3::from(self.east),DVec3::from(self.north),DVec3::from(self.up));
        ensure!(e.dot(n).abs() < 1e-6 && e.cross(n).dot(u) > 0.999999, "Terrain basis is not orthonormal");
        Ok(())
    }
    pub fn ground(&self, point: DVec3) -> Option<DVec3> {
        let origin = DVec3::from(self.origin);
        let delta = point - origin;
        let e = delta.dot(self.east.into()); let n = delta.dot(self.north.into());
        let x = (e - self.west_m) / self.cell_m - 0.5;
        let y = (n - self.south_m) / self.cell_m - 0.5;
        let ix = x.floor() as i32; let iy = y.floor() as i32;
        if ix >= 0 && iy >= 0 && ix + 1 < self.width as i32 && iy + 1 < self.height as i32 {
            let a = iy as usize * self.width + ix as usize;
            let heights = [a, a + 1, a + self.width, a + self.width + 1].map(|i| self.heights_q10[i]);
            if heights.iter().all(|&h| h != i16::MIN) {
                let [a,b,c,d] = heights.map(|h| h as f64 * 0.1);
                let u = x - ix as f64; let v = y - iy as f64;
                let h = if u + v <= 1.0 { a * (1.0-u-v) + b*u + c*v }
                    else { d*(u+v-1.0) + b*(1.0-v) + c*(1.0-u) };
                return Some(origin + DVec3::from(self.east)*e + DVec3::from(self.north)*n + DVec3::from(self.up)*h);
            }
        }
        let mut value = 0.0; let mut weight = 0.0;
        // Continuous inverse-distance blending avoids hard cell-height jumps.
        // Expand only where a building/water-data gap lacks nearby ground points.
        for radius in [1, 2, 4, 8] {
            value = 0.0; weight = 0.0;
            for yy in iy - radius..=iy + radius + 1 {
                for xx in ix - radius..=ix + radius + 1 {
                    if xx < 0 || yy < 0 || xx >= self.width as i32 || yy >= self.height as i32 { continue; }
                    let h = self.heights_q10[yy as usize * self.width + xx as usize];
                    if h == i16::MIN { continue; }
                    let w = 1.0 / ((x - xx as f64).powi(2) + (y - yy as f64).powi(2) + 0.05).powi(2);
                    value += h as f64 * 0.1 * w; weight += w;
                }
            }
            if weight > 0.0 { break; }
        }
        (weight > 0.0).then(|| origin + DVec3::from(self.east) * e + DVec3::from(self.north) * n + DVec3::from(self.up) * (value / weight))
    }
}

fn insert(grid: &TerrainGrid, tile: &PreparedTile, preferred: &mut [f64], fallback: &mut [f64], coverage: &mut [u8]) {
    let anchor = DVec3::from(tile.anchor_units.map(|v| v as f64 / tile.units_per_meter as f64)) - DVec3::from(grid.origin);
    let east = DVec3::from(grid.east); let north = DVec3::from(grid.north); let up = DVec3::from(grid.up);
    let mut lo = [i32::MAX; 2]; let mut hi = [i32::MIN; 2];
    for p in &tile.instances {
        let d = anchor + DVec3::from(p.ofs_m.map(|v| v as f64));
        let x = ((d.dot(east) - grid.west_m) / grid.cell_m).floor() as i32;
        let y = ((d.dot(north) - grid.south_m) / grid.cell_m).floor() as i32;
        if x < 0 || y < 0 || x >= grid.width as i32 || y >= grid.height as i32 { continue; }
        lo[0] = lo[0].min(x); lo[1] = lo[1].min(y); hi[0] = hi[0].max(x); hi[1] = hi[1].max(y);
        if p.label == 1 { continue; } // Buildings do not define ground height.
        let i = y as usize * grid.width + x as usize;
        let h = d.dot(up);
        if matches!(p.label, 2..=5 | 8 | 9) { preferred[i] = preferred[i].min(h); }
        fallback[i] = fallback[i].min(h);
    }
    if lo[0] <= hi[0] {
        for y in lo[1]..=hi[1] { for x in lo[0]..=hi[0] { coverage[y as usize * grid.width + x as usize] = 1; } }
    }
}

pub fn build_terrain(dataset: &Dataset, cache: &Path) -> Result<TerrainGrid> {
    let center = dataset.nodes[dataset.root as usize].center();
    let (lat, lon, _) = hypc::ecef_to_geodetic(center.x, center.y, center.z);
    let (slat, clat) = lat.to_radians().sin_cos(); let (slon, clon) = lon.to_radians().sin_cos();
    let origin = DVec3::from(hypc::geodetic_to_ecef(lat, lon, 0.0));
    let east = DVec3::new(-slon, clon, 0.0);
    let north = DVec3::new(-slat * clon, -slat * slon, clat);
    let up = east.cross(north);
    let mut west = f64::INFINITY; let mut south = f64::INFINITY;
    let mut east_max = f64::NEG_INFINITY; let mut north_max = f64::NEG_INFINITY;
    let mut leaf_coarse = Vec::new();
    for (id, node) in dataset.nodes.iter().enumerate() {
        // Last per-source LOD chain node, before the spatial binary hierarchy.
        if node.children.len() == 1 && node.spacing_m >= 16.0 {
            leaf_coarse.push(id as u32);
            let d = node.center() - origin; let r = node.radius();
            west = west.min(d.dot(east) - r); east_max = east_max.max(d.dot(east) + r);
            south = south.min(d.dot(north) - r); north_max = north_max.max(d.dot(north) + r);
        }
    }
    ensure!(!leaf_coarse.is_empty(), "No coarse terrain sources");
    let cell_m = 64.0;
    west = (west / cell_m).floor() * cell_m; south = (south / cell_m).floor() * cell_m;
    let width = ((east_max - west) / cell_m).ceil() as usize;
    let height = ((north_max - south) / cell_m).ceil() as usize;
    ensure!(width <= 4096 && height <= 4096, "Navigation terrain extent exceeds regional grid limit");
    let mut grid = TerrainGrid { origin: origin.into(), east: east.into(), north: north.into(), up: up.into(), west_m: west, south_m: south, cell_m, width, height, heights_q10: vec![i16::MIN; width * height], coverage: vec![] };
    let mut preferred = vec![f64::INFINITY; width * height]; let mut fallback = preferred.clone();
    let mut coverage = vec![0; width * height];
    for id in leaf_coarse { insert(&grid, &dataset.read_node(cache, id)?, &mut preferred, &mut fallback, &mut coverage); }
    grid.coverage = coverage;
    for i in 0..grid.heights_q10.len() {
        let h = if preferred[i].is_finite() { preferred[i] } else { fallback[i] };
        if h.is_finite() && (-3276.7..=3276.7).contains(&h) { grid.heights_q10[i] = (h * 10.0).round() as i16; }
    }
    let original = grid.clone();
    for y in 0..height { for x in 0..width {
        let i = y * width + x;
        if grid.coverage[i] == 1 && grid.heights_q10[i] == i16::MIN {
            let p = origin + east * (west + (x as f64 + 0.5) * cell_m) + north * (south + (y as f64 + 0.5) * cell_m);
            if let Some(ground) = original.ground(p) { grid.heights_q10[i] = ((ground - origin).dot(up) * 10.0).round() as i16; }
        }
    } }
    grid.validate()?;
    Ok(grid)
}

pub fn upgrade_terrain(cache: &Path) -> Result<()> {
    let lock = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(cache.join("build.lock"))?;
    lock.try_lock()?;
    let mut dataset = Dataset::open(cache)?;
    dataset.terrain = Some(build_terrain(&dataset, cache)?);
    let temporary = cache.join("catalog.terrain.part");
    serde_json::to_writer(std::io::BufWriter::new(std::fs::File::create(&temporary)?), &dataset)?;
    std::fs::rename(temporary, cache.join("catalog.json"))?;
    Dataset::open(cache)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn grid() -> TerrainGrid {
        TerrainGrid { origin: [6378137.0,0.0,0.0], east: [0.0,1.0,0.0], north: [0.0,0.0,1.0], up: [1.0,0.0,0.0], west_m: -64.0, south_m: -64.0, cell_m: 64.0,
            width: 2, height: 2, heights_q10: vec![100,200,300,400], coverage: vec![1;4] }
    }
    #[test]
    fn navigation_ground_matches_the_rendered_triangle_surface() {
        let g = grid(); g.validate().unwrap(); let origin = DVec3::from(g.origin);
        let a = g.ground(origin + DVec3::new(0.0,-16.0,-16.0)).unwrap();
        // u=v=0.25 in the first triangle: 0.5*10 + 0.25*20 + 0.25*30.
        assert!((a.x - origin.x - 17.5).abs() < 1e-8);
        let left = g.ground(origin + DVec3::new(0.0,-1e-4,0.0)).unwrap();
        let right = g.ground(origin + DVec3::new(0.0,1e-4,0.0)).unwrap();
        assert!(left.distance(right) < 0.001);
        assert!(g.ground(origin + DVec3::new(0.0,100000.0,100000.0)).is_none());
    }
    #[test]
    fn missing_ground_and_invalid_basis_are_explicit() {
        let mut g = grid(); g.heights_q10.fill(i16::MIN);
        assert!(g.ground(g.origin.into()).is_none());
        g.north = g.east; assert!(g.validate().is_err());
    }
}
