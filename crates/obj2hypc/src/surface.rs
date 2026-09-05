//! Sample triangle interiors on a global metric lattice, independent of mesh
//! tessellation and tile origins. Boundary vertices are not a sampling measure.
use anyhow::{ensure, Context, Result};
use std::io::{BufRead, BufReader, Read};

pub struct Mesh {
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<[usize; 3]>,
}

pub fn read_obj<R: Read>(reader: R, faces: bool) -> Result<Mesh> {
    let mut mesh = Mesh {
        vertices: Vec::new(),
        triangles: Vec::new(),
    };
    for (line_no, line) in BufReader::new(reader).lines().enumerate() {
        let line = line?;
        let mut fields = line.split_whitespace();
        match fields.next() {
            Some("v") => {
                let mut p = [0.0; 3];
                for value in &mut p {
                    *value = fields
                        .next()
                        .context("Incomplete OBJ vertex")?
                        .parse::<f64>()?;
                }
                ensure!(
                    p.iter().all(|v| v.is_finite()),
                    "Non-finite vertex on line {}",
                    line_no + 1
                );
                mesh.vertices.push(p);
            }
            Some("f") if faces => {
                let mut tri = [0; 3];
                for vertex in &mut tri {
                    let raw = fields
                        .next()
                        .context("Incomplete OBJ triangle")?
                        .split('/')
                        .next()
                        .unwrap()
                        .parse::<i64>()?;
                    ensure!(raw != 0, "OBJ indices are one-based");
                    let index = if raw < 0 {
                        mesh.vertices.len() as i64 + raw
                    } else {
                        raw - 1
                    };
                    ensure!(
                        index >= 0,
                        "Invalid relative OBJ index on line {}",
                        line_no + 1
                    );
                    *vertex = usize::try_from(index)?;
                }
                ensure!(
                    fields.next().is_none_or(|s| s.starts_with('#')),
                    "Surface sampling requires triangulated OBJ faces (line {})",
                    line_no + 1
                );
                mesh.triangles.push(tri);
            }
            _ => {}
        }
    }
    ensure!(
        mesh.triangles
            .iter()
            .flatten()
            .all(|&i| i < mesh.vertices.len()),
        "OBJ face index out of bounds"
    );
    Ok(mesh)
}

/// For each triangle, choose its dominant normal axis, project to the other
/// two ECEF axes, rasterize the *global* cell centers, and interpolate the third
/// coordinate. O(F + C) time, O(S) output space; C is the number of projected
/// bounding-box cells tested, S the accepted samples. No per-tile grid phase.
pub fn sample(
    vertices: &[[f64; 3]],
    triangles: &[[usize; 3]],
    spacing_m: f64,
) -> Result<Vec<[f64; 3]>> {
    ensure!(
        spacing_m.is_finite() && spacing_m >= 0.01,
        "Surface spacing must be at least 0.01 m"
    );
    ensure!(
        !triangles.is_empty(),
        "No triangle faces; use --sampling vertices for point-only OBJ input"
    );
    ensure!(
        vertices
            .iter()
            .flatten()
            .all(|v| v.is_finite() && v.abs() <= 1e9),
        "Surface sampling requires finite terrestrial ECEF coordinates"
    );
    ensure!(
        triangles.iter().flatten().all(|&i| i < vertices.len()),
        "Triangle index out of bounds"
    );
    let mut points = Vec::new();
    for &[ia, ib, ic] in triangles {
        let a = vertices[ia];
        let b = vertices[ib];
        let c = vertices[ic];
        let ab: [f64; 3] = std::array::from_fn(|i| b[i] - a[i]);
        let ac: [f64; 3] = std::array::from_fn(|i| c[i] - a[i]);
        let n = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        let max_n = n.iter().map(|v| v.abs()).fold(0.0, f64::max);
        if max_n < 1e-14 {
            continue;
        }
        // Stable tie handling for equally inclined planes.
        let w = n
            .iter()
            .position(|v| v.abs() >= max_n * (1.0 - 1e-10))
            .unwrap();
        let u = (w + 1) % 3;
        let v = (w + 2) % 3;
        let determinant = ab[u] * ac[v] - ab[v] * ac[u];
        let low = |axis: usize| {
            ((a[axis].min(b[axis]).min(c[axis]) / spacing_m) - 0.5 - 1e-8).ceil() as i64
        };
        let high = |axis: usize| {
            ((a[axis].max(b[axis]).max(c[axis]) / spacing_m) - 0.5 + 1e-8).floor() as i64
        };
        let (lo_u, hi_u, lo_v, hi_v) = (low(u), high(u), low(v), high(v));
        let cells = (hi_u - lo_u + 1).max(0) as u64 * (hi_v - lo_v + 1).max(0) as u64;
        ensure!(
            cells <= 100_000_000,
            "Triangle exceeds sampling work limit; check source CRS/spacing"
        );
        for iu in lo_u..=hi_u {
            let pu = (iu as f64 + 0.5) * spacing_m;
            for iv in lo_v..=hi_v {
                let pv = (iv as f64 + 0.5) * spacing_m;
                let du = pu - a[u];
                let dv = pv - a[v];
                let s = (du * ac[v] - dv * ac[u]) / determinant;
                let t = (ab[u] * dv - ab[v] * du) / determinant;
                if s >= -1e-9 && t >= -1e-9 && s + t <= 1.0 + 1e-9 {
                    let mut p = [0.0; 3];
                    p[u] = pu;
                    p[v] = pv;
                    p[w] = a[w] + s * ab[w] + t * ac[w];
                    points.push(p);
                }
            }
        }
    }
    ensure!(
        !points.is_empty(),
        "No surface samples at the requested spacing"
    );
    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn lattice(mut p: Vec<[f64; 3]>) -> Vec<[i64; 3]> {
        let mut q: Vec<_> = p
            .drain(..)
            .map(|p| p.map(|v| (v * 1000.0).round() as i64))
            .collect();
        q.sort_unstable();
        q.dedup();
        q
    }
    #[test]
    fn sampling_is_invariant_to_diagonal_and_tile_cut() {
        for axis in 0..3 {
            // Plane far from zero, split at an arbitrary non-lattice tile edge.
            let point = |x: f64, y: f64| {
                let mut p = [3_780_000.0, 900_000.0, 5_030_000.0];
                p[(axis + 1) % 3] += x;
                p[(axis + 2) % 3] += y;
                p
            };
            let vertices = [
                point(0.0, 0.0),
                point(10.0, 0.0),
                point(10.0, 10.0),
                point(0.0, 10.0),
                point(4.37, 0.0),
                point(4.37, 10.0),
            ];
            let one = lattice(sample(&vertices, &[[0, 1, 2], [0, 2, 3]], 0.5).unwrap());
            let other = lattice(sample(&vertices, &[[0, 1, 3], [1, 2, 3]], 0.5).unwrap());
            let tiles = lattice(
                sample(
                    &vertices,
                    &[[0, 4, 5], [0, 5, 3], [4, 1, 2], [4, 2, 5]],
                    0.5,
                )
                .unwrap(),
            );
            assert_eq!(one.len(), 400);
            assert_eq!(one, other);
            assert_eq!(one, tiles);
        }
    }
    #[test]
    fn inclined_plane_and_reversed_winding_keep_the_same_samples() {
        let p = |x: f64, y: f64| {
            [
                3_780_000.0 + x,
                900_000.0 + y,
                5_030_000.0 + 0.3 * x - 0.2 * y,
            ]
        };
        let vertices = [
            p(0.0, 0.0),
            p(10.0, 0.0),
            p(10.0, 10.0),
            p(0.0, 10.0),
            p(4.37, 0.0),
            p(4.37, 10.0),
        ];
        let one = lattice(sample(&vertices, &[[0, 1, 2], [0, 2, 3]], 0.5).unwrap());
        let cut = lattice(
            sample(
                &vertices,
                &[[5, 4, 0], [3, 5, 0], [2, 1, 4], [5, 2, 4]],
                0.5,
            )
            .unwrap(),
        );
        assert_eq!(one, cut);
        assert_eq!(one.len(), 400);
    }

    #[test]
    fn obj_indices_and_failures() {
        let m = read_obj(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3/1 -2/2 -1/3\n".as_bytes(),
            true,
        )
        .unwrap();
        assert_eq!(m.triangles, vec![[0, 1, 2]]);
        assert!(read_obj("v 0 0 0\nf 1 2 3\n".as_bytes(), true).is_err());
        assert!(sample(&m.vertices, &m.triangles, 0.0).is_err());
        assert!(sample(&m.vertices, &[], 0.5).is_err());
    }
}
