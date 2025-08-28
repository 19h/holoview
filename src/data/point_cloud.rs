use crate::camera::Camera;
use crate::data::types::{PointInstance, TileGpu};
use anyhow::Result;
use hypc::{
    ecef_to_geodetic, read_file, smc1_decode_rle, wgs84, HypcTile, Smc1CoordSpace, Smc1Encoding,
};
use rayon::prelude::*;
use std::path::Path;

// ASSUMPTION: The `PointInstance` struct is defined in `crate::data::types`.
// For this file to be self-contained and compilable, we define the module and the
// struct here.
pub mod data {
    pub mod types {
        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        pub struct PointInstance {
            pub ofs_m: [f32; 3],
            pub label: u32,
        }

        // Unchanged struct, provided for context.
        pub struct TileGpu {
            pub key: u64,
            pub units_per_meter: f32,
            pub anchor_units: [i64; 3],
            pub instances_len: u32,
            pub vtx: wgpu::Buffer,
            pub ubo: wgpu::Buffer,
            pub bind: wgpu::BindGroup,
        }
    }
}

// wgpu::util::DeviceExt is a trait, so we need to bring it into scope.
mod wgpu_util {
    pub use wgpu::util::DeviceExt;
}
use wgpu_util::*;

#[inline(always)]
fn build_ecef_to_enu(lat_rad: f64, lon_rad: f64) -> [[f64; 3]; 3] {
    let (sφ, cφ) = lat_rad.sin_cos();
    let (sλ, cλ) = lon_rad.sin_cos();
    // Rows are ê^T, n̂^T, û^T in ECEF components.
    [
        [-sλ, cλ, 0.0],
        [-sφ * cλ, -sφ * sλ, cφ],
        [cφ * cλ, cφ * sλ, sφ],
    ]
}

#[inline(always)]
fn mul_mat3_vec3(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Read one HYPC tile from disk and upload to GPU (instances + per-tile UBO).
pub fn load_hypc_tile(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    camera: &Camera,
    path: &Path,
    viewport_size: [f32; 2], // Initial viewport size
) -> Result<TileGpu> {
    let tile: HypcTile = read_file(path)?;
    let upm_f32 = tile.units_per_meter as f32;
    let inv_upm_f32 = upm_f32.recip();
    let inv_upm_f64 = (tile.units_per_meter as f64).recip();

    // SMC1 decode (only if needed)
    let (smc_w, smc_h, smc_raw): (u32, u32, Option<Vec<u8>>) =
        if let Some(smc1) = tile.smc1.as_ref() {
            if smc1.coord_space == Smc1CoordSpace::Crs84BboxNorm && tile.geot.is_some() {
                let data = match smc1.encoding {
                    Smc1Encoding::Raw => smc1.data.clone(),
                    Smc1Encoding::Rle => smc1_decode_rle(&smc1.data)?,
                };
                (smc1.width as u32, smc1.height as u32, Some(data))
            } else {
                (0, 0, None)
            }
        } else {
            (0, 0, None)
        };

    // GEOT in degrees
    let geot_deg = tile.geot.map(|g| g.to_deg());

    // Precompute anchor in meters (f64) once
    let upm64 = tile.units_per_meter as f64;
    let anchor_m = [
        tile.anchor_ecef_units[0] as f64 / upm64,
        tile.anchor_ecef_units[1] as f64 / upm64,
        tile.anchor_ecef_units[2] as f64 / upm64,
    ];

    // Fast path: either direct labels, or SMC path with linearized geodesy
    let has_direct_labels = tile
        .labels
        .as_ref()
        .map_or(false, |v| v.len() == tile.points_units.len());

    // Prepare instance buffer in parallel
    let instances: Vec<PointInstance> =
        if has_direct_labels || smc_raw.is_none() || geot_deg.is_none() {
            // No geodesy work; just scale offsets and copy labels if present.
            let labels = tile.labels.as_deref();
            tile.points_units
                .par_iter()
                .enumerate()
                .map(|(i, p)| {
                    let ofs_m = [
                        (p[0] as f32) * inv_upm_f32,
                        (p[1] as f32) * inv_upm_f32,
                        (p[2] as f32) * inv_upm_f32,
                    ];
                    let label = labels.map(|ls| ls[i]).unwrap_or(0) as u32;
                    PointInstance { ofs_m, label }
                })
                .collect()
        } else {
            let (lon_min, lon_max, lat_min, lat_max) = geot_deg.unwrap();
            let inv_dlon = 1.0 / (lon_max - lon_min + 1e-12);
            let inv_dlat = 1.0 / (lat_max - lat_min + 1e-12);

            let smc = smc_raw.as_ref().unwrap();
            let sw = smc_w as usize;
            let sh = smc_h as usize;

            tile.points_units
                .par_iter()
                .map(|p| {
                    // 1. Reconstruct the point's full ECEF coordinate in meters (f64 for precision).
                    let point_ecef_m = [
                        anchor_m[0] + (p[0] as f64 * inv_upm_f64),
                        anchor_m[1] + (p[1] as f64 * inv_upm_f64),
                        anchor_m[2] + (p[2] as f64 * inv_upm_f64),
                    ];

                    // 2. Convert the ECEF coordinate to a precise geodetic coordinate.
                    let (lat_deg, lon_deg, _h) =
                        ecef_to_geodetic(point_ecef_m[0], point_ecef_m[1], point_ecef_m[2]);

                    // 3. Normalize the geodetic coordinate into a [0,1] UV coordinate using the tile's GEOT bbox.
                    let u = ((lon_deg - lon_min) * inv_dlon).clamp(0.0, 1.0);
                    let v = ((lat_deg - lat_min) * inv_dlat).clamp(0.0, 1.0);

                    // 4. Sample the semantic mask texture.
                    let ix = (u * (smc_w.saturating_sub(1)) as f64).round() as usize;
                    let iy = (v * (smc_h.saturating_sub(1)) as f64).round() as usize;
                    let label = smc[iy * sw + ix] as u32;

                    // 5. Create the PointInstance. The offset is still the original ECEF offset for rendering.
                    PointInstance {
                        ofs_m: [
                            (p[0] as f32) * inv_upm_f32,
                            (p[1] as f32) * inv_upm_f32,
                            (p[2] as f32) * inv_upm_f32,
                        ],
                        label,
                    }
                })
                .collect()
        };

    // Tile-level analysis and logging is confined to debug builds.
    #[cfg(debug_assertions)]
    {
        // --- Start: Tile-level orientation calculation via PCA ---

        // 1. Get geodetic coordinates of the anchor to define the local tangent plane.
        let (anchor_lat_deg, anchor_lon_deg, _) =
            ecef_to_geodetic(anchor_m[0], anchor_m[1], anchor_m[2]);
        let anchor_lat_rad = anchor_lat_deg.to_radians();
        let anchor_lon_rad = anchor_lon_deg.to_radians();

        // 2. Build the transformation matrix from ECEF to the local ENU frame.
        let ecef_to_enu_mat = build_ecef_to_enu(anchor_lat_rad, anchor_lon_rad);

        // 3. Calculate PCA-based orientation. This requires iterating through points to build covariance matrix.
        let num_points = instances.len() as f64;
        let mut mean_e = 0.0;
        let mut mean_n = 0.0;
        let mut enu_coords = Vec::with_capacity(instances.len());

        for inst in &instances {
            let ofs_m_f64 = [inst.ofs_m[0] as f64, inst.ofs_m[1] as f64, inst.ofs_m[2] as f64];
            let ofs_enu = mul_mat3_vec3(&ecef_to_enu_mat, ofs_m_f64);
            let (e, n) = (ofs_enu[0], ofs_enu[1]);
            enu_coords.push((e, n));
            mean_e += e;
            mean_n += n;
        }
        mean_e /= num_points;
        mean_n /= num_points;

        let mut cov_ee = 0.0;
        let mut cov_nn = 0.0;
        let mut cov_en = 0.0;

        for (e, n) in enu_coords {
            let de = e - mean_e;
            let dn = n - mean_n;
            cov_ee += de * de;
            cov_nn += dn * dn;
            cov_en += de * dn;
        }
        cov_ee /= num_points;
        cov_nn /= num_points;
        cov_en /= num_points;

        // 4. Calculate angle of the primary eigenvector.
        // angle = 0.5 * atan2(2 * cov_xy, cov_xx - cov_yy)
        let pca_angle_rad = 0.5 * cov_en.mul_add(2.0, 0.0).atan2(cov_ee - cov_nn);
        let pca_angle_deg = pca_angle_rad.to_degrees();

        // --- End: Tile-level orientation calculation via PCA ---

        // AABB calculation for logging purposes.
        use std::f32::{INFINITY, NEG_INFINITY};
        let (min, max) = instances
            .par_iter()
            .map(|pi| (pi.ofs_m, pi.ofs_m))
            .reduce(
                || ([INFINITY; 3], [NEG_INFINITY; 3]),
                |(a_min, a_max), (b_min, b_max)| {
                    (
                        [
                            a_min[0].min(b_min[0]),
                            a_min[1].min(b_min[1]),
                            a_min[2].min(b_min[2]),
                        ],
                        [
                            a_max[0].max(b_max[0]),
                            a_max[1].max(b_max[1]),
                            a_max[2].max(b_max[2]),
                        ],
                    )
                },
            );

        log::debug!(
            "HYPC {:?}: pts={}, upm={}, anchor_ecef_m=({:.3},{:.3},{:.3}), ofs_AABB_m=min({:.2},{:.2},{:.2}) max({:.2},{:.2},{:.2}), pca_orientation_deg_from_N={:.1}",
            path.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
            tile.points_units.len(),
            tile.units_per_meter,
            anchor_m[0], anchor_m[1], anchor_m[2],
            min[0], min[1], min[2],
            max[0], max[1], max[2],
            pca_angle_deg
        );
    }

    // GPU upload
    let vtx = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("HYPC Instances"),
        contents: bytemuck::cast_slice(&instances),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });

    let tile_ubo_data = camera.make_tile_uniform(
        tile.anchor_ecef_units,
        tile.units_per_meter,
        viewport_size,
        1.0, // Default point size
    );

    let ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("HYPC Tile UBO"),
        contents: bytemuck::bytes_of(&tile_ubo_data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("HYPC Tile BindGroup"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: ubo.as_entire_binding(),
        }],
    });

    Ok(TileGpu {
        key: tile.tile_key,
        units_per_meter: tile.units_per_meter,
        anchor_units: tile.anchor_ecef_units,
        instances_len: instances.len() as u32,
        vtx,
        ubo,
        bind,
    })
}
