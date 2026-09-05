use crate::camera::Camera;
use crate::data::types::{PointInstance, TileGpu};
use anyhow::Result;
use hypc::{ecef_to_geodetic, read_file, smc1_decode_rle, HypcTile, Smc1CoordSpace, Smc1Encoding};
use rayon::prelude::*;
use std::path::Path;

// wgpu::util::DeviceExt is a trait, so we need to bring it into scope.
mod wgpu_util {
    pub use wgpu::util::DeviceExt;
}
use wgpu_util::*;

/// CPU decoding is independent of the render thread and GPU.
#[derive(Debug)]
pub struct PreparedTile {
    pub key: Option<[u8; 32]>,
    pub units_per_meter: u32,
    pub anchor_units: [i64; 3],
    pub instances: Vec<PointInstance>,
}

pub fn prepare_hypc_tile(path: &Path) -> Result<PreparedTile> {
    let tile: HypcTile = read_file(path)?;
    let inv_upm_f64 = (tile.units_per_meter as f64).recip();

    // SMC1 decode (only if needed)
    let (smc_w, smc_h, smc_raw): (u32, u32, Option<Vec<u8>>) =
        if let Some(smc1) = tile.smc1.as_ref() {
            if smc1.coord_space == Smc1CoordSpace::Crs84BboxNorm && tile.geot.is_some() {
                let data = match smc1.encoding {
                    Smc1Encoding::Raw => smc1.data.clone(),
                    Smc1Encoding::Rle => smc1_decode_rle(&smc1.data)?,
                };
                anyhow::ensure!(
                    smc1.width > 0
                        && smc1.height > 0
                        && data.len() == smc1.width as usize * smc1.height as usize,
                    "Invalid SMC1 dimensions/payload length"
                );
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

    // Direct labels take precedence over geographic mask sampling.
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
                        (p[0] as f64 * inv_upm_f64) as f32,
                        (p[1] as f64 * inv_upm_f64) as f32,
                        (p[2] as f64 * inv_upm_f64) as f32,
                    ];
                    let label = labels.map(|ls| ls[i]).unwrap_or(0) as u32;
                    PointInstance { ofs_m, label }
                })
                .collect()
        } else {
            let (lon_min, lon_max, lat_min, lat_max) = geot_deg.unwrap();
            anyhow::ensure!(
                lon_max > lon_min && lat_max > lat_min,
                "Geographic mask requires a non-degenerate GEOT extent"
            );
            let inv_dlon = 1.0 / (lon_max - lon_min);
            let inv_dlat = 1.0 / (lat_max - lat_min);

            let smc = smc_raw.as_ref().unwrap();
            let sw = smc_w as usize;

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
                            (p[0] as f64 * inv_upm_f64) as f32,
                            (p[1] as f64 * inv_upm_f64) as f32,
                            (p[2] as f64 * inv_upm_f64) as f32,
                        ],
                        label,
                    }
                })
                .collect()
        };

    // Tile-level analysis and logging is confined to debug builds.
    #[cfg(debug_assertions)]
    {
        // AABB calculation for logging purposes.
        use std::f32::{INFINITY, NEG_INFINITY};
        let (min, max) = instances.par_iter().map(|pi| (pi.ofs_m, pi.ofs_m)).reduce(
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
            "HYPC {:?}: pts={}, upm={}, anchor_ecef_m=({:.3},{:.3},{:.3}), ofs_AABB_m=min({:.2},{:.2},{:.2}) max({:.2},{:.2},{:.2})",
            path.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
            tile.points_units.len(),
            tile.units_per_meter,
            anchor_m[0], anchor_m[1], anchor_m[2],
            min[0], min[1], min[2],
            max[0], max[1], max[2]
        );
    }

    anyhow::ensure!(!instances.is_empty(), "Empty HYPC tile: {}", path.display());
    Ok(PreparedTile {
        key: tile.tile_key,
        units_per_meter: tile.units_per_meter,
        anchor_units: tile.anchor_ecef_units,
        instances,
    })
}

/// Compatibility entry point for individual tiles and rendering regressions.
pub fn load_hypc_tile(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    camera: &Camera,
    path: &Path,
    viewport_size: [f32; 2],
) -> Result<TileGpu> {
    let tile = prepare_hypc_tile(path)?;
    Ok(upload_tile(device, layout, camera, &tile, viewport_size))
}

pub fn upload_tile(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    camera: &Camera,
    tile: &PreparedTile,
    viewport_size: [f32; 2],
) -> TileGpu {
    let instances = &tile.instances;
    // GPU upload
    let vtx = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("HYPC Instances"),
        contents: bytemuck::cast_slice(&instances),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });

    let tile_ubo_data = camera.make_tile_uniform(
        tile.anchor_units,
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

    TileGpu {
        key: tile.key,
        units_per_meter: tile.units_per_meter,
        anchor_units: tile.anchor_units,
        instances_len: instances.len() as u32,
        vtx,
        ubo,
        bind,
    }
}
