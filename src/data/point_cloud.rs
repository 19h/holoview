use crate::camera::Camera;
use crate::data::types::{PointInstance, TileGpu};
use anyhow::Result;
use hypc::{
    ecef_to_geodetic,
    read_file,
    smc1_decode_rle,
    HypcTile,
    Smc1CoordSpace,
    Smc1Encoding,
};
use log::debug;
use std::path::Path;

// wgpu::util::DeviceExt is a trait, so we need to bring it into scope.
mod wgpu_util {
    pub use wgpu::util::DeviceExt;
}
use wgpu_util::*;

/// Read one HYPC tile from disk and upload to GPU (instances + per-tile UBO).
pub fn load_hypc_tile(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    cam: &Camera,
    path: &Path,
    viewport_size: [f32; 2], // Initial viewport size
) -> Result<TileGpu> {
    let tile: HypcTile = read_file(path)?;
    let upm_f = tile.units_per_meter as f32;
    let upm_f64 = tile.units_per_meter as f64;

    let (smc_w, smc_h, smc_raw): (u32, u32, Option<Vec<u8>>) = if let Some(smc1) = tile.smc1.as_ref() {
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

    let geot_deg: Option<(f64, f64, f64, f64)> = tile.geot.map(|g| g.to_deg());

    let mut instances = Vec::<PointInstance>::with_capacity(tile.points_units.len());

    // Debug: measure per-tile offset AABB in meters
    let mut ofs_min = [f32::INFINITY; 3];
    let mut ofs_max = [f32::NEG_INFINITY; 3];
    let has_direct_labels = tile
        .labels
        .as_ref()
        .map_or(false, |v| v.len() == tile.points_units.len());

    for (i, p) in tile.points_units.iter().enumerate() {
        let ofs_m = [
            (p[0] as f32) / upm_f,
            (p[1] as f32) / upm_f,
            (p[2] as f32) / upm_f,
        ];

        ofs_min[0] = ofs_min[0].min(ofs_m[0]);
        ofs_max[0] = ofs_max[0].max(ofs_m[0]);
        ofs_min[1] = ofs_min[1].min(ofs_m[1]);
        ofs_max[1] = ofs_max[1].max(ofs_m[1]);
        ofs_min[2] = ofs_min[2].min(ofs_m[2]);
        ofs_max[2] = ofs_max[2].max(ofs_m[2]);

        let mut label: u8 = if has_direct_labels {
            tile.labels.as_ref().unwrap()[i]
        } else {
            0
        };

        if label == 0 {
            if let (Some(ref raw), Some((lon_min, lon_max, lat_min, lat_max))) = (smc_raw.as_ref(), geot_deg) {
                let px_m = tile.anchor_ecef_units[0] as f64 / upm_f64 + ofs_m[0] as f64;
                let py_m = tile.anchor_ecef_units[1] as f64 / upm_f64 + ofs_m[1] as f64;
                let pz_m = tile.anchor_ecef_units[2] as f64 / upm_f64 + ofs_m[2] as f64;

                let (lat_deg, lon_deg, _h) = ecef_to_geodetic(px_m, py_m, pz_m);

                let u = ((lon_deg - lon_min) / (lon_max - lon_min + 1e-12)).clamp(0.0, 1.0);
                let v = ((lat_deg - lat_min) / (lat_max - lat_min + 1e-12)).clamp(0.0, 1.0);

                let ix = (u * smc_w as f64) as i32;
                let iy = (v * smc_h as f64) as i32;
                let ix = ix.clamp(0, smc_w.saturating_sub(1) as i32) as usize;
                let iy = iy.clamp(0, smc_h.saturating_sub(1) as i32) as usize;

                let idx = iy * smc_w as usize + ix;
                if let Some(l) = raw.get(idx) {
                    label = *l;
                }
            }
        }

        instances.push(PointInstance {
            ofs_m,
            label: label as u32,
        });
    }

    // Log once the key facts for this tile
    let upm64 = tile.units_per_meter as f64;
    let a_m = [
        tile.anchor_ecef_units[0] as f64 / upm64,
        tile.anchor_ecef_units[1] as f64 / upm64,
        tile.anchor_ecef_units[2] as f64 / upm64,
    ];
    let _ext = [
        ofs_max[0] - ofs_min[0],
        ofs_max[1] - ofs_min[1],
        ofs_max[2] - ofs_min[2],
    ];
    debug!(
        "HYPC {}: pts={}, upm={}, anchor_ecef_m=({:.3},{:.3},{:.3}), ofs_AABB_m=min({:.2},{:.2},{:.2}) max({:.2},{:.2},{:.2})",
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?"),
        tile.points_units.len(),
        tile.units_per_meter,
        a_m[0],
        a_m[1],
        a_m[2],
        ofs_min[0],
        ofs_min[1],
        ofs_min[2],
        ofs_max[0],
        ofs_max[1],
        ofs_max[2]
    );

    let vtx = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("HYPC Instances"),
        contents: bytemuck::cast_slice(&instances),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });

    let tile_ubo_data = cam.make_tile_uniform(
        tile.anchor_ecef_units,
        tile.units_per_meter,
        viewport_size,
        4.0, // Default point size
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
