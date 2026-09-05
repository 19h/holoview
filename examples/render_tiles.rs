//! Headless capture using the production HYPC loader, camera and point pipeline.
//! cargo run --release --example render_tiles -- hypc target/alignment/scene.ppm 75
#[path = "support/offscreen.rs"]
mod offscreen;
use anyhow::{Context, Result};
use glam::{DVec3, Mat4};
use holographic_viewer::{camera::Camera, data::point_cloud::load_hypc_tile};
use std::{io::Write, path::Path};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let root = args
        .get(1)
        .context("Expected tile directory and output.ppm [elevation_deg]")?;
    let output = args.get(2).context("Expected output.ppm")?;
    let elevation: f64 = args.get(3).map_or(Ok(75.0), |v| v.parse())?;
    let gpu = offscreen::Offscreen::new(1536)?;
    let mut camera = Camera::new(
        52.513,
        13.375,
        1400.0,
        Mat4::perspective_rh(60f32.to_radians(), 1.0, 0.1, 100_000.0),
    );
    let mut paths = std::fs::read_dir(root)?
        .map(|v| v.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.retain(|p| p.extension().is_some_and(|v| v == "hypc"));
    paths.sort();
    anyhow::ensure!(!paths.is_empty(), "No HYPC tiles");
    let mut tiles = Vec::new();
    for path in paths {
        tiles.push(load_hypc_tile(
            &gpu.device,
            &gpu.pipeline.tile_layout,
            &camera,
            &path,
            [gpu.size as f32; 2],
        )?);
    }
    let n: f64 = tiles.iter().map(|t| t.instances_len as f64).sum();
    let center = tiles.iter().fold(DVec3::ZERO, |sum, t| {
        sum + DVec3::from_array(t.anchor_units.map(|v| v as f64 / t.units_per_meter as f64))
            * t.instances_len as f64
    }) / n;
    let spread = tiles
        .iter()
        .map(|t| {
            (DVec3::from_array(t.anchor_units.map(|v| v as f64 / t.units_per_meter as f64))
                - center)
                .length()
        })
        .fold(0.0, f64::max);
    camera.set_target_and_radius(center.into(), (spread * 3.5).max(800.0));
    // Optional explicit shared pose for reproducible before/after captures:
    // latitude longitude ellipsoidal_height_m radius_m azimuth_deg.
    if args.len() > 5 {
        anyhow::ensure!(
            args.len() == 10,
            "Expected five pose values: lat lon height radius azimuth"
        );
        let pose = args[5..]
            .iter()
            .map(|v| v.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()?;
        camera.set_target_and_radius(hypc::geodetic_to_ecef(pose[0], pose[1], pose[2]), pose[3]);
        camera.azimuth_rad = pose[4].to_radians();
    }
    camera.elevation_rad = elevation.to_radians();
    camera.update();
    let mode = args.get(4).map(String::as_str).unwrap_or("raw");
    let mut params = holographic_viewer::renderer::pipelines::post_stack::PostParams::default();
    if mode != "post" {
        params.crt_on = false;
        params.rgb_on = false;
        params.sem_on = mode == "labels";
        if mode == "labels" {
            params.debug_mode = 2;
        }
    }
    let pixels = if mode == "raw" {
        gpu.render(&tiles, &camera, 0.65)?
    } else {
        gpu.render_with_post(&tiles, &camera, 0.65, Some(params))?
    };
    anyhow::ensure!(pixels.chunks_exact(4).any(|p| p[3] != 0), "Empty render");
    if let Some(parent) = Path::new(output).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(output)?);
    write!(f, "P6\n{} {}\n255\n", gpu.size, gpu.size)?;
    for p in pixels.chunks_exact(4) {
        f.write_all(&p[..3])?;
    }
    f.flush()?;
    eprintln!(
        "Captured {} tiles, {} points to {}",
        tiles.len(),
        n as u64,
        output
    );
    Ok(())
}
