//! Native city viewer and reproducible navigation/capture harness.
use anyhow::{Context, Result};
use holographic_viewer::app::App;
use std::{path::PathBuf, sync::Arc};
use winit::{event::{ElementState, Event, WindowEvent}, event_loop::{ControlFlow, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::WindowBuilder};

#[derive(Default)]
struct Options {
    dataset: Option<PathBuf>, frames: Option<u64>, capture: Option<PathBuf>, report: Option<PathBuf>,
    pose: Option<[f64; 6]>, tour: bool, gpu_mib: Option<u64>, stress: bool, verify_coverage: bool,
}
fn options() -> Result<Options> {
    let mut args = std::env::args().skip(1);
    let mut result = Options::default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dataset" => result.dataset = Some(args.next().context("--dataset requires a directory")?.into()),
            "--frames" => result.frames = Some(args.next().context("--frames requires a count")?.parse()?),
            "--capture" => result.capture = Some(args.next().context("--capture requires a PNG path")?.into()),
            "--report" => result.report = Some(args.next().context("--report requires a JSON path")?.into()),
            "--tour" => result.tour = true,
            "--stress" => result.stress = true,
            "--verify-coverage" => result.verify_coverage = true,
            "--gpu-mib" => result.gpu_mib = Some(args.next().context("--gpu-mib requires a budget")?.parse()?),
            "--pose" => {
                let values = args.next().context("--pose requires lat,lon,height_m,radius_m,azimuth_deg,elevation_deg")?
                    .split(',').map(str::parse::<f64>).collect::<std::result::Result<Vec<_>, _>>()?;
                let pose: [f64; 6] = values.try_into().map_err(|_| anyhow::anyhow!("Pose requires six numbers"))?;
                anyhow::ensure!(pose.iter().all(|v| v.is_finite()) && (-90.0..=90.0).contains(&pose[0]) && (-180.0..=180.0).contains(&pose[1]) && (5.0..=150_000.0).contains(&pose[3]) && (3.0..=89.9).contains(&pose[5]), "Invalid camera pose");
                result.pose = Some(pose);
            }
            "--help" | "-h" => {
                println!("Holographic City Viewer\nUsage: holographic_viewer [--dataset] DIRECTORY\n\nOpen a prepared city catalog or HYPC folder.\nDrag/WASD/arrows: pan; Cmd/Ctrl/right/middle-drag/QE: orbit; RF/PageUp/PageDown: tilt; wheel/+/-: zoom.\nShift-drag: orbit. Shift+movement: faster. F12: save a frame.\n\nVerification options:\n  --frames N --report report.json --capture final.png\n  --pose lat,lon,height_m,radius_m,azimuth_deg,elevation_deg\n  --tour  (reproducible 900-frame city/close-up/navigation route)");
                std::process::exit(0);
            }
            _ if !arg.starts_with('-') && result.dataset.is_none() => result.dataset = Some(arg.into()),
            _ => anyhow::bail!("Unknown option: {arg}"),
        }
    }
    anyhow::ensure!(result.frames.is_none_or(|n| n > 0), "Frame count must be positive");
    if result.tour && result.frames.is_none() { result.frames = Some(900); }
    if result.stress && result.frames.is_none() { result.frames = Some(2400); }
    anyhow::ensure!(result.gpu_mib.is_none_or(|m| (64..=8192).contains(&m)), "GPU cache budget must be 64..8192 MiB");
    if result.dataset.is_none() {
        result.dataset = ["../files/ber2025/city-lod", "files/ber2025/city-lod", "hypc"].iter().map(PathBuf::from).find(|p| p.is_dir());
    }
    Ok(result)
}

fn pose(app: &mut App, p: [f64; 6]) {
    app.camera.azimuth_rad = p[4].to_radians();
    app.camera.elevation_rad = p[5].to_radians();
    app.camera.set_target_and_radius(hypc::geodetic_to_ecef(p[0], p[1], p[2]), p[3]);
}

fn tour(app: &mut App, frame: u64) {
    match frame {
        0 | 780 => app.fit_dataset(),
        120 => pose(app, [52.516275, 13.3777, 40.0, 1500.0, 180.0, 70.0]),
        420 => pose(app, [52.5362, 13.2000, 40.0, 1200.0, 180.0, 45.0]),
        600 => pose(app, [52.4455, 13.5745, 38.0, 1500.0, 140.0, 60.0]),
        _ => {}
    }
    if (120..300).contains(&frame) {
        let t = (frame - 120) as f64 / 180.0;
        app.camera.radius_m = 1500.0 * (30.0f64 / 1500.0).powf(t);
        app.camera.elevation_rad = (70.0 - 45.0 * t).to_radians(); app.camera.update();
    } else if (300..420).contains(&frame) {
        let (right, _, _) = app.camera.navigation_basis(); app.camera.translate_surface(right * 3.0);
        app.camera.azimuth_rad += 0.02;
        app.camera.elevation_rad = (25.0 - 18.0 * (frame - 300) as f64 / 120.0).to_radians(); app.camera.update();
    } else if (420..600).contains(&frame) {
        let (_, forward, _) = app.camera.navigation_basis(); app.camera.translate_surface(forward * 9.0);
    } else if (600..780).contains(&frame) {
        let t = (frame - 600) as f64 / 180.0;
        app.camera.radius_m = 1500.0 * (400.0f64 / 1500.0).powf(t);
        app.camera.elevation_rad = (60.0 + 29.9 * t).to_radians(); app.camera.update();
    }
}

fn stress(app: &mut App, frame: u64) {
    // Widely separated visits followed by a return to force eviction/reuse.
    let sites = [
        (52.516275, 13.3777), (52.5362, 13.2000), (52.4455, 13.5745),
        (52.5219, 13.4132), (52.4730, 13.4030), (52.555, 13.35),
        (52.505, 13.45), (52.515, 13.30), (52.46, 13.32), (52.57, 13.41),
        (52.49, 13.52), (52.53, 13.48),
    ];
    let visit = (frame / 90) as usize;
    let (lat, lon) = sites[visit % sites.len()];
    if frame % 90 == 0 { pose(app, [lat, lon, 42.0, 900.0, 180.0, 50.0]); }
    let t = (frame % 90) as f64 / 90.0;
    app.camera.radius_m = 900.0 * (120.0f64 / 900.0).powf(t);
    app.camera.azimuth_rad += 0.006; app.camera.update();
}

fn report(app: &mut App, path: &std::path::Path) -> Result<()> {
    if let Some(probe) = &mut app.renderer.probe { probe.poll(&app.renderer.gfx.device, true); }
    let mut times: Vec<_> = app.metrics.iter().skip(5).filter_map(|v| v["frame_ms"].as_f64()).collect();
    times.sort_by(f64::total_cmp);
    let percentile = |q: f64| times.get(((times.len().saturating_sub(1)) as f64 * q).round() as usize).copied().unwrap_or(0.0);
    let value = serde_json::json!({
        "viewport":[app.renderer.gfx.size.width, app.renderer.gfx.size.height],
        "frames":app.metrics.len(), "frame_ms_p50":percentile(0.5), "frame_ms_p95":percentile(0.95), "frame_ms_p99":percentile(0.99),
        "terrain_gpu_bytes":app.renderer.terrain.as_ref().map(|t| t.bytes),
        "samples":app.metrics,
        "gpu_probe":app.renderer.probe.as_ref().map(|p| &p.samples),
        "gpu_probe_skipped":app.renderer.probe.as_ref().map(|p| p.skipped),
    });
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) { std::fs::create_dir_all(parent)?; }
    serde_json::to_writer_pretty(std::fs::File::create(path)?, &value)?;
    log::info!("Frame time p50={:.2} p95={:.2} p99={:.2} ms; report {}", percentile(0.5), percentile(0.95), percentile(0.99), path.display());
    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let options = options()?;
    let event_loop = EventLoop::new()?;
    let window = Arc::new(WindowBuilder::new().with_title("Holographic City Viewer")
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720)).build(&event_loop)?);
    let mut app = pollster::block_on(App::new(window.clone()))?;
    if let Some(mib) = options.gpu_mib { app.stream_config.gpu_bytes = mib * 1024 * 1024; }
    if let Some(dataset) = &options.dataset {
        if let Err(error) = app.load_dataset(dataset) { app.load_error = Some(format!("{error:#}")); log::error!("{error:#}"); }
    }
    if let Some(p) = options.pose { pose(&mut app, p); }
    app.record_metrics = options.report.is_some();
    app.automated_navigation = options.tour || options.stress;
    if options.verify_coverage {
        let renderer = &mut app.renderer;
        renderer.probe = Some(holographic_viewer::renderer::probe::FrameProbe::new(&renderer.gfx.device, &renderer.gfx.queue,
            &renderer.targets.dlin, &renderer.reconstruction.depth, [renderer.gfx.size.width, renderer.gfx.size.height]));
    }
    let mut frame_number = 0u64;
    let mut reported = false;
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        match event {
            Event::WindowEvent { window_id, event } if window_id == window.id() => {
                if !app.handle_event(&window, &event) {
                    match event {
                        WindowEvent::CloseRequested => elwt.exit(),
                        WindowEvent::DroppedFile(path) => {
                            if let Err(error) = app.load_dataset(&path) { app.load_error = Some(format!("{error:#}")); }
                        }
                        WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                            if event.physical_key == PhysicalKey::Code(KeyCode::Escape) { elwt.exit(); }
                            if event.physical_key == PhysicalKey::Code(KeyCode::F12) { app.capture_next = Some(options.capture.clone().unwrap_or_else(|| "holoviewer-capture.png".into())); }
                        }
                        WindowEvent::RedrawRequested => {
                            if options.frames.is_some_and(|n| frame_number >= n) { return; }
                            if options.tour { tour(&mut app, frame_number); }
                            if options.stress { stress(&mut app, frame_number); }
                            if frame_number + 1 == options.frames.unwrap_or(240) { app.capture_next = options.capture.clone(); }
                            match app.render(&window) {
                                Ok(_) => {
                                    frame_number += 1;
                                    if options.frames.is_some_and(|n| frame_number >= n) {
                                        if let Some(path) = &options.report { if let Err(e) = report(&mut app, path) { log::error!("Report: {e:#}"); } }
                                        reported = true;
                                        elwt.exit();
                                    }
                                }
                                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => app.resize(app.renderer.gfx.size),
                                Err(wgpu::SurfaceError::OutOfMemory) => { log::error!("GPU out of memory"); elwt.exit(); }
                                Err(e) => log::error!("Render error: {e:?}"),
                            }
                        }
                        _ => {}
                    }
                }
            }
            Event::AboutToWait => window.request_redraw(),
            Event::LoopExiting if !reported => {
                if let Some(path) = &options.report { if let Err(e) = report(&mut app, path) { log::error!("Report: {e:#}"); } }
            }
            _ => {}
        }
    })?;
    Ok(())
}
