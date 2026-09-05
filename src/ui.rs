// holographic-viewer/src/ui.rs
//! UI rendering using egui.

use crate::renderer::pipelines::post_stack::PostParams;
use egui::{Area, Frame, RichText};

/// Draws the HUD overlay, including corner brackets and status text.
pub fn draw_hud(egui_ctx: &egui::Context, altitude: i32, total_points: u32) {
    // Draw corner brackets and central dot
    {
        let painter = egui_ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("hud_lines"),
        ));

        let rect = egui_ctx.screen_rect();
        let color = egui::Color32::from_rgba_unmultiplied(45, 247, 255, 200);
        let (thickness, margin, length) = (2.0, 26.0, 140.0);

        // Top‑left bracket
        painter.line_segment(
            [egui::pos2(margin, margin), egui::pos2(margin + length, margin)],
            (thickness, color),
        );
        painter.line_segment(
            [egui::pos2(margin, margin), egui::pos2(margin, margin + length)],
            (thickness, color),
        );

        // Top‑right bracket
        painter.line_segment(
            [
                egui::pos2(rect.max.x - margin - length, margin),
                egui::pos2(rect.max.x - margin, margin),
            ],
            (thickness, color),
        );
        painter.line_segment(
            [
                egui::pos2(rect.max.x - margin, margin),
                egui::pos2(rect.max.x - margin, margin + length),
            ],
            (thickness, color),
        );

        // Bottom‑left bracket
        painter.line_segment(
            [
                egui::pos2(margin, rect.max.y - margin),
                egui::pos2(margin + length, rect.max.y - margin),
            ],
            (thickness, color),
        );
        painter.line_segment(
            [
                egui::pos2(margin, rect.max.y - margin - length),
                egui::pos2(margin, rect.max.y - margin),
            ],
            (thickness, color),
        );

        // Bottom‑right bracket
        painter.line_segment(
            [
                egui::pos2(rect.max.x - margin - length, rect.max.y - margin),
                egui::pos2(rect.max.x - margin, rect.max.y - margin),
            ],
            (thickness, color),
        );
        painter.line_segment(
            [
                egui::pos2(rect.max.x - margin, rect.max.y - margin - length),
                egui::pos2(rect.max.x - margin, rect.max.y - margin),
            ],
            (thickness, color),
        );

        // Central dot
        painter.circle_filled(egui::pos2(rect.center().x, 16.0), 3.0, color);
    }

    // Draw status text in the top‑left corner
    {
        Area::new("hud_text".into())
            .interactable(false)
            .movable(false)
            .order(egui::Order::Foreground)
            .fixed_pos(egui::pos2(40.0, 42.0))
            .show(egui_ctx, |ui| {
                Frame::none().show(ui, |ui| {
                    let text_color = egui::Color32::from_rgb(45, 247, 255);

                    ui.label(
                        RichText::new("HOLOGRAPHIC  SCAN  ACTIVE")
                            .monospace()
                            .color(text_color)
                            .size(16.0)
                            .strong(),
                    );
                    ui.label(
                        RichText::new(format!("RESOLUTION: {:>11} POINTS", total_points))
                            .monospace()
                            .color(text_color),
                    );
                    ui.label(
                        RichText::new(format!("ALTITUDE: {}M", altitude))
                            .monospace()
                            .color(text_color),
                    );
                    ui.label(
                        RichText::new("STATUS:  SCAN  COMPLETE")
                            .monospace()
                            .color(text_color),
                    );
                });
            });
    }
}

pub fn draw_debug_panel(
    egui_ctx: &egui::Context,
    params: &mut PostParams,
    gamma_deg: f64,
) {
    Area::new("debug_panel".into())
        .fixed_pos(egui::pos2(40.0, 140.0))
        .show(egui_ctx, |ui| {
            Frame::dark_canvas(ui.style()).show(ui, |ui| {
                let defaults = PostParams::default();

                ui.horizontal(|ui| {
                    ui.heading("Debug");
                    if ui.button("Reset All").clicked() {
                        *params = defaults;
                    }
                });

                ui.horizontal(|ui| {
                    ui.checkbox(&mut params.edl_on, "EDL");
                    ui.checkbox(&mut params.sem_on, "Semantic");
                    ui.checkbox(&mut params.rgb_on, "RGB shift");
                    ui.checkbox(&mut params.crt_on, "CRT");
                });
                ui.separator();

                ui.collapsing("Grid", |ui| {
                    ui.checkbox(&mut params.grid_on, "Visible");
                    ui.separator();
                    ui.label("Alignment");
                    ui.radio_value(&mut params.grid_utm_align, false, "True North");
                    ui.radio_value(&mut params.grid_utm_align, true, "UTM Grid North");
                    ui.label(format!("Convergence (γ): {:.4}°", gamma_deg));
                });

                ui.collapsing("EDL", |ui| {
                    if ui.button("Reset").clicked() {
                        params.edl_strength = defaults.edl_strength;
                        params.edl_radius_px = defaults.edl_radius_px;
                    }
                    ui.separator();
                    ui.label("Strength");
                    ui.add(egui::Slider::new(&mut params.edl_strength, 0.0..=5.0));
                    ui.label("Radius (px)");
                    ui.add(egui::Slider::new(&mut params.edl_radius_px, 0.5..=4.0));
                });

                ui.collapsing("Semantic", |ui| {
                    if ui.button("Reset").clicked() {
                        params.sem_amount = defaults.sem_amount;
                    }
                    ui.separator();
                    ui.label("Amount");
                    ui.add(egui::Slider::new(&mut params.sem_amount, 0.0..=1.0));
                });

                ui.collapsing("RGB Shift", |ui| {
                    if ui.button("Reset").clicked() {
                        params.rgb_amount = defaults.rgb_amount;
                        params.rgb_angle = defaults.rgb_angle;
                    }
                    ui.separator();
                    ui.label("Amount");
                    ui.add(egui::Slider::new(&mut params.rgb_amount, 0.0..=0.01));
                    ui.label("Angle");
                    ui.add(egui::Slider::new(
                        &mut params.rgb_angle,
                        0.0..=std::f32::consts::TAU,
                    ));
                });

                ui.collapsing("CRT", |ui| {
                    if ui.button("Reset").clicked() {
                        params.crt_intensity = defaults.crt_intensity;
                        params.crt_vignette = defaults.crt_vignette;
                    }
                    ui.separator();
                    ui.label("Intensity");
                    ui.add(egui::Slider::new(&mut params.crt_intensity, 0.0..=1.0));
                    ui.label("Vignette");
                    ui.add(egui::Slider::new(&mut params.crt_vignette, 0.0..=1.0));
                });
                ui.separator();

                ui.label("Debug View");
                ui.radio_value(&mut params.debug_mode, 0, "Off");
                ui.radio_value(&mut params.debug_mode, 1, "Depth");
                ui.radio_value(&mut params.debug_mode, 2, "Labels");
                ui.radio_value(&mut params.debug_mode, 3, "Tag");
            });
        });
}

#[derive(PartialEq)]
pub enum MapAction { None, Fit }

pub fn draw_map_ui(ctx: &egui::Context, camera: &mut crate::camera::Camera,
    scene: Option<&mut crate::data::streaming::StreamScene>, frame_ms: f64,
    preparing: bool, error: Option<&str>, params: &mut PostParams) -> MapAction {
    let mut action = MapAction::None;
    egui::Window::new("City map")
        .anchor(egui::Align2::LEFT_TOP, [16.0, 16.0])
        .resizable(false).collapsible(false).default_width(260.0)
        .show(ctx, |ui| {
            if let Some(scene) = scene {
                ui.label(format!("{} source tiles · {:.2} billion points",
                    scene.stats.source_tiles, scene.stats.source_points as f64 / 1e9));
                ui.horizontal(|ui| {
                    if ui.button("Fit city").clicked() { action = MapAction::Fit; }
                    if ui.button("North").clicked() { camera.azimuth_rad = std::f64::consts::PI; camera.update(); }
                    if ui.button("−").clicked() { camera.radius_m = (camera.radius_m * 1.6).min(150_000.0); camera.update(); }
                    if ui.button("+").clicked() { camera.radius_m = (camera.radius_m / 1.6).max(5.0); camera.update(); }
                });
                let mut tilt = camera.elevation_rad.to_degrees();
                if ui.add(egui::Slider::new(&mut tilt, 3.0..=89.9).text("Elevation °")).changed() {
                    camera.elevation_rad = tilt.to_radians(); camera.update();
                }
                let (lat, lon, _) = hypc::ecef_to_geodetic(camera.target_ecef.x, camera.target_ecef.y, camera.target_ecef.z);
                ui.label(format!("{lat:.5}°, {lon:.5}° · range {:.0} m", camera.radius_m));
                ui.collapsing("Places", |ui| {
                    for (name, lat, lon, h) in [
                        ("Brandenburg Gate", 52.516275, 13.3777, 40.0),
                        ("Alexanderplatz", 52.5219, 13.4132, 40.0),
                        ("Tempelhofer Feld", 52.4730, 13.4030, 48.0),
                        ("Spandau", 52.5362, 13.2000, 40.0),
                        ("Köpenick", 52.4455, 13.5745, 38.0),
                    ] {
                        if ui.button(name).clicked() {
                            camera.set_target_and_radius(hypc::geodetic_to_ecef(lat, lon, h), 1200.0);
                        }
                    }
                });
                ui.collapsing("Rendering", |ui| {
                    ui.checkbox(&mut params.fill_on, "Reconstruct small surface gaps");
                    ui.checkbox(&mut params.edl_on, "Depth shading");
                    ui.checkbox(&mut params.sem_on, "Semantic colours");
                    ui.checkbox(&mut params.grid_on, "Ground grid");
                    ui.checkbox(&mut params.crt_on, "CRT effect");
                    ui.checkbox(&mut params.rgb_on, "RGB shift");
                    ui.add(egui::Slider::new(&mut scene.config.target_spacing_px, 0.8..=6.0).text("LOD spacing px"));
                });
                ui.collapsing("Performance", |ui| {
                    ui.monospace(format!("{:.1} ms/frame · {:.0} FPS", frame_ms, 1000.0 / frame_ms.max(0.01)));
                    ui.monospace(format!("{} draws · {:.2} M points", scene.stats.draw_calls, scene.stats.visible_points as f64 / 1e6));
                    ui.monospace(format!("GPU {:.1} / {:.0} MiB", scene.stats.gpu_bytes as f64 / 1048576.0, scene.config.gpu_bytes as f64 / 1048576.0));
                    ui.monospace(format!("{} resident · {} in flight", scene.stats.resident_nodes, scene.stats.pending_nodes));
                    ui.monospace(format!("Selection {:.2} ms · update {:.2} ms", scene.stats.selection_ms, scene.stats.update_ms));
                    ui.monospace(format!("{} evictions · {} stale requests", scene.stats.evictions, scene.stats.cancelled));
                    if scene.stats.failures > 0 {
                        ui.colored_label(egui::Color32::YELLOW, format!("{} nodes unavailable; coarse coverage retained", scene.stats.failures));
                        if ui.button("Retry unavailable detail").clicked() { scene.retry_failed(); }
                    }
                });
            } else if preparing {
                ui.spinner();
                ui.label("Preparing city detail levels…");
                ui.label("This one-time preparation runs in the background.");
            } else {
                ui.label("Drop a HYPC folder or LOD catalog here.");
            }
            if let Some(error) = error { ui.colored_label(egui::Color32::LIGHT_RED, error); }
        });
    egui::Area::new("navigation_help".into())
        .anchor(egui::Align2::CENTER_BOTTOM, [0.0, -16.0])
        .show(ctx, |ui| {
            egui::Frame::dark_canvas(ui.style()).rounding(6.0).show(ui, |ui| {
                ui.label("Drag / WASD / arrows: pan   ·   Right-drag / Q E: rotate   ·   R F: tilt   ·   Scroll / + −: zoom   ·   Shift: faster");
            });
        });
    action
}
