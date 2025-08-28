//! UI rendering using egui.

use egui::{Area, Frame, RichText};
use crate::renderer::pipelines::post_stack::PostParams;

/// Draws the HUD overlay, including corner brackets and status text.
pub fn draw_hud(egui_ctx: &egui::Context, altitude: i32, total_points: u32) {
    // Draw corner brackets and central dot
    {
        let painter = egui_ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("hud_lines"),
        ));

        let rect   = egui_ctx.screen_rect();
        let color  = egui::Color32::from_rgba_unmultiplied(45, 247, 255, 200);
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

pub fn draw_debug_panel(egui_ctx: &egui::Context, params: &mut PostParams, point_size: &mut f32) {
    Area::new("debug_panel".into())
        .fixed_pos(egui::pos2(40.0, 140.0))
        .show(egui_ctx, |ui| {
            Frame::dark_canvas(ui.style()).show(ui, |ui| {
                ui.heading("Debug");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut params.grid_on, "Grid");
                    ui.checkbox(&mut params.edl_on,  "EDL");
                    ui.checkbox(&mut params.sem_on,  "Semantic");
                    ui.checkbox(&mut params.rgb_on,  "RGB shift");
                    ui.checkbox(&mut params.crt_on,  "CRT");
                });
                ui.separator();
                ui.label("Point size (px)");
                ui.add(egui::Slider::new(point_size, 1.0..=12.0));
                ui.separator();
                ui.label("Debug View");
                ui.radio_value(&mut params.debug_mode, 0, "Off");
                ui.radio_value(&mut params.debug_mode, 1, "Depth");
                ui.radio_value(&mut params.debug_mode, 2, "Labels");
                ui.radio_value(&mut params.debug_mode, 3, "Tag");
            });
        });
}
