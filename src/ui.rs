// src/ui.rs
use crate::renderer::context::GpuContext;
use winit::window::Window;

#[allow(clippy::too_many_arguments)]
pub fn draw_hud(
    egui_ctx: &mut egui::Context,
    egui_state: &mut egui_winit::State,
    egui_renderer: &mut egui_wgpu::Renderer,
    window: &Window,
    gpu_context: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    swap_view: &wgpu::TextureView,
    total_points: u32,
    altitude: i32,
) {
    let egui_input = egui_state.take_egui_input(window);
    egui_ctx.begin_frame(egui_input);

    // Corner brackets & dot painter
    {
        let painter = egui_ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("hud_lines"),
        ));

        let rect = egui_ctx.screen_rect();
        let color = egui::Color32::from_rgba_unmultiplied(45, 247, 255, 200);
        let (thickness, margin, length) = (2.0, 26.0, 140.0);

        // Top-left
        painter.line_segment([egui::pos2(margin, margin), egui::pos2(margin + length, margin)], (thickness, color));
        painter.line_segment([egui::pos2(margin, margin), egui::pos2(margin, margin + length)], (thickness, color));
        // Top-right
        painter.line_segment([egui::pos2(rect.max.x - margin - length, margin), egui::pos2(rect.max.x - margin, margin)], (thickness, color));
        painter.line_segment([egui::pos2(rect.max.x - margin, margin), egui::pos2(rect.max.x - margin, margin + length)], (thickness, color));
        // Bottom-left
        painter.line_segment([egui::pos2(margin, rect.max.y - margin), egui::pos2(margin + length, rect.max.y - margin)], (thickness, color));
        painter.line_segment([egui::pos2(margin, rect.max.y - margin - length), egui::pos2(margin, rect.max.y - margin)], (thickness, color));
        // Bottom-right
        painter.line_segment([egui::pos2(rect.max.x - margin - length, rect.max.y - margin), egui::pos2(rect.max.x - margin, rect.max.y - margin)], (thickness, color));
        painter.line_segment([egui::pos2(rect.max.x - margin, rect.max.y - margin - length), egui::pos2(rect.max.x - margin, rect.max.y - margin)], (thickness, color));
        // Top-center dot
        painter.circle_filled(egui::pos2(rect.center().x, 16.0), 3.0, color);
    }

    // Top-left status text
    {
        use egui::{Area, Frame, RichText};
        Area::new("hud_text".into())
            .interactable(false)
            .movable(false)
            .order(egui::Order::Foreground)
            .fixed_pos(egui::pos2(40.0, 42.0))
            .show(egui_ctx, |ui| {
                Frame::none().show(ui, |ui| {
                    let text_color = egui::Color32::from_rgb(45, 247, 255);
                    ui.label(RichText::new("HOLOGRAPHIC  SCAN  ACTIVE").monospace().color(text_color).size(16.0).strong());
                    ui.label(RichText::new(format!("RESOLUTION: {:>11} POINTS", total_points)).monospace().color(text_color));
                    ui.label(RichText::new(format!("ALTITUDE: {}M", altitude)).monospace().color(text_color));
                    ui.label(RichText::new("STATUS:  SCAN  COMPLETE").monospace().color(text_color));
                });
            });
    }

    // Render egui to the swapchain
    let egui_output = egui_ctx.end_frame();
    let shapes = egui_ctx.tessellate(egui_output.shapes, egui_ctx.pixels_per_point());

    let screen_descriptor = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [gpu_context.config.width, gpu_context.config.height],
        pixels_per_point: egui_state.egui_ctx().pixels_per_point(),
    };

    for (id, delta) in &egui_output.textures_delta.set {
        egui_renderer.update_texture(&gpu_context.device, &gpu_context.queue, *id, delta);
    }

    egui_renderer.update_buffers(
        &gpu_context.device,
        &gpu_context.queue,
        encoder,
        &shapes,
        &screen_descriptor,
    );

    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("HUD"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: swap_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        egui_renderer.render(&mut render_pass, &shapes, &screen_descriptor);
    }

    for id in &egui_output.textures_delta.free {
        egui_renderer.free_texture(id);
    }
}
