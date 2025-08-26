// src/main.rs
use anyhow::Result;
use holographic_viewer::app::App;
use std::sync::Arc;
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

fn main() -> Result<()> {
    env_logger::init();

    let event_loop = EventLoop::new()?;

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Holographic City Viewer — Rust")
            .with_inner_size(LogicalSize::new(1280, 720))
            .build(&event_loop)?,
    );

    let mut app = pollster::block_on(App::new(window.clone()));

    // Load all .hypc tiles found under ./hypc (or ../hypc)
    if std::path::Path::new("hypc").exists() {
        if let Err(e) = app.build_all_tiles("hypc") {
            log::error!("Failed to build tiles from 'hypc': {}", e);
        }
    } else if std::path::Path::new("../hypc").exists() {
        if let Err(e) = app.build_all_tiles("../hypc") {
            log::error!("Failed to build tiles from '../hypc': {}", e);
        }
    }

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(winit::event_loop::ControlFlow::Poll);

        match event {
            Event::WindowEvent {
                window_id,
                event,
            } if window_id == window.id() => {
                // Let egui consume events first, then pass to app
                if !app.handle_event(&event) {
                    match event {
                        WindowEvent::CloseRequested => elwt.exit(),
                        WindowEvent::KeyboardInput { event, .. } => {
                            if event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                                elwt.exit();
                            }
                        }
                        WindowEvent::RedrawRequested => {
                            if let Err(e) = app.render() {
                                match e {
                                    wgpu::SurfaceError::Lost => app.resize(app.get_size()),
                                    wgpu::SurfaceError::OutOfMemory => elwt.exit(),
                                    _ => eprintln!("Render error: {:?}", e),
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    })?;

    #[allow(unreachable_code)]
    Ok(())
}
