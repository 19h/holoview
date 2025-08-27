//! Executable entry point for the Holographic Viewer application.

use anyhow::Result;
use holographic_viewer::app::App;
use std::sync::Arc;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

fn main() -> Result<()> {
    // Setup logging with a sensible default if RUST_LOG is unset
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    // Create the winit event loop and window
    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Holographic City Viewer")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
            .build(&event_loop)?,
    );

    // Block on async initialization
    let mut app = pollster::block_on(App::new(window.clone()))?;

    // Load data
    if let Err(e) = app.build_all_tiles("hypc") {
        log::error!("Failed to build tiles: {}", e);
    }

    // Run the event loop
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent {
                window_id,
                event,
            } if window_id == window.id() => {
                // Pass events to the app. If not consumed, handle window-level events.
                if !app.handle_event(&window, &event) {
                    match event {
                        WindowEvent::CloseRequested => elwt.exit(),
                        WindowEvent::KeyboardInput { event, .. } => {
                            if event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                                elwt.exit();
                            }
                        }
                        WindowEvent::RedrawRequested => {
                            match app.render(&window) {
                                Ok(_) => {}
                                Err(wgpu::SurfaceError::Lost) => app.resize(app.renderer.gfx.size),
                                Err(wgpu::SurfaceError::OutOfMemory) => {
                                    log::error!("WGPU Out of Memory! Exiting.");
                                    elwt.exit();
                                }
                                Err(e) => log::error!("Render error: {:?}", e),
                            }
                        }
                        _ => {}
                    }
                }
            }
            Event::AboutToWait => {
                // Redraw continuously
                window.request_redraw();
            }
            _ => {}
        }
    })?;

    Ok(())
}
