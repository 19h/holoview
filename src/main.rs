//! Entry point for the Holographic Viewer application.

use anyhow::Result;
use holographic_viewer::app::App;
use std::{
    sync::Arc,
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

fn main() -> Result<()> {
    // Initialize logging; default to "info" if RUST_LOG is unset.
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    // Create the event loop and window.
    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Holographic City Viewer")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
            .build(&event_loop)?,
    );

    // Initialise the application (async → sync).
    let mut app = pollster::block_on(App::new(window.clone()))?;

    // Load tiles; log any errors.
    if let Err(err) = app.build_all_tiles("hypc") {
        log::error!("Failed to build tiles: {}", err);
    }

    // Run the winit event loop.
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { window_id, event } if window_id == window.id() => {
                // Forward events to the app; handle unconsumed window events.
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
                                Err(wgpu::SurfaceError::Lost) => {
                                    app.resize(app.renderer.gfx.size);
                                }
                                Err(wgpu::SurfaceError::OutOfMemory) => {
                                    log::error!("WGPU out of memory – exiting.");
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
                // Request a redraw each frame.
                window.request_redraw();
            }
            _ => {}
        }
    })?;

    Ok(())
}
