use anyhow::{anyhow, Result};
use std::sync::Arc;
use winit::window::Window;

/// Holds all GPU resources needed for rendering.
pub struct GfxContext {
    pub surface: wgpu::Surface<'static>,
    pub device:  wgpu::Device,
    pub queue:   wgpu::Queue,
    pub config:  wgpu::SurfaceConfiguration,
    pub size:    winit::dpi::PhysicalSize<u32>,
}

impl GfxContext {
    /// Creates a new graphics context bound to the given window.
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

        // The surface must outlive the window; `Arc` guarantees this.
        let surface = instance.create_surface(window.clone())?;

        // Choose a high‑performance adapter compatible with the surface.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:         wgpu::PowerPreference::HighPerformance,
                compatible_surface:       Some(&surface),
                force_fallback_adapter:   false,
            })
            .await
            .ok_or_else(|| anyhow!("Failed to find a suitable GPU adapter."))?;

        // Request a device and its command queue.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label:            Some("Device"),
                    required_features: wgpu::Features::empty(),
                    // Use default limits for broad compatibility.
                    required_limits:   wgpu::Limits::default(),
                },
                None, // no trace
            )
            .await?;

        // Determine the surface format (prefer sRGB).
        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        // Configure the surface.
        let config = wgpu::SurfaceConfiguration {
            usage:                       wgpu::TextureUsages::RENDER_ATTACHMENT,
            format:                      surface_format,
            width:                       size.width.max(1),
            height:                      size.height.max(1),
            present_mode:                wgpu::PresentMode::Fifo, // V‑sync
            alpha_mode:                  caps.alpha_modes[0],
            view_formats:                vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
        })
    }

    /// Resizes the swap chain when the window size changes.
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }
}
