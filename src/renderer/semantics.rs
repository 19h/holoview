// src/renderer/semantics.rs
//! Semantic mask texture management for point cloud rendering.
//!
//! This module handles uploading semantic masks (SMC1 format) to GPU textures,
//! including proper alignment for WGPU copy operations and fallback textures
//! for cases where no semantic data is available.

use crate::data::types::SemanticMask;

/// Manages semantic mask textures and associated GPU resources.
pub struct SemanticsIO {
    // We don't store persistent resources since wgpu types aren't clonable
    // Instead, we'll create them on demand
}

impl SemanticsIO {
    /// Creates a new SemanticsIO instance with fallback resources.
    ///
    /// # Arguments
    /// * `device` - The GPU device
    /// * `queue` - The GPU command queue
    ///
    /// # Returns
    /// A new SemanticsIO instance with initialized fallback texture and sampler.
    pub fn new(_device: &wgpu::Device, _queue: &wgpu::Queue) -> Self {
        // No persistent resources needed
        Self {}
    }

    /// Uploads a semantic mask to a GPU texture in R8Unorm format.
    ///
    /// # Arguments
    /// * `device` - The GPU device
    /// * `queue` - The GPU command queue
    /// * `semantic_mask` - Optional semantic mask to upload
    ///
    /// # Returns
    /// A tuple containing:
    /// - Optional texture (Some if new texture was created, None for fallback)
    /// - Texture view to use for rendering
    /// - Sampler to use for the texture
    ///
    /// # Implementation Notes
    /// - Uses R8Unorm format (single byte per pixel)
    /// - Ensures proper row alignment (256 bytes) for WGPU copy operations
    /// - Falls back to 1x1 texture if no mask is provided
    pub fn upload_r8(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        semantic_mask: Option<&SemanticMask>,
    ) -> (Option<wgpu::Texture>, wgpu::TextureView, wgpu::Sampler) {
        let Some(mask) = semantic_mask else {
            // No semantic mask provided, create fallback resources
            let fallback_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("SMC1 Fallback 1x1"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &fallback_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &[0u8],
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(1),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );

            let fallback_view = fallback_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("SMC1 Fallback Sampler"),
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });

            return (None, fallback_view, sampler);
        };

        let width = mask.width as u32;
        let height = mask.height as u32;

        // Create texture for the semantic labels
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SMC1 Labels Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Calculate aligned bytes per row (must be multiple of COPY_BYTES_PER_ROW_ALIGNMENT)
        let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT; // 256
        let bytes_per_row = ((width + alignment - 1) / alignment) * alignment;
        
        // Ensure our calculation is correct
        debug_assert_eq!(bytes_per_row % alignment, 0);

        // Create staging buffer with proper alignment
        let mut staging_buffer = vec![0u8; (bytes_per_row * height) as usize];

        // Copy semantic data row by row with padding
        for row in 0..height as usize {
            let src_start = row * width as usize;
            let src_end = (row + 1) * width as usize;
            let dst_start = row * bytes_per_row as usize;
            let dst_end = dst_start + width as usize;

            staging_buffer[dst_start..dst_end].copy_from_slice(&mask.data[src_start..src_end]);
            // Remaining bytes in the row are already zeroed
        }

        // Upload to GPU
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &staging_buffer,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Create a new sampler for each upload
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SMC1 Nearest Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        (Some(texture), view, sampler)
    }
}
