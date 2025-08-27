//! Core data types for the holographic viewer, focused on GPU data representation.

/// Defines the per-instance data uploaded to the GPU vertex buffer.
/// Must match the layout of instance inputs in `hypc_points.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct PointInstance {
    /// Per-point offset from the tile's anchor, in meters.
    pub ofs_m: [f32; 3],
    /// Per-point semantic label (0-255).
    pub label: u32,
}

/// Defines the per-tile uniform buffer data, respecting std140 layout.
/// Must match the layout of `TileUniform` in `hypc_points.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TileUniformStd140 {
    /// High part of the (tile_anchor - camera_position) vector.
    pub delta_hi: [f32; 3],
    pub _pad0: f32,
    /// Low part of the (tile_anchor - camera_position) vector.
    pub delta_lo: [f32; 3],
    pub _pad1: f32,
    /// Combined view-projection matrix for camera-relative ECEF rendering.
    pub view_proj: [[f32; 4]; 4],
    /// Size of the viewport in physical pixels.
    pub viewport_size: [f32; 2],
    /// Base size of the point sprite in pixels.
    pub point_size_px: f32,
    pub _pad2: f32,
}

/// A 32-byte, zero-padded UTF-8 tile identifier.
pub type TileKey32 = [u8; 32];

/// Holds all GPU resources and metadata for a single, renderable HYPC tile.
#[derive(Debug)]
pub struct TileGpu {
    pub key: Option<TileKey32>,
    pub units_per_meter: u32,
    pub anchor_units: [i64; 3],
    pub instances_len: u32,

    /// Vertex buffer containing `PointInstance` data.
    pub vtx: wgpu::Buffer,
    /// Uniform buffer containing `TileUniformStd140` data.
    pub ubo: wgpu::Buffer,
    /// Bind group connecting the UBO to the pipeline.
    pub bind: wgpu::BindGroup,
}
