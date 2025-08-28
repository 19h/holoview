use crate::data::types::TileUniformStd140 as TileUniform;
use glam::{Mat3, Mat4, Vec3};
use hypc::{geodetic_to_ecef, split_f64_to_f32_pair};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};

#[derive(Debug, Clone)]
pub struct Camera {
    // Geodetic position
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub h_m: f64,

    // ENU orientation (radians)
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,

    // Projection matrix
    pub proj: Mat4,
}

impl Camera {
    /// Creates a new camera with default orientation.
    pub fn new(lat_deg: f64, lon_deg: f64, h_m: f64, proj: Mat4) -> Self {
        Self {
            lat_deg,
            lon_deg,
            h_m,
            yaw: 0.0,
            pitch: (-40.0_f32).to_radians(), // default down‑look
            roll: 0.0,
            proj,
        }
    }

    /// Returns camera position in ECEF meters.
    #[inline]
    pub fn ecef_m(&self) -> [f64; 3] {
        geodetic_to_ecef(self.lat_deg, self.lon_deg, self.h_m)
    }

    /// Returns rotation matrix from ECEF to ENU for the camera position.
    pub fn ecef_to_enu_matrix(&self) -> Mat3 {
        let lat_rad = self.lat_deg.to_radians();
        let lon_rad = self.lon_deg.to_radians();
        let (sin_lat, cos_lat) = lat_rad.sin_cos();
        let (sin_lon, cos_lon) = lon_rad.sin_cos();

        // East, North, Up basis vectors.
        let east  = Vec3::new(-sin_lon as f32, cos_lon as f32, 0.0);
        let north = Vec3::new((-sin_lat * cos_lon) as f32, (-sin_lat * sin_lon) as f32, cos_lat as f32);
        let up    = Vec3::new((cos_lat * cos_lon) as f32, (cos_lat * sin_lon) as f32, sin_lat as f32);

        Mat3::from_cols(east, north, up).transpose()
    }

    /// Returns combined view‑projection matrix in ECEF meters.
    pub fn view_proj_ecef(&self) -> Mat4 {
        self.proj * self.view_ecef()
    }

    /// Rotation‑only view matrix: transforms from ECEF to camera frame.
    pub fn view_ecef(&self) -> Mat4 {
        // ECEF → ENU rotation at the camera location.
        let r_ecef_to_enu = self.ecef_to_enu_matrix();

        // Camera orientation (yaw then pitch) in ENU space.
        let r_cam = Mat3::from_rotation_z(self.yaw) * Mat3::from_rotation_x(self.pitch);

        // Inverse camera rotation (ENU → camera frame).
        let view_enu = Mat4::from_mat3(r_cam).transpose();

        // Combine rotations: ECEF → ENU → camera.
        let to_enu = Mat4::from_mat3(r_ecef_to_enu);
        view_enu * to_enu
    }

    /// Builds a per‑tile uniform buffer.
    pub fn make_tile_uniform(
        &self,
        tile_anchor_units: [i64; 3],
        units_per_meter: u32,
        viewport_size: [f32; 2],
        point_size_px: f32,
    ) -> TileUniform {
        // Camera position in ECEF (meters).
        let cam_ecef = self.ecef_m();

        // Convert tile anchor from integer units to meters.
        let upm = units_per_meter as f64;
        let anchor_m = [
            tile_anchor_units[0] as f64 / upm,
            tile_anchor_units[1] as f64 / upm,
            tile_anchor_units[2] as f64 / upm,
        ];

        // Difference between tile anchor and camera position.
        let dx = anchor_m[0] - cam_ecef[0];
        let dy = anchor_m[1] - cam_ecef[1];
        let dz = anchor_m[2] - cam_ecef[2];

        // Split 64‑bit differences into high/low 32‑bit components.
        let (hix, lox) = split_f64_to_f32_pair(dx);
        let (hiy, loy) = split_f64_to_f32_pair(dy);
        let (hiz, loz) = split_f64_to_f32_pair(dz);

        // Assemble the uniform buffer.
        TileUniform {
            delta_hi: [hix, hiy, hiz],
            _pad0: 0.0,
            delta_lo: [lox, loy, loz],
            _pad1: 0.0,
            view_proj: self.view_proj_ecef().to_cols_array_2d(),
            viewport_size,
            point_size_px,
            _pad2: 0.0,
        }
    }
}

pub struct CameraController {
    mouse_down: bool,
    last_mouse: Option<(f64, f64)>,
}

impl CameraController {
    /// Creates a new controller with default state.
    pub fn new() -> Self {
        Self {
            mouse_down: false,
            last_mouse: None,
        }
    }

    /// Handles window events and updates the camera.
    pub fn handle_event(&mut self, event: &WindowEvent, camera: &mut Camera) {
        match event {
            WindowEvent::MouseInput { button, state, .. } => {
                if *button == MouseButton::Left {
                    self.mouse_down = *state == ElementState::Pressed;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.handle_cursor_orbit((position.x, position.y), camera);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 120.0,
                };

                self.handle_scroll(scroll, camera);
            }
            _ => {}
        }
    }

    /// Adjusts camera altitude based on scroll input.
    fn handle_scroll(&mut self, delta: f32, camera: &mut Camera) {
        // Positive delta = scroll up. Make this increase altitude (feel free to flip if preferred).
        let zoom = 1.1_f64.powf(delta as f64);

        camera.h_m *= zoom;
        camera.h_m = camera.h_m.clamp(10.0, 1_000_000.0);
    }

    /// Rotates the camera while the left mouse button is held.
    fn handle_cursor_orbit(&mut self, xy: (f64, f64), camera: &mut Camera) {
        if let Some(last) = self.last_mouse {
            if self.mouse_down {
                let dx = ((xy.0 - last.0) as f32) * 0.005;
                let dy = ((xy.1 - last.1) as f32) * 0.005;

                camera.yaw -= dx;
                camera.pitch -= dy;
                camera.pitch = camera
                    .pitch
                    .clamp(-89.9_f32.to_radians(), -1.0_f32.to_radians());
            }
        }

        self.last_mouse = Some(xy);
    }
}
