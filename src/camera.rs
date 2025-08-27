use crate::data::types::TileUniformStd140 as TileUniform;
use glam::{Mat3, Mat4, Vec3};
use hypc::{geodetic_to_ecef, split_f64_to_f32_pair};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};

#[derive(Debug, Clone)]
pub struct Camera {
    // Geodetic pose
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub h_m: f64,

    // Orientation in ENU (yaw/pitch/roll in radians)
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,

    // Projection
    pub proj: Mat4,
}

impl Camera {
    pub fn new(lat_deg: f64, lon_deg: f64, h_m: f64, proj: Mat4) -> Self {
        Self {
            lat_deg,
            lon_deg,
            h_m,
            yaw: 0.0,
            pitch: -40.0f32.to_radians(), // Default pitch down
            roll: 0.0,
            proj,
        }
    }

    #[inline]
    pub fn ecef_m(&self) -> [f64; 3] {
        geodetic_to_ecef(self.lat_deg, self.lon_deg, self.h_m)
    }

    /// ECEF->ENU rotation at the camera location.
    fn ecef_to_enu_matrix(&self) -> Mat3 {
        let lat = self.lat_deg.to_radians();
        let lon = self.lon_deg.to_radians();
        let (sl, cl) = lat.sin_cos();
        let (so, co) = lon.sin_cos();

        let e = Vec3::new(-so as f32, co as f32, 0.0);
        let n = Vec3::new((-sl * co) as f32, (-sl * so) as f32, cl as f32);
        let u = Vec3::new((cl * co) as f32, (cl * so) as f32, sl as f32);

        Mat3::from_cols(e, n, u).transpose()
    }

    /// View*Projection that expects camera-relative ECEF meters.
    pub fn view_proj_ecef(&self) -> Mat4 {
        self.proj * self.view_ecef()
    }

    /// Pure View matrix (rotation only): ECEF -> camera frame.
    pub fn view_ecef(&self) -> Mat4 {
        let r_ecef_to_enu = self.ecef_to_enu_matrix();
        let r_cam = Mat3::from_rotation_z(self.yaw) * Mat3::from_rotation_x(self.pitch);
        let view_enu = Mat4::from_mat3(r_cam).transpose(); // inverse of camera orientation
        let to_enu = Mat4::from_mat3(r_ecef_to_enu);
        view_enu * to_enu
    }

    /// Build the per-tile uniform given the tile's anchor and rendering parameters.
    pub fn make_tile_uniform(
        &self,
        tile_anchor_units: [i64; 3],
        units_per_meter: u32,
        viewport_size: [f32; 2],
        point_size_px: f32,
    ) -> TileUniform {
        let cam_ecef = self.ecef_m();
        let upm = units_per_meter as f64;
        let anchor_m = [
            tile_anchor_units[0] as f64 / upm,
            tile_anchor_units[1] as f64 / upm,
            tile_anchor_units[2] as f64 / upm,
        ];
        let dx = anchor_m[0] - cam_ecef[0];
        let dy = anchor_m[1] - cam_ecef[1];
        let dz = anchor_m[2] - cam_ecef[2];
        let (hix, lox) = split_f64_to_f32_pair(dx);
        let (hiy, loy) = split_f64_to_f32_pair(dy);
        let (hiz, loz) = split_f64_to_f32_pair(dz);

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
    pub fn new() -> Self {
        Self {
            mouse_down: false,
            last_mouse: None,
        }
    }

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
                let scroll_delta = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 120.0,
                };
                self.handle_scroll(scroll_delta, camera);
            }
            _ => {}
        }
    }

    fn handle_scroll(&mut self, delta: f32, camera: &mut Camera) {
        let zoom_factor = 1.1f64.powf(-delta as f64);
        camera.h_m *= zoom_factor;
        camera.h_m = camera.h_m.clamp(10.0, 1_000_000.0);
    }

    fn handle_cursor_orbit(&mut self, xy: (f64, f64), camera: &mut Camera) {
        if let Some(last) = self.last_mouse {
            if self.mouse_down {
                let dx = (xy.0 - last.0) as f32 * 0.005;
                let dy = (xy.1 - last.1) as f32 * 0.005;

                camera.yaw -= dx;
                camera.pitch -= dy;

                camera.pitch = camera.pitch.clamp(-89.9f32.to_radians(), -1.0f32.to_radians());
            }
        }
        self.last_mouse = Some(xy);
    }
}
