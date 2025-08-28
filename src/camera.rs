use crate::data::types::TileUniformStd140 as TileUniform;
use glam::{DMat3, DVec3, Mat3, Mat4, Vec3};
use hypc::{ecef_to_geodetic, geodetic_to_ecef, split_f64_to_f32_pair};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};

/// This matrix converts clip-space coordinates from OpenGL conventions (Y-up, Z in [-1, 1])
/// to WebGPU conventions (Y-down, Z in [0, 1]).
#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Mat4 = Mat4::from_cols_array(&[
    -1.0,  0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0,  0.0, 0.5, 0.0,
    0.0,  0.0, 0.5, 1.0,
]);

#[derive(Debug, Clone)]
pub struct Camera {
    // --- Orbital Parameters (Primary State) ---
    /// The ECEF coordinate (meters) the camera orbits around.
    pub target_ecef: DVec3,
    /// Distance from the camera to the target (meters).
    pub radius_m: f64,
    /// Azimuth angle around the target's local "up" vector (radians).
    pub azimuth_rad: f64,
    /// Elevation angle from the target's local tangent plane (radians).
    pub elevation_rad: f64,

    // --- Derived Properties (Updated by `update()`) ---
    /// Camera position in ECEF meters.
    position_ecef: DVec3,
    /// Geodetic latitude in degrees.
    pub lat_deg: f64,
    /// Geodetic longitude in degrees.
    pub lon_deg: f64,
    /// Geodetic height above the ellipsoid in meters.
    pub h_m: f64,

    // --- Projection Matrix ---
    pub proj: Mat4,
}

impl Camera {
    /// Creates a new orbital camera.
    pub fn new(target_lat_deg: f64, target_lon_deg: f64, radius_m: f64, proj: Mat4) -> Self {
        let target_ecef = DVec3::from(geodetic_to_ecef(target_lat_deg, target_lon_deg, 0.0));

        let mut camera = Self {
            target_ecef,
            radius_m,
            azimuth_rad: 180.0f64.to_radians(),
            elevation_rad: 30.0f64.to_radians(),
            position_ecef: DVec3::ZERO, // placeholder
            lat_deg: 0.0,               // placeholder
            lon_deg: 0.0,               // placeholder
            h_m: 0.0,                   // placeholder
            proj,
        };

        camera.update(); // Calculate initial position
        camera
    }

    /// Recalculates the camera's ECEF position and geodetic coordinates from its
    /// orbital parameters. This must be called after any orbital parameter changes.
    pub fn update(&mut self) {
        // 1. Get the geodetic coordinates of the target to define its local tangent plane.
        let (target_lat, target_lon, _) =
            ecef_to_geodetic(self.target_ecef.x, self.target_ecef.y, self.target_ecef.z);
        let (sin_lat, cos_lat) = target_lat.to_radians().sin_cos();
        let (sin_lon, cos_lon) = target_lon.to_radians().sin_cos();

        // 2. Create the rotation matrix from the local ENU frame at the target back to ECEF.
        let east = DVec3::new(-sin_lon, cos_lon, 0.0);
        let north = DVec3::new(-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat);
        let up = DVec3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat);
        let enu_to_ecef = DMat3::from_cols(east, north, up);

        // 3. Calculate the camera's offset from the target in the local ENU frame
        //    using spherical coordinates (azimuth, elevation, radius).
        let (sin_az, cos_az) = self.azimuth_rad.sin_cos();
        let (sin_el, cos_el) = self.elevation_rad.sin_cos();
        let offset_enu = DVec3::new(
            self.radius_m * cos_el * sin_az, // East
            self.radius_m * cos_el * cos_az, // North
            self.radius_m * sin_el,          // Up
        );

        // 4. Transform the ENU offset back to ECEF and add it to the target's
        //    position to get the final camera position.
        self.position_ecef = self.target_ecef + enu_to_ecef * offset_enu;

        // 5. Update the derived geodetic coordinates for external use (e.g., UI).
        let (lat, lon, h) =
            ecef_to_geodetic(self.position_ecef.x, self.position_ecef.y, self.position_ecef.z);
        self.lat_deg = lat;
        self.lon_deg = lon;
        self.h_m = h;
    }

    /// Sets a new orbit target and radius, then updates the camera state.
    pub fn set_target_and_radius(&mut self, target_ecef: [f64; 3], radius_m: f64) {
        self.target_ecef = DVec3::from(target_ecef);
        self.radius_m = radius_m;
        self.update();
    }

    /// Returns camera position in ECEF meters.
    #[inline]
    pub fn ecef_m(&self) -> [f64; 3] {
        self.position_ecef.into()
    }

    /// Returns rotation matrix from ECEF to ENU for the camera position.
    pub fn ecef_to_enu_matrix(&self) -> Mat3 {
        let lat_rad = self.lat_deg.to_radians();
        let lon_rad = self.lon_deg.to_radians();
        let (sin_lat, cos_lat) = lat_rad.sin_cos();
        let (sin_lon, cos_lon) = lon_rad.sin_cos();

        // East, North, Up basis vectors.
        let east = Vec3::new(-sin_lon as f32, cos_lon as f32, 0.0);
        let north = Vec3::new(
            (-sin_lat * cos_lon) as f32,
            (-sin_lat * sin_lon) as f32,
            cos_lat as f32,
        );
        let up = Vec3::new(
            (cos_lat * cos_lon) as f32,
            (cos_lat * sin_lon) as f32,
            sin_lat as f32,
        );

        Mat3::from_cols(east, north, up).transpose()
    }

    /// Returns combined view‑projection matrix in ECEF meters.
    pub fn view_proj_ecef(&self) -> Mat4 {
        OPENGL_TO_WGPU_MATRIX * self.proj * self.view_ecef()
    }

    /// Returns a rotation-only view matrix that transforms from ECEF to the camera's
    /// local frame. The translation is handled separately in the shader for precision.
    pub fn view_ecef(&self) -> Mat4 {
        // The "forward" vector points from the camera to the target.
        let f = (self.target_ecef - self.position_ecef)
            .normalize()
            .as_vec3();

        // The geodetic "up" vector at the camera's current position.
        let (lat_rad, lon_rad) = (self.lat_deg.to_radians(), self.lon_deg.to_radians());
        let (sin_lat, cos_lat) = (lat_rad.sin() as f32, lat_rad.cos() as f32);
        let (sin_lon, cos_lon) = (lon_rad.sin() as f32, lon_rad.cos() as f32);
        let world_up = Vec3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat);

        // The "side" vector is orthogonal to forward and world_up.
        // f.cross(world_up) gives the "left" vector.
        let s = f.cross(world_up).normalize();

        // The camera's local "up" vector is orthogonal to the side and forward vectors.
        // s.cross(f) gives the "down" vector.
        let u = s.cross(f);

        // The view matrix is the inverse of the camera's basis matrix. For an orthonormal
        // matrix, the inverse is the transpose. The basis columns are [right, up, back].
        // We must use the opposites of s (left) and u (down) to get right and up.
        let rot_mat = Mat3::from_cols(-s, -u, -f).transpose();
        Mat4::from_mat3(rot_mat)
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

    /// Adjusts camera orbit radius based on scroll input.
    fn handle_scroll(&mut self, delta: f32, camera: &mut Camera) {
        // Positive delta = scroll up = zoom in = decrease radius.
        let zoom = 1.1_f64.powf(-delta as f64);
        camera.radius_m *= zoom;
        camera.radius_m = camera.radius_m.clamp(10.0, 1_000_000.0);
        camera.update();
    }

    /// Rotates the camera around the target while the left mouse button is held.
    fn handle_cursor_orbit(&mut self, xy: (f64, f64), camera: &mut Camera) {
        if let Some(last) = self.last_mouse {
            if self.mouse_down {
                let dx = (xy.0 - last.0) * 0.005;
                let dy = (last.1 - xy.1) * 0.005;

                camera.azimuth_rad -= dx;
                camera.elevation_rad -= dy;

                // Clamp elevation to prevent flipping over the poles.
                // 1 degree to 89 degrees.
                camera.elevation_rad = camera
                    .elevation_rad
                    .clamp(1.0f64.to_radians(), 89.0f64.to_radians());

                camera.update();
            }
        }
        self.last_mouse = Some(xy);
    }
}
