use crate::data::types::TileUniformStd140 as TileUniform;
use glam::{DMat3, DVec3, Mat3, Mat4, Vec3};
use hypc::{ecef_to_geodetic, geodetic_to_ecef, split_f64_to_f32_pair};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use std::collections::HashSet;

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
        let (lat, lon, h) = ecef_to_geodetic(
            self.position_ecef.x,
            self.position_ecef.y,
            self.position_ecef.z,
        );
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
        self.proj * self.view_ecef()
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
        // f.cross(world_up) gives the right vector.
        let s = f.cross(world_up).normalize();

        // The camera's local "up" vector is orthogonal to the side and forward vectors.
        // s.cross(f) gives the camera up vector.
        let u = s.cross(f);

        // The view matrix is the inverse of the camera's basis matrix. For an orthonormal
        // matrix, the inverse is the transpose. The basis columns are [right, up, back].
        // Projection already uses WebGPU depth [0, 1]; no OpenGL remapping.
        let rot_mat = Mat3::from_cols(s, u, -f).transpose();
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
            splat_radius_m: 0.0,
        }
    }
}

/// Navigation state is advanced with elapsed time, independently of OS key repeat.
pub struct CameraController {
    pan_down: bool,
    orbit_down: bool,
    last_mouse: Option<(f64, f64)>,
    viewport: [f64; 2],
    keys: HashSet<KeyCode>,
}

impl Default for CameraController {
    fn default() -> Self { Self::new() }
}

impl CameraController {
    pub fn new() -> Self {
        Self {
            pan_down: false, orbit_down: false, last_mouse: None,
            viewport: [1280.0, 720.0], keys: HashSet::new(),
        }
    }

    pub fn set_viewport(&mut self, width: u32, height: u32) {
        self.viewport = [width.max(1) as f64, height.max(1) as f64];
    }

    /// Always forward releases and focus loss, even when UI owns the input.
    pub fn release_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::Focused(false) => {
                self.keys.clear(); self.pan_down = false; self.orbit_down = false;
                self.last_mouse = None;
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Released => {
                if let PhysicalKey::Code(key) = event.physical_key { self.keys.remove(&key); }
            }
            WindowEvent::MouseInput { button, state: ElementState::Released, .. } => {
                if *button == MouseButton::Left { self.pan_down = false; }
                if matches!(button, MouseButton::Right | MouseButton::Middle) { self.orbit_down = false; }
            }
            _ => {}
        }
    }

    pub fn handle_event(&mut self, event: &WindowEvent, camera: &mut Camera) {
        self.release_event(event);
        match event {
            WindowEvent::Resized(size) => self.set_viewport(size.width, size.height),
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if event.state == ElementState::Pressed { self.keys.insert(key); }
                }
            }
            WindowEvent::MouseInput { button, state, .. } => {
                if *button == MouseButton::Left { self.pan_down = *state == ElementState::Pressed; }
                if matches!(button, MouseButton::Right | MouseButton::Middle) {
                    self.orbit_down = *state == ElementState::Pressed;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let xy = (position.x, position.y);
                if let Some(last) = self.last_mouse {
                    if self.orbit_down || (self.pan_down && self.shift()) {
                        camera.azimuth_rad -= (xy.0 - last.0) * 0.004;
                        camera.elevation_rad += (xy.1 - last.1) * 0.004;
                        Self::clamp_angles(camera);
                    } else if self.pan_down {
                        self.pan_pixels(last, xy, camera);
                    }
                }
                self.last_mouse = Some(xy);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y as f64,
                    MouseScrollDelta::PixelDelta(pos) => pos.y / 80.0,
                };
                self.zoom(scroll, camera);
            }
            _ => {}
        }
    }

    fn shift(&self) -> bool {
        self.keys.contains(&KeyCode::ShiftLeft) || self.keys.contains(&KeyCode::ShiftRight)
    }

    fn axis(&self, positive: &[KeyCode], negative: &[KeyCode]) -> f64 {
        (positive.iter().any(|k| self.keys.contains(k)) as i32
            - negative.iter().any(|k| self.keys.contains(k)) as i32) as f64
    }

    pub fn update(&mut self, camera: &mut Camera, dt_s: f64) {
        let dt = dt_s.clamp(0.0, 0.1);
        let x = self.axis(&[KeyCode::KeyD, KeyCode::ArrowRight], &[KeyCode::KeyA, KeyCode::ArrowLeft]);
        let y = self.axis(&[KeyCode::KeyW, KeyCode::ArrowUp], &[KeyCode::KeyS, KeyCode::ArrowDown]);
        let length = x.hypot(y).max(1.0);
        let speed = camera.radius_m * if self.shift() { 1.5 } else { 0.5 };
        if x != 0.0 || y != 0.0 {
            let (right, forward, _) = camera.navigation_basis();
            camera.translate_surface((right * x + forward * y) * (speed * dt / length));
        }
        let rotate = self.axis(&[KeyCode::KeyE], &[KeyCode::KeyQ]);
        let tilt = self.axis(&[KeyCode::KeyR, KeyCode::PageUp], &[KeyCode::KeyF, KeyCode::PageDown]);
        if rotate != 0.0 || tilt != 0.0 {
            camera.azimuth_rad += rotate * dt;
            camera.elevation_rad += tilt * dt * 0.7;
            Self::clamp_angles(camera);
        }
        let zoom = self.axis(&[KeyCode::Equal, KeyCode::NumpadAdd], &[KeyCode::Minus, KeyCode::NumpadSubtract]);
        if zoom != 0.0 { self.zoom(zoom * dt * 8.0, camera); }
    }

    fn clamp_angles(camera: &mut Camera) {
        camera.azimuth_rad = camera.azimuth_rad.rem_euclid(std::f64::consts::TAU);
        camera.elevation_rad = camera.elevation_rad.clamp(3f64.to_radians(), 89.9f64.to_radians());
        camera.update();
    }

    fn ground_hit(&self, xy: (f64, f64), camera: &Camera) -> Option<DVec3> {
        let x = (2.0 * xy.0 / self.viewport[0] - 1.0) / camera.proj.x_axis.x as f64;
        let y = (1.0 - 2.0 * xy.1 / self.viewport[1]) / camera.proj.y_axis.y as f64;
        let ray = camera.view_ecef().transpose().transform_vector3(Vec3::new(x as f32, y as f32, -1.0)).as_dvec3().normalize();
        let (_, _, up) = camera.navigation_basis();
        let denom = ray.dot(up);
        if denom >= -0.025 { return None; }
        let eye = DVec3::from(camera.ecef_m());
        let t = (camera.target_ecef - eye).dot(up) / denom;
        if !(0.0..camera.radius_m * 40.0).contains(&t) { return None; }
        Some(eye + ray * t)
    }

    fn pan_pixels(&self, from: (f64, f64), to: (f64, f64), camera: &mut Camera) {
        if let (Some(a), Some(b)) = (self.ground_hit(from, camera), self.ground_hit(to, camera)) {
            camera.translate_surface(a - b);
        } else {
            let (right, forward, _) = camera.navigation_basis();
            let m_per_px = 2.0 * camera.radius_m / (camera.proj.y_axis.y as f64 * self.viewport[1]);
            camera.translate_surface((right * (from.0 - to.0) + forward * (to.1 - from.1)) * m_per_px);
        }
    }

    fn zoom(&self, delta: f64, camera: &mut Camera) {
        let before = self.last_mouse.and_then(|xy| self.ground_hit(xy, camera));
        camera.radius_m = (camera.radius_m * (-delta * 0.16).exp()).clamp(5.0, 150_000.0);
        camera.update();
        if let (Some(before), Some(xy)) = (before, self.last_mouse) {
            if let Some(after) = self.ground_hit(xy, camera) {
                camera.translate_surface(before - after);
            }
        }
    }
}

impl Camera {
    /// Screen right and forward on the target tangent plane, with geodetic up.
    pub fn navigation_basis(&self) -> (DVec3, DVec3, DVec3) {
        let (lat, lon, _) = ecef_to_geodetic(self.target_ecef.x, self.target_ecef.y, self.target_ecef.z);
        let (slat, clat) = lat.to_radians().sin_cos();
        let (slon, clon) = lon.to_radians().sin_cos();
        let east = DVec3::new(-slon, clon, 0.0);
        let north = DVec3::new(-slat * clon, -slat * slon, clat);
        let up = DVec3::new(clat * clon, clat * slon, slat);
        let (s, c) = self.azimuth_rad.sin_cos();
        (-east * c + north * s, -east * s - north * c, up)
    }

    /// Reproject each pan onto the same geodetic height, avoiding tangent-plane drift.
    pub fn translate_surface(&mut self, displacement: DVec3) {
        let (_, _, height) = ecef_to_geodetic(self.target_ecef.x, self.target_ecef.y, self.target_ecef.z);
        let p = self.target_ecef + displacement;
        let (lat, lon, _) = ecef_to_geodetic(p.x, p.y, p.z);
        self.target_ecef = DVec3::from(geodetic_to_ecef(lat, lon, height));
        self.update();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_uses_webgpu_depth_without_a_second_remap() {
        let camera = Camera::new(
            52.513,
            13.375,
            800.0,
            Mat4::perspective_rh(60f32.to_radians(), 1.0, 10.0, 10000.0),
        );
        let forward = (camera.target_ecef - camera.position_ecef)
            .normalize()
            .as_vec3();
        for (distance, expected) in [(10.0, 0.0), (10000.0, 1.0)] {
            let clip = camera.view_proj_ecef() * (forward * distance).extend(1.0);
            assert!((clip.z / clip.w - expected).abs() < 2e-6);
            assert!((clip.w - distance).abs() < distance * 2e-6);
        }
    }
    fn navigation_camera() -> Camera {
        let mut camera = Camera::new(52.516275, 13.3777, 1000.0,
            Mat4::perspective_infinite_reverse_rh(60f32.to_radians(), 16.0 / 9.0, 0.1));
        camera.set_target_and_radius(geodetic_to_ecef(52.516275, 13.3777, 40.0), 1000.0);
        camera.elevation_rad = 60f64.to_radians(); camera.update(); camera
    }

    #[test]
    fn navigation_is_time_based_and_diagonal_speed_is_normalized() {
        let run = |fps: usize, diagonal: bool| {
            let mut c = navigation_camera(); let start = c.target_ecef;
            let mut input = CameraController::new(); input.keys.insert(KeyCode::KeyW);
            if diagonal { input.keys.insert(KeyCode::KeyD); }
            for _ in 0..fps { input.update(&mut c, 1.0 / fps as f64); }
            (c.target_ecef, c.target_ecef.distance(start))
        };
        let a = run(60, false); let b = run(144, false); let diagonal = run(60, true);
        assert!(a.0.distance(b.0) < 0.01);
        assert!((a.1 - 500.0).abs() < 0.02);
        assert!((a.1 - diagonal.1).abs() < 0.02);
    }

    #[test]
    fn dragging_preserves_height_and_moves_ground_with_cursor() {
        let mut c = navigation_camera(); let input = CameraController::new();
        let before = input.ground_hit((640.0, 360.0), &c).unwrap();
        input.pan_pixels((640.0, 360.0), (740.0, 410.0), &mut c);
        let after = input.ground_hit((740.0, 410.0), &c).unwrap();
        assert!(before.distance(after) < 0.02);
        for _ in 0..1000 { let (right, _, _) = c.navigation_basis(); c.translate_surface(right * 100.0); }
        let (_, _, height) = ecef_to_geodetic(c.target_ecef.x, c.target_ecef.y, c.target_ecef.z);
        assert!((height - 40.0).abs() < 0.001);
    }

    #[test]
    fn cursor_anchored_zoom_and_focus_loss() {
        let mut c = navigation_camera(); let mut input = CameraController::new();
        input.last_mouse = Some((800.0, 450.0));
        let before = input.ground_hit(input.last_mouse.unwrap(), &c).unwrap();
        input.zoom(1.0, &mut c);
        let after = input.ground_hit(input.last_mouse.unwrap(), &c).unwrap();
        assert!(before.distance(after) < 0.02);
        assert!(c.radius_m < 1000.0);
        input.keys.insert(KeyCode::KeyW); input.pan_down = true; input.orbit_down = true;
        input.release_event(&WindowEvent::Focused(false));
        let before = c.target_ecef; input.update(&mut c, 0.1);
        assert_eq!(c.target_ecef, before); assert!(!input.pan_down && !input.orbit_down);
    }

    #[test]
    fn reverse_z_retains_depth_order_from_street_to_city_scale() {
        let camera = navigation_camera();
        let forward = (camera.target_ecef - camera.position_ecef).normalize().as_vec3();
        let mut last = f32::INFINITY;
        for distance in [0.1, 5.0, 100.0, 10_000.0, 150_000.0] {
            let p = camera.view_proj_ecef() * (forward * distance).extend(1.0);
            let z = p.z / p.w;
            assert!(z > 0.0 && z < last);
            assert!((z - 0.1 / distance).abs() < 1e-5);
            last = z;
        }
    }

}
