// src/camera.rs
use glam::{Mat4, Vec3};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};

pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_rad: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    pub fn projection_matrix_gl(&self, aspect_ratio: f32) -> Mat4 {
        Mat4::perspective_rh_gl(self.fov_y_rad, aspect_ratio, self.near, self.far)
    }
}

pub struct CameraController {
    mouse_down: bool,
    last_mouse: Option<(f64, f64)>,
    zoom_factor: f32,
    default_distance: f32,
}

impl CameraController {
    pub fn new(default_distance: f32) -> Self {
        Self {
            mouse_down: false,
            last_mouse: None,
            zoom_factor: 1.0,
            default_distance,
        }
    }

    pub fn set_default_distance(&mut self, distance: f32) {
        self.default_distance = distance;
    }

    pub fn reset_zoom(&mut self) {
        self.zoom_factor = 1.0;
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
                    MouseScrollDelta::PixelDelta(pos) => (pos.y as f32) / 120.0,
                };
                self.handle_scroll(scroll_delta, camera);
            }
            _ => {}
        }
    }

    fn handle_scroll(&mut self, delta: f32, camera: &mut Camera) {
        const ZOOM_STEP: f32 = 1.20;
        let scale = ZOOM_STEP.powf(-delta);

        let mut offset = camera.position - camera.target;
        let old_len = offset.length();

        if old_len < 1e-6 {
            let fallback = self.default_distance.max(camera.near * 2.0);
            offset = Vec3::new(0.0, -1.0, 0.0) * fallback;
        }

        offset *= scale;

        let new_len = offset.length();
        let min_distance = (camera.near * 2.0).max(0.10);
        let max_distance = (camera.far * 0.90).max(min_distance);

        if new_len < min_distance {
            offset = offset.normalize_or_zero() * min_distance;
        } else if new_len > max_distance {
            offset = offset.normalize_or_zero() * max_distance;
        }

        camera.position = camera.target + offset;
        if self.default_distance > 0.0 {
            self.zoom_factor = self.default_distance / offset.length();
        }
    }

    fn handle_cursor_orbit(&mut self, xy: (f64, f64), camera: &mut Camera) {
        if let Some(last) = self.last_mouse {
            if self.mouse_down {
                let dx = (xy.0 - last.0) as f32 * 0.01;
                let dy = (xy.1 - last.1) as f32 * 0.01;

                let to_target = (camera.target - camera.position).normalize_or_zero();
                let right = to_target.cross(camera.up).normalize_or_zero();

                let yaw = Mat4::from_axis_angle(camera.up, -dx);
                let mut dir = yaw.transform_vector3(to_target);

                let pitch = Mat4::from_axis_angle(right, -dy);
                dir = pitch.transform_vector3(dir);

                // Avoid flipping over by clamping pitch
                const MIN_Z_NORMALIZED: f32 = 0.05;
                if dir.z.abs() < MIN_Z_NORMALIZED {
                    dir.z = dir.z.signum() * MIN_Z_NORMALIZED;
                    dir = dir.normalize();
                }

                let current_dist = (camera.position - camera.target).length();
                camera.position = camera.target - dir * current_dist;
            }
        }
        self.last_mouse = Some(xy);
    }
}
