//! GPU regression: the same cloud, encoded under different anchors and lattice
//! resolutions, must rasterize at the same location through the actual loader.
#[path = "../examples/support/offscreen.rs"]
mod offscreen;
use glam::Mat4;
use holographic_viewer::{camera::Camera, data::point_cloud::load_hypc_tile};
use hypc::{geodetic_to_ecef, quantize_units, HypcTile};

#[test]
fn tile_partition_does_not_change_rendered_location() {
    let gpu = offscreen::Offscreen::new(512).expect("GPU required for this test");
    let dir = std::env::temp_dir().join(format!("hypc-render-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (lat, lon) in [(52.513, 13.375), (-33.8, 151.2), (0.0, -179.99)] {
        let p = geodetic_to_ecef(lat, lon, 45.0);
        let mut camera = Camera::new(
            lat,
            lon,
            180.0,
            Mat4::perspective_infinite_reverse_rh(60f32.to_radians(), 1.0, 0.1),
        );
        camera.set_target_and_radius(p, 180.0);
        let mut tiles = Vec::new();
        for (index, (upm, shift)) in [
            (1000, [0.0, 0.0, 0.0]),
            (2000, [310.123, -270.777, 183.213]),
        ]
        .into_iter()
        .enumerate()
        {
            let anchor = std::array::from_fn(|axis| quantize_units(p[axis] + shift[axis], upm));
            let point =
                std::array::from_fn(|axis| (quantize_units(p[axis], upm) - anchor[axis]) as i32);
            let tile = HypcTile {
                units_per_meter: upm,
                anchor_ecef_units: anchor,
                tile_key: None,
                points_units: vec![point],
                labels: None,
                geot: None,
                smc1: None,
            };
            let path = dir.join(format!("{index}.hypc"));
            hypc::write_file(&path, &tile).unwrap();
            tiles.push(
                load_hypc_tile(
                    &gpu.device,
                    &gpu.pipeline.tile_layout,
                    &camera,
                    &path,
                    [512.0; 2],
                )
                .unwrap(),
            );
        }
        for elevation in [5.0_f64, 30.0, 85.0] {
            camera.elevation_rad = elevation.to_radians();
            camera.update();
            let a = gpu.render(&tiles[..1], &camera, 6.0).unwrap();
            let b = gpu.render(&tiles[1..], &camera, 6.0).unwrap();
            let centroid = |data: &[u8]| {
                let (mut x, mut y, mut weight) = (0.0, 0.0, 0.0);
                for (i, px) in data.chunks_exact(4).enumerate() {
                    let w = px[3] as f64;
                    x += (i % 512) as f64 * w;
                    y += (i / 512) as f64 * w;
                    weight += w;
                }
                assert!(weight > 0.0, "Point must be visible");
                (x / weight, y / weight)
            };
            let ac = centroid(&a);
            let bc = centroid(&b);
            let error = (ac.0 - bc.0).hypot(ac.1 - bc.1);
            assert!(
                error < 0.02,
                "Anchor-dependent screen displacement {error} px"
            );
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn semantic_geometry_remains_visible_at_large_eye_depth() {
    use holographic_viewer::renderer::pipelines::post_stack::PostParams;
    let gpu = offscreen::Offscreen::new(512).unwrap();
    let path = std::env::temp_dir().join(format!("hypc-depth-test-{}.hypc", std::process::id()));
    let p = geodetic_to_ecef(52.513, 13.375, 45.0);
    let tile = HypcTile {
        units_per_meter: 1000,
        anchor_ecef_units: p.map(|v| quantize_units(v, 1000)),
        tile_key: None,
        points_units: vec![[0; 3]],
        labels: Some(vec![1]),
        geot: None,
        smc1: None,
    };
    hypc::write_file(&path, &tile).unwrap();
    for distance in [100.0, 1_000_000.0] {
        let mut camera = Camera::new(
            52.513,
            13.375,
            distance,
            Mat4::perspective_infinite_reverse_rh(60f32.to_radians(), 1.0, 10.0),
        );
        camera.set_target_and_radius(p, distance);
        let tile = load_hypc_tile(
            &gpu.device,
            &gpu.pipeline.tile_layout,
            &camera,
            &path,
            [512.0; 2],
        )
        .unwrap();
        let params = PostParams {
            edl_on: false,
            sem_on: true,
            sem_amount: 1.0,
            rgb_on: false,
            crt_on: false,
            ..Default::default()
        };
        let image = gpu
            .render_with_post(&[tile], &camera, 6.0, Some(params))
            .unwrap();
        assert!(
            image.chunks_exact(4).any(|px| px[0] > 250
                && (200..=215).contains(&px[1])
                && (98..=106).contains(&px[2])),
            "Building label was classified as background at {distance} m"
        );
    }
    std::fs::remove_file(path).unwrap();
}
