use std::{fs, path::Path, process::Command};

fn run(input: &Path, output: &Path, options: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_obj2hypc"))
        .arg("--input-dir")
        .arg(input)
        .arg("--output-dir")
        .arg(output)
        .args(["--sampling", "vertices"])
        .args(options)
        .output()
        .unwrap()
}

#[test]
fn cli_requires_georeferencing_and_preserves_cross_tile_points() {
    let root = std::env::temp_dir().join(format!("hypc-converter-test-{}", std::process::id()));
    let input = root.join("source");
    let output = root.join("output");
    fs::create_dir_all(&input).unwrap();
    fs::write(
        input.join("west.obj"),
        "v 389400 5819300 40\nv 389700 5819300 45\n",
    )
    .unwrap();
    fs::write(
        input.join("east.obj"),
        "v 389700 5819300 45\nv 390000 5819300 80\n",
    )
    .unwrap();
    let ambiguous = run(&input, &output, &[]);
    assert!(!ambiguous.status.success());
    assert!(!output.join("west.hypc").exists());
    let local = run(&input, &output, &["--input-cs", "local-m"]);
    assert!(!local.status.success());
    let converted = run(&input, &output, &["--source-crs", "EPSG:25833"]);
    assert!(
        converted.status.success(),
        "{}",
        String::from_utf8_lossy(&converted.stderr)
    );
    let a = hypc::read_file(output.join("west.hypc")).unwrap();
    let b = hypc::read_file(output.join("east.hypc")).unwrap();
    for axis in 0..3 {
        assert_eq!(
            a.anchor_ecef_units[axis] + a.points_units[1][axis] as i64,
            b.anchor_ecef_units[axis] + b.points_units[0][axis] as i64
        );
    }
    assert!(a.geot.is_some() && b.geot.is_some());
    assert!(output.join("west.provenance.json").exists());
    let bytes = fs::read(output.join("west.hypc")).unwrap();
    let duplicate = run(&input, &output, &["--source-crs", "EPSG:25833"]);
    assert!(!duplicate.status.success());
    assert_eq!(bytes, fs::read(output.join("west.hypc")).unwrap());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn accepts_2025_bom_index_and_coordinate_filenames() {
    let root = std::env::temp_dir().join(format!("hypc-2025-test-{}", std::process::id()));
    let input = root.join("source");
    let output = root.join("output");
    fs::create_dir_all(&input).unwrap();
    fs::write(
        input.join("3898_58196_-002.obj"),
        "v 389872.7 5819778.5 35.5\n",
    )
    .unwrap();
    let index = root.join("index.json");
    fs::write(&index, "\u{feff}{\"type\":\"FeatureCollection\",\"features\":[{\"type\":\"Feature\",\"properties\":{\"url\":\"3898_58196_-002.zip\"},\"geometry\":{\"type\":\"Polygon\",\"coordinates\":[[[13.37,52.51],[13.38,52.51],[13.38,52.52],[13.37,52.52],[13.37,52.51]]]}}]}").unwrap();
    let result = run(
        &input,
        &output,
        &[
            "--source-crs",
            "EPSG:25833",
            "--feature-index",
            index.to_str().unwrap(),
        ],
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let tile = hypc::read_file(output.join("3898_58196_-002.hypc")).unwrap();
    assert_eq!(tile.points_units.len(), 1);
    assert!(tile.geot.is_some());
    fs::remove_dir_all(root).unwrap();
}
