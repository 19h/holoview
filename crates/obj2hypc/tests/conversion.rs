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
