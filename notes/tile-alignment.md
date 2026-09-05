# Tile alignment: implementation and verification

The converter and renderer introduced several independent tile boundaries. The final implementation replaces the per-tile coordinate correction, mesh-vertex sampling, and per-tile semantic sampling. All 63 local HYPC files in `hypc`, `hypc2`, and `hypcx` have been regenerated and installed. The HYPC v2 representation remains compatible.

[Before/after, identical camera and current renderer](alignment/before-after.png) · [Final colored render](alignment/final-after.png) · [49-tile geometry render](alignment/final-hypc2.png) · [8-tile geometry render](alignment/final-hypcx.png)

## Causes and implemented changes

1. **Coordinates:** the old converter centered each mesh's projected bounding box on its geographic bounding-box center and approximated metre offsets as latitude/longitude increments. This discarded projection convergence and introduced tile-dependent translation/rotation. The new converter requires explicit georeferencing and uses PROJ with a shared source CRS. Berlin sources use EPSG:25833. No bounding-box fitting or estimated registration remains.
2. **Sampling:** raw OBJ vertices inherit triangulation density and concentrations at tile cuts. Surface mode now samples triangle interiors on a globally anchored ECEF lattice, with 0.5 m spacing by default. Changing the triangle diagonal, winding or tile cut does not change samples on the same plane. Quantized duplicates are removed. Point-only OBJ input remains available through explicit `--sampling vertices`.
3. **Semantics:** clipping the OSM node set omitted geometry crossing/enclosing tiles; clamping polygon/polyline vertices to the mask perimeter manufactured border features. Complete ways are retained, selected by buffered geometry bounds, and rasterized without vertex clamping. The production labels are now stored directly per point using a globally anchored 0.000005° semantic grid, approximately 0.34 × 0.56 m in Berlin. Deterministic precedence removes dependence on feature ordering. Legacy SMC1 masks remain auxiliary; direct labels take precedence in the viewer.
4. **Rendering:** the eye-dome lighting pass previously used half-precision nonlinear depth. It now receives 32-bit linear eye depth in metres. Geometry/background classification uses the geometry tag at every distance. An erroneous second OpenGL-to-WebGPU depth remapping was removed. Offset conversion rounds to f32 only after f64 scaling; the shader adds the low residual before the high camera-relative term.
5. **File integrity:** HYPC parsing accepts unaligned input byte slices. Writes validate sizes and use a buffered temporary file followed by atomic replacement. Invalid coordinates, ambiguous CRS selection, impossible quantization, and conversion failures are reported rather than silently accepted.

## Quantitative evidence

| Collection | Tiles | Original vertices | Installed surface points | Shared semantic-cell comparisons | Label disagreements |
|---|---:|---:|---:|---:|---:|
| hypc | 6 | 3,982,546 | 4,536,831 | 2,774 | 0 |
| hypc2 | 49 | 33,242,668 | 39,269,530 | 33,891 | 0 |
| hypcx | 8 | 5,893,881 | 5,828,083 | 4,246 | 0 |
| Total | 63 | 43,119,095 | 49,634,444 | 40,911 | 0 |

The rebuild manifest contains SHA-256 hashes of sources, backups and installed files. Every installed file was checked against its final manifest hash.

Independent `cs2cs` controls covered 258,048 original vertices: maximum per-axis ECEF error was less than 0.500 mm. Another 258,048 surface samples were checked by point-to-triangle distance against independently converted, quantized source meshes. Maximum distance was 1.07 mm, below the combined 1.73 mm quantization bound. These are conversion consistency measurements, not survey-accuracy claims.

For the five default-dataset boundaries with source vertices agreeing within 2 mm, old median separations were 7.367–7.450 m. The corrected vertex references have zero median separation; their largest separation is 2.18 mm, including source mismatch. Broader source correspondence checks include all seven adjacent default-dataset boundaries. Source meshes need not have identical boundary vertices.

At longitude 13.375°, latitude 52.513°, PROJ reports grid convergence −1.28955261°. The scale of the observed defect follows `330 m × sin(1.28955261°) ≈ 7.43 m`.

For lattice resolution `u` in units/m, `q = round(u p)`, `a = round(u centroid)`, and `d = q − a`. Decoding gives `(a+d)/u = q/u`, independently of the anchor. Each axis has rounding error at most `0.5/u m`; a pair's combined Euclidean bound is `sqrt(3)/2 × (1/u₁ + 1/u₂) m`, excluding source disagreement.

## Verification and reproduction

Seventeen automated tests passed: 11 converter unit tests, one converter CLI integration test, two HYPC tests, one camera test and two real-GPU integration tests. The GPU tests exercise the production loader and shader on Metal, including different anchors/resolutions, three geographic locations, three elevations, correct near/far projection, and semantic visibility at 1,000,000 m eye depth. Anchor-dependent screen displacement stayed below the test limit of 0.02 pixels.

The production point and post-processing pipelines rendered all three final collections. The images above were inspected for the tile patterns identified in the supplied screenshot. Geometry-only and colored passes were checked separately.

Build dependencies: Rust; PROJ >= 9.6.2 for the converter (`pkg-config` locates the installed library). Viewing existing HYPC files does not require PROJ. Audit scripts require Python, NumPy, SciPy and the PROJ `cs2cs` executable.

```sh
cargo test
cargo test --manifest-path crates/hypc/Cargo.toml --all-features
cargo test --manifest-path crates/obj2hypc/Cargo.toml
cargo build --release --manifest-path crates/obj2hypc/Cargo.toml
python3 scripts/rebuild_berlin_tiles.py \
  --dataset hypc --dataset hypc2 --dataset hypcx \
  --sources zips3 --sources zips2 --sources ../files/berlin_tiles \
  --feature-index crates/obj2hypc/mesh-index-2023.json \
  --osm-pbf crates/obj2hypc/berlin-2025-08-24.osm.pbf --install
cargo run --release
```

The rebuild script stages and validates vertex references, surface samples and shared semantic labels before installation. Original files are retained under `target/alignment/rebuild/<collection>/original`; preserve that directory before running `cargo clean`. Detailed manifests and numerical audit results are under the same rebuild directory. `examples/render_tiles.rs` captures the production pipeline without a window and accepts an explicit camera pose for comparisons.

## Assumption register and bounded scope

| ID | Assumption / scope | Falsification probe | Dependent results |
|---|---|---|---|
| A1 | The supplied Berlin OBJ horizontal coordinates use EPSG:25833. | Compare the source geographic footprint and independent control coordinates; verify against a surveyed control point for absolute placement. | Geographic placement and convergence diagnosis. |
| A2 | Source Z is preserved with explicit height offset 0 m. Its vertical datum and coordinate epoch are **unknown**. | Obtain source vertical metadata and benchmark/geoid controls. | Absolute altitude remains unverified; cross-tile consistency does not establish absolute accuracy. |
| A3 | Source triangle surfaces define the retained geometry. | Point-to-triangle controls and differently partitioned planar fixtures. | Surface fidelity. Features below the chosen sampling spacing may be omitted; spacing is configurable. |
| A4 | The regional OSM extract and supported way classes define semantic content. | Compare shared global cells and inspect complete source ways. | Seam consistency, not exhaustive OSM interpretation; multipolygon relations are not newly implemented. |
| A5 | The numerical GPU evidence applies to the exercised Metal backend and camera cases. | Run the GPU tests on another supported backend and inspect corresponding captures. | Reported rendering verification. |

Additional findings: **high impact**—nonlinear half-precision depth hid shape information and per-tile mask phase caused color discontinuities even after coordinate repair; both are addressed. **Medium impact**—surface sampling increases these collections' point count by about 15.1%, and absolute vertical datum remains unknown [A2–A3]. **Low impact**—unaligned HYPC byte slices previously failed decoding; regression coverage now includes them.

Surface sampling costs O(F + C + S log S) time and O(S) output space, where F is triangle count, C is tested projected bounding-box cells and S is accepted samples. Semantic indexing costs O(G log G); point labeling costs O(N + K(log G + H)) with caching, where K is occupied semantic cells and H is the candidate-geometry evaluation cost. Proximity verification uses spatial indices and exact triangle-distance evaluation; bounds expansion is used when initial centroid candidates are insufficient.

Quality gates: QG1—technical content only; QG2—assumptions/probes above; QG3—coordinates, data sampling, semantic borders, rendering and installed assets covered; QG4—units and error bounds stated; QG5—source limitations explicitly separated from conversion consistency; QG6—local source files, test results, manifests and primary references below; QG7—bounded additional findings documented.

Primary references: [PROJ UTM documentation](https://proj.org/en/stable/operations/projections/utm.html), [PROJ Transverse Mercator formulation](https://proj.org/en/stable/operations/projections/tmerc.html), [Rust PROJ normalized axis order](https://docs.rs/proj/0.31.0/proj/struct.Proj.html), [Berlin download portal](https://www.businesslocationcenter.de/berlin3d-downloadportal/). The current portal configuration identifies EPSG:25833; applying it to the older local mesh is additionally supported by the supplied 2023 feature index and coordinate controls [A1].
