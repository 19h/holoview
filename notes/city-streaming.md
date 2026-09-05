# Full Berlin streaming and navigation — 2026-09-05

The native viewer opens the complete available Berlin 2025 dataset: **8,086 source tiles, 3,697,194,950 points**, represented by 56,601 hierarchy nodes. The two publisher-denied tiles remain unavailable; see [download provenance](../../files/ber2025/README.md). No source geometry was changed for this streaming implementation.

Run from `holographic-viewer`:

```sh
cargo run --release --bin holographic_viewer
# Explicit equivalent:
cargo run --release --bin holographic_viewer -- --dataset ../files/ber2025/city-lod
```

The prepared catalog and 41,610,148,504-byte point pack are installed at `../files/ber2025/city-lod`. Finest detail is read from `../files/ber2025/full/hypc` (48,078,260,595 source bytes). The catalog stores the absolute source directory: retain that directory, or regenerate the cache after relocation. Preparation is a one-time operation, separate from interactive loading:

```sh
cargo run --release --bin prepare_city -- SOURCE_HYPC_DIRECTORY NEW_CACHE_DIRECTORY 4
```

| Input | Action |
|---|---|
| Left drag; WASD; arrow keys | Move along the map |
| Cmd + left drag; Ctrl + left drag; right/middle drag; Shift + left drag | Rotate horizontally; drag vertically to tilt |
| Q / E | Rotate |
| R / F; Page Up / Page Down | Tilt |
| Wheel / trackpad scroll; + / − | Zoom |
| Shift + movement | Faster movement |
| Fit city; North; Places | Reframe, reset heading, or visit named locations |
| F12 | Save a framebuffer capture |

Keyboard motion integrates elapsed seconds and normalizes diagonals. Panning preserves geodetic height before following the approximate navigation surface. Cursor-anchored zoom uses the target tangent plane. Elevation spans 3–89.9°, and orbit range spans 5–150,000 m. Focus loss clears held inputs; UI-consumed releases and modifier events still synchronize navigation state.

## Runtime design

* Global ECEF voxel sampling retains representative source points and their labels. Per-tile levels at 1, 2, 4, 8 and 16 m feed a spatial binary hierarchy. Source HYPC remains the finest level.
* Projected sample spacing and view-frustum tests prioritize refinement. Hysteresis suppresses repeated level changes near a threshold. The default selected draw cap is 3,000,000 points and 1,500 draws; sustained slow frames reduce the point budget and sustained fast frames restore it.
* A resident ancestor remains visible until all its visible immediate children are resident. Missing or corrupt detail therefore retains ancestor coverage. An exhaustive 64-state residency test verifies complete, nonduplicated source-descendant coverage on a branching fixture.
* Approximately 8 MiB of overview points are pinned. Four workers read detail asynchronously; requests are bounded by 96 MiB and eight outstanding nodes. GPU point instances use an independent 512 MiB LRU budget. Upload scheduling stops after 16 MiB or approximately 2 ms per frame, except that one complete node can exceed a single-frame upload target. Frame uploads are not disk reads.
* Camera translation and logarithmic zoom rate predict 0.3 s ahead; an expanded frustum prefetches surrounding detail. Stale queued work is cancelled. Point uniforms are batched into one GPU upload. Reversed infinite depth and split camera-relative ECEF coordinates maintain depth ordering and positional precision across the zoom range.
* A 64 m navigation grid derived from coarse non-building samples supplies immediate ground support and camera clearance. Its 11,409,948-byte mesh is always resident. It is approximate, displayed beneath actual point geometry, and does not overwrite HYPC. Bounded screen-space reconstruction fills small sample gaps; the Rendering panel exposes its toggle.

For N hierarchy nodes and V nodes visited, selection is O(V log V) time and O(V) auxiliary space; V ≤ N. Catalog memory is O(N). Resident point memory and pending work are independently bounded by the configured byte budgets. LRU victim search is O(R) per eviction for R resident nodes. Preprocessing uses expected O(P) voxel insertion plus O(K log K) ordering per sampling pass for P input points and K occupied cells; working memory is bounded by the worker count and largest tile, plus hierarchy metadata. Pixel reconstruction cost scales with viewport area and its fixed sampling stencil, not city point count.

## Measured evidence

Apple M4 Max, Metal, release build, 2560 × 1440 framebuffer, local SSD, existing filesystem cache. These are observed runs, not hardware-independent latency bounds. Frame times measure intervals between frame starts and include presentation pacing; timestamp probes separately measure the geometry pass. The report excludes the first five frames from timing percentiles.

| Run | Frames | Median | p95 | p99 |
|---|---:|---:|---:|---:|
| City / close-up / low-angle / distant-site tour | 900 | 8.33 ms | 9.26 ms | 16.03 ms |
| 64 MiB point-cache eviction stress | 2,400 | 8.34 ms | 9.80 ms | 16.31 ms |
| Brandenburg Gate close-up | 240 | 8.33 ms | 9.00 ms | 16.61 ms |

Dataset activation took 0.039 s in the recorded tour. Every one of its 900 GPU readbacks contained actual source-point pixels; none were skipped. Minimum source coverage was 839,578 pixels and minimum displayed coverage was 843,656 pixels. These counts detect blank frames; they do not prove every pixel is geographically accurate. The resident-cut invariant supplies the separate hierarchy coverage guarantee.

The stress run performed 18,556 evictions, reported zero loading failures, and peaked at 67,108,800 point-cache bytes, below 64 × 2²⁰ = 67,108,864 bytes. Pending point bytes peaked at 34,234,496. Terrain, render targets, uniforms, driver allocations and OS filesystem cache are additional memory, excluded from this point-cache cap.

The close-up converged to its selected detail at frame 54, with 2,984,614 visible points and no pending refinement. Immediate camera response and persistent coarse coverage do not imply zero latency for finest detail; refinement is asynchronous and constrained by the selected quality budget.

Native macOS event injection exercised 17 actions through the actual window event loop. Cmd-drag and Ctrl-drag each changed azimuth by −22.0016° and elevation by +5.50038°, with zero target translation. WASD, all arrows, Q/E, R/F, left drag, right drag and wheel zoom also produced their corresponding motion. Automated regression checks passed: 19 library tests and two production-GPU integration tests.

Primary implementation and machine-readable evidence:

* [Hierarchy preparation and validation](../src/data/dataset.rs), [stream scheduler](../src/data/streaming.rs), [navigation](../src/camera.rs).
* [Tour measurements](../../files/ber2025/validation/final-tour.json), [eviction measurements](../../files/ber2025/validation/final-stress.json), [close-up measurements](../../files/ber2025/validation/final-close.json).
* [Native input evidence](../../files/ber2025/validation/native-verification-latest.json), [native event driver](../scripts/verify_native_input.swift).
* [Full-data integrity verification](../../files/ber2025/validation/full-city-integrity.json), [independent verifier](../scripts/verify_city_data.py).
* [Overview capture](../../files/ber2025/validation/final-overview.png), [close-up capture](../../files/ber2025/validation/final-close.png).

```sh
cargo test --lib --tests
cargo run --release --bin holographic_viewer -- --tour --verify-coverage --report tour.json --capture overview.png
cargo run --release --bin holographic_viewer -- --stress --gpu-mib 64 --report stress.json
python3 scripts/verify_city_data.py ../files/ber2025/city-lod ../files/ber2025/converted-manifest.json integrity.json
```

## Assumption register and bounded scope

| ID | Assumption and dependent result | Falsification probe / limit | Impact |
|---|---|---|---|
| A1 | Available indexed data defines this installed city: 8,086 tiles. | Compare catalog source paths against all conversion receipts. Two indexed objects return HTTP 403; their geometry is unknown. | High |
| A2 | Source files and cache remain unchanged. Detail fidelity and resumed reads depend on this. | Independently SHA-256-check every HYPC against its conversion receipt and CRC32-check every packed node; runtime also checks source size/mtime and decoded CRC. | High |
| A3 | Recorded local SSD / M4 Max behavior represents this run. Timing results depend on it. | Cold-cache storage, other GPUs, thermal state and high-contention workloads remain unmeasured. Frame feedback tests exercise sustained slowdown and recovery. | High |
| A4 | Coarse non-building samples approximate a traversable support surface. Camera clearance and underlay depend on it. | Grid interpolation/continuity tests and low-angle tour; bridges, tunnels, roof misclassification and steep local terrain can disagree. This is not a collision mesh or surveyed DTM. Vertical datum remains unknown; source heights are preserved. | High |
| A5 | Projected sample spacing is a useful visual-quality proxy. Detail selection depends on it. | Close-up and overview inspection, point/draw-budget tests and moving tour. Thin features may disappear at coarse LOD; lower spacing increases requested refinement within the point cap. | Medium |
| A6 | Source semantics are usable for navigation and coloring. | Prior shared-cell semantic audit; unlabeled source regions remain unknown. Inferred underlay coverage can bridge small acquisition gaps and must not be interpreted as observed geometry. | Medium |

Additional bounded findings: cache storage is substantial (41.61 GB beyond 48.08 GB HYPC), while runtime point residency is independent of those disk totals [medium]. A portable cache would require relative source-root relocation support [medium]. Terrain support avoids the former below-ground ECEF-box-center camera target, but does not establish absolute height accuracy [high].

Quality gates: QG1 descriptive technical work; QG2 assumptions and probes above; QG3 full available-city hierarchy, native controls, refinement and bounded streaming verified; QG4 metre/radian/second/byte definitions and reproducible measurements; QG5 coverage, corruption, focus-loss, budget and depth edge cases tested with remaining observational limits stated; QG6 primary source files, receipts, captures and measured JSON linked; QG7 bounded extensions and impacts recorded. Completion of the full-data integrity pass is recorded in its linked result.
