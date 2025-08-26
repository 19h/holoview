# hypc

[![Crates.io](https://img.shields.io/crates/v/hypc.svg)](https://crates.io/crates/hypc)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Rust crate for reading and writing **HPC1** point clouds, with first-class support for **SMC1** semantic masks and **GEOT** georeferencing chunks.

This crate provides a simple, fast, and safe way to interact with `.hpc` files. It is designed to be lightweight, with minimal dependencies (`anyhow` for error handling and `miniz_oxide` for zlib compression).

## Features

*   **HPC1 Core Support**: Read and write the base HPC1 point cloud format, including header and quantized position data.
*   **TileKey Integration**: Natively handles `TileKey` metadata, allowing for slippy-map style `(zoom, x, y)` keys or 64-bit name hashes to be embedded directly in the file header.
*   **Semantic Masks (SMC1)**: Full support for reading and writing the `SMC1` chunk. This allows for a 2D semantic label grid to be associated with the point cloud, useful for classification and segmentation tasks. Supports both raw and zlib-compressed mask data.
*   **Georeferencing (GEOT)**: Full support for reading and writing the `GEOT` chunk. This provides a geographic bounding box (`CRS:84`) for the point cloud, enabling coordinate transformations between the local decode space and longitude/latitude.
*   **High-Level Helpers**: Provides convenient methods for common tasks, such as:
    *   Looking up a semantic class ID from a point's `(x, y)` coordinate.
    *   Converting between geographic coordinates (lon/lat) and the point cloud's local decode space.
    *   Looking up a semantic class ID directly from a lon/lat coordinate.
*   **Robust Parsing**: Gracefully skips unknown trailing chunks, ensuring forward compatibility with future extensions to the format.

## Format Specification

The `.hpc` file format as implemented by this crate consists of a main HPC1 header and payload, followed by zero or more tagged chunks.

### Main HPC1 Format

The file begins with a fixed-size 52-byte header, followed by the point data payload.

| Offset | Size (bytes) | Type          | Field        | Description                                                               |
| :----- | :----------- | :------------ | :----------- | :------------------------------------------------------------------------ |
| 0      | 4            | `[u8; 4]`     | **Magic**    | Must be `b"HPC1"`.                                                        |
| 4      | 4            | `u32` (LE)    | **Version**  | Currently must be `1`.                                                    |
| 8      | 4            | `u32` (LE)    | **Flags**    | Bitfield. Bit 0 (`1 << 0`) indicates a `TileKey` is present.              |
| 12     | 4            | `u32` (LE)    | **Count**    | The number of points in the cloud.                                        |
| 16     | 1            | `u8`          | **QBits**    | Quantization bits per component. Currently must be `16`.                  |
| 17     | 11           | `[u8; 11]`    | **Reserved** | Used for `TileKey` payload if `Flags` bit 0 is set. Otherwise, zeroed.    |
| 28     | 12           | `[f32; 3]`    | **Decode Min** | The minimum `[x, y, z]` corner of the point cloud's bounding box.         |
| 40     | 12           | `[f32; 3]`    | **Decode Max** | The maximum `[x, y, z]` corner of the point cloud's bounding box.         |
| 52     | `Count * 6`  | `[u16; N*3]`  | **Payload**  | Quantized positions. Each point is `(qx, qy, qz)` as little-endian `u16`. |

#### TileKey Payload

When `Flags` bit 0 is set, the 11-byte `Reserved` field is used to store a `TileKey`. The first byte of the reserved field acts as a type discriminator.

**`TileKey::XY` (Type `0`)**
| Offset in Reserved | Size | Type      | Description |
|:------------------ |:---- |:--------- |:----------- |
| 0                  | 1    | `u8`      | Type (`0`) |
| 1                  | 1    | `u8`      | Zoom level |
| 2                  | 4    | `u32` (LE)| X coordinate |
| 6                  | 4    | `u32` (LE)| Y coordinate |
| 10                 | 1    | `u8`      | Scheme |

**`TileKey::NameHash64` (Type `4`)**
| Offset in Reserved | Size | Type      | Description |
|:------------------ |:---- |:--------- |:----------- |
| 0                  | 1    | `u8`      | Type (`4`) |
| 1                  | 8    | `u64` (LE)| 64-bit hash value |
| 9                  | 2    | `[u8; 2]` | Unused (zeroed) |

### Trailing Chunks

After the main HPC1 payload, any number of tagged chunks can appear. The format is designed to be extensible; unknown chunks are skipped using their provided length.

Each chunk follows a simple `[TAG][LENGTH][PAYLOAD]` structure:
*   **Tag**: A 4-byte ASCII identifier (e.g., `b"SMC1"`, `b"GEOT"`).
*   **Length**: A `u32` (LE) specifying the size of the payload in bytes.
*   **Payload**: The chunk-specific data.

#### SMC1 Chunk (Semantic Mask)

The `SMC1` chunk provides a 2D classification grid that maps onto the point cloud's XY plane.

**Tag**: `b"SMC1"`

**Payload Layout (Version 1)**:
| Field            | Type          | Description                                                               |
| :--------------- | :------------ | :------------------------------------------------------------------------ |
| Version          | `u8`          | `1` for the current version.                                              |
| Encoding         | `u8`          | `0` for Raw (uncompressed), `1` for Zlib-compressed.                      |
| Width            | `u16` (LE)    | Width of the mask grid in pixels.                                         |
| Height           | `u16` (LE)    | Height of the mask grid in pixels.                                        |
| Coord Space      | `u8`          | `0` indicates the mask maps to the point cloud's decode XY space.         |
| Class Count      | `u8`          | Number of entries in the palette.                                         |
| Reserved         | `u16` (LE)    | Zeroed.                                                                   |
| **Palette**      | `[Entry]`     | `Class Count` entries. Each entry is `(class_id: u8, precedence: u8)` followed by 2 reserved bytes. |
| Data Length      | `u32` (LE)    | The length of the following data block in bytes.                          |
| **Data**         | `[u8]`        | The mask data (row-major), possibly zlib-compressed. After decompression, its size is `Width * Height`. Each byte is a `class_id`. |

#### GEOT Chunk (Georeferencing)

The `GEOT` chunk provides a geographic bounding box for the point cloud, linking it to real-world coordinates.

**Tag**: `b"GEOT"`

**Payload Layout (Version 1, CRS:84)**:
This is a fixed-size 24-byte payload.
| Field            | Type         | Description                                                            |
| :--------------- | :----------- | :--------------------------------------------------------------------- |
| Version          | `u8`         | `1`.                                                                   |
| CRS ID           | `u8`         | `1` for `CRS:84` (WGS 84, lon/lat in degrees).                         |
| Mode             | `u8`         | `0` for `BBOX_DEG_Q7` (bounding box in degrees with Q7 quantization).   |
| Reserved         | `u8`         | Zeroed.                                                                |
| Lon Min (Q7)     | `i32` (LE)   | Minimum longitude quantized by `1e7`. `(lon_min * 1e7)`                |
| Lat Min (Q7)     | `i32` (LE)   | Minimum latitude quantized by `1e7`. `(lat_min * 1e7)`                 |
| Delta Lon (Q7)   | `u32` (LE)   | Longitude extent quantized by `1e7`. `(lon_max - lon_min) * 1e7`        |
| Delta Lat (Q7)   | `u32` (LE)   | Latitude extent quantized by `1e7`. `(lat_max - lat_min) * 1e7`         |

## Usage

### Add to Your Project

Add `hypc` to your `Cargo.toml`:
```toml
[dependencies]
hypc = "0.1.0"
```

### Reading an `.hpc` File

The easiest way to read a file is with `hypc::read_file`. This returns a `HypcPointCloud` struct containing all parsed data.

```rust
use hypc::HypcPointCloud;

fn main() -> anyhow::Result<()> {
    // Read the entire file into memory.
    let pc: HypcPointCloud = hypc::read_file("path/to/your/cloud.hpc")?;

    println!("Successfully read {} points.", pc.positions.len());

    // The point positions are dequantized and available as f32 triplets.
    if let Some(first_point) = pc.positions.first() {
        println!("First point position: {:?}", first_point);
    }

    // Check for optional TileKey metadata.
    if let Some(tile_key) = pc.tile_key {
        println!("TileKey present: {:?}", tile_key);
    }

    // Check for optional chunks.
    if pc.has_semantics() {
        let sm = pc.semantic_mask.as_ref().unwrap();
        println!(
            "Semantic mask found: {}x{} with {} classes.",
            sm.width,
            sm.height,
            sm.palette.len()
        );
    }

    if pc.has_geot() {
        let bbox = pc.geog_bbox_deg.as_ref().unwrap();
        println!(
            "Geographic BBox (CRS:84): lon({:.6}, {:.6}), lat({:.6}, {:.6})",
            bbox.lon_min, bbox.lon_max, bbox.lat_min, bbox.lat_max
        );
    }

    Ok(())
}
```

### Using Semantic and Geographic Helpers

The real power of `hypc` comes from the high-level methods that combine data from different chunks. For example, you can find the semantic class of any geographic coordinate.

```rust
fn main() -> anyhow::Result<()> {
    let pc = hypc::read_file("path/to/your/georeferenced_and_classified_cloud.hpc")?;

    // We have a geographic coordinate in Dublin, Ireland.
    let lon_dublin = -6.2603;
    let lat_dublin = 53.3498;

    // The `class_of_lonlat` method automatically:
    // 1. Checks if a GEOT chunk exists.
    // 2. Converts the lon/lat to the point cloud's local decode XY space.
    // 3. Checks if an SMC1 chunk exists.
    // 4. Looks up the class ID in the semantic mask at the converted coordinate.
    // 5. Returns 0 if any step fails (e.g., no chunk, coordinate out of bounds).
    let class_id = pc.class_of_lonlat(lon_dublin, lat_dublin);

    if class_id != 0 {
        // You can map the class ID back to a meaningful name using the palette.
        let class_name = match class_id {
            1 => "Building",
            2 => "Vegetation",
            3 => "Water",
            _ => "Unknown",
        };
        println!(
            "The class at ({}, {}) is ID {} ({})",
            lon_dublin, lat_dublin, class_id, class_name
        );
    } else {
        println!(
            "No classification data available at ({}, {})",
            lon_dublin, lat_dublin
        );
    }

    Ok(())
}
```

### Writing an `.hpc` File

To write a file, you construct a `HypcWrite` struct with all the necessary data and pass it to `hypc::write_file`.

Note that for writing, you must provide the positions in their **quantized** `u16` form.

```rust
use hypc::{
    HypcWrite, SemanticMask, Smc1Encoding, GeoCrs, GeoExtentDeg, TileKey,
};

fn main() -> anyhow::Result<()> {
    // 1. Define point data. Positions must be quantized u16s.
    // This represents two points: (0, 32767, 65535) and (100, 200, 300).
    let quantized_positions: Vec<u16> = vec![0, 32767, 65535, 100, 200, 300];
    let decode_min = [0.0, 0.0, -10.0];
    let decode_max = [100.0, 100.0, 50.0];

    // 2. Define optional semantic mask data.
    let smc1 = SemanticMask {
        width: 2,
        height: 2,
        // Class IDs for a 2x2 grid.
        data: vec![1, 2, 1, 3], // Building, Vegetation, Building, Water
        palette: vec![(1, 10), (2, 20), (3, 30)], // (class_id, precedence)
        coord_space: 0,
        // Let the writer handle zlib compression for smaller file size.
        encoding: Smc1Encoding::Zlib,
    };

    // 3. Define optional georeferencing data.
    let geog_bbox_deg = GeoExtentDeg {
        lon_min: -6.26,
        lat_min: 53.34,
        lon_max: -6.25,
        lat_max: 53.35,
    };

    // 4. Define optional TileKey.
    let tile_key = TileKey::XY { zoom: 15, x: 16383, y: 10878, scheme: 0 };

    // 5. Assemble the write parameters.
    let params = HypcWrite {
        quant_bits: 16,
        quantized_positions: &quantized_positions,
        decode_min,
        decode_max,
        tile_key: Some(tile_key),
        geog_crs: Some(GeoCrs::Crs84),
        geog_bbox_deg: Some(geog_bbox_deg),
        smc1: Some(&smc1),
    };

    // 6. Write to file.
    hypc::write_file("path/to/output.hpc", &params)?;
    println!("Successfully wrote output.hpc");

    Ok(())
}
```

## License

This project is licensed under the [MIT License](LICENSE).
