//! hypc: HPC1 reader/writer with SMC1 and GEOT support.
//!
//! Byte-level format derived from the provided reference implementation.
//!
//! Layout (header):
//!   b"HPC1"               (4)
//!   version: u32          (4) = 1
//!   flags: u32            (4) bit0 => tilekey present
//!   count: u32            (4) number of points
//!   qbits: u8             (1) must be 16
//!   reserved: [u8; 11]    (11) tilekey payload if flags bit0 set
//!   decode_min: [f32; 3]  (12)
//!   decode_max: [f32; 3]  (12)
//!   payload: count * 3 * (qbits/8) bytes (for qbits=16 => count*6)
//!
//! Trailing chunks:
//!   "SMC1" + u32(len) + payload(v1)  -> semantic mask; encoding 0(raw) or 1(zlib)
//!   "GEOT" + fixed 24 bytes (v1)     -> bbox in CRS:84 with Q7 (1e-7 deg) quantization.
//!
//! Unknown chunks are assumed to be "TAG + u32 length + payload" for skipping.

use anyhow::{anyhow, bail, Context, Result};
use miniz_oxide::deflate::compress_to_vec_zlib;
use miniz_oxide::inflate::decompress_to_vec_zlib;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

pub const FLAG_TILEKEY_PRESENT: u32 = 1 << 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeoCrs {
    /// CRS:84 (lon, lat) in degrees.
    Crs84,
}

#[derive(Debug, Clone, Copy)]
pub struct GeoExtentDeg {
    pub lon_min: f64,
    pub lat_min: f64,
    pub lon_max: f64,
    pub lat_max: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Smc1Encoding {
    Raw = 0,
    Zlib = 1,
}

impl From<u8> for Smc1Encoding {
    fn from(v: u8) -> Self {
        match v {
            1 => Smc1Encoding::Zlib,
            _ => Smc1Encoding::Raw,
        }
    }
}

impl From<Smc1Encoding> for u8 {
    fn from(e: Smc1Encoding) -> u8 {
        match e {
            Smc1Encoding::Raw => 0,
            Smc1Encoding::Zlib => 1,
        }
    }
}

/// Palette entry: (class_id, precedence)
pub type Smc1ClassEntry = (u8, u8);

#[derive(Debug, Clone)]
pub struct SemanticMask {
    pub width: u16,
    pub height: u16,
    /// Decompressed labels, row-major, one byte per pixel.
    pub data: Vec<u8>,
    pub palette: Vec<Smc1ClassEntry>,
    /// 0 => decode XY (supported by helpers here)
    pub coord_space: u8,
    /// Encoding as stored on disk. When reading, `data` is decompressed, but this
    /// field preserves the on-disk encoding indicator.
    pub encoding: Smc1Encoding,
}

#[derive(Debug, Clone)]
pub enum TileKey {
    /// slippy-like XY plus zoom and a dataset scheme byte.
    XY { zoom: u8, x: u32, y: u32, scheme: u8 },
    /// A 64-bit name hash.
    NameHash64 { hash: u64 },
}

fn read_exact_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}
fn read_exact_u16<R: Read>(r: &mut R) -> Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}
fn read_exact_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_exact_i32<R: Read>(r: &mut R) -> Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}
fn read_exact_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

/// Encode TileKey into flags+reserved field.
pub fn pack_tilekey(tile_key: &TileKey) -> (u32, [u8; 11]) {
    let mut reserved = [0u8; 11];
    match *tile_key {
        TileKey::XY {
            zoom,
            x,
            y,
            scheme,
        } => {
            reserved[0] = 0u8;
            reserved[1] = zoom;
            reserved[2..6].copy_from_slice(&x.to_le_bytes());
            reserved[6..10].copy_from_slice(&y.to_le_bytes());
            reserved[10] = scheme;
        }
        TileKey::NameHash64 { hash } => {
            reserved[0] = 4u8;
            reserved[1..9].copy_from_slice(&hash.to_le_bytes());
        }
    }
    (FLAG_TILEKEY_PRESENT, reserved)
}

/// Decode TileKey from flags+reserved if present/recognized.
pub fn unpack_tilekey(flags: u32, reserved: &[u8; 11]) -> Option<TileKey> {
    if flags & FLAG_TILEKEY_PRESENT == 0 {
        return None;
    }
    match reserved[0] {
        0 => {
            let zoom = reserved[1];
            let x = u32::from_le_bytes(reserved[2..6].try_into().ok()?);
            let y = u32::from_le_bytes(reserved[6..10].try_into().ok()?);
            let scheme = reserved[10];
            Some(TileKey::XY {
                zoom,
                x,
                y,
                scheme,
            })
        }
        4 => {
            let hash = u64::from_le_bytes(reserved[1..9].try_into().ok()?);
            Some(TileKey::NameHash64 { hash })
        }
        _ => None,
    }
}

/// In-memory representation of an HPC1 point cloud.
#[derive(Debug, Clone)]
pub struct HypcPointCloud {
    pub positions: Vec<[f32; 3]>,
    pub decode_min: [f32; 3],
    pub decode_max: [f32; 3],
    pub quant_bits: u8, // currently always 16 on read/write
    pub tile_key: Option<TileKey>,
    pub geog_crs: Option<GeoCrs>,
    pub geog_bbox_deg: Option<GeoExtentDeg>,
    pub semantic_mask: Option<SemanticMask>,
}

impl HypcPointCloud {
    /// O(1) class lookup from a position in decode space (x,y).
    pub fn class_of_xy(&self, x: f32, y: f32) -> u8 {
        let Some(ref sm) = self.semantic_mask else {
            return 0;
        };
        if sm.coord_space != 0 {
            return 0;
        }
        let w = sm.width as f32;
        let h = sm.height as f32;
        let ix = (((x - self.decode_min[0]) * w)
            / (self.decode_max[0] - self.decode_min[0]).max(f32::EPSILON))
            .floor() as i32;
        let iy = (((y - self.decode_min[1]) * h)
            / (self.decode_max[1] - self.decode_min[1]).max(f32::EPSILON))
            .floor() as i32;
        if ix < 0 || iy < 0 || ix >= sm.width as i32 || iy >= sm.height as i32 {
            return 0;
        }
        sm.data[(iy as usize) * sm.width as usize + ix as usize]
    }

    /// Convert lon/lat (deg) to decode XY using GEOT v1 bbox.
    pub fn lonlat_to_decode_xy(&self, lon: f64, lat: f64) -> Option<(f32, f32)> {
        let bb = self.geog_bbox_deg.as_ref()?;
        let sx = (self.decode_max[0] - self.decode_min[0]) as f64;
        let sy = (self.decode_max[1] - self.decode_min[1]) as f64;

        let x = self.decode_min[0] as f64
            + (lon - bb.lon_min) / (bb.lon_max - bb.lon_min + f64::EPSILON) * sx;
        let y = self.decode_min[1] as f64
            + (lat - bb.lat_min) / (bb.lat_max - bb.lat_min + f64::EPSILON) * sy;

        Some((x as f32, y as f32))
    }

    /// Inverse of `lonlat_to_decode_xy`.
    pub fn decode_xy_to_lonlat(&self, x: f32, y: f32) -> Option<(f64, f64)> {
        let bb = self.geog_bbox_deg.as_ref()?;
        let sx = (self.decode_max[0] - self.decode_min[0]) as f64;
        let sy = (self.decode_max[1] - self.decode_min[1]) as f64;

        let lon =
            bb.lon_min + ((x as f64 - self.decode_min[0] as f64) / sx) * (bb.lon_max - bb.lon_min);
        let lat =
            bb.lat_min + ((y as f64 - self.decode_min[1] as f64) / sy) * (bb.lat_max - bb.lat_min);
        Some((lon, lat))
    }

    /// Convenience: class lookup directly in geographic space (deg).
    pub fn class_of_lonlat(&self, lon: f64, lat: f64) -> u8 {
        let Some((x, y)) = self.lonlat_to_decode_xy(lon, lat) else {
            return 0;
        };
        self.class_of_xy(x, y)
    }

    pub fn has_semantics(&self) -> bool {
        self.semantic_mask.is_some()
    }
    pub fn has_geot(&self) -> bool {
        self.geog_bbox_deg.is_some()
    }
}

fn parse_smc1_payload(buf: &[u8]) -> Result<SemanticMask> {
    let mut off = 0usize;

    let version = *buf
        .get(off)
        .ok_or_else(|| anyhow!("SMC1: truncated at version"))?;
    off += 1;
    if version != 1 {
        bail!("SMC1: unsupported version {}", version);
    }

    let encoding_raw = *buf
        .get(off)
        .ok_or_else(|| anyhow!("SMC1: truncated at encoding"))?;
    off += 1;
    let encoding = Smc1Encoding::from(encoding_raw);

    let w = u16::from_le_bytes([
        *buf.get(off).ok_or_else(|| anyhow!("SMC1: truncated at width[0]"))?,
        *buf.get(off + 1).ok_or_else(|| anyhow!("SMC1: truncated at width[1]"))?,
    ]);
    off += 2;
    let h = u16::from_le_bytes([
        *buf.get(off).ok_or_else(|| anyhow!("SMC1: truncated at height[0]"))?,
        *buf.get(off + 1).ok_or_else(|| anyhow!("SMC1: truncated at height[1]"))?,
    ]);
    off += 2;

    let coord_space = *buf
        .get(off)
        .ok_or_else(|| anyhow!("SMC1: truncated at coord_space"))?;
    off += 1;

    let class_count = *buf
        .get(off)
        .ok_or_else(|| anyhow!("SMC1: truncated at class_count"))?;
    off += 1;

    // reserved u16
    off += 2;
    if off > buf.len() {
        bail!("SMC1: header exceeds payload");
    }

    let mut palette = Vec::with_capacity(class_count as usize);
    for _ in 0..class_count {
        let cid = *buf
            .get(off)
            .ok_or_else(|| anyhow!("SMC1: truncated in class table"))?;
        let prec = *buf
            .get(off + 1)
            .ok_or_else(|| anyhow!("SMC1: truncated in class table"))?;
        off += 2;
        // reserved per entry
        off += 2;
        palette.push((cid, prec));
    }

    if off + 4 > buf.len() {
        bail!("SMC1: truncated at data_len");
    }
    let data_len = u32::from_le_bytes([
        buf[off],
        buf[off + 1],
        buf[off + 2],
        buf[off + 3],
    ]) as usize;
    off += 4;

    if off + data_len > buf.len() {
        bail!("SMC1: data length exceeds payload");
    }
    let data_raw = &buf[off..off + data_len];

    let data = match encoding {
        Smc1Encoding::Zlib => decompress_to_vec_zlib(data_raw)
            .map_err(|_| anyhow!("SMC1: zlib inflate failed"))?,
        Smc1Encoding::Raw => data_raw.to_vec(),
    };

    Ok(SemanticMask {
        width: w,
        height: h,
        data,
        palette,
        coord_space,
        encoding,
    })
}

/// Read HPC1 + optional chunks from a path.
pub fn read_file(path: &str) -> Result<HypcPointCloud> {
    let f = File::open(path).with_context(|| format!("open {}", path))?;
    read(f)
}

/// Read HPC1 + optional chunks from an arbitrary reader (must implement Seek for skipping).
pub fn read<R: Read + Seek>(mut r: R) -> Result<HypcPointCloud> {
    // Magic
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"HPC1" {
        bail!("bad magic (expected HPC1)");
    }

    let version = read_exact_u32(&mut r)?;
    if version != 1 {
        bail!("unsupported HPC1 version {}", version);
    }
    let flags = read_exact_u32(&mut r)?;
    let count = read_exact_u32(&mut r)? as usize;
    let qbits = read_exact_u8(&mut r)?;
    if qbits != 16 {
        bail!("unsupported quant_bits != 16");
    }
    let mut reserved = [0u8; 11];
    r.read_exact(&mut reserved)?;
    let dmin = [
        read_exact_f32(&mut r)?,
        read_exact_f32(&mut r)?,
        read_exact_f32(&mut r)?,
    ];
    let dmax = [
        read_exact_f32(&mut r)?,
        read_exact_f32(&mut r)?,
        read_exact_f32(&mut r)?,
    ];

    // Quantized payload
    let mut payload = vec![0u8; count * 6];
    r.read_exact(&mut payload)?;

    let step = [
        (dmax[0] - dmin[0]) / 65535.0,
        (dmax[1] - dmin[1]) / 65535.0,
        (dmax[2] - dmin[2]) / 65535.0,
    ];
    let mut positions = Vec::with_capacity(count);
    let mut o = 0usize;
    for _ in 0..count {
        let qx = u16::from_le_bytes([payload[o], payload[o + 1]]) as u32;
        o += 2;
        let qy = u16::from_le_bytes([payload[o], payload[o + 1]]) as u32;
        o += 2;
        let qz = u16::from_le_bytes([payload[o], payload[o + 1]]) as u32;
        o += 2;
        let x = dmin[0] + (qx as f32) * step[0];
        let y = dmin[1] + (qy as f32) * step[1];
        let z = dmin[2] + (qz as f32) * step[2];
        positions.push([x, y, z]);
    }

    // Optional chunks
    let mut geog_crs = None;
    let mut geog_bbox_deg = None;
    let mut semantic_mask: Option<SemanticMask> = None;

    loop {
        let mut tag = [0u8; 4];
        match r.read_exact(&mut tag) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }

        match &tag {
            b"GEOT" => {
                // GEOT v1 fixed-size footer
                let v = read_exact_u8(&mut r)?;
                let crs = read_exact_u8(&mut r)?;
                let mode = read_exact_u8(&mut r)?;
                let _rsv = read_exact_u8(&mut r)?;
                if v == 1 && crs == 1 && mode == 0 {
                    let lon_min_q7 = read_exact_i32(&mut r)? as i64;
                    let lat_min_q7 = read_exact_i32(&mut r)? as i64;
                    let dlon_q7 = read_exact_u32(&mut r)? as u64;
                    let dlat_q7 = read_exact_u32(&mut r)? as u64;

                    let q = 1e-7_f64;
                    let lon_min = (lon_min_q7 as f64) * q;
                    let lat_min = (lat_min_q7 as f64) * q;
                    let lon_max = lon_min + (dlon_q7 as f64) * q;
                    let lat_max = lat_min + (dlat_q7 as f64) * q;

                    geog_crs = Some(GeoCrs::Crs84);
                    geog_bbox_deg = Some(GeoExtentDeg {
                        lon_min,
                        lat_min,
                        lon_max,
                        lat_max,
                    });
                } else {
                    // Unknown GEOT variant -> stop chunk parsing gracefully.
                    break;
                }
            }
            b"SMC1" => {
                let len = read_exact_u32(&mut r)? as usize;
                let mut buf = vec![0u8; len];
                r.read_exact(&mut buf)?;
                let sm = parse_smc1_payload(&buf)?;
                semantic_mask = Some(sm);
            }
            _ => {
                // Try generic length-prefixed skip
                match read_exact_u32(&mut r) {
                    Ok(len) => {
                        if r.seek(SeekFrom::Current(len as i64)).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        }
    }

    Ok(HypcPointCloud {
        positions,
        decode_min: dmin,
        decode_max: dmax,
        quant_bits: qbits,
        tile_key: unpack_tilekey(flags, &reserved),
        geog_crs,
        geog_bbox_deg,
        semantic_mask,
    })
}

/// Parameters for writing an HPC1 file.
pub struct HypcWrite<'a> {
    /// Must be 16 today.
    pub quant_bits: u8,
    /// Flat `u16` triplets [x0,y0,z0, x1,y1,z1, ...], length must be 3*N.
    pub quantized_positions: &'a [u16],
    pub decode_min: [f32; 3],
    pub decode_max: [f32; 3],

    pub tile_key: Option<TileKey>,

    /// If present and `geog_crs` is `Some(Crs84)`, a GEOT v1 bbox footer is emitted.
    pub geog_crs: Option<GeoCrs>,
    pub geog_bbox_deg: Option<GeoExtentDeg>,

    /// Optional SMC1 semantic mask to append.
    /// When `encoding == Smc1Encoding::Zlib`, the data will be zlib-compressed on write.
    pub smc1: Option<&'a SemanticMask>,
}

fn write_all_u32_le<W: Write>(w: &mut W, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}
fn write_all_i32_le<W: Write>(w: &mut W, v: i32) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}
fn write_all_u16_le<W: Write>(w: &mut W, v: u16) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}
fn write_all_f32_le<W: Write>(w: &mut W, v: f32) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}
fn write_all_u8<W: Write>(w: &mut W, v: u8) -> Result<()> {
    w.write_all(&[v])?;
    Ok(())
}

fn write_geot_footer<W: Write>(
    w: &mut W,
    crs: GeoCrs,
    bbox: GeoExtentDeg,
) -> Result<()> {
    match crs {
        GeoCrs::Crs84 => {
            const Q: f64 = 1e7;
            let lon_min_q7 =
                (bbox.lon_min * Q).round().clamp(i32::MIN as f64, i32::MAX as f64) as i32;
            let lat_min_q7 =
                (bbox.lat_min * Q).round().clamp(i32::MIN as f64, i32::MAX as f64) as i32;
            let dlon_q7 =
                ((bbox.lon_max - bbox.lon_min) * Q).round().clamp(0.0, u32::MAX as f64) as u32;
            let dlat_q7 =
                ((bbox.lat_max - bbox.lat_min) * Q).round().clamp(0.0, u32::MAX as f64) as u32;

            w.write_all(b"GEOT")?;
            write_all_u8(w, 1)?; // version
            write_all_u8(w, 1)?; // crs_id = 1 (CRS:84)
            write_all_u8(w, 0)?; // mode = 0 (BBOX_DEG_Q7)
            write_all_u8(w, 0)?; // reserved
            write_all_i32_le(w, lon_min_q7)?;
            write_all_i32_le(w, lat_min_q7)?;
            write_all_u32_le(w, dlon_q7)?;
            write_all_u32_le(w, dlat_q7)?;
            Ok(())
        }
    }
}

fn write_smc1_chunk<W: Write>(w: &mut W, sm: &SemanticMask) -> Result<()> {
    // Build payload
    let mut payload = Vec::<u8>::new();
    // version
    payload.push(1u8);
    // encoding
    payload.push(u8::from(sm.encoding));
    // width/height
    payload.extend_from_slice(&sm.width.to_le_bytes());
    payload.extend_from_slice(&sm.height.to_le_bytes());
    // coord_space
    payload.push(sm.coord_space);
    // class_count
    payload.push(
        sm.palette
            .len()
            .try_into()
            .map_err(|_| anyhow!("SMC1 palette too large"))?,
    );
    // reserved u16
    payload.extend_from_slice(&0u16.to_le_bytes());
    // class entries
    for &(cid, prec) in &sm.palette {
        payload.push(cid);
        payload.push(prec);
        payload.extend_from_slice(&0u16.to_le_bytes());
    }

    // data block
    let data_bytes = match sm.encoding {
        Smc1Encoding::Raw => sm.data.clone(),
        Smc1Encoding::Zlib => compress_to_vec_zlib(&sm.data, 6),
    };
    payload.extend_from_slice(&(data_bytes.len() as u32).to_le_bytes());
    payload.extend_from_slice(&data_bytes);

    // Chunk header + payload
    w.write_all(b"SMC1")?;
    write_all_u32_le(w, payload.len() as u32)?;
    w.write_all(&payload)?;
    Ok(())
}

/// Write HPC1 + optional chunks to a path.
pub fn write_file(path: &str, params: &HypcWrite) -> Result<()> {
    let mut f = File::create(path).with_context(|| format!("create {}", path))?;
    write(&mut f, params)
}

/// Write HPC1 + optional chunks to an arbitrary writer.
pub fn write<W: Write>(w: &mut W, params: &HypcWrite) -> Result<()> {
    if params.quant_bits != 16 {
        bail!("only quant_bits=16 is supported");
    }
    if params.quantized_positions.len() % 3 != 0 {
        bail!("quantized_positions length must be a multiple of 3");
    }
    let count = (params.quantized_positions.len() / 3) as u32;

    // Header
    w.write_all(b"HPC1")?;
    write_all_u32_le(w, 1)?; // version
    let (flags, reserved) = if let Some(ref tk) = params.tile_key {
        let (f, r) = pack_tilekey(tk);
        (f, r)
    } else {
        (0u32, [0u8; 11])
    };
    write_all_u32_le(w, flags)?;
    write_all_u32_le(w, count)?;
    write_all_u8(w, 16)?;
    w.write_all(&reserved)?;

    // decode min/max
    write_all_f32_le(w, params.decode_min[0])?;
    write_all_f32_le(w, params.decode_min[1])?;
    write_all_f32_le(w, params.decode_min[2])?;
    write_all_f32_le(w, params.decode_max[0])?;
    write_all_f32_le(w, params.decode_max[1])?;
    write_all_f32_le(w, params.decode_max[2])?;

    // payload
    for &q in params.quantized_positions {
        write_all_u16_le(w, q)?;
    }

    // optional chunks: SMC1 then GEOT (order matches current producer)
    if let Some(sm) = params.smc1 {
        write_smc1_chunk(w, sm)?;
    }
    if let (Some(crs), Some(bb)) = (params.geog_crs, params.geog_bbox_deg) {
        write_geot_footer(w, crs, bb)?;
    }
    Ok(())
}
