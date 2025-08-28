//! HYPC: internal dependency-free global point cloud format using WGS-84 ECEF.
//!
//! - Stores an i64 ECEF anchor (integer "units") and i32 offsets per point.
//! - Default units: 1000 units/meter (millimetres).
//! - Optional per-point labels (u8).
//! - Optional GEOT chunk: CRS:84 bbox (deg, Q7: 1e-7 deg ticks).
//! - Optional SMC1 chunk: semantic mask grid (u8), Raw or RLE encoding.
//!
//! File layout (little-endian):
//!   00  : [u8;4]  magic = b"HYPC"
//!   04  : u32     version = 2
//!   08  : u32     flags (bitfield)
//!                 bit 0 => tile key present (32 bytes)
//!                 bit 1 => per-point labels present
//!                 bit 2 => GEOT chunk present
//!                 bit 3 => SMC1 chunk present
//!   0C  : u32     points_count
//!   10  : u32     units_per_meter (default: 1000, mm)
//!   14  : i64[3]  anchor_ecef_units
//!   ..  : [u8;32] tile_key            (if bit0)
//!   ..  : for each point: i32 dx, i32 dy, i32 dz, [u8 label]? (if bit1)
//!   ..  : GEOT chunk                  (if bit2)
//!   ..  : SMC1 chunk                  (if bit3)
//!
//! GEOT chunk:
//!   "GEOT" [i32 lon_min_q7, lon_max_q7, lat_min_q7, lat_max_q7]
//!
//! SMC1 chunk:
//!   "SMC1" u16 width u16 height u8 coord_space u8 encoding u16 palette_len
//!          (palette_len pairs: u8 class, u8 precedence)
//!          u32 payload_size
//!          [payload_size bytes of pixel data] (Raw or RLE)
//!
//! RLE format: repeated [u16 run_len][u8 value] (little-endian)

use std::fs::File;
use std::io::{self, ErrorKind, Write};
use std::path::Path;

pub const HYPC_MAGIC: [u8; 4] = *b"HYPC";
pub const HYPC_VERSION: u32 = 2;

/// Represents a geographic bounding box using Q7 fixed-point encoding.
#[derive(Debug, Clone, Copy)]
pub struct GeoExtentQ7 {
    /// Minimum longitude in Q7 format (1e-7 degrees)
    pub lon_min_q7: i32,
    /// Maximum longitude in Q7 format (1e-7 degrees)
    pub lon_max_q7: i32,
    /// Minimum latitude in Q7 format (1e-7 degrees)
    pub lat_min_q7: i32,
    /// Maximum latitude in Q7 format (1e-7 degrees)
    pub lat_max_q7: i32,
}

impl GeoExtentQ7 {
    /// Creates a GeoExtentQ7 from floating-point degree coordinates.
    #[inline]
    pub fn from_deg(lon_min: f64, lon_max: f64, lat_min: f64, lat_max: f64) -> Self {
        Self {
            lon_min_q7: (lon_min * 1e7).round() as i32,
            lon_max_q7: (lon_max * 1e7).round() as i32,
            lat_min_q7: (lat_min * 1e7).round() as i32,
            lat_max_q7: (lat_max * 1e7).round() as i32,
        }
    }

    /// Converts Q7 fixed-point coordinates back to floating-point degrees.
    #[inline]
    pub fn to_deg(self) -> (f64, f64, f64, f64) {
        (
            self.lon_min_q7 as f64 * 1e-7,
            self.lon_max_q7 as f64 * 1e-7,
            self.lat_min_q7 as f64 * 1e-7,
            self.lat_max_q7 as f64 * 1e-7,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Smc1Encoding {
    Raw = 0,
    Rle = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Smc1CoordSpace {
    /// UV in "decode" space (legacy/local); not used by HYPC.
    DecodeXY = 0,

    /// Normalized CRS:84 bbox coordinates (lon/lat -> [0,1] within GEOT bbox).
    Crs84BboxNorm = 1,
}

#[derive(Debug, Clone)]
pub struct Smc1Chunk {
    pub width: u16,
    pub height: u16,
    pub coord_space: Smc1CoordSpace,
    pub encoding: Smc1Encoding,
    pub palette: Vec<(u8, u8)>, // (class, precedence)
    pub data: Vec<u8>,          // raw (w*h) if Raw; RLE payload if Rle
}

#[derive(Debug, Clone)]
pub struct HypcTile {
    pub units_per_meter: u32,
    pub anchor_ecef_units: [i64; 3],
    pub tile_key: Option<[u8; 32]>,
    pub points_units: Vec<[i32; 3]>,
    pub labels: Option<Vec<u8>>,
    pub geot: Option<GeoExtentQ7>,
    pub smc1: Option<Smc1Chunk>,
}

#[inline(always)]
fn need(buf: &[u8], want: usize) -> io::Result<()> {
    if buf.len() < want {
        Err(io::Error::new(ErrorKind::UnexpectedEof, "truncated HYPC"))
    } else {
        Ok(())
    }
}

#[inline(always)]
fn take<'a>(buf: &mut &'a [u8], n: usize) -> io::Result<&'a [u8]> {
    need(buf, n)?;
    let (head, tail) = buf.split_at(n);
    *buf = tail;
    Ok(head)
}

#[inline(always)]
fn le_u8(buf: &mut &[u8]) -> io::Result<u8> {
    Ok(take(buf, 1)?[0])
}

#[inline(always)]
fn le_u16(buf: &mut &[u8]) -> io::Result<u16> {
    let b = take(buf, 2)?;
    Ok(u16::from_le_bytes([b[0], b[1]]))
}

#[inline(always)]
fn le_u32(buf: &mut &[u8]) -> io::Result<u32> {
    let b = take(buf, 4)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

#[inline(always)]
fn le_i32(buf: &mut &[u8]) -> io::Result<i32> {
    let b = take(buf, 4)?;
    Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

#[inline(always)]
fn le_i64(buf: &mut &[u8]) -> io::Result<i64> {
    let b = take(buf, 8)?;
    Ok(i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
}

#[cold]
fn bad(msg: &str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, msg)
}

/// Parse HYPC from a contiguous byte slice. This is the single source of truth for parsing.
pub fn parse_hypc_bytes(mut p: &[u8]) -> io::Result<HypcTile> {
    // Header
    if take(&mut p, 4)? != b"HYPC" {
        return Err(bad("bad HYPC magic"));
    }

    let version = le_u32(&mut p)?;
    if version != HYPC_VERSION {
        return Err(bad("unsupported HYPC version"));
    }

    let flags = le_u32(&mut p)?;
    let has_key    = (flags & (1 << 0)) != 0;
    let has_labels = (flags & (1 << 1)) != 0;
    let has_geot   = (flags & (1 << 2)) != 0;
    let has_smc1   = (flags & (1 << 3)) != 0;

    let count = le_u32(&mut p)? as usize;
    let units_per_meter = le_u32(&mut p)?;
    if units_per_meter == 0 {
        return Err(bad("units_per_meter must be > 0"));
    }

    let anchor_ecef_units = [
        le_i64(&mut p)?,
        le_i64(&mut p)?,
        le_i64(&mut p)?,
    ];

    let tile_key = if has_key {
        let t = take(&mut p, 32)?;
        let mut k = [0u8; 32];
        k.copy_from_slice(t);
        Some(k)
    } else {
        None
    };

    // Points (+ optional interleaved label bytes)
    let pts_rec = 12usize + if has_labels { 1 } else { 0 };
    let pts_bytes = count.checked_mul(pts_rec).ok_or_else(|| bad("points size overflow"))?;
    need(p, pts_bytes)?;

    let (points_units, labels): (Vec<[i32; 3]>, Option<Vec<u8>>) = if has_labels {
        // Single pass decode of interleaved [i32 i32 i32 u8] records.
        let mut pts = Vec::<[i32; 3]>::with_capacity(count);
        let mut ls  = Vec::<u8>::with_capacity(count);
        unsafe {
            pts.set_len(count);
        } // we'll write every slot exactly once

        for i in 0..count {
            // Read 12 bytes of i32s
            let dx = le_i32(&mut p)?;
            let dy = le_i32(&mut p)?;
            let dz = le_i32(&mut p)?;

            // Label
            let l = le_u8(&mut p)?;

            // Write (avoids bounds checks)
            unsafe {
                *pts.get_unchecked_mut(i) = [dx, dy, dz];
            }
            ls.push(l);
        }

        (pts, Some(ls))
    } else {
        // Fast path: points block is tightly packed 12N bytes; zero‑copy reinterpret + to_vec().
        let raw = take(&mut p, count * 12)?;

        #[cfg(target_endian = "little")]
        {
            // Safety:
            // - alignment: header is 44 or 76 bytes (both %4 == 0), so this slice is 4‑aligned.
            // - repr: [i32;3] has no padding beyond 12 bytes.
            // - endianness: little.
            let as_i32x3: &[[i32; 3]] = bytemuck::try_cast_slice(raw)
                .map_err(|_| bad("misaligned points block"))?;

            (as_i32x3.to_vec(), None)
        }

        #[cfg(not(target_endian = "little"))]
        {
            // Fallback: portable decode (still a single pass).
            let mut pts = Vec::<[i32; 3]>::with_capacity(count);

            for chunk in raw.chunks_exact(12) {
                let dx = i32::from_le_bytes(chunk[0..4].try_into().unwrap());
                let dy = i32::from_le_bytes(chunk[4..8].try_into().unwrap());
                let dz = i32::from_le_bytes(chunk[8..12].try_into().unwrap());
                pts.push([dx, dy, dz]);
            }

            (pts, None)
        }
    };

    // GEOT
    let geot = if has_geot {
        if take(&mut p, 4)? != b"GEOT" {
            return Err(bad("expected GEOT tag"));
        }

        Some(GeoExtentQ7 {
            lon_min_q7: le_i32(&mut p)?,
            lon_max_q7: le_i32(&mut p)?,
            lat_min_q7: le_i32(&mut p)?,
            lat_max_q7: le_i32(&mut p)?,
        })
    } else {
        None
    };

    // SMC1
    let smc1 = if has_smc1 {
        if take(&mut p, 4)? != b"SMC1" {
            return Err(bad("expected SMC1 tag"));
        }

        let width  = le_u16(&mut p)?;
        let height = le_u16(&mut p)?;

        let coord_space = match le_u8(&mut p)? {
            0 => Smc1CoordSpace::DecodeXY,
            1 => Smc1CoordSpace::Crs84BboxNorm,
            x => return Err(bad(&format!("unknown SMC1 coord space {}", x))),
        };

        let encoding = match le_u8(&mut p)? {
            0 => Smc1Encoding::Raw,
            1 => Smc1Encoding::Rle,
            x => return Err(bad(&format!("unknown SMC1 encoding {}", x))),
        };

        let palette_len = le_u16(&mut p)? as usize;
        let mut palette = Vec::<(u8, u8)>::with_capacity(palette_len);

        for _ in 0..palette_len {
            let class = le_u8(&mut p)?;
            let precedence = le_u8(&mut p)?;
            palette.push((class, precedence));
        }

        let payload_size = le_u32(&mut p)? as usize;
        let data = take(&mut p, payload_size)?.to_vec();

        Some(Smc1Chunk {
            width,
            height,
            coord_space,
            encoding,
            palette,
            data,
        })
    } else {
        None
    };

    Ok(HypcTile {
        units_per_meter,
        anchor_ecef_units,
        tile_key,
        points_units,
        labels,
        geot,
        smc1,
    })
}

/// Fast path: prefer mmap; fall back to a single read.
#[cfg(feature = "mmap")]
pub fn read_file<P: AsRef<Path>>(path: P) -> io::Result<HypcTile> {
    let file = File::open(path)?;
    let map = unsafe { memmap2::MmapOptions::new().map(&file)? };
    parse_hypc_bytes(&map)
}

#[cfg(not(feature = "mmap"))]
pub fn read_file<P: AsRef<Path>>(path: P) -> io::Result<HypcTile> {
    let bytes = std::fs::read(path)?;
    parse_hypc_bytes(&bytes)
}

pub fn write_file<P: AsRef<Path>>(path: P, tile: &HypcTile) -> io::Result<()> {
    let mut flags = 0u32;

    if tile.tile_key.is_some() {
        flags |= 1 << 0;
    }

    if tile.labels.is_some() {
        flags |= 1 << 1;
    }

    if tile.geot.is_some() {
        flags |= 1 << 2;
    }

    if tile.smc1.is_some() {
        flags |= 1 << 3;
    }

    let mut file = File::create(path)?;

    file.write_all(&HYPC_MAGIC)?;

    write_u32(&mut file, HYPC_VERSION)?;
    write_u32(&mut file, flags)?;

    write_u32(&mut file, tile.points_units.len() as u32)?;
    write_u32(&mut file, tile.units_per_meter)?;

    write_i64(&mut file, tile.anchor_ecef_units[0])?;
    write_i64(&mut file, tile.anchor_ecef_units[1])?;
    write_i64(&mut file, tile.anchor_ecef_units[2])?;

    if let Some(key) = tile.tile_key {
        file.write_all(&key)?;
    }

    if let Some(labels) = tile.labels.as_ref() {
        if labels.len() != tile.points_units.len() {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "labels length != points length",
            ));
        }

        for (index, point) in tile.points_units.iter().enumerate() {
            write_i32(&mut file, point[0])?;
            write_i32(&mut file, point[1])?;
            write_i32(&mut file, point[2])?;

            file.write_all(&[labels[index]])?;
        }
    } else {
        for point in tile.points_units.iter() {
            write_i32(&mut file, point[0])?;
            write_i32(&mut file, point[1])?;
            write_i32(&mut file, point[2])?;
        }
    }

    if let Some(geot) = tile.geot.as_ref() {
        file.write_all(b"GEOT")?;

        write_i32(&mut file, geot.lon_min_q7)?;
        write_i32(&mut file, geot.lon_max_q7)?;
        write_i32(&mut file, geot.lat_min_q7)?;
        write_i32(&mut file, geot.lat_max_q7)?;
    }

    if let Some(smc1) = tile.smc1.as_ref() {
        file.write_all(b"SMC1")?;

        write_u16(&mut file, smc1.width)?;
        write_u16(&mut file, smc1.height)?;

        file.write_all(&[smc1.coord_space as u8])?;
        file.write_all(&[smc1.encoding as u8])?;

        write_u16(&mut file, smc1.palette.len() as u16)?;

        for &(class, precedence) in &smc1.palette {
            file.write_all(&[class, precedence])?;
        }

        write_u32(&mut file, smc1.data.len() as u32)?;

        file.write_all(&smc1.data)?;
    }

    file.flush()?;

    Ok(())
}

pub fn smc1_encode_rle(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::<u8>::with_capacity(raw.len() / 2);
    if raw.is_empty() {
        return out;
    }

    let mut i = 0usize;
    while i < raw.len() {
        let value = raw[i];
        let mut run_length = 1usize;

        while i + run_length < raw.len()
            && raw[i + run_length] == value
            && run_length < u16::MAX as usize
        {
            run_length += 1;
        }

        out.extend_from_slice(&(run_length as u16).to_le_bytes());
        out.push(value);
        i += run_length;
    }

    out
}

pub fn smc1_decode_rle(rle: &[u8]) -> io::Result<Vec<u8>> {
    let mut out = Vec::<u8>::new();
    let mut i = 0usize;

    while i + 3 <= rle.len() {
        let run = u16::from_le_bytes([rle[i], rle[i + 1]]) as usize;
        let v = rle[i + 2];
        out.resize(out.len() + run, v);
        i += 3;
    }

    if i != rle.len() {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "RLE payload truncated",
        ));
    }

    Ok(out)
}

pub mod wgs84 {
    /// Semi-major axis (equatorial radius) in meters.
    pub const A: f64 = 6_378_137.0;

    /// Flattening factor (1 / 298.257223563).
    pub const F: f64 = 1.0 / 298.257_223_563;

    /// First eccentricity squared.
    pub const E2: f64 = F * (2.0 - F);

    /// Semi-minor axis (polar radius) in meters.
    pub const B: f64 = A * (1.0 - F);

    /// Second eccentricity squared.
    pub const E2P: f64 = (A * A - B * B) / (B * B);
}

#[inline]
pub fn geodetic_to_ecef(lat_deg: f64, lon_deg: f64, h_m: f64) -> [f64; 3] {
    // Convert latitude and longitude from degrees to radians
    let lat_rad = lat_deg.to_radians();
    let lon_rad = lon_deg.to_radians();

    // Compute sine and cosine of latitude and longitude
    let (sin_lat, cos_lat) = lat_rad.sin_cos();
    let (sin_lon, cos_lon) = lon_rad.sin_cos();

    // Compute the radius of curvature in the prime vertical (N)
    let n = wgs84::A / (1.0 - wgs84::E2 * sin_lat * sin_lat).sqrt();

    // Compute ECEF coordinates
    let x = (n + h_m) * cos_lat * cos_lon;
    let y = (n + h_m) * cos_lat * sin_lon;
    let z = (n * (1.0 - wgs84::E2) + h_m) * sin_lat;

    [x, y, z]
}

#[inline]
pub fn ecef_to_geodetic(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    // Compute the distance from the Z-axis
    let p = (x * x + y * y).sqrt();

    // Compute longitude (λ)
    let lon = y.atan2(x);

    // Initial latitude estimate (θ)
    let theta = (z * wgs84::A).atan2(p * wgs84::B);
    let (sin_theta, cos_theta) = theta.sin_cos();

    // Compute latitude (φ)
    let lat_numerator = z + wgs84::E2P * wgs84::B * sin_theta * sin_theta * sin_theta;
    let lat_denominator = p - wgs84::E2 * wgs84::A * cos_theta * cos_theta * cos_theta;
    let lat = lat_numerator.atan2(lat_denominator);

    // Compute the radius of curvature in the prime vertical (N)
    let sin_lat = lat.sin();
    let n = wgs84::A / (1.0 - wgs84::E2 * sin_lat * sin_lat).sqrt();

    // Compute ellipsoidal height (h)
    let h = p / lat.cos() - n;

    (lat.to_degrees(), lon.to_degrees(), h)
}

#[inline]
pub fn quantize_units(meters: f64, units_per_meter: u32) -> i64 {
    (meters * (units_per_meter as f64)).round() as i64
}

#[inline]
pub fn split_f64_to_f32_pair(v: f64) -> (f32, f32) {
    let hi = v as f32;
    let lo = (v - hi as f64) as f32;
    (hi, lo)
}

#[inline]
fn write_u16<W: Write>(w: &mut W, v: u16) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

#[inline]
fn write_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

#[inline]
fn write_i32<W: Write>(w: &mut W, v: i32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

#[inline]
fn write_i64<W: Write>(w: &mut W, v: i64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
