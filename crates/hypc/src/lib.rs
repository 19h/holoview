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
use std::io::{self, ErrorKind, Read, Write};
use std::path::Path;

pub const HYPC_MAGIC: [u8; 4] = *b"HYPC";
pub const HYPC_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy)]
pub struct GeoExtentQ7 {
    pub lon_min_q7: i32,
    pub lon_max_q7: i32,
    pub lat_min_q7: i32,
    pub lat_max_q7: i32,
}

impl GeoExtentQ7 {
    #[inline]
    pub fn from_deg(lon_min: f64, lon_max: f64, lat_min: f64, lat_max: f64) -> Self {
        Self {
            lon_min_q7: (lon_min * 1e7).round() as i32,
            lon_max_q7: (lon_max * 1e7).round() as i32,
            lat_min_q7: (lat_min * 1e7).round() as i32,
            lat_max_q7: (lat_max * 1e7).round() as i32,
        }
    }

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

pub fn read_file<P: AsRef<Path>>(path: P) -> io::Result<HypcTile> {
    let mut f = File::open(path)?;

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if magic != HYPC_MAGIC {
        return Err(io::Error::new(ErrorKind::InvalidData, "bad HYPC magic"));
    }

    let version = read_u32(&mut f)?;
    if version != HYPC_VERSION {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("unsupported HYPC version {}", version),
        ));
    }

    let flags = read_u32(&mut f)?;
    let has_key = (flags & (1 << 0)) != 0;
    let has_labels = (flags & (1 << 1)) != 0;
    let has_geot = (flags & (1 << 2)) != 0;
    let has_smc1 = (flags & (1 << 3)) != 0;

    let count = read_u32(&mut f)? as usize;
    let units_per_meter = read_u32(&mut f)?;
    if units_per_meter == 0 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "units_per_meter must be > 0",
        ));
    }

    let anchor_ecef_units = [
        read_i64(&mut f)?,
        read_i64(&mut f)?,
        read_i64(&mut f)?,
    ];

    let tile_key = if has_key {
        let mut k = [0u8; 32];
        f.read_exact(&mut k)?;
        Some(k)
    } else {
        None
    };

    let mut points_units = Vec::<[i32; 3]>::with_capacity(count);
    let mut labels = if has_labels {
        Some(Vec::<u8>::with_capacity(count))
    } else {
        None
    };

    for _ in 0..count {
        let dx = read_i32(&mut f)?;
        let dy = read_i32(&mut f)?;
        let dz = read_i32(&mut f)?;
        points_units.push([dx, dy, dz]);

        if let Some(ref mut ls) = labels {
            let mut b = [0u8; 1];
            f.read_exact(&mut b)?;
            ls.push(b[0]);
        }
    }

    let geot = if has_geot {
        let mut tag = [0u8; 4];
        f.read_exact(&mut tag)?;
        if &tag != b"GEOT" {
            return Err(io::Error::new(ErrorKind::InvalidData, "expected GEOT tag"));
        }

        let lon_min_q7 = read_i32(&mut f)?;
        let lon_max_q7 = read_i32(&mut f)?;
        let lat_min_q7 = read_i32(&mut f)?;
        let lat_max_q7 = read_i32(&mut f)?;

        Some(GeoExtentQ7 {
            lon_min_q7,
            lon_max_q7,
            lat_min_q7,
            lat_max_q7,
        })
    } else {
        None
    };

    let smc1 = if has_smc1 {
        let mut tag = [0u8; 4];
        f.read_exact(&mut tag)?;
        if &tag != b"SMC1" {
            return Err(io::Error::new(ErrorKind::InvalidData, "expected SMC1 tag"));
        }

        let width = read_u16(&mut f)?;
        let height = read_u16(&mut f)?;

        let coord_space = match read_u8(&mut f)? {
            0 => Smc1CoordSpace::DecodeXY,
            1 => Smc1CoordSpace::Crs84BboxNorm,
            x => {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    format!("unknown SMC1 coord space {}", x),
                ));
            }
        };

        let encoding = match read_u8(&mut f)? {
            0 => Smc1Encoding::Raw,
            1 => Smc1Encoding::Rle,
            x => {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    format!("unknown SMC1 encoding {}", x),
                ));
            }
        };

        let palette_len = read_u16(&mut f)? as usize;
        let mut palette = Vec::<(u8, u8)>::with_capacity(palette_len);
        for _ in 0..palette_len {
            let class = read_u8(&mut f)?;
            let precedence = read_u8(&mut f)?;
            palette.push((class, precedence));
        }

        let payload_size = read_u32(&mut f)? as usize;
        let mut data = vec![0u8; payload_size];
        f.read_exact(&mut data)?;

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

    let mut f = File::create(path)?;

    f.write_all(&HYPC_MAGIC)?;

    write_u32(&mut f, HYPC_VERSION)?;
    write_u32(&mut f, flags)?;

    write_u32(&mut f, tile.points_units.len() as u32)?;
    write_u32(&mut f, tile.units_per_meter)?;

    write_i64(&mut f, tile.anchor_ecef_units[0])?;
    write_i64(&mut f, tile.anchor_ecef_units[1])?;
    write_i64(&mut f, tile.anchor_ecef_units[2])?;

    if let Some(key) = tile.tile_key {
        f.write_all(&key)?;
    }

    if let Some(labels) = tile.labels.as_ref() {
        if labels.len() != tile.points_units.len() {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "labels length != points length",
            ));
        }

        for (i, point) in tile.points_units.iter().enumerate() {
            write_i32(&mut f, point[0])?;
            write_i32(&mut f, point[1])?;
            write_i32(&mut f, point[2])?;

            f.write_all(&[labels[i]])?;
        }
    } else {
        for point in tile.points_units.iter() {
            write_i32(&mut f, point[0])?;
            write_i32(&mut f, point[1])?;
            write_i32(&mut f, point[2])?;
        }
    }

    if let Some(geot) = tile.geot.as_ref() {
        f.write_all(b"GEOT")?;

        write_i32(&mut f, geot.lon_min_q7)?;
        write_i32(&mut f, geot.lon_max_q7)?;
        write_i32(&mut f, geot.lat_min_q7)?;
        write_i32(&mut f, geot.lat_max_q7)?;
    }

    if let Some(smc1) = tile.smc1.as_ref() {
        f.write_all(b"SMC1")?;

        write_u16(&mut f, smc1.width)?;
        write_u16(&mut f, smc1.height)?;

        f.write_all(&[smc1.coord_space as u8])?;
        f.write_all(&[smc1.encoding as u8])?;

        write_u16(&mut f, smc1.palette.len() as u16)?;

        for &(class, precedence) in &smc1.palette {
            f.write_all(&[class, precedence])?;
        }

        write_u32(&mut f, smc1.data.len() as u32)?;

        f.write_all(&smc1.data)?;
    }

    f.flush()?;

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
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let (sl, cl) = lat.sin_cos();
    let (so, co) = lon.sin_cos();
    let n = wgs84::A / (1.0 - wgs84::E2 * sl * sl).sqrt();
    let x = (n + h_m) * cl * co;
    let y = (n + h_m) * cl * so;
    let z = (n * (1.0 - wgs84::E2) + h_m) * sl;
    [x, y, z]
}

#[inline]
pub fn ecef_to_geodetic(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    use wgs84::*;

    let p = (x * x + y * y).sqrt();
    let lon = y.atan2(x);
    let theta = (z * A).atan2(p * B);
    let (st, ct) = theta.sin_cos();
    let lat = (z + E2P * B * st * st * st).atan2(p - E2 * A * ct * ct * ct);
    let sin_lat = lat.sin();
    let n = A / (1.0 - E2 * sin_lat * sin_lat).sqrt();
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
fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

#[inline]
fn read_u16<R: Read>(r: &mut R) -> io::Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

#[inline]
fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

#[inline]
fn read_i32<R: Read>(r: &mut R) -> io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}

#[inline]
fn read_i64<R: Read>(r: &mut R) -> io::Result<i64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(i64::from_le_bytes(b))
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
