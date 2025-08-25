use anyhow::Result;
use glam::Vec3;
use ply_rs::{parser, ply};
use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
};

// === SMC1 additions ===
use miniz_oxide::inflate::decompress_to_vec_zlib;

#[derive(Debug, Clone)]
pub enum TileKey {
    XY { zoom: u8, x: u32, y: u32, scheme: u8 },
    NameHash64 { hash: u64 },
}

#[derive(Debug, Clone)]
pub enum GeoCrs {
    Crs84, // degrees
}

#[derive(Debug, Clone)]
pub struct GeoExtentDeg {
    pub lon_min: f64,
    pub lat_min: f64,
    pub lon_max: f64,
    pub lat_max: f64,
}

// === SMC1 additions ===
#[derive(Debug, Clone)]
pub struct SemanticMask {
    pub width: u16,
    pub height: u16,
    pub data: Vec<u8>, // decompressed labels, row-major
    pub palette: Vec<(u8 /*class*/, u8 /*precedence*/)>,
    pub coord_space: u8, // 0 = decode XY (current writer)
    pub encoding: u8,    // 0 = raw, 1 = zlib (as written)
}

#[derive(Debug, Clone)]
pub struct QuantizedPointCloud {
    pub positions: Vec<glam::Vec3>, // decoded in centered+scaled local space
    pub decode_min: glam::Vec3,
    pub decode_max: glam::Vec3,
    pub kept: usize,
    pub tile_key: Option<TileKey>,
    pub geog_crs: Option<GeoCrs>,
    pub geog_bbox_deg: Option<GeoExtentDeg>,

    // === SMC1 additions ===
    pub semantic_mask: Option<SemanticMask>,
}

// Helpers to parse little-endian scalars from a Read
#[inline(always)]
fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
#[inline(always)]
fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}
#[inline(always)]
fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

// Parse tile key flags/reserved mirroring the writer
fn parse_tilekey(flags: u32, reserved: &[u8; 11]) -> Option<TileKey> {
    const FLAG_TILEKEY_PRESENT: u32 = 1 << 0;
    if flags & FLAG_TILEKEY_PRESENT == 0 {
        return None;
    }
    match reserved[0] {
        0 => Some(TileKey::XY {
            zoom: reserved[1],
            x: u32::from_le_bytes(reserved[2..6].try_into().unwrap()),
            y: u32::from_le_bytes(reserved[6..10].try_into().unwrap()),
            scheme: reserved[10],
        }),
        4 => Some(TileKey::NameHash64 {
            hash: u64::from_le_bytes(reserved[1..9].try_into().unwrap()),
        }),
        _ => None,
    }
}

// === SMC1 parsing (versioned, with palette & precedence)
fn parse_smc1<R: Read>(r: &mut R) -> Result<SemanticMask> {
    let payload_len = read_u32(r)? as usize; // total payload bytes
    let mut buf = vec![0u8; payload_len];
    r.read_exact(&mut buf)?;
    let mut off = 0usize;

    let version = buf[off];
    off += 1;
    if version != 1 {
        anyhow::bail!("SMC1: unsupported version {}", version);
    }
    let encoding = buf[off];
    off += 1;

    let width = u16::from_le_bytes([buf[off], buf[off + 1]]);
    off += 2;
    let height = u16::from_le_bytes([buf[off], buf[off + 1]]);
    off += 2;

    let coord_space = buf[off];
    off += 1;

    let class_count = buf[off];
    off += 1;

    off += 2; // reserved (u16)

    let mut palette = Vec::with_capacity(class_count as usize);
    for _ in 0..class_count {
        let cid = buf[off];
        let prec = buf[off + 1];
        off += 2;
        off += 2; // reserved per entry
        palette.push((cid, prec));
    }

    let data_len = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]) as usize;
    off += 4;

    if off + data_len > buf.len() {
        anyhow::bail!("SMC1: payload length exceeds buffer");
    }

    let data_raw = &buf[off..off + data_len];
    let data = if encoding == 1 {
        decompress_to_vec_zlib(data_raw).map_err(|_| anyhow::anyhow!("SMC1 zlib inflate failed"))?
    } else {
        data_raw.to_vec()
    };

    Ok(SemanticMask {
        width,
        height,
        data,
        palette,
        coord_space,
        encoding,
    })
}

impl QuantizedPointCloud {
    /// Mirrors JS: load → AABB → analytic center/scale → stride sample → quantize → decode
    pub fn load_ply_quantized(path: &str, point_budget: usize, target_extent: f32) -> Result<Self> {
        let f = File::open(path)?;
        let mut rd = BufReader::new(f);

        let vertex_parser =
            parser::Parser::<linked_hash_map::LinkedHashMap<String, ply::Property>>::new();
        let ply = vertex_parser.read_ply(&mut rd)?;
        let verts = ply
            .payload
            .get("vertex")
            .ok_or_else(|| anyhow::anyhow!("No 'vertex' element"))?;

        let mut pos: Vec<f32> = Vec::with_capacity(verts.len() * 3);
        for rec in verts {
            let x = match rec.get("x") {
                Some(ply::Property::Float(v)) => *v,
                Some(ply::Property::Double(v)) => *v as f32,
                _ => 0.0,
            };
            let y = match rec.get("y") {
                Some(ply::Property::Float(v)) => *v,
                Some(ply::Property::Double(v)) => *v as f32,
                _ => 0.0,
            };
            let z = match rec.get("z") {
                Some(ply::Property::Float(v)) => *v,
                Some(ply::Property::Double(v)) => *v as f32,
                _ => 0.0,
            };
            pos.extend([x, y, z]);
        }

        if pos.len() < 3 {
            return Ok(Self {
                positions: vec![],
                decode_min: Vec3::ZERO,
                decode_max: Vec3::ZERO,
                kept: 0,
                tile_key: None,
                geog_crs: None,
                geog_bbox_deg: None,
                semantic_mask: None,
            });
        }

        let n = pos.len() / 3;

        // AABB
        let mut minx = pos[0];
        let mut miny = pos[1];
        let mut minz = pos[2];
        let mut maxx = minx;
        let mut maxy = miny;
        let mut maxz = minz;
        let mut i = 3usize;
        while i < pos.len() {
            let x = pos[i];
            let y = pos[i + 1];
            let z = pos[i + 2];
            minx = f32::min(minx, x);
            miny = f32::min(miny, y);
            minz = f32::min(minz, z);
            maxx = f32::max(maxx, x);
            maxy = f32::max(maxy, y);
            maxz = f32::max(maxz, z);
            i += 3;
        }

        let sx = maxx - minx;
        let sy = maxy - miny;
        let sz = maxz - minz;
        let eps = 1e-20f32;
        let scale = target_extent / sx.max(sy).max(sz).max(eps);
        let dx = 0.5 * sx * scale;
        let dy = 0.5 * sy * scale;
        let dz = 0.5 * sz * scale;
        let decode_min = Vec3::new(-dx, -dy, -dz);
        let decode_max = Vec3::new(dx, dy, dz);

        let stride = ((n + point_budget - 1) / point_budget).max(1);
        let m = n / stride;
        let step = stride * 3;

        let u = 65535.0f32;
        let qx = u / sx.max(eps);
        let qy = u / sy.max(eps);
        let qz = u / sz.max(eps);
        let stepx = (sx * scale) / u;
        let stepy = (sy * scale) / u;
        let stepz = (sz * scale) / u;

        let mut out = Vec::with_capacity(m);
        let mut src = 0usize;
        for _ in 0..m {
            let qxv = ((pos[src] - minx) * qx + 0.5).floor() as u32;
            let qyv = ((pos[src + 1] - miny) * qy + 0.5).floor() as u32;
            let qzv = ((pos[src + 2] - minz) * qz + 0.5).floor() as u32;
            let x = -dx + (qxv as f32) * stepx;
            let y = -dy + (qyv as f32) * stepy;
            let z = -dz + (qzv as f32) * stepz;
            out.push(Vec3::new(x, y, z));
            src += step;
        }

        Ok(Self {
            positions: out,
            decode_min,
            decode_max,
            kept: m,
            tile_key: None,
            geog_crs: None,
            geog_bbox_deg: None,
            semantic_mask: None,
        })
    }

    /// Read `.hypc` (HPC1) and optional chunks (SMC1 v1, GEOT v1).
    pub fn load_hypc(path: &str) -> Result<Self> {
        let mut f = File::open(path)?;

        // HPC1 header
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"HPC1" {
            return Err(anyhow::anyhow!("{}: bad magic", path));
        }

        let version = read_u32(&mut f)?;
        if version != 1 {
            return Err(anyhow::anyhow!("unsupported HPC1 version {}", version));
        }

        let flags = read_u32(&mut f)?;
        let count = read_u32(&mut f)? as usize;
        let qbits = read_u8(&mut f)?;
        if qbits != 16 {
            return Err(anyhow::anyhow!("unsupported quant_bits != 16"));
        }

        let mut reserved = [0u8; 11];
        f.read_exact(&mut reserved)?;

        let dmin = glam::Vec3::new(read_f32(&mut f)?, read_f32(&mut f)?, read_f32(&mut f)?);
        let dmax = glam::Vec3::new(read_f32(&mut f)?, read_f32(&mut f)?, read_f32(&mut f)?);

        let mut payload = vec![0u8; count * 6];
        f.read_exact(&mut payload)?;

        // Decode quantized positions
        let step = (dmax - dmin) / 65535.0;
        let mut positions = Vec::with_capacity(count);
        let mut o = 0usize;
        for _ in 0..count {
            let qx = u16::from_le_bytes([payload[o], payload[o + 1]]) as u32;
            o += 2;
            let qy = u16::from_le_bytes([payload[o], payload[o + 1]]) as u32;
            o += 2;
            let qz = u16::from_le_bytes([payload[o], payload[o + 1]]) as u32;
            o += 2;
            let x = dmin.x + (qx as f32) * step.x;
            let y = dmin.y + (qy as f32) * step.y;
            let z = dmin.z + (qz as f32) * step.z;
            positions.push(glam::Vec3::new(x, y, z));
        }

        // Optional trailing chunks
        let mut geog_crs = None;
        let mut geog_bbox_deg = None;
        let mut semantic_mask: Option<SemanticMask> = None;

        loop {
            let mut tag = [0u8; 4];
            match f.read_exact(&mut tag) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }

            match &tag {
                b"GEOT" => {
                    // GEOT v1 fixed-size footer
                    let v = read_u8(&mut f)?;
                    let crs = read_u8(&mut f)?;
                    let mode = read_u8(&mut f)?;
                    let _rsv = read_u8(&mut f)?;
                    if v == 1 && crs == 1 && mode == 0 {
                        let lon_min_q7 = {
                            let mut b = [0u8; 4];
                            f.read_exact(&mut b)?;
                            i32::from_le_bytes(b) as i64
                        };
                        let lat_min_q7 = {
                            let mut b = [0u8; 4];
                            f.read_exact(&mut b)?;
                            i32::from_le_bytes(b) as i64
                        };
                        let dlon_q7 = {
                            let mut b = [0u8; 4];
                            f.read_exact(&mut b)?;
                            u32::from_le_bytes(b) as u64
                        };
                        let dlat_q7 = {
                            let mut b = [0u8; 4];
                            f.read_exact(&mut b)?;
                            u32::from_le_bytes(b) as u64
                        };
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
                        // Unknown GEOT variant — bail out of chunk loop gracefully.
                        break;
                    }
                }
                b"SMC1" => {
                    let sm = parse_smc1(&mut f)?;
                    semantic_mask = Some(sm);
                }
                _ => {
                    // Generic "chunk with length" skip (MAGIC + u32(len) + payload)
                    // We already consumed MAGIC; attempt to read len and skip payload.
                    match read_u32(&mut f) {
                        Ok(len) => {
                            // Seek forward by `len`; if it fails, stop chunk parsing.
                            if f.seek(SeekFrom::Current(len as i64)).is_err() {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
            }
        }

        Ok(Self {
            positions,
            decode_min: dmin,
            decode_max: dmax,
            kept: count,
            tile_key: parse_tilekey(flags, &reserved),
            geog_crs,
            geog_bbox_deg,
            semantic_mask,
        })
    }

    /// O(1) class lookup from a position in decode space (x,y).
    pub fn class_of_xy(&self, x: f32, y: f32) -> u8 {
        let Some(ref sm) = self.semantic_mask else {
            return 0;
        };
        if sm.coord_space != 0 {
            // Only decode-space masks are supported by this lookup.
            return 0;
        }
        let w = sm.width as f32;
        let h = sm.height as f32;
        let ix = (((x - self.decode_min.x) * w)
            / (self.decode_max.x - self.decode_min.x).max(f32::EPSILON))
            .floor() as i32;
        let iy = (((y - self.decode_min.y) * h)
            / (self.decode_max.y - self.decode_min.y).max(f32::EPSILON))
            .floor() as i32;
        if ix < 0 || iy < 0 || ix >= sm.width as i32 || iy >= sm.height as i32 {
            return 0;
        }
        sm.data[(iy as usize) * sm.width as usize + ix as usize]
    }

    /// Convert lon/lat (deg) to decode XY, using GEOT v1 bbox (if present).
    pub fn lonlat_to_decode_xy(&self, lon: f64, lat: f64) -> Option<(f32, f32)> {
        let bb = self.geog_bbox_deg.as_ref()?;
        let sx = (self.decode_max.x - self.decode_min.x) as f64;
        let sy = (self.decode_max.y - self.decode_min.y) as f64;

        let x = self.decode_min.x as f64
            + (lon - bb.lon_min) / (bb.lon_max - bb.lon_min + f64::EPSILON) * sx;
        let y = self.decode_min.y as f64
            + (lat - bb.lat_min) / (bb.lat_max - bb.lat_min + f64::EPSILON) * sy;

        Some((x as f32, y as f32))
    }

    /// Convert decode XY to lon/lat (deg), inverse of `lonlat_to_decode_xy`.
    pub fn decode_xy_to_lonlat(&self, x: f32, y: f32) -> Option<(f64, f64)> {
        let bb = self.geog_bbox_deg.as_ref()?;
        let sx = (self.decode_max.x - self.decode_min.x) as f64;
        let sy = (self.decode_max.y - self.decode_min.y) as f64;

        let lon = bb.lon_min + ((x as f64 - self.decode_min.x as f64) / sx) * (bb.lon_max - bb.lon_min);
        let lat = bb.lat_min + ((y as f64 - self.decode_min.y as f64) / sy) * (bb.lat_max - bb.lat_min);
        Some((lon, lat))
    }

    /// Convenience: class lookup directly in geographic space (deg).
    pub fn class_of_lonlat(&self, lon: f64, lat: f64) -> u8 {
        let Some((x, y)) = self.lonlat_to_decode_xy(lon, lat) else {
            return 0;
        };
        self.class_of_xy(x, y)
    }

    /// Convenience flags
    pub fn has_semantics(&self) -> bool {
        self.semantic_mask.is_some()
    }
    pub fn has_geot(&self) -> bool {
        self.geog_bbox_deg.is_some()
    }
}
