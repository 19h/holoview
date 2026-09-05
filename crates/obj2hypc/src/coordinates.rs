//! One dataset coordinate transform, independent of tile extents and sampling.
use anyhow::{ensure, Context, Result};
use proj::Proj;

pub struct ProjectedTransform {
    horizontal: Proj,
    height_offset_m: f64,
}

impl ProjectedTransform {
    pub fn new(source_crs: &str, height_offset_m: f64) -> Result<Self> {
        ensure!(height_offset_m.is_finite(), "Non-finite height offset");
        Ok(Self {
            horizontal: Proj::new_known_crs(source_crs, "OGC:CRS84", None)
                .context("Creating horizontal CRS transformation")?,
            height_offset_m,
        })
    }

    /// PROJ normalizes axis order to easting/northing and longitude/latitude.
    /// Height is explicit and shared: h = source_z + height_offset_m.
    pub fn convert(&self, p: [f64; 3]) -> Result<([f64; 3], f64, f64)> {
        ensure!(
            p.iter().all(|v| v.is_finite()),
            "Non-finite source coordinate"
        );
        let (lon, lat) = self.horizontal.convert((p[0], p[1]))?;
        let h = p[2] + self.height_offset_m;
        ensure!(
            lon.is_finite()
                && lat.is_finite()
                && h.is_finite()
                && (-180.0..=180.0).contains(&lon)
                && (-90.0..=90.0).contains(&lat),
            "Invalid transformed coordinate"
        );
        Ok((hypc::geodetic_to_ecef(lat, lon, h), lon, lat))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn berlin_control_point() {
        // Independent cs2cs 9.8.1: EPSG:25833 -> OGC:CRS84, degrees.
        let t = ProjectedTransform::new("EPSG:25833", 0.0).unwrap();
        let (ecef, lon, lat) = t
            .convert([389569.2241239654, 5819247.970530123, 40.443061542])
            .unwrap();
        assert!((lon - 13.3727154880).abs() < 1e-9);
        assert!((lat - 52.5121500550).abs() < 1e-9);
        let (_, _, h) = hypc::ecef_to_geodetic(ecef[0], ecef[1], ecef[2]);
        assert!((h - 40.443061542).abs() < 1e-6);
    }

    #[test]
    fn rejects_invalid_coordinates_and_crs() {
        assert!(ProjectedTransform::new("invalid", 0.0).is_err());
        assert!(ProjectedTransform::new("EPSG:25833", f64::NAN).is_err());
        let t = ProjectedTransform::new("EPSG:25833", 0.0).unwrap();
        assert!(t.convert([f64::INFINITY, 0.0, 0.0]).is_err());
    }
}
