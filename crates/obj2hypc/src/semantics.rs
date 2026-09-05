//! Dataset-independent semantic sampling. HYPC direct labels avoid independently
//! phased per-tile bbox textures at shared edges.
use crate::{class_precedence, pad_degrees_for, SemOverlayPerTile};
use rstar::{RTree, RTreeObject, AABB};
use std::collections::HashMap;

/// A globally anchored angular grid: approximately 0.34 x 0.56 m in Berlin.
/// Grid phase and label tie breaking never depend on tile bounds or load order.
pub const CELL_DEG: f64 = 0.000005;

#[derive(Clone)]
enum Kind {
    Area(usize),
    Segment(usize, usize),
}
#[derive(Clone)]
struct Feature {
    bounds: AABB<[f64; 2]>,
    kind: Kind,
    class: u8,
}
impl RTreeObject for Feature {
    type Envelope = AABB<[f64; 2]>;
    fn envelope(&self) -> Self::Envelope {
        self.bounds
    }
}

pub struct LabelField<'a> {
    overlay: &'a SemOverlayPerTile,
    tree: RTree<Feature>,
    cache: HashMap<(i64, i64), u8>,
}

impl<'a> LabelField<'a> {
    pub fn new(overlay: &'a SemOverlayPerTile) -> Self {
        let bounds = |coords: &[(f64, f64)], pad_m: f64| {
            let mut lo = [f64::INFINITY; 2];
            let mut hi = [f64::NEG_INFINITY; 2];
            for &(x, y) in coords {
                lo[0] = lo[0].min(x);
                lo[1] = lo[1].min(y);
                hi[0] = hi[0].max(x);
                hi[1] = hi[1].max(y);
            }
            let (dy, dx) = pad_degrees_for((lo[1] + hi[1]) * 0.5, pad_m * 1.01);
            AABB::from_corners([lo[0] - dx, lo[1] - dy], [hi[0] + dx, hi[1] + dy])
        };
        let mut features = Vec::new();
        for (i, area) in overlay.areas.iter().enumerate() {
            if area.ring.len() >= 3 {
                features.push(Feature {
                    bounds: bounds(&area.ring, 0.0),
                    kind: Kind::Area(i),
                    class: area.class,
                });
            }
        }
        for (i, line) in overlay.roads.iter().enumerate() {
            if !line.width_m.is_finite() || line.width_m <= 0.0 {
                continue;
            }
            for (j, segment) in line.pts.windows(2).enumerate() {
                features.push(Feature {
                    bounds: bounds(segment, line.width_m as f64 * 0.5),
                    kind: Kind::Segment(i, j),
                    class: line.class,
                });
            }
        }
        Self {
            overlay,
            tree: RTree::bulk_load(features),
            cache: HashMap::new(),
        }
    }

    pub fn label(&mut self, lon: f64, lat: f64) -> u8 {
        let key = (
            (lon / CELL_DEG).floor() as i64,
            (lat / CELL_DEG).floor() as i64,
        );
        if let Some(&label) = self.cache.get(&key) {
            return label;
        }
        let x = (key.0 as f64 + 0.5) * CELL_DEG;
        let y = (key.1 as f64 + 0.5) * CELL_DEG;
        let phi = y.to_radians();
        let denom = (1.0 - hypc::wgs84::E2 * phi.sin().powi(2)).sqrt();
        let mx = (hypc::wgs84::A / denom) * phi.cos() * std::f64::consts::PI / 180.0;
        let my =
            hypc::wgs84::A * (1.0 - hypc::wgs84::E2) / denom.powi(3) * std::f64::consts::PI / 180.0;
        let mut label = 0;
        for feature in self
            .tree
            .locate_in_envelope_intersecting(&AABB::from_point([x, y]))
        {
            // Class ID resolves equal precedence independently of R-tree order.
            if (class_precedence(feature.class), feature.class) <= (class_precedence(label), label)
            {
                continue;
            }
            let contains = match feature.kind {
                Kind::Area(i) => inside(&self.overlay.areas[i].ring, x, y),
                Kind::Segment(i, j) => {
                    let line = &self.overlay.roads[i];
                    let a = line.pts[j];
                    let b = line.pts[j + 1];
                    let ax = (a.0 - x) * mx;
                    let ay = (a.1 - y) * my;
                    let dx = (b.0 - a.0) * mx;
                    let dy = (b.1 - a.1) * my;
                    let d2 = dx * dx + dy * dy;
                    let t = if d2 > 0.0 {
                        (-(ax * dx + ay * dy) / d2).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    (ax + t * dx).powi(2) + (ay + t * dy).powi(2)
                        <= (line.width_m as f64 * 0.5).powi(2)
                }
            };
            if contains {
                label = feature.class;
            }
        }
        self.cache.insert(key, label);
        label
    }
}

fn inside(ring: &[(f64, f64)], x: f64, y: f64) -> bool {
    let mut result = false;
    let mut j = ring.len() - 1;
    for i in 0..ring.len() {
        let (xi, yi) = ring[i];
        let (xj, yj) = ring[j];
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            result = !result;
        }
        j = i;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Polygon, Polyline};
    use std::sync::Arc;
    #[test]
    fn shared_cells_ignore_tile_feature_order_and_remote_geometry() {
        let area = Polygon {
            class: 6,
            ring: Arc::new(vec![(13.0, 52.0), (14.0, 52.0), (14.0, 53.0), (13.0, 53.0)]),
        };
        let road = Polyline {
            class: 2,
            width_m: 8.0,
            pts: Arc::new(vec![(13.4, 52.0), (13.4, 53.0)]),
        };
        let a = SemOverlayPerTile {
            areas: vec![area.clone()],
            roads: vec![road.clone()],
        };
        let b = SemOverlayPerTile {
            areas: vec![
                Polygon {
                    class: 1,
                    ring: Arc::new(vec![(15.0, 54.0), (16.0, 54.0), (16.0, 55.0)]),
                },
                area,
            ],
            roads: vec![road],
        };
        let mut left = LabelField::new(&a);
        let mut right = LabelField::new(&b);
        for i in 0..1000 {
            let x = 13.399 + i as f64 * 0.000002;
            assert_eq!(left.label(x, 52.5), right.label(x, 52.5));
        }
        assert_eq!(left.label(13.4, 52.5), 2);
        assert_eq!(left.label(13.5, 52.5), 6);
    }
}
