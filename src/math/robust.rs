// src/math/robust.rs
//! Robust statistical functions for point cloud alignment and processing.

use std::cmp::Ordering;

/// Computes the percentile of a mutable slice in-place.
///
/// # Arguments
/// * `v` - The values to analyze (will be partially sorted)
/// * `q` - The percentile in [0,1], where 0 = min, 0.5 = median, 1 = max
///
/// # Returns
/// The percentile value, or NaN if the slice is empty.
///
/// # Complexity
/// O(n) average case using quickselect via `select_nth_unstable_by`.
pub fn percentile_in_place(v: &mut [f32], q: f32) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    
    let q = q.clamp(0.0, 1.0);
    let k = ((v.len() - 1) as f32 * q).round() as usize;
    
    let (_, nth, _) = v.select_nth_unstable_by(k, |a, b| {
        a.partial_cmp(b).unwrap_or(Ordering::Equal)
    });
    
    *nth
}

/// Computes the median of a mutable slice in-place.
///
/// # Arguments
/// * `v` - The values to analyze (will be partially sorted)
///
/// # Returns
/// The median value, or NaN if the slice is empty.
///
/// # Complexity
/// O(n) average case using quickselect.
pub fn median_in_place(v: &mut [f32]) -> f32 {
    percentile_in_place(v, 0.5)
}

/// Computes the median by taking ownership of the input vector.
///
/// # Arguments
/// * `v` - The values to analyze (consumed)
///
/// # Returns
/// The median value, or None if the vector is empty.
///
/// # Complexity
/// O(n) average case using quickselect.
pub fn median(mut v: Vec<f32>) -> Option<f32> {
    if v.is_empty() {
        return None;
    }
    
    let k = v.len() / 2;
    v.select_nth_unstable_by(k, |a, b| {
        a.partial_cmp(b).unwrap_or(Ordering::Equal)
    });
    
    Some(v[k])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_odd() {
        let values = vec![3.0, 1.0, 5.0, 2.0, 4.0];
        assert_eq!(median(values), Some(3.0));
    }

    #[test]
    fn test_median_even() {
        // Note: This returns the lower middle element for even-sized arrays
        let values = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(median(values), Some(2.0));
    }

    #[test]
    fn test_percentile_quartiles() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile_in_place(&mut values, 0.0), 1.0);
        assert_eq!(percentile_in_place(&mut values, 0.25), 2.0);
        assert_eq!(percentile_in_place(&mut values, 0.5), 3.0);
        assert_eq!(percentile_in_place(&mut values, 0.75), 4.0);
        assert_eq!(percentile_in_place(&mut values, 1.0), 5.0);
    }

    #[test]
    fn test_empty() {
        assert_eq!(median(vec![]), None);
        assert!(percentile_in_place(&mut [], 0.5).is_nan());
    }
}
