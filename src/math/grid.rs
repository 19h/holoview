// src/math/grid.rs
//! Uniform spatial grid for efficient 2D neighbor queries.
//!
//! Used for accelerating point matching in tile alignment.

use std::collections::HashMap;

/// A uniform 2D spatial grid for efficient neighbor queries.
///
/// # Type Parameters
/// Points are stored as indices into an external array, enabling
/// efficient memory usage and flexible point data storage.
pub struct Grid {
    /// Cell size in world units (meters for our use case).
    cell: f32,
    
    /// Mapping from grid cell coordinates to point indices.
    map: HashMap<(i32, i32), Vec<usize>>,
}

impl Grid {
    /// Builds a spatial grid from 3D points (using only XY coordinates).
    ///
    /// # Arguments
    /// * `points_xy` - Array of 3D points (z-component is ignored)
    /// * `cell` - Cell size in world units
    ///
    /// # Returns
    /// A new Grid instance with all points indexed.
    ///
    /// # Complexity
    /// O(n) for n points.
    pub fn build(points_xy: &[[f32; 3]], cell: f32) -> Self {
        let mut map = HashMap::new();
        
        for (i, p) in points_xy.iter().enumerate() {
            let ix = (p[0] / cell).floor() as i32;
            let iy = (p[1] / cell).floor() as i32;
            
            map.entry((ix, iy))
                .or_insert_with(Vec::new)
                .push(i);
        }
        
        Self { cell, map }
    }
    
    /// Returns an iterator over point indices within radius r of position (x, y).
    ///
    /// # Arguments
    /// * `x` - X coordinate of query point
    /// * `y` - Y coordinate of query point
    /// * `r` - Search radius in world units
    ///
    /// # Returns
    /// Iterator yielding indices of points potentially within radius r.
    /// Note: This returns all points in grid cells that overlap the search circle,
    /// so the caller should perform exact distance checks if needed.
    ///
    /// # Complexity
    /// Expected O(1 + ρπr²) for uniform point density ρ.
    /// Worst case O(n) if all points fall in searched cells.
    pub fn neighbors<'a>(
        &'a self,
        x: f32,
        y: f32,
        r: f32,
    ) -> impl Iterator<Item = &usize> + 'a {
        let rx = (r / self.cell).ceil() as i32;
        let ix = (x / self.cell).floor() as i32;
        let iy = (y / self.cell).floor() as i32;
        
        (-rx..=rx)
            .flat_map(move |dx| {
                (-rx..=rx)
                    .filter_map(move |dy| {
                        self.map.get(&(ix + dx, iy + dy))
                    })
                    .flatten()
            })
    }
    
    /// Returns the number of indexed points.
    pub fn point_count(&self) -> usize {
        self.map.values().map(|v| v.len()).sum()
    }
    
    /// Returns the number of occupied grid cells.
    pub fn cell_count(&self) -> usize {
        self.map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_build() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [10.0, 10.0, 0.0],
        ];
        
        let grid = Grid::build(&points, 1.0);
        assert_eq!(grid.point_count(), 4);
    }

    #[test]
    fn test_grid_neighbors() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [10.0, 10.0, 0.0],
        ];
        
        let grid = Grid::build(&points, 1.0);
        
        // Query near origin should find first 3 points
        let neighbors: Vec<_> = grid.neighbors(0.5, 0.5, 2.5).cloned().collect();
        assert!(neighbors.contains(&0));
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2));
        assert!(!neighbors.contains(&3));
    }
}
