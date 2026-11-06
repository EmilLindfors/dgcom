//! Land masking for 2D meshes.
//!
//! Provides wet/dry cell classification based on coastline and bathymetry data,
//! similar to ROMS land masking.
//!
//! # Example
//!
//! ```ignore
//! use dg::mesh::{Mesh2D, LandMask2D};
//! use dg::io::{CoastlineData, GeoTiffBathymetry, LocalProjection, GeoBoundingBox};
//!
//! // Load geographic data
//! let bbox = GeoBoundingBox::new(8.0, 63.5, 9.5, 64.0);
//! let coastline = CoastlineData::load("data/GSHHS_f_L1.shp", &bbox)?;
//! let bathy = GeoTiffBathymetry::load("data/bathymetry.tif")?;
//! let proj = LocalProjection::new(63.75, 8.75);
//!
//! // Land masks: from the coastline, or from the bathymetry (wet where at
//! // least 1 m deep)
//! let by_coast = LandMask2D::from_coastline(&mesh, &ops, &coastline, &proj);
//! let by_depth = LandMask2D::from_bathymetry(&mesh, &ops, &bathy, &proj, 1.0);
//!
//! // Check if element is wet
//! if by_coast.is_wet(5) && by_depth.is_wet(5) {
//!     // Process water element
//! }
//! ```

use crate::io::{CoastlineData, CoordinateProjection, GeoTiffBathymetry};
use crate::mesh::Mesh2D;
use crate::operators::DGOperators2D;
use crate::types::ElementIndex;

/// Land mask for 2D mesh elements.
///
/// Classifies each element as wet (water) or dry (land) based on
/// coastline polygons and/or bathymetry data.
#[derive(Clone)]
pub struct LandMask2D {
    /// True if element is wet (water), false if land
    pub wet: Vec<bool>,
    /// Fraction of element nodes that are wet (0.0 to 1.0)
    pub wet_fraction: Vec<f64>,
    /// Number of elements
    pub n_elements: usize,
}

impl LandMask2D {
    /// Create a mask where all elements are wet.
    pub fn all_wet(n_elements: usize) -> Self {
        Self {
            wet: vec![true; n_elements],
            wet_fraction: vec![1.0; n_elements],
            n_elements,
        }
    }

    /// Create a mask where all elements are dry.
    pub fn all_dry(n_elements: usize) -> Self {
        Self {
            wet: vec![false; n_elements],
            wet_fraction: vec![0.0; n_elements],
            n_elements,
        }
    }

    /// Create land mask from coastline data only.
    ///
    /// An element is considered wet if its centroid is in water.
    pub fn from_coastline<P: CoordinateProjection>(
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        coastline: &CoastlineData,
        projection: &P,
    ) -> Self {
        let mut wet = Vec::with_capacity(mesh.n_elements);
        let mut wet_fraction = Vec::with_capacity(mesh.n_elements);

        for k in ElementIndex::iter(mesh.n_elements) {
            // Check all nodes in element
            let mut wet_nodes = 0;

            for i in 0..ops.n_nodes {
                let r = ops.nodes_r[i];
                let s = ops.nodes_s[i];
                let [x, y] = mesh.reference_to_physical(k, r, s);
                let (lat, lon) = projection.xy_to_geo(x, y);

                if coastline.is_water(lat, lon) {
                    wet_nodes += 1;
                }
            }

            let frac = wet_nodes as f64 / ops.n_nodes as f64;
            wet_fraction.push(frac);

            // Element is wet if majority of nodes are in water
            wet.push(frac > 0.5);
        }

        Self {
            wet,
            wet_fraction,
            n_elements: mesh.n_elements,
        }
    }

    /// Create land mask from bathymetry data only.
    ///
    /// An element is considered wet if its depth is greater than the minimum depth.
    pub fn from_bathymetry<P: CoordinateProjection>(
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        bathymetry: &GeoTiffBathymetry,
        projection: &P,
        min_depth: f64,
    ) -> Self {
        let mut wet = Vec::with_capacity(mesh.n_elements);
        let mut wet_fraction = Vec::with_capacity(mesh.n_elements);

        for k in ElementIndex::iter(mesh.n_elements) {
            let mut wet_nodes = 0;

            for i in 0..ops.n_nodes {
                let r = ops.nodes_r[i];
                let s = ops.nodes_s[i];
                let [x, y] = mesh.reference_to_physical(k, r, s);
                let (lat, lon) = projection.xy_to_geo(x, y);

                // Check if depth exists and is deep enough
                if let Some(depth) = bathymetry.get_depth(lat, lon) {
                    // Depth is negative (below sea level), so -depth > min_depth
                    if -depth >= min_depth {
                        wet_nodes += 1;
                    }
                }
            }

            let frac = wet_nodes as f64 / ops.n_nodes as f64;
            wet_fraction.push(frac);
            wet.push(frac > 0.5);
        }

        Self {
            wet,
            wet_fraction,
            n_elements: mesh.n_elements,
        }
    }

    /// Check if an element is wet.
    #[inline]
    pub fn is_wet(&self, k: usize) -> bool {
        self.wet[k]
    }

    /// Check if an element is dry.
    #[inline]
    pub fn is_dry(&self, k: usize) -> bool {
        !self.wet[k]
    }

    /// Get the wet fraction for an element (0.0 = all land, 1.0 = all water).
    #[inline]
    pub fn wet_fraction(&self, k: usize) -> f64 {
        self.wet_fraction[k]
    }

    /// Get the number of wet elements.
    pub fn wet_count(&self) -> usize {
        self.wet.iter().filter(|&&w| w).count()
    }

    /// Get the number of dry elements.
    pub fn dry_count(&self) -> usize {
        self.wet.iter().filter(|&&w| !w).count()
    }

    /// Get indices of all wet elements.
    pub fn wet_elements(&self) -> Vec<usize> {
        self.wet
            .iter()
            .enumerate()
            .filter_map(|(i, &w)| if w { Some(i) } else { None })
            .collect()
    }

    /// Get indices of all dry elements.
    pub fn dry_elements(&self) -> Vec<usize> {
        self.wet
            .iter()
            .enumerate()
            .filter_map(|(i, &w)| if !w { Some(i) } else { None })
            .collect()
    }

    /// Set the wet status for an element.
    pub fn set_wet(&mut self, k: usize, is_wet: bool) {
        self.wet[k] = is_wet;
        self.wet_fraction[k] = if is_wet { 1.0 } else { 0.0 };
    }

    /// Get statistics about the land mask.
    pub fn statistics(&self) -> LandMaskStatistics {
        let wet_count = self.wet_count();
        let partial_count = self
            .wet_fraction
            .iter()
            .filter(|&&f| f > 0.0 && f < 1.0)
            .count();

        LandMaskStatistics {
            total_elements: self.n_elements,
            wet_elements: wet_count,
            dry_elements: self.n_elements - wet_count,
            partial_elements: partial_count,
        }
    }
}

/// Statistics about a land mask.
#[derive(Debug, Clone)]
pub struct LandMaskStatistics {
    /// Total number of elements
    pub total_elements: usize,
    /// Number of wet elements
    pub wet_elements: usize,
    /// Number of dry elements
    pub dry_elements: usize,
    /// Number of elements with partial wet fraction
    pub partial_elements: usize,
}

impl std::fmt::Display for LandMaskStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Land Mask Statistics:")?;
        writeln!(f, "  Total elements: {}", self.total_elements)?;
        writeln!(
            f,
            "  Wet elements: {} ({:.1}%)",
            self.wet_elements,
            100.0 * self.wet_elements as f64 / self.total_elements as f64
        )?;
        writeln!(
            f,
            "  Dry elements: {} ({:.1}%)",
            self.dry_elements,
            100.0 * self.dry_elements as f64 / self.total_elements as f64
        )?;
        write!(f, "  Partial elements: {}", self.partial_elements)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_wet() {
        let mask = LandMask2D::all_wet(10);
        assert_eq!(mask.wet_count(), 10);
        assert_eq!(mask.dry_count(), 0);
        assert!(mask.is_wet(5));
    }

    #[test]
    fn test_all_dry() {
        let mask = LandMask2D::all_dry(10);
        assert_eq!(mask.wet_count(), 0);
        assert_eq!(mask.dry_count(), 10);
        assert!(mask.is_dry(5));
    }

    #[test]
    fn test_set_wet() {
        let mut mask = LandMask2D::all_wet(10);
        mask.set_wet(5, false);
        assert!(mask.is_dry(5));
        assert!(mask.is_wet(4));
    }
}
