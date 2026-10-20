//! The Stokes drift of a wave state as a field to sample anywhere on the mesh
//! and at any depth, for particles (TODO F.4 coupling; [`crate::particles`]).
//!
//! The drift of a spectrum in depth `d` at height `z` (≤ 0) below the mean
//! surface is (Kenyon 1969; [`SpectralGrid::stokes_drift`])
//!
//! ```text
//! u_s(z) = Σ_i b_i cosh(2k_i(z + d))/sinh²(k_i d),   b_i = σ_i k_i Δσ_i Δθ Σ_j E_ij e_θj,
//! ```
//!
//! so per node the spectrum compresses, without approximation, to one vector
//! `b_i` per frequency with its wavenumber `k_i`. [`StokesDriftField`] keeps
//! these and the depth at every node. At a point it interpolates them by the
//! element's nodal basis, as the particle trackers sample the flow, and sums
//! the exact profile: `n_freq` exponentials per sample, whatever the number of
//! directions. At a node this is [`WaveModel2D::stokes_drift`] to round-off.

use crate::operators::MeshTransfer2D;
use crate::types::ElementIndex;

use super::model::WaveModel2D;
use super::spectrum::{SpectralGrid, stokes_decay};
use super::state::WaveSolution;

/// The Stokes drift of one wave state (see the module docs).
#[derive(Clone, Debug)]
pub struct StokesDriftField {
    n_freq: usize,
    n_points: usize,
    /// Nodes per element
    n_nodes: usize,
    /// Depth per node (m), as the wave model's
    depth: Vec<f64>,
    /// Per frequency and node (`[i · n_points + p]`): k and the vector b
    k: Vec<f64>,
    bx: Vec<f64>,
    by: Vec<f64>,
}

impl StokesDriftField {
    /// The drift of the wave state `n` of `model`.
    pub fn new(model: &WaveModel2D, n: &WaveSolution) -> Self {
        let grid: &SpectralGrid = &model.grid;
        let (nf, nd, np) = (grid.n_freq(), grid.n_dir(), model.n_points());
        let mut field = Self {
            n_freq: nf,
            n_points: np,
            n_nodes: model.ops.n_nodes,
            depth: model.depth().to_vec(),
            k: vec![0.0; nf * np],
            bx: vec![0.0; nf * np],
            by: vec![0.0; nf * np],
        };
        let mut e = vec![0.0; grid.n_components()];
        for p in 0..np {
            model.energy_spectrum_into(n, p, &mut e);
            for i in 0..nf {
                let k = model.wavenumber(i, p);
                let w = grid.sigma[i] * k * grid.d_sigma[i] * grid.d_theta;
                let row = &e[i * nd..(i + 1) * nd];
                let bx: f64 = row.iter().zip(&grid.cos_theta).map(|(e, c)| e * c).sum();
                let by: f64 = row.iter().zip(&grid.sin_theta).map(|(e, s)| e * s).sum();
                let ip = i * np + p;
                field.k[ip] = k;
                field.bx[ip] = w * bx;
                field.by[ip] = w * by;
            }
        }
        field
    }

    /// No drift anywhere, on `n_elements` elements of `n_nodes` nodes.
    pub fn zero(n_elements: usize, n_nodes: usize) -> Self {
        let np = n_elements * n_nodes;
        Self {
            n_freq: 0,
            n_points: np,
            n_nodes,
            depth: vec![1.0; np],
            k: Vec::new(),
            bx: Vec::new(),
            by: Vec::new(),
        }
    }

    /// The field on another mesh (`n_nodes` per element): the depth, and each
    /// frequency's `k` and `b`, interpolated by `transfer` from this field's
    /// nodes onto the other mesh's (the wave model on a coarse mesh, the
    /// particles on the circulation's; [`super::WaveCoupling2D::stokes_drift`]).
    /// The depth and `k` are kept positive where a higher-order basis
    /// overshoots.
    pub fn transferred(&self, transfer: &MeshTransfer2D, n_nodes: usize) -> Self {
        assert_eq!(
            transfer.n_source_points(),
            self.n_points,
            "a transfer from this field's mesh"
        );
        let np = transfer.n_target_points();
        assert_eq!(np % n_nodes, 0, "{n_nodes} nodes per target element");
        let rows = |field: &[f64], floor: Option<f64>| -> Vec<f64> {
            let mut out = vec![0.0; self.n_freq * np];
            for (source, target) in field
                .chunks_exact(self.n_points)
                .zip(out.chunks_exact_mut(np))
            {
                transfer.apply_into(source, target);
                if let Some(floor) = floor {
                    target.iter_mut().for_each(|x| *x = x.max(floor));
                }
            }
            out
        };
        let depth_floor = self.depth.iter().cloned().fold(f64::INFINITY, f64::min);
        let k_floor = self.k.iter().cloned().fold(f64::INFINITY, f64::min);
        Self {
            n_freq: self.n_freq,
            n_points: np,
            n_nodes,
            depth: transfer
                .apply(&self.depth)
                .into_iter()
                .map(|d| d.max(depth_floor))
                .collect(),
            k: rows(&self.k, Some(k_floor)),
            bx: rows(&self.bx, None),
            by: rows(&self.by, None),
        }
    }

    /// Nodes of the mesh the field lives on.
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// The wave model's depth (m) at the point of element `element` where the
    /// nodal basis takes the values `weights`.
    pub fn depth(&self, element: ElementIndex, weights: &[f64]) -> f64 {
        let base = element.as_usize() * self.n_nodes;
        dot(weights, &self.depth[base..base + self.n_nodes])
    }

    /// The Stokes drift (m/s, mesh axes) at `depth_below` (m, ≥ 0) under the
    /// surface at the point of element `element` where the nodal basis takes
    /// the values `weights`. Below the bed it is the bed's.
    pub fn at(&self, element: ElementIndex, weights: &[f64], depth_below: f64) -> [f64; 2] {
        debug_assert_eq!(weights.len(), self.n_nodes);
        let base = element.as_usize() * self.n_nodes;
        let nodes = base..base + self.n_nodes;
        let d = dot(weights, &self.depth[nodes.clone()]);
        let z = -depth_below.clamp(0.0, d);
        let (mut ux, mut uy) = (0.0, 0.0);
        for i in 0..self.n_freq {
            let row = i * self.n_points + base..i * self.n_points + nodes.end;
            let (bx, by) = (
                dot(weights, &self.bx[row.clone()]),
                dot(weights, &self.by[row.clone()]),
            );
            if bx == 0.0 && by == 0.0 {
                continue;
            }
            let decay = stokes_decay(dot(weights, &self.k[row]), d, z);
            ux += bx * decay;
            uy += by * decay;
        }
        [ux, uy]
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::mesh::{Bathymetry2D, Mesh2D};
    use crate::operators::{DGOperators2D, GeometricFactors2D};

    const G: f64 = 9.81;

    fn model(bed: impl Fn(f64, f64) -> f64) -> WaveModel2D {
        let mesh = Mesh2D::uniform_periodic(0.0, 400.0, 0.0, 300.0, 4, 3);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, bed);
        WaveModel2D::new(
            Arc::new(mesh),
            Arc::new(ops),
            Arc::new(geom),
            &bathymetry,
            SpectralGrid::new(0.05, 0.6, 24, 24),
            G,
        )
    }

    /// At every node, and at every depth down to below the bed, the field is
    /// the spectral sum of the wave model, in shallow and deep water; between
    /// nodes of a uniform sea it is the same everywhere.
    #[test]
    fn the_field_is_the_spectral_stokes_drift() {
        let m = model(|x, _| -(3.0 + 0.1 * x));
        let mut n = m.uniform_state(&m.grid.jonswap(1.5, 7.0, 3.3, 0.7, 4.0));
        // Not uniform: scale each node's spectrum differently
        for p in 0..n.n_points {
            let factor = 1.0 + 0.01 * p as f64;
            n.spectrum_mut(p).iter_mut().for_each(|x| *x *= factor);
        }
        let field = StokesDriftField::new(&m, &n);
        let nn = m.ops.n_nodes;
        let mut weights = vec![0.0; nn];
        for z in [0.0, 0.3, 2.0, 10.0, 100.0] {
            let expected = m.stokes_drift(&n, -z);
            for p in 0..m.n_points() {
                weights.fill(0.0);
                weights[p % nn] = 1.0;
                let got = field.at(ElementIndex::new(p / nn), &weights, z);
                for d in 0..2 {
                    let scale = expected[p][0].hypot(expected[p][1]);
                    assert!(
                        (got[d] - expected[p][d]).abs() <= 1e-12 * scale + 1e-300,
                        "node {p}, z {z}: {got:?} against {:?}",
                        expected[p]
                    );
                }
            }
        }
        // A uniform sea over a flat bed is sampled the same between the nodes
        let m = model(|_, _| -30.0);
        let n = m.uniform_state(&m.grid.jonswap(1.5, 7.0, 3.3, 0.7, 4.0));
        let field = StokesDriftField::new(&m, &n);
        let at_node = m.stokes_drift(&n, -1.0)[0];
        let w = m.ops.interpolation_weights(0.31, -0.62);
        let got = field.at(ElementIndex::new(5), &w, 1.0);
        assert!((got[0] - at_node[0]).abs() < 1e-14 && (got[1] - at_node[1]).abs() < 1e-14);
        assert!(at_node[0] > 0.0 && at_node[1] > 0.0, "towards 0.7 rad");
        assert_eq!(
            StokesDriftField::zero(12, nn).at(ElementIndex::new(3), &w, 0.0),
            [0.0, 0.0]
        );
    }
}
