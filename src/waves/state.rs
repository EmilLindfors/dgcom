//! The wave model's state: action density per spectral component and DG node.

use crate::time::Integrable;

/// Action density `N(σ, θ)` (m²·s/rad², i.e. variance density over σ) at every DG
/// node for every spectral component, `data[p · n_components + c]` with
/// component `c = i n_θ + j` and node `p = k n_nodes + i` (element-major, as
/// the 2D solutions). Node-major: each node's spectrum is contiguous, as the
/// node passes (implicit refraction and shifting, the sources) and every
/// diagnostic read it, and a frequency's directions at a node are contiguous
/// for the DG propagation's vector kernel (`waves/propagation.rs`).
#[derive(Clone, Debug)]
pub struct WaveSolution {
    pub n_components: usize,
    pub n_points: usize,
    pub data: Vec<f64>,
}

impl WaveSolution {
    /// No energy anywhere.
    pub fn new(n_components: usize, n_points: usize) -> Self {
        Self {
            n_components,
            n_points,
            data: vec![0.0; n_components * n_points],
        }
    }

    /// The spectrum at node `p`, one value per component.
    pub fn spectrum(&self, p: usize) -> &[f64] {
        &self.data[p * self.n_components..(p + 1) * self.n_components]
    }

    pub fn spectrum_mut(&mut self, p: usize) -> &mut [f64] {
        &mut self.data[p * self.n_components..(p + 1) * self.n_components]
    }

    /// Component `c` at every node, in node order.
    pub fn component(&self, c: usize) -> impl ExactSizeIterator<Item = f64> + '_ {
        self.data[c..].iter().step_by(self.n_components).copied()
    }

    /// The same action spectrum `n[c]` at every node.
    pub fn fill_uniform(&mut self, n: &[f64]) {
        assert_eq!(n.len(), self.n_components);
        for spectrum in self.data.chunks_exact_mut(self.n_components) {
            spectrum.copy_from_slice(n);
        }
    }
}

impl Integrable for WaveSolution {
    fn scale(&mut self, c: f64) {
        self.data.iter_mut().for_each(|x| *x *= c);
    }

    fn axpy(&mut self, c: f64, other: &Self) {
        self.data
            .iter_mut()
            .zip(&other.data)
            .for_each(|(x, y)| *x += c * y);
    }
}
