//! Drag of fish-farm net cages in the 3D model.
//!
//! The 3D form of [`crate::source::CageDrag2D`]: the netting removes
//! momentum per unit volume at `S = −½ C_d a |u| u` wherever it hangs, from
//! the surface (the collar floats, so the net rides `η`) down to the net
//! depth `d`. On σ-layer `l` of a column of depth `D` that is a linear sink
//!
//! ```text
//!     ∂u_l/∂t = −λ_l u_l,    λ_l = ½ C_d a φ χ_l |u_l|,
//! ```
//!
//! with `φ` the node's share of the cage footprint (the weights of
//! [`CageDrag2D`]) and `χ_l` the fraction of the layer above the net bottom
//! `z = η − d` ([`net_fraction`]). Only the layers the net reaches feel it,
//! each with its own speed. Over the column, `Σ_l Δσ_l χ_l = min(d, D)/D`, so
//! for a velocity uniform in depth the column-mean rate
//! `Λ̄ = Σ_l Δσ_l λ_l` is the 2D cage drag's `½ C_d a φ |ū| min(d, D)/D`.
//!
//! # Time discretisation (with [`crate::time::ModeSplitIntegrator`])
//!
//! As for the bottom drag ([`crate::physics::bottom_drag`]), the rates
//! `λ_l` are frozen at `tⁿ` over each baroclinic step
//! ([`crate::time::ModeSplitPhysics::layer_drag_into`]):
//! - the depth mean feels `−Λ̄ ū` in the barotropic pass, point-implicitly
//!   (`(hu, hv) ← (hu, hv)/(1 + Δt_s Λ̄)` per stage);
//! - the slow forcing `G` carries the rest, `−D Σ_l Δσ_l λ_l (u_l − ū)`,
//!   which vanishes without vertical shear: unsheared flow feels exactly the
//!   2D cage drag;
//! - the vertical diffusion takes `−λ_l u_lⁿ⁺¹` on every layer, implicitly.
//!   It shapes the profile only: the splitter resets the depth mean to the
//!   pass's `ū` afterwards.
//!
//! Freezing `λ` makes the drag first order in time, like the point-implicit
//! 2D drag. The 2D module must not carry the cages as well.
//!
//! The rear net sees the local velocity, so the reduction of the current
//! inside and behind the cage is left to the resolved flow, as in 2D.

use crate::source::{CageDrag2D, CageNode};
use crate::vertical::SigmaGrid;

/// Fraction of σ-layer `level` that lies above the bottom of a net hanging
/// `net_depth` (m) below the surface, in a column of depth `depth > 0`.
#[inline]
pub fn net_fraction(sigma: &SigmaGrid, level: usize, net_depth: f64, depth: f64) -> f64 {
    net_fraction_on([sigma.sigma_w(), sigma.d_sigma()], level, net_depth, depth)
}

/// [`net_fraction`] on the σ-surfaces and layer thicknesses
/// `[σ_w, Δσ]`.
#[inline]
pub fn net_fraction_on(
    [sigma_w, d_sigma]: [&[f64]; 2],
    level: usize,
    net_depth: f64,
    depth: f64,
) -> f64 {
    let net_bottom = (-net_depth / depth).max(-1.0);
    let (bottom, top) = (sigma_w[level].max(net_bottom), sigma_w[level + 1]);
    ((top - bottom) / d_sigma[level]).max(0.0)
}

/// `Σ φ ½ C_d a χ_l` (1/m) of the cage entries of one node on σ-layer
/// `level`, in a column of depth `depth > 0`: the layer's drag rate per
/// unit speed, `λ_l = c_l |u_l|`.
#[inline]
pub fn layer_coefficient(entries: &[CageNode], sigma: &SigmaGrid, level: usize, depth: f64) -> f64 {
    layer_coefficient_on(
        entries.len(),
        |e| (entries[e].coefficient, entries[e].net_depth),
        [sigma.sigma_w(), sigma.d_sigma()],
        level,
        depth,
    )
}

/// [`layer_coefficient`] of `n` entries, `entry(e) = (coefficient, net
/// depth)`, on `[σ_w, Δσ]`.
#[inline]
pub fn layer_coefficient_on(
    n: usize,
    entry: impl Fn(usize) -> (f64, f64),
    levels: [&[f64]; 2],
    level: usize,
    depth: f64,
) -> f64 {
    (0..n)
        .map(|e| {
            let (coefficient, net_depth) = entry(e);
            coefficient * net_fraction_on(levels, level, net_depth, depth)
        })
        .sum()
}

/// The cage-drag rates `λ_l = c_l |u_l|` (1/s) of every layer of a column of
/// depth `depth` (m) under `n` cage entries (`entry(e) = (coefficient, net
/// depth)`) into `rate`, on `[σ_w, Δσ]`: with the layers' velocities
/// `[u, v]`, or the depth mean's `mean` in a column shallower than
/// `thin_depth`; zero where it is dry. The speeds are
/// [`crate::math::hypot`]'s (the GPU's bits).
#[allow(clippy::too_many_arguments)]
pub fn cage_rates(
    n: usize,
    entry: impl Fn(usize) -> (f64, f64),
    levels: [&[f64]; 2],
    depth: f64,
    thin_depth: f64,
    [u, v]: [&[f64]; 2],
    mean: [f64; 2],
    rate: &mut [f64],
) {
    if depth <= 0.0 {
        rate.fill(0.0);
        return;
    }
    let thin = depth < thin_depth;
    for (l, r) in rate.iter_mut().enumerate() {
        let [u, v] = if thin { mean } else { [u[l], v[l]] };
        *r = layer_coefficient_on(n, &entry, levels, l, depth) * crate::math::hypot(u, v);
    }
}

/// The cage entries of every node of element `k` (`n_nodes` nodes), in
/// order: call `f(i, entries)` for each node `i` that has any.
pub fn for_each_caged_node(
    cages: &CageDrag2D,
    k: usize,
    n_nodes: usize,
    mut f: impl FnMut(usize, &[CageNode]),
) {
    let nodes = cages.nodes();
    let (first, end) = (k * n_nodes, (k + 1) * n_nodes);
    let mut start = cages.first_at_or_after(first);
    while start < nodes.len() && nodes[start].node < end {
        let node = nodes[start].node;
        let stop = start + nodes[start..].partition_point(|n| n.node == node);
        f(node - first, &nodes[start..stop]);
        start = stop;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vertical::{SigmaGrid, UniformStretching};

    #[test]
    fn layers_above_the_net_bottom_are_inside() {
        // Four layers of a 20 m column, net to 7.5 m: the top layer whole,
        // the second half, the lower two not at all
        let sigma = SigmaGrid::new(4, UniformStretching);
        let fractions: Vec<f64> = (0..4).map(|l| net_fraction(&sigma, l, 7.5, 20.0)).collect();
        let expect = [0.0, 0.0, 0.5, 1.0];
        for (f, e) in fractions.iter().zip(expect) {
            assert!((f - e).abs() < 1e-14, "{fractions:?}");
        }
        // A net deeper than the column reaches every layer
        for l in 0..4 {
            assert!((net_fraction(&sigma, l, 30.0, 20.0) - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn the_column_sum_is_the_2d_net_share() {
        // Σ Δσ_l χ_l = min(d, D)/D on stretched levels too
        let sigma = SigmaGrid::new(
            7,
            crate::vertical::SongHaidvogelStretching::new(5.0, 0.4, 10.0),
        );
        for (d, depth) in [(3.0, 40.0), (17.3, 40.0), (40.0, 40.0), (55.0, 40.0)] {
            let sum: f64 = (0..7)
                .map(|l| sigma.d_sigma()[l] * net_fraction(&sigma, l, d, depth))
                .sum();
            let expect = f64::min(d, depth) / depth;
            assert!((sum - expect).abs() < 1e-14, "d {d}: {sum} vs {expect}");
        }
    }

    #[test]
    fn overlapping_cages_add_layer_by_layer() {
        let sigma = SigmaGrid::new(2, UniformStretching);
        let entry = |coefficient, net_depth| CageNode {
            node: 0,
            coefficient,
            net_depth,
        };
        // Shallow and deep net over a 10 m column of two layers
        let entries = [entry(0.01, 5.0), entry(0.02, 10.0)];
        assert!((layer_coefficient(&entries, &sigma, 1, 10.0) - 0.03).abs() < 1e-15);
        assert!((layer_coefficient(&entries, &sigma, 0, 10.0) - 0.02).abs() < 1e-15);
    }
}
