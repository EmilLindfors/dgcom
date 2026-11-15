//! The GLL subcells of a quadrilateral element, shared by the 2D wet/dry
//! scheme and the 3D layers that must move the same water.
//!
//! In `SWEFormulation2D::WetDry` an element with a shallow node replaces its
//! volume term by a finite-volume update on the GLL subcells, per GLL line
//! and in both directions (see `swe_2d_split_form`): along a line of nodes
//! `0..n₁`, subcell `a` has width `w_a` (the GLL weight) and exchanges the
//! flux `F̂_{a+½}` with subcell `a + 1` through an interface whose
//! contravariant direction `m_{a+½}` is the telescoping sum of the
//! flux-differencing metric ([`telescoped_interfaces`]). The element's
//! nodal mass rate is then
//!
//! ```text
//!     J_i dh_i/dt = −Σ_dir (F̂_{a+½} − F̂_{a−½}) / w_a  − (face terms),
//! ```
//!
//! with the end "interfaces" the node's own physical flux, which the surface
//! term replaces by `F*`. The 3D layers of such an element carry their
//! volume, tracers and momentum through the same interfaces
//! ([`crate::solver::rhs::LayerTransport`]), so that their column sums are
//! the 2D update node by node.
//!
//! Interfaces are numbered per element as `slot = (dir · n₁ + line) · (n₁ − 1)
//! + a` for the interface between nodes `a` and `a + 1` of `line` in
//! direction `dir` (0: r-lines, nodes `line · n₁ + a`; 1: s-lines, nodes
//! `a · n₁ + line`).

use crate::operators::{DGOperators2D, GeometricFactors2D};

/// Number of interior subcell interfaces of an element with `n_1d` nodes per
/// direction: `n₁ − 1` per line, `n₁` lines, two directions.
#[inline]
pub fn subcell_interfaces(n_1d: usize) -> usize {
    2 * n_1d * (n_1d - 1)
}

/// Slot of the interface between nodes `a` and `a + 1` of `line` in
/// direction `dir` (see the module docs).
#[inline]
pub fn subcell_slot(n_1d: usize, dir: usize, line: usize, a: usize) -> usize {
    (dir * n_1d + line) * (n_1d - 1) + a
}

/// Node of position `a` on `line` in direction `dir`.
#[inline]
pub fn line_node(n_1d: usize, dir: usize, line: usize, a: usize) -> usize {
    if dir == 0 {
        line * n_1d + a
    } else {
        a * n_1d + line
    }
}

/// The subcell interface directions of one line (`n₁ + 1` values into `out`)
/// from the contravariant vectors `metric` of the line's direction at its
/// nodes: `out[0]` and `out[n₁]` are the end nodes' own, and
/// `out[a + 1] = Σ_{b ≤ a} Σ_{c≠b} 2 w_b D_bc ½(Ja_b + Ja_c)` (the telescoping
/// sum of the flux-differencing metric, so the subcell update is the split
/// form's on constant states; by summation by parts its last step lands on
/// the end node's metric). On an affine element every interface takes the
/// one metric.
pub fn telescoped_interfaces(
    w: &[f64],
    d1: &[f64],
    metric: &[(f64, f64)],
    affine: bool,
    out: &mut [(f64, f64)],
) {
    let n1 = w.len();
    if affine {
        out[..=n1].fill(metric[0]);
        return;
    }
    out[0] = metric[0];
    // From zero: the summation-by-parts boundary term makes the first step
    // the first node's metric
    let mut m = (0.0, 0.0);
    for a in 0..n1 - 1 {
        for c in (0..n1).filter(|&c| c != a) {
            let q2 = 2.0 * w[a] * d1[a * n1 + c];
            m.0 += q2 * 0.5 * (metric[a].0 + metric[c].0);
            m.1 += q2 * 0.5 * (metric[a].1 + metric[c].1);
        }
        out[a + 1] = m;
    }
    out[n1] = metric[n1 - 1];
}

/// The contravariant vectors of direction `dir` (`J∇r` on r-lines, `J∇s` on
/// s-lines) at the nodes of `line` of element `k` into `metric` (`n₁`
/// values), as the 2D wet/dry kernel takes them; returns whether the element
/// is affine (one metric).
pub fn line_metric(
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    k: usize,
    dir: usize,
    line: usize,
    metric: &mut [(f64, f64)],
) -> bool {
    let n1 = ops.n_1d;
    let element = geom.element_geometry(k);
    if element.affine {
        let (r, s) = element.metric.contravariant();
        metric[..n1].fill(if dir == 0 { r } else { s });
        return true;
    }
    for (a, m) in metric[..n1].iter_mut().enumerate() {
        let (r, s) = geom.contravariant(k, line_node(n1, dir, line, a));
        *m = if dir == 0 { r } else { s };
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh2D;

    #[test]
    fn slots_cover_every_interface_once() {
        for n1 in 2..=5 {
            let mut seen = vec![false; subcell_interfaces(n1)];
            for dir in 0..2 {
                for line in 0..n1 {
                    for a in 0..n1 - 1 {
                        let slot = subcell_slot(n1, dir, line, a);
                        assert!(!seen[slot]);
                        seen[slot] = true;
                    }
                }
            }
            assert!(seen.iter().all(|&s| s));
        }
    }

    /// On an affine element every interface is the element's metric; on a
    /// general quadrilateral the ends are the end nodes' and the interior
    /// interfaces telescope (the last step lands on the end metric, the
    /// discrete metric identity of the averaged form).
    #[test]
    fn telescoped_interfaces_end_on_the_line_metric() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 2.0, 0.0, 1.0, 2, 1);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let n1 = ops.n_1d;
        let mut metric = vec![(0.0, 0.0); n1];
        let mut out = vec![(0.0, 0.0); n1 + 1];
        assert!(line_metric(&ops, &geom, 0, 0, 1, &mut metric));
        telescoped_interfaces(
            &ops.weights_1d,
            &ops.dr_1d_row_major,
            &metric,
            true,
            &mut out,
        );
        assert!(out.iter().all(|&m| m == metric[0]));
        // The general formula on the same constant metric telescopes to it
        telescoped_interfaces(
            &ops.weights_1d,
            &ops.dr_1d_row_major,
            &metric,
            false,
            &mut out,
        );
        for m in &out {
            assert!((m.0 - metric[0].0).abs() < 1e-13 && (m.1 - metric[0].1).abs() < 1e-13);
        }
    }
}
