//! Adaptive implicit vertical advection.
//!
//! The explicit vertical advection of the 3D stages is stable up to a
//! vertical Courant number `|Ω|Δt/H_z` of about one. In a column just deeper
//! than the thin depth the layers are centimetres thick, and a flooding tide
//! moves water through them fast enough to set the step for the whole
//! domain: 2.2 s falling to 0.9 s on the Mausund sub-domain of the Frøya
//! coastline mesh, against 5–7 s for the internal waves elsewhere (TODO P1.3).
//!
//! Following Shchepetkin (2015), the vertical volume flux `Ω` at every
//! σ-surface is split into an explicit part `Ω_e` that the stages advect with
//! their own reconstruction ([`crate::solver::rhs::VerticalAdvection`]) and an
//! implicit remainder `Ω_i = Ω − Ω_e`, nonzero only where the explicit
//! Courant number would be too large. The split is set by the Courant number
//! of the cells on either side of the surface,
//!
//! ```text
//!     c_l = Δt (max(Ω_{l+½}, 0) + max(−Ω_{l−½}, 0)) / H_z,l     (the outflow of layer l),
//!     Ω_i = f(c) Ω,   c = max(c_l, c_{l+1}),
//!     f(c) = (c − c_min)² / (c (c + c_max − 2 c_min))   for c > c_min, else 0.
//! ```
//!
//! `f` is zero up to `c_min`, so below it the explicit scheme is untouched,
//! and `c (1 − f(c))` rises monotonically (its derivative is
//! `(c_max − c_min)²/(c + c_max − 2c_min)² > 0`) towards `c_max`: the explicit
//! part's outflow Courant number of every cell stays below `c_max` however
//! long the step (each surface's share of the cell's outflow is scaled by
//! `1 − f` at a Courant number at least the cell's own, and `1 − f` falls
//! with `c`).
//!
//! The Courant numbers are for the longest forward-Euler step of the 3D
//! stages' scheme, `Δt` over its SSP coefficient (`Δt` for SSP-RK3, `Δt/2`
//! for SSP-RK(4,3), `BarotropicFlux::euler_dt`): each stage is a convex
//! combination of such steps, so the explicit part keeps every stage within
//! the bounds below.
//!
//! The implicit part is first-order upwind and backward Euler. Each SSP
//! stage `u⁽ⁱ⁾ = Σ_k α_ik u⁽ᵏ⁾ + β_i Δt L(u⁽ⁱ⁻¹⁾)` advects with `Ω_e` of
//! `u⁽ⁱ⁻¹⁾`, and is then relaxed with `Ω_i` over `β_i Δt` (the integrator's
//! `step_with_relaxation`), solving for the stage's concentration `x` in every
//! column
//!
//! ```text
//!     H_z,l x_l + β_iΔt [Ω⁺_{l+½} x_l − Ω⁻_{l+½} x_{l+1} − Ω⁺_{l−½} x_{l−1} + Ω⁻_{l−½} x_l] = q_l,
//! ```
//!
//! with `q` the stage's inventory, `H_z` its layer thickness and
//! `Ω⁺ = max(Ω_i, 0)`, `Ω⁻ = max(−Ω_i, 0)`. Then:
//!
//! - **Constancy**: the stage's thickness is that of the explicit continuity
//!   plus `β_iΔt δΩ_i`, so a uniform field's inventory is `C (H_z + β_iΔt δΩ_i)`
//!   and `x = C` solves the system.
//! - **Conservation**: the matrix's columns sum to `H_z` (every flux leaves
//!   one cell and enters its neighbour), so `Σ_l H_z,l x_l = Σ_l q_l`.
//! - **Positivity and boundedness**: the matrix is an M-matrix (positive
//!   diagonal, non-positive off-diagonals, strictly diagonally dominant by
//!   columns) for any step, so its inverse is non-negative.
//! - The stage values the limiters, the density and the pressure gradient
//!   see are true concentrations: the stage's inventory over its own
//!   thickness.
//!
//! The implicit part diffuses with `κ ≈ |Ω_i|H_z/2` where it acts, as ROMS's
//! implicit part does; it is not reconstructed about the vertical reference
//! stratification ([`crate::physics::Hydrostatic3D::with_vertical_reference`]),
//! whose correction needs the explicit centred value. Columns below `c_min`,
//! where every stratified rest state lives, are advected as before, bit for
//! bit.
//!
//! Shchepetkin, A. F. (2015). An adaptive, Courant-number-dependent implicit
//! scheme for vertical advection in oceanic modeling. *Ocean Modelling* 91,
//! 38–69.

use crate::mesh::data::Bathymetry2D;
use crate::operators::GeometricFactors2D;
use crate::solver::core::blocks::{Pooled, for_each_block};
use crate::solver::rhs::transport_3d::{layer_thickness_of, layer_volume_of};
use crate::solver::state::Solution3D;
use crate::types::ElementIndex;

/// The adaptive split of the vertical advection into an explicit part and an
/// implicit remainder (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImplicitVerticalAdvection {
    /// Outflow Courant number up to which the advection is fully explicit.
    pub courant_min: f64,
    /// Bound of the explicit part's outflow Courant number.
    pub courant_max: f64,
}

impl Default for ImplicitVerticalAdvection {
    /// Fully explicit up to an outflow Courant number of 0.4, the explicit
    /// part below 0.8: within the stability range of every
    /// [`crate::solver::rhs::VerticalAdvection`] with the SSP-RK3 stages
    /// (upwind's is 1), and with SSP-RK(4,3)'s at their forward-Euler step.
    fn default() -> Self {
        Self::new(0.4, 0.8)
    }
}

impl ImplicitVerticalAdvection {
    /// Explicit up to `courant_min`, the explicit part bounded by
    /// `courant_max`.
    ///
    /// # Panics
    /// Unless `0 ≤ courant_min < courant_max`.
    pub fn new(courant_min: f64, courant_max: f64) -> Self {
        assert!(
            (0.0..courant_max).contains(&courant_min),
            "ImplicitVerticalAdvection: need 0 ≤ courant_min < courant_max, got \
             {courant_min} and {courant_max}"
        );
        Self {
            courant_min,
            courant_max,
        }
    }

    /// The implicit fraction `f(c)` at outflow Courant number `courant`.
    pub fn implicit_fraction(&self, courant: f64) -> f64 {
        let (a, b) = (self.courant_min, self.courant_max);
        if courant > a {
            (courant - a).powi(2) / (courant * (courant + b - 2.0 * a))
        } else {
            0.0
        }
    }

    /// Write the implicit part `Ω_i` of a column's `omega` (`n + 1`
    /// σ-surfaces, bed first, zero at both ends) to `implicit`, for layers
    /// of thickness `thickness` (m) and a step `dt`. Returns the largest
    /// outflow Courant number of the column's layers.
    pub fn split_column(
        &self,
        omega: &[f64],
        thickness: &[f64],
        dt: f64,
        implicit: &mut [f64],
    ) -> f64 {
        let n = thickness.len();
        debug_assert_eq!(omega.len(), n + 1);
        debug_assert_eq!(implicit.len(), n + 1);
        let courant = |l: usize| dt * (omega[l + 1].max(0.0) - omega[l].min(0.0)) / thickness[l];
        implicit[0] = 0.0;
        implicit[n] = 0.0;
        if n == 0 {
            return 0.0;
        }
        let mut below = courant(0);
        let mut largest = below;
        for j in 1..n {
            let above = courant(j);
            implicit[j] = self.implicit_fraction(below.max(above)) * omega[j];
            largest = largest.max(above);
            below = above;
        }
        largest
    }
}

/// Relax the inventory fields of the stage value `stage` (`H_z u`, `H_z v`,
/// `H_z T`, `H_z S` and the turbulence's `H_w k`, `H_w ψ`) with the implicit
/// vertical fluxes `omega_implicit` (`Ω_i` at the σ-surfaces,
/// `[element][node][n_levels + 1]`) over `dt` (see the module docs), on the
/// stage's thicknesses (its `η`): the layers' σ-thicknesses `d_sigma`, the
/// w-cells' `d_sigma_w` ([`crate::solver::rhs::w_cell_thicknesses`]).
///
/// - Elements not marked in `element_means`: every column on its own.
/// - Marked elements, whose fields the mode splitter carries as element
///   means per level ([`crate::solver::rhs::from_inventory`]): one column of
///   the element's volumes and mass-weighted fluxes per level, its change of
///   inventory spread over the nodes by their volumes (only the means are
///   read back).
///
/// The turbulence's bed and surface w-points hold the closure's boundary
/// values: the solve takes them as their interior neighbours' (as the
/// explicit advection does) and leaves their inventories alone.
pub(crate) fn apply_implicit_vertical_advection(
    stage: &mut Solution3D,
    omega_implicit: &[f64],
    d_sigma: &[f64],
    d_sigma_w: &[f64],
    bathymetry: &Bathymetry2D,
    geom: &GeometricFactors2D,
    element_means: &[bool],
    dt: f64,
) {
    let (ne, nn, nl) = (stage.n_elements, stage.n_nodes, stage.n_levels);
    let n = ne * nn * nl;
    let Solution3D {
        eta,
        u,
        v,
        temp,
        salt,
        tke,
        gls,
        ..
    } = stage;
    let eta: &[f64] = &eta.data;
    let has_turbulence = !tke.is_empty();
    for_each_block(
        ne,
        [
            &mut u[..n],
            &mut v[..n],
            &mut temp[..n],
            &mut salt[..n],
            &mut tke[..],
            &mut gls[..],
        ],
        || {
            Pooled::take(
                |s: &StageScratch| s.fits(nn, nl),
                || StageScratch::new(nn, nl),
            )
        },
        |scratch, k, [u, v, temp, salt, tke, gls]| {
            let omega = &omega_implicit[k * nn * (nl + 1)..(k + 1) * nn * (nl + 1)];
            if omega.iter().all(|&w| w == 0.0) {
                return;
            }
            let columns = ElementColumns {
                k,
                n_levels: nl,
                eta: &eta[k * nn..(k + 1) * nn],
                bed: bathymetry.element(ElementIndex::new(k)),
                omega,
                geom,
                means: element_means[k],
            };
            for field in [u, v, temp, salt] {
                columns.relax(field, d_sigma, false, dt, scratch);
            }
            if has_turbulence {
                for field in [tke, gls] {
                    columns.relax(field, d_sigma_w, true, dt, scratch);
                }
            }
        },
    );
}

/// The columns of one element, for [`apply_implicit_vertical_advection`].
struct ElementColumns<'a> {
    k: usize,
    n_levels: usize,
    /// `η` and the bed of the element's nodes.
    eta: &'a [f64],
    bed: &'a [f64],
    /// `Ω_i` of the element's columns, `[node][n_levels + 1]`.
    omega: &'a [f64],
    geom: &'a GeometricFactors2D,
    /// Whether the element's fields are carried as element means.
    means: bool,
}

impl ElementColumns<'_> {
    /// `Ω_i` of node `i`'s column.
    fn omega_at(&self, i: usize) -> &[f64] {
        let nw = self.n_levels + 1;
        &self.omega[i * nw..(i + 1) * nw]
    }

    /// Add `weight` times the upward and downward fluxes through the
    /// surfaces of node `i`'s `n_cells` cells to `up` and `down`: the layers'
    /// `Ω_i`, or for the w-cells (one more) the mean of each layer's two, as
    /// [`crate::solver::rhs::LayerTransport::stagger_from`] has it.
    fn add_fluxes(&self, i: usize, n_cells: usize, weight: f64, up: &mut [f64], down: &mut [f64]) {
        let omega = self.omega_at(i);
        let staggered = n_cells > self.n_levels;
        for j in 1..n_cells {
            let w = if staggered {
                0.5 * (omega[j - 1] + omega[j])
            } else {
                omega[j]
            };
            up[j] += weight * w.max(0.0);
            down[j] += weight * (-w).max(0.0);
        }
    }

    /// Relax the inventory `field` (`[node][cell]`, cells of σ-thickness
    /// `cells`) over `dt`; `w_points`: the end cells hold the closure's
    /// boundary values (see [`apply_implicit_vertical_advection`]).
    ///
    /// Every cell's inventory changes by `V_l (x_l − c_l)`, the solve's
    /// change from the concentration `c_l` it started from: the fluxes'
    /// divergence, so the column (or element) total is kept, also where the
    /// end w-points start from their neighbours' values.
    fn relax(
        &self,
        field: &mut [f64],
        cells: &[f64],
        w_points: bool,
        dt: f64,
        scratch: &mut StageScratch,
    ) {
        let (nc, nn) = (cells.len(), self.eta.len());
        let StageScratch {
            layers,
            w_cells,
            volumes,
            start,
        } = scratch;
        let solve = if w_points { w_cells } else { layers };
        debug_assert_eq!(solve.len(), nc);
        let depth = |i: usize| (self.eta[i] - self.bed[i]).max(0.0);
        // The end w-points carry their neighbours' values, as the explicit
        // advection does
        let starting_values = |solve: &mut ColumnSolve, start: &mut [f64]| {
            start[..nc].copy_from_slice(&solve.values);
        };
        if !self.means {
            for i in 0..nn {
                if self.omega_at(i).iter().all(|&w| w == 0.0) {
                    continue;
                }
                solve.up.fill(0.0);
                solve.down.fill(0.0);
                self.add_fluxes(i, nc, 1.0, &mut solve.up, &mut solve.down);
                let column = &mut field[i * nc..(i + 1) * nc];
                for l in 0..nc {
                    solve.volume[l] = layer_thickness_of(depth(i), cells[l]);
                    solve.values[l] = column[l] / solve.volume[l];
                }
                starting_values(solve, start);
                solve.solve(dt, w_points);
                for (l, q) in column.iter_mut().enumerate() {
                    *q += solve.volume[l] * (solve.values[l] - start[l]);
                }
            }
            return;
        }
        // One column of the element's means: mass-weighted volumes, fluxes
        // and inventories
        solve.up.fill(0.0);
        solve.down.fill(0.0);
        solve.volume.fill(0.0);
        solve.values.fill(0.0);
        for i in 0..nn {
            let mass = self.geom.node_mass(self.k, i);
            self.add_fluxes(i, nc, mass, &mut solve.up, &mut solve.down);
            for l in 0..nc {
                let volume = layer_volume_of(depth(i), cells[l]);
                volumes[i * nc + l] = volume;
                solve.volume[l] += mass * volume;
                solve.values[l] += mass * field[i * nc + l];
            }
        }
        for l in 0..nc {
            // The inventories hold the cells' water, none at a dry node
            solve.volume[l] = solve.volume[l].max(f64::MIN_POSITIVE);
            solve.values[l] /= solve.volume[l];
        }
        starting_values(solve, start);
        solve.solve(dt, w_points);
        // Each node's share of the change of the element's inventory, by its
        // volume: the mass-weighted sum over the nodes is the change
        for l in 0..nc {
            let change = solve.values[l] - start[l];
            for i in 0..nn {
                field[i * nc + l] += change * volumes[i * nc + l];
            }
        }
    }
}

/// Buffers of [`apply_implicit_vertical_advection`] for one thread.
struct StageScratch {
    layers: ColumnSolve,
    w_cells: ColumnSolve,
    /// The volume of every node's cell, `[node][cell]`.
    volumes: Vec<f64>,
    /// The concentration of every cell the solve starts from.
    start: Vec<f64>,
}

impl StageScratch {
    fn new(nn: usize, nl: usize) -> Self {
        Self {
            layers: ColumnSolve::new(nl),
            w_cells: ColumnSolve::new(nl + 1),
            volumes: vec![0.0; nn * (nl + 1)],
            start: vec![0.0; nl + 1],
        }
    }

    fn fits(&self, nn: usize, nl: usize) -> bool {
        self.layers.len() == nl && self.volumes.len() == nn * (nl + 1)
    }
}

/// Buffers of one column's implicit solve.
#[derive(Debug, Default)]
pub(crate) struct ColumnSolve {
    /// Upward and downward volume flux through each σ-surface (≥ 0).
    pub up: Vec<f64>,
    pub down: Vec<f64>,
    /// The cells' volumes.
    pub volume: Vec<f64>,
    /// Concentrations: the right-hand side's, then the solution.
    pub values: Vec<f64>,
    c_prime: Vec<f64>,
}

impl ColumnSolve {
    /// Buffers for columns of `n` cells.
    pub(crate) fn new(n: usize) -> Self {
        Self {
            up: vec![0.0; n + 1],
            down: vec![0.0; n + 1],
            volume: vec![0.0; n],
            values: vec![0.0; n],
            c_prime: vec![0.0; n],
        }
    }

    /// Number of cells.
    pub(crate) fn len(&self) -> usize {
        self.volume.len()
    }

    /// Solve
    /// `V_l x_l + dt [up_{l+1} x_l − down_{l+1} x_{l+1} − up_l x_{l−1} + down_l x_l] = V_l c_l`
    /// for `x`, with `c` and then `x` in `self.values` (the volumes, the
    /// fluxes through the σ-surfaces and the values set). An M-matrix,
    /// diagonally dominant by columns: the Thomas algorithm needs no pivoting
    /// and every pivot is positive.
    ///
    /// With `zero_gradient_ends` (at least three cells) the two end cells are
    /// not unknowns: what flows out of them carries their neighbour's new
    /// value, as for the turbulence's bed and surface w-points, whose own
    /// values are the closure's boundary values. Their neighbours' rows then
    /// have that inflow on the diagonal; where it would take the diagonal
    /// below half the cell's volume (an inflow far beyond the outflow) it is
    /// taken at the neighbour's starting value instead, keeping the M-matrix
    /// at the cost of constancy there. The end cells' values then change by
    /// the flux through their one interior surface, so the column's inventory
    /// is kept.
    pub(crate) fn solve(&mut self, dt: f64, zero_gradient_ends: bool) {
        let n = self.len();
        let Self {
            up,
            down,
            volume,
            values,
            c_prime,
        } = self;
        if n == 0 || (zero_gradient_ends && n < 3) {
            return;
        }
        let (lo, hi) = if zero_gradient_ends {
            (1, n - 2)
        } else {
            (0, n - 1)
        };
        let (start_lo, start_hi) = (values[lo], values[hi]);
        let diagonal = |l: usize| volume[l] + dt * (up[l + 1] + down[l]);
        // The rows next to zero-gradient ends: the inflow from the end on
        // the diagonal (`implicit_*`), or lagged on the right-hand side
        let (mut b_lo, mut extra_lo) = (diagonal(lo), 0.0);
        let (mut implicit_below, mut implicit_above) = (true, true);
        if zero_gradient_ends {
            if b_lo - dt * up[lo] >= 0.5 * volume[lo] {
                b_lo -= dt * up[lo];
            } else {
                implicit_below = false;
                extra_lo += dt * up[lo] * start_lo;
            }
        }
        let (mut b_hi, mut extra_hi) = if hi == lo {
            (b_lo, extra_lo)
        } else {
            (diagonal(hi), 0.0)
        };
        if zero_gradient_ends {
            if b_hi - dt * down[hi + 1] >= 0.5 * volume[hi] {
                b_hi -= dt * down[hi + 1];
            } else {
                implicit_above = false;
                extra_hi += dt * down[hi + 1] * start_hi;
            }
        }
        // Row l: a_l x_{l−1} + b_l x_l + c_l x_{l+1} = V_l c_l (+ extra)
        let a = |l: usize| if l > lo { -dt * up[l] } else { 0.0 };
        let c = |l: usize| if l < hi { -dt * down[l + 1] } else { 0.0 };
        let row = |l: usize, value: f64| {
            if l == hi {
                (b_hi, volume[l] * value + extra_hi)
            } else if l == lo {
                (b_lo, volume[l] * value + extra_lo)
            } else {
                (diagonal(l), volume[l] * value)
            }
        };
        let (b, rhs) = row(lo, values[lo]);
        c_prime[lo] = c(lo) / b;
        values[lo] = rhs / b;
        for l in lo + 1..=hi {
            let (b, rhs) = row(l, values[l]);
            let pivot = b - a(l) * c_prime[l - 1];
            c_prime[l] = c(l) / pivot;
            values[l] = (rhs - a(l) * values[l - 1]) / pivot;
        }
        for l in (lo..hi).rev() {
            values[l] -= c_prime[l] * values[l + 1];
        }
        if zero_gradient_ends {
            // The bed end loses what flows up through surface 1 and gains
            // what flows down; the surface end the reverse through n − 1
            let carried_up = if implicit_below { values[1] } else { start_lo };
            values[0] += dt * (down[1] * values[1] - up[1] * carried_up) / volume[0];
            let carried_down = if implicit_above {
                values[n - 2]
            } else {
                start_hi
            };
            values[n - 1] +=
                dt * (up[n - 1] * values[n - 2] - down[n - 1] * carried_down) / volume[n - 1];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `f` is zero up to `c_min`, continuous, below one, and leaves the
    /// explicit part `c(1 − f)` rising towards `c_max`.
    #[test]
    fn the_explicit_part_stays_below_the_bound() {
        let split = ImplicitVerticalAdvection::new(0.4, 0.8);
        assert_eq!(split.implicit_fraction(0.0), 0.0);
        assert_eq!(split.implicit_fraction(0.4), 0.0);
        let mut last = 0.0;
        for i in 1..=100_000 {
            let c = 1e-3 * i as f64;
            let f = split.implicit_fraction(c);
            assert!((0.0..1.0).contains(&f), "f({c}) = {f}");
            let explicit = c * (1.0 - f);
            assert!(explicit < 0.8, "explicit Courant {explicit} at {c}");
            assert!(explicit >= last - 1e-15, "not monotone at {c}");
            last = explicit;
        }
        // Approaches the bound
        assert!(last > 0.79, "explicit Courant {last} at c = 100");
        // Continuous at c_min
        assert!(split.implicit_fraction(0.4 + 1e-9) < 1e-17);
    }

    /// Every cell's explicit outflow Courant number stays below `c_max`,
    /// for columns of very different layers and flows.
    #[test]
    fn every_cell_keeps_its_explicit_outflow_below_the_bound() {
        let split = ImplicitVerticalAdvection::default();
        let thickness = [0.01, 0.05, 0.3, 2.0, 0.02, 0.5, 0.004];
        let n = thickness.len();
        for (dt, scale) in [(1.0, 1e-3), (5.0, 0.01), (100.0, 0.05), (3600.0, 0.2)] {
            let mut omega = vec![0.0; n + 1];
            for (j, w) in omega.iter_mut().enumerate().take(n).skip(1) {
                *w = scale * ((j as f64 * 1.7).sin() + 0.3 * (j as f64).cos());
            }
            let mut implicit = vec![0.0; n + 1];
            let largest = split.split_column(&omega, &thickness, dt, &mut implicit);
            assert_eq!((implicit[0], implicit[n]), (0.0, 0.0));
            let explicit: Vec<f64> = omega.iter().zip(&implicit).map(|(w, i)| w - i).collect();
            for l in 0..n {
                let c = dt * (explicit[l + 1].max(0.0) - explicit[l].min(0.0)) / thickness[l];
                assert!(
                    c < 0.8,
                    "dt {dt}: cell {l} explicit Courant {c} (largest {largest})"
                );
            }
            // The split keeps each surface's direction
            for (w, i) in omega.iter().zip(&implicit) {
                assert!(w * i >= 0.0 && i.abs() <= w.abs());
            }
        }
    }

    /// Below `c_min` everywhere the split is exactly zero.
    #[test]
    fn slow_columns_stay_explicit() {
        let split = ImplicitVerticalAdvection::default();
        let omega = [0.0, 1e-4, -2e-4, 3e-4, 0.0];
        let thickness = [1.0, 1.0, 1.0, 1.0];
        let mut implicit = [1.0; 5];
        let largest = split.split_column(&omega, &thickness, 100.0, &mut implicit);
        assert!(largest < 0.4);
        assert_eq!(implicit, [0.0; 5]);
    }

    fn column(volume: &[f64], omega: &[f64]) -> ColumnSolve {
        let mut s = ColumnSolve::new(volume.len());
        s.volume.copy_from_slice(volume);
        for (j, &w) in omega.iter().enumerate() {
            s.up[j] = w.max(0.0);
            s.down[j] = (-w).max(0.0);
        }
        s
    }

    /// The solve conserves the column's inventory, keeps a field whose
    /// inventory is the constancy one uniform, and stays within the
    /// right-hand side's range (an M-matrix), for steps of any length.
    #[test]
    fn the_column_solve_is_conservative_constant_and_bounded() {
        let volume = [0.02, 0.5, 1.5, 0.1, 3.0];
        let omega = [0.0, 0.3, -0.2, 0.05, 0.4, 0.0];
        for dt in [0.1, 1.0, 10.0, 1e4] {
            // Conservation and boundedness
            let mut s = column(&volume, &omega);
            let c0 = [3.0, -1.0, 2.5, 7.0, 0.0];
            s.values.copy_from_slice(&c0);
            let before: f64 = volume.iter().zip(&c0).map(|(v, c)| v * c).sum();
            s.solve(dt, false);
            let after: f64 = volume.iter().zip(&s.values).map(|(v, c)| v * c).sum();
            assert!(
                (after - before).abs() < 1e-12 * before.abs().max(1.0),
                "dt {dt}"
            );
            // Constancy: inventory C (V + dt δΩ), concentration over V
            let mut s = column(&volume, &omega);
            for l in 0..volume.len() {
                s.values[l] = 4.2 * (volume[l] + dt * (omega[l + 1] - omega[l])) / volume[l];
            }
            s.solve(dt, false);
            // Round-off of inventories up to dt·|δΩ|/V ≈ 2.5e5 times the volume
            let scale = 1.0 + dt * 0.5 / 0.02;
            for &x in &s.values {
                assert!(
                    (x - 4.2).abs() < 1e-15 * scale * 4.2,
                    "dt {dt}: {:?}",
                    s.values
                );
            }
            // Zero-gradient ends at other values: the interior stays uniform,
            // the column keeps its inventory (upwelling, as near a bed: the
            // ends' inflow never takes over a diagonal)
            let upwelling = [0.0, 0.1, 0.2, 0.3, 0.2, 0.0];
            let mut s = column(&volume, &upwelling);
            for l in 0..volume.len() {
                s.values[l] =
                    4.2 * (volume[l] + dt * (upwelling[l + 1] - upwelling[l])) / volume[l];
            }
            (s.values[0], s.values[4]) = (50.0, -3.0);
            let before: f64 = volume.iter().zip(&s.values).map(|(v, c)| v * c).sum();
            s.solve(dt, true);
            for &x in &s.values[1..4] {
                assert!(
                    (x - 4.2).abs() < 1e-15 * scale * 4.2,
                    "dt {dt}: {:?}",
                    s.values
                );
            }
            let after: f64 = volume.iter().zip(&s.values).map(|(v, c)| v * c).sum();
            assert!(
                (after - before).abs() < 1e-14 * scale * before.abs(),
                "dt {dt}: {before} → {after}"
            );
            // Non-negative data stay non-negative
            let mut s = column(&volume, &omega);
            s.values.copy_from_slice(&[0.0, 0.0, 1.0, 0.0, 0.0]);
            s.solve(dt, false);
            assert!(
                s.values.iter().all(|&x| x >= 0.0),
                "dt {dt}: {:?}",
                s.values
            );
        }
    }

    /// A pulse carried up a uniform column by a uniform implicit flux
    /// converges at first order in the step and the layer thickness
    /// (backward Euler, first-order upwind) to the exact translation.
    #[test]
    fn a_pulse_advected_implicitly_converges_at_first_order() {
        let (length, speed, time) = (1.0, 1.0, 0.25);
        let profile = |z: f64| (-((z - 0.3) / 0.08).powi(2)).exp();
        let error = |n: usize| {
            let dz = length / n as f64;
            // Courant number 2: fully beyond the explicit limit
            let dt = 2.0 * dz / speed;
            let steps = (time / dt).round() as usize;
            let mut s = ColumnSolve::new(n);
            s.volume.fill(dz);
            for j in 1..n {
                s.up[j] = speed;
            }
            for (l, c) in s.values.iter_mut().enumerate() {
                *c = profile((l as f64 + 0.5) * dz);
            }
            for _ in 0..steps {
                s.solve(dt, false);
            }
            let t = steps as f64 * dt;
            s.values
                .iter()
                .enumerate()
                .map(|(l, c)| (c - profile((l as f64 + 0.5) * dz - speed * t)).abs() * dz)
                .sum::<f64>()
        };
        let errors: Vec<f64> = [400, 800, 1600, 3200].iter().map(|&n| error(n)).collect();
        for pair in errors.windows(2) {
            let rate = (pair[0] / pair[1]).log2();
            assert!(
                (0.8..1.2).contains(&rate),
                "convergence rate {rate:.2} ({errors:?})"
            );
        }
    }
}
