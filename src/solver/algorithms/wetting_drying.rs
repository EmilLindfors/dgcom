//! Wetting and drying for the 2D shallow water equations.
//!
//! Three pieces, each where it belongs in a time step:
//!
//! 1. **Positivity** (after every RK stage, [`apply_wet_dry_correction_all`]).
//!    Zhang–Shu scaling of each element towards its mean until h ≥ 0 at every
//!    node (Zhang & Shu 2010; Xing, Zhang & Shu 2010). It conserves mass and
//!    leaves a lake-at-rest shoreline, h = max(0, η − B), untouched. Elements
//!    whose mean depth is below `h_dry` lose their momentum. With a
//!    positivity-preserving flux (HLL, Rusanov) under `positivity_cfl_swe_2d`
//!    the element means stay non-negative; a negative mean is emptied (which
//!    creates mass) and counted.
//! 2. **Velocity desingularization** (same pass). Kurganov & Petrova (2007):
//!
//!    ```text
//!    u = √2 h (hu) / √(h⁴ + max(h⁴, ε⁴)),   hu ← h u,   ε = h_dry
//!    ```
//!
//!    The identity for h ≥ ε; below it the velocity vanishes like h², so a
//!    near-dry node cannot carry a huge hu/h (which collapses the time step
//!    and pollutes the fluxes). A velocity cap follows as a last resort.
//! 3. **Thin-layer relaxation** (point-implicit in every RK stage,
//!    [`apply_implicit_damping_2d`] via
//!    `TimeIntegrator::step_with_relaxation`). A drag in films thinner than
//!    h_dry,
//!
//!    ```text
//!    ∂(hu)/∂t = −r(h) hu,   r(h) = (h_dry/h − 1)² / τ   (0 < h < h_dry)
//!    ```
//!
//!    and r = 0 for h ≥ h_dry, discretized as `hu ← hu / (1 + Δt r(h))`. It
//!    replaces multiplying hu by a blending factor α(h) after every stage,
//!    whose damping per unit time grew as Δt shrank and which erased currents
//!    up to 10·h_min deep (notes/review-2026-09-25.md §1.7). Bottom friction laws and net-cage
//!    drag (`CageDrag2D`) are applied in the same implicit update.
//!
//! Water at least h_dry deep is never touched by 2. or 3.
//!
//! # References
//! - Zhang & Shu (2010), JCP 229, positivity-preserving high-order DG
//! - Xing, Zhang & Shu (2010), Adv. Water Resour. 33, positivity-preserving
//!   well-balanced DG for the SWE
//! - Kurganov & Petrova (2007), Commun. Math. Sci. 5, desingularization
//! - Medeiros & Hagen (2013), review of wetting and drying algorithms

use crate::operators::GeometricFactors2D;
use crate::solver::SWESolution2D;
use crate::solver::limiters::{element_mass, element_mean, positivity_limit_element};
use crate::solver::state::SWEState2D;
use crate::source::{BottomFriction2D, CageDrag2D, CageNode};
use crate::types::Depth;

/// Configuration for wetting/drying treatment.
#[derive(Clone, Debug)]
pub struct WetDryConfig {
    /// Dry threshold and desingularization depth ε (default 1e-3 m): elements
    /// with a smaller mean depth are dry, and nodal velocities are
    /// desingularized below it. Scale it with the problem: about 1e-4 of a
    /// characteristic depth. 1 mm suits tidal flats and coastal runs; on
    /// Thacker's 0.1 m laboratory bowl it dominates the error (7.4 % with
    /// 1 mm, 4.5 % with 1e-5 m).
    pub h_dry: Depth,
    /// Thin-layer relaxation time τ (s) at h = h_dry/2 (see
    /// [`Self::thin_layer_rate`]).
    pub relaxation_time: f64,
    /// Maximum allowed velocity magnitude (m/s)
    pub max_velocity: f64,
    /// Gravitational acceleration
    pub g: f64,
}

impl WetDryConfig {
    /// Default dry threshold (m).
    pub const DEFAULT_H_DRY: f64 = 1e-3;

    /// Configuration with dry threshold `h_dry`, a 1 s thin-layer relaxation
    /// time and a 20 m/s velocity cap.
    ///
    /// # Arguments
    /// * `h_dry` - Dry threshold / desingularization depth (typically 1e-3 m)
    /// * `g` - Gravitational acceleration
    pub fn new(h_dry: Depth, g: f64) -> Self {
        Self {
            h_dry,
            relaxation_time: 1.0,
            max_velocity: 20.0, // very fast tidal current
            g,
        }
    }

    /// Set the thin-layer relaxation time τ (s) at h = h_dry/2.
    pub fn with_relaxation_time(mut self, relaxation_time: f64) -> Self {
        assert!(relaxation_time > 0.0, "relaxation time must be positive");
        self.relaxation_time = relaxation_time;
        self
    }

    /// Set the maximum velocity.
    pub fn with_max_velocity(mut self, max_velocity: f64) -> Self {
        self.max_velocity = max_velocity;
        self
    }

    /// Whether a depth is below the dry threshold.
    #[inline]
    pub fn is_dry(&self, h: f64) -> bool {
        h < self.h_dry.meters()
    }

    /// Kurganov–Petrova desingularized velocity,
    /// `(u, v) = √2 h (hu, hv) / √(h⁴ + max(h⁴, ε⁴))` with ε = h_dry.
    ///
    /// Equal to (hu/h, hv/h) for h ≥ h_dry, zero for h ≤ 0.
    #[inline]
    pub fn desingularized_velocity(&self, h: f64, hu: f64, hv: f64) -> (f64, f64) {
        if h <= 0.0 {
            return (0.0, 0.0);
        }
        let eps = self.h_dry.meters();
        if h >= eps {
            return (hu / h, hv / h);
        }
        // √(h⁴ + ε⁴) = ε²√(1 + r⁴) with r = h/ε, so ε⁴ never underflows
        let r = h / eps;
        let factor = std::f64::consts::SQRT_2 * r / (eps * (1.0 + r * r * r * r).sqrt());
        (factor * hu, factor * hv)
    }

    /// Thin-layer drag rate r(h) = (h_dry/h − 1)² / τ (1/s) below h_dry:
    /// zero for h ≥ h_dry, 1/τ at h_dry/2, infinite at h ≤ 0.
    #[inline]
    pub fn thin_layer_rate(&self, h: f64) -> f64 {
        let h_dry = self.h_dry.meters();
        if h >= h_dry {
            return 0.0;
        }
        if h <= 0.0 {
            return f64::INFINITY;
        }
        let excess = h_dry / h - 1.0;
        excess * excess / self.relaxation_time
    }

    /// Desingularize and cap the velocity of one node in place (h unchanged).
    /// Nodes with h ≥ h_dry below the cap are left bitwise unchanged.
    #[inline]
    fn correct_node(&self, h: f64, hu: &mut f64, hv: &mut f64) {
        if h <= 0.0 {
            *hu = 0.0;
            *hv = 0.0;
            return;
        }
        if h < self.h_dry.meters() {
            let (u, v) = self.desingularized_velocity(h, *hu, *hv);
            *hu = h * u;
            *hv = h * v;
        }
        let speed = (*hu * *hu + *hv * *hv).sqrt() / h;
        if speed > self.max_velocity {
            let scale = self.max_velocity / speed;
            *hu *= scale;
            *hv *= scale;
        }
    }
}

impl Default for WetDryConfig {
    fn default() -> Self {
        Self::new(Depth::new(Self::DEFAULT_H_DRY), 9.81)
    }
}

/// Apply the nodal part of the wetting/drying correction to one state:
/// negative depths become dry (h = 0, no momentum), then the velocity is
/// desingularized and capped.
///
/// Clipping a single node does not conserve mass; for solutions use
/// [`apply_wet_dry_correction_all`], which limits whole elements.
pub fn apply_wet_dry_correction(state: &mut SWEState2D, config: &WetDryConfig) {
    state.h = state.h.max(0.0);
    config.correct_node(state.h, &mut state.hu, &mut state.hv);
}

/// Wet/dry correction of one element (SoA slices). Returns `true` if its mean
/// depth was negative (see `positivity_limit_element`).
#[inline]
fn wet_dry_element(
    h: &mut [f64],
    hu: &mut [f64],
    hv: &mut [f64],
    weights: &[f64],
    inv_total_weight: f64,
    config: &WetDryConfig,
) -> bool {
    let h_dry = config.h_dry.meters();
    // Common case: wet, positive element; the positivity step would be a no-op
    let needs_limiting = h.iter().any(|&h| h < h_dry);
    let clipped = needs_limiting && {
        let avg = element_mean(h, hu, hv, weights, inv_total_weight);
        positivity_limit_element(h, hu, hv, avg, h_dry)
    };
    for ((&h, hu), hv) in h.iter().zip(hu.iter_mut()).zip(hv.iter_mut()) {
        config.correct_node(h, hu, hv);
    }
    clipped
}

/// Apply the wetting/drying correction to an entire solution, in place.
///
/// Per element: Zhang–Shu positivity towards h ≥ 0 (dry elements lose their
/// momentum), then Kurganov–Petrova desingularization and the velocity cap at
/// every node. Call it after every RK stage (`SWEPhysics2D::post_process`);
/// the thin-layer relaxation needs Δt and is applied separately
/// ([`apply_implicit_damping_2d`]).
///
/// Returns the number of elements whose negative mean depth was emptied
/// (creating mass); zero under the positivity CFL with HLL or Rusanov.
pub fn apply_wet_dry_correction_all(
    solution: &mut SWESolution2D,
    geom: &GeometricFactors2D,
    config: &WetDryConfig,
) -> usize {
    let n = solution.n_nodes;
    let [h, hu, hv] = &mut solution.data;
    h.chunks_exact_mut(n)
        .zip(hu.chunks_exact_mut(n))
        .zip(hv.chunks_exact_mut(n))
        .enumerate()
        .map(|(k, ((h, hu), hv))| {
            let (mass, inv_area) = (element_mass(geom, k), 1.0 / geom.area[k]);
            wet_dry_element(h, hu, hv, mass, inv_area, config) as usize
        })
        .sum()
}

/// Parallel version of [`apply_wet_dry_correction_all`] (identical result),
/// in place.
#[cfg(feature = "parallel")]
pub fn apply_wet_dry_correction_all_parallel(
    solution: &mut SWESolution2D,
    geom: &GeometricFactors2D,
    config: &WetDryConfig,
) -> usize {
    use rayon::prelude::*;

    let n = solution.n_nodes;
    let [h, hu, hv] = &mut solution.data;
    h.par_chunks_exact_mut(n)
        .zip(hu.par_chunks_exact_mut(n))
        .zip(hv.par_chunks_exact_mut(n))
        .enumerate()
        .map(|(k, ((h, hu), hv))| {
            let (mass, inv_area) = (element_mass(geom, k), 1.0 / geom.area[k]);
            wet_dry_element(h, hu, hv, mass, inv_area, config) as usize
        })
        .sum()
}

/// Stiff momentum damping applied point-implicitly: bottom friction, net-cage
/// drag and the wet/dry thin-layer relaxation.
#[derive(Clone, Copy)]
pub struct ImplicitDamping2D<'a> {
    /// Bottom friction law, if any
    pub friction: Option<&'a dyn BottomFriction2D>,
    /// Net-cage drag, if any
    pub cages: Option<&'a CageDrag2D>,
    /// Wet/dry configuration (thin-layer relaxation and desingularization), if any
    pub wet_dry: Option<&'a WetDryConfig>,
    /// Desingularization depth for the friction velocity without `wet_dry`
    pub h_min: Depth,
}

impl ImplicitDamping2D<'_> {
    /// Whether this applies any damping at all.
    pub fn is_active(&self) -> bool {
        self.friction.is_some() || self.wet_dry.is_some() || self.cages.is_some()
    }

    /// Damp the momentum of mesh node `node` of a stage value with depth
    /// `h`, whose RHS was evaluated at `from`; `cages` are the node's
    /// cage-drag entries.
    #[inline]
    fn damp_node(
        &self,
        node: usize,
        h: f64,
        hu: &mut f64,
        hv: &mut f64,
        from: SWEState2D,
        dt: f64,
        cages: &[CageNode],
    ) {
        if h <= 0.0 {
            *hu = 0.0;
            *hv = 0.0;
            return;
        }
        let mut rate = 0.0;
        if self.friction.is_some() || !cages.is_empty() {
            // |u| frozen at the RHS input keeps drag balances exact
            let (u, v) = match self.wet_dry {
                Some(wd) => wd.desingularized_velocity(from.h, from.hu, from.hv),
                None => from.velocity(self.h_min),
            };
            let speed = (u * u + v * v).sqrt();
            if speed > 0.0 {
                if let Some(friction) = self.friction {
                    rate += friction.damping_rate(node, h, speed);
                }
                rate += CageDrag2D::damping_rate(cages, h, speed);
            }
        }
        if let Some(wet_dry) = self.wet_dry {
            rate += wet_dry.thin_layer_rate(h);
        }
        // rate may be +inf (factor 0); dt > 0 keeps dt·rate from being NaN
        let factor = 1.0 / (1.0 + dt * rate);
        *hu *= factor;
        *hv *= factor;
    }
}

/// Apply stiff momentum damping point-implicitly to the RK stage value
/// `stage`, whose RHS was evaluated at `from` with weight `dt`:
///
/// ```text
/// (hu, hv) ← (hu, hv) / (1 + dt·Λ),   Λ = Λ_friction(h, |u_from|) + Λ_cage(h, |u_from|) + r(h)
/// ```
///
/// with h the stage depth. Momentum is only ever shrunk, never sign-flipped,
/// for any `dt`. This is the `relax` step of
/// `TimeIntegrator::step_with_relaxation`, which explains why |u| is taken
/// from the RHS input. Allocation-free.
pub fn apply_implicit_damping_2d(
    stage: &mut SWESolution2D,
    from: &SWESolution2D,
    dt: f64,
    damping: &ImplicitDamping2D,
) {
    if !damping.is_active() {
        return;
    }
    check_mesh(stage, damping);
    let [h, hu, hv] = &mut stage.data;
    let [fh, fhu, fhv] = &from.data;
    damp_nodes(h, hu, hv, (fh, fhu, fhv), 0, dt, damping);
}

/// The cage drag and a spatially varying friction must come from the
/// stage's mesh and order.
#[inline]
fn check_mesh(stage: &SWESolution2D, damping: &ImplicitDamping2D) {
    let n_total = stage.h_data().len();
    if let Some(cages) = damping.cages {
        assert_eq!(
            cages.n_total_nodes(),
            n_total,
            "CageDrag2D was built for a different mesh or order"
        );
    }
    if let Some(n) = damping.friction.and_then(|f| f.n_total_nodes()) {
        assert_eq!(
            n, n_total,
            "the friction field was built for a different mesh or order"
        );
    }
}

/// Parallel version of [`apply_implicit_damping_2d`] (identical result).
#[cfg(feature = "parallel")]
pub fn apply_implicit_damping_2d_parallel(
    stage: &mut SWESolution2D,
    from: &SWESolution2D,
    dt: f64,
    damping: &ImplicitDamping2D,
) {
    use rayon::prelude::*;

    const CHUNK: usize = 4096;
    if !damping.is_active() {
        return;
    }
    check_mesh(stage, damping);
    let [h, hu, hv] = &mut stage.data;
    let [fh, fhu, fhv] = &from.data;
    h.par_chunks(CHUNK)
        .zip(hu.par_chunks_mut(CHUNK))
        .zip(hv.par_chunks_mut(CHUNK))
        .zip(fh.par_chunks(CHUNK))
        .zip(fhu.par_chunks(CHUNK))
        .zip(fhv.par_chunks(CHUNK))
        .enumerate()
        .for_each(|(c, (((((h, hu), hv), fh), fhu), fhv))| {
            damp_nodes(h, hu, hv, (fh, fhu, fhv), c * CHUNK, dt, damping);
        });
}

/// Damp the nodes `offset..offset + h.len()` (global node numbering).
#[inline]
fn damp_nodes(
    h: &[f64],
    hu: &mut [f64],
    hv: &mut [f64],
    (fh, fhu, fhv): (&[f64], &[f64], &[f64]),
    offset: usize,
    dt: f64,
    damping: &ImplicitDamping2D,
) {
    // Wet water under bottom friction (almost every node of a tidal run):
    // a block of nodes at a time, see `damp_wet_block`
    if let (Some(friction), Some(wet_dry), None) =
        (damping.friction, damping.wet_dry, damping.cages)
    {
        let h_dry = wet_dry.h_dry.meters();
        let mut start = 0;
        while start < h.len() {
            let r = start..(start + WET_BLOCK).min(h.len());
            let from = (&fh[r.clone()], &fhu[r.clone()], &fhv[r.clone()]);
            let (h, hu, hv) = (&h[r.clone()], &mut hu[r.clone()], &mut hv[r.clone()]);
            if !damp_wet_block(friction, h_dry, h, hu, hv, from, offset + start, dt) {
                damp_nodes_one_by_one(h, hu, hv, from, offset + start, dt, damping);
            }
            start = r.end;
        }
        return;
    }
    damp_nodes_one_by_one(h, hu, hv, (fh, fhu, fhv), offset, dt, damping);
}

/// Nodes per [`damp_wet_block`] (an element up to order 7 in one block).
const WET_BLOCK: usize = 64;

/// [`ImplicitDamping2D::damp_node`] with bottom friction for a block of nodes
/// that are all deeper than `h_dry` (> 0), in the stage value and in the RHS
/// input, without cages: then the velocity is the plain quotient and there is
/// no thin-layer rate, and the nodes' square roots and divisions run side by
/// side (the friction law's too, [`BottomFriction2D::damping_rates`]) instead
/// of one node's chain after another's. The same result bit for bit. Returns
/// false, having changed nothing, for any other block.
#[inline]
fn damp_wet_block(
    friction: &dyn BottomFriction2D,
    h_dry: f64,
    h: &[f64],
    hu: &mut [f64],
    hv: &mut [f64],
    (fh, fhu, fhv): (&[f64], &[f64], &[f64]),
    first_node: usize,
    dt: f64,
) -> bool {
    let n = h.len();
    debug_assert!(n <= WET_BLOCK);
    let (hu, hv) = (&mut hu[..n], &mut hv[..n]);
    let (fh, fhu, fhv) = (&fh[..n], &fhu[..n], &fhv[..n]);
    let wet = |h: f64| (h >= h_dry) & (h > 0.0);
    if !h
        .iter()
        .zip(fh)
        .fold(true, |all, (&h, &fh)| all & wet(h) & wet(fh))
    {
        return false;
    }
    let (mut speed, mut rate) = ([0.0; WET_BLOCK], [0.0; WET_BLOCK]);
    for i in 0..n {
        // |u| frozen at the RHS input keeps drag balances exact
        let (u, v) = (fhu[i] / fh[i], fhv[i] / fh[i]);
        speed[i] = (u * u + v * v).sqrt();
    }
    friction.damping_rates(first_node, h, &speed[..n], &mut rate[..n]);
    for i in 0..n {
        let factor = 1.0 / (1.0 + dt * rate[i]);
        hu[i] *= factor;
        hv[i] *= factor;
    }
    true
}

/// [`damp_nodes`] node by node: any depth, with cages.
#[inline]
fn damp_nodes_one_by_one(
    h: &[f64],
    hu: &mut [f64],
    hv: &mut [f64],
    (fh, fhu, fhv): (&[f64], &[f64], &[f64]),
    offset: usize,
    dt: f64,
    damping: &ImplicitDamping2D,
) {
    // Cursor into the sparse, node-sorted cage entries
    let cage_nodes = damping.cages.map_or(&[][..], |c| c.nodes());
    let mut next = damping.cages.map_or(0, |c| c.first_at_or_after(offset));
    for i in 0..h.len() {
        let start = next;
        while next < cage_nodes.len() && cage_nodes[next].node == offset + i {
            next += 1;
        }
        let from = SWEState2D::new(fh[i], fhu[i], fhv[i]);
        damping.damp_node(
            offset + i,
            h[i],
            &mut hu[i],
            &mut hv[i],
            from,
            dt,
            &cage_nodes[start..next],
        );
    }
}

/// The wet/dry correction of element `k` (its SoA rows), the element kernel
/// of [`apply_wet_dry_correction_all`], for local time stepping. Returns
/// `true` if its mean depth was negative and it was emptied.
pub fn apply_wet_dry_correction_element(
    k: usize,
    [h, hu, hv]: [&mut [f64]; 3],
    geom: &GeometricFactors2D,
    config: &WetDryConfig,
) -> bool {
    let (mass, inv_area) = (element_mass(geom, k), 1.0 / geom.area[k]);
    wet_dry_element(h, hu, hv, mass, inv_area, config)
}

impl ImplicitDamping2D<'_> {
    /// [`apply_implicit_damping_2d`] of element `k` (its SoA rows `stage`,
    /// with rates frozen at its rows `from`) over `dt`, for local time
    /// stepping. Same per-node arithmetic, so the same result.
    pub fn damp_element(&self, k: usize, stage: [&mut [f64]; 3], from: [&[f64]; 3], dt: f64) {
        if !self.is_active() {
            return;
        }
        let [h, hu, hv] = stage;
        let [fh, fhu, fhv] = from;
        damp_nodes(h, hu, hv, (fh, fhu, fhv), k * h.len(), dt, self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh2D;
    use crate::operators::DGOperators2D;
    use crate::source::ManningFriction2D;
    use crate::types::ElementIndex;

    /// Geometry of `n` unit squares in a row.
    fn unit_geometry(ops: &DGOperators2D, n: usize) -> GeometricFactors2D {
        let mesh = Mesh2D::uniform_rectangle(0.0, n as f64, 0.0, 1.0, n, 1);
        GeometricFactors2D::compute(&mesh, ops)
    }

    fn weighted_mass(solution: &SWESolution2D, ops: &DGOperators2D) -> f64 {
        solution
            .element_h(ElementIndex::new(0))
            .iter()
            .zip(ops.weights.iter())
            .map(|(&h, &w)| h * w)
            .sum()
    }

    #[test]
    fn test_desingularized_velocity() {
        let config = WetDryConfig::new(Depth::new(1e-3), 9.81);

        // Identity for h ≥ h_dry
        let (u, v) = config.desingularized_velocity(0.5, 0.25, -0.1);
        assert!((u - 0.5).abs() < 1e-15 && (v + 0.2).abs() < 1e-15);
        let (u, _) = config.desingularized_velocity(1e-3, 1e-3, 0.0);
        assert!((u - 1.0).abs() < 1e-12);

        // Kurganov–Petrova formula below it
        let (h, hu) = (4e-4, 1e-3);
        let (u, _) = config.desingularized_velocity(h, hu, 0.0);
        let expected = 2f64.sqrt() * h * hu / (h.powi(4) + 1e-12).sqrt();
        assert!((u - expected).abs() < 1e-12 * expected, "{u} vs {expected}");

        // Bounded as h → 0 with fixed momentum: |u| ≤ √2 |hu| h / ε²
        for h in [1e-5, 1e-8, 1e-12, 1e-300] {
            let (u, _) = config.desingularized_velocity(h, 1e-3, 0.0);
            assert!(u.is_finite() && u <= 2f64.sqrt() * 1e-3 * h / 1e-6 * (1.0 + 1e-12));
        }
        assert_eq!(config.desingularized_velocity(0.0, 1.0, 1.0), (0.0, 0.0));
        assert_eq!(config.desingularized_velocity(-1e-3, 1.0, 1.0), (0.0, 0.0));
    }

    #[test]
    fn test_velocity_cap() {
        let config = WetDryConfig::new(Depth::new(0.01), 9.81).with_max_velocity(10.0);

        // Normal velocity (not capped)
        let mut state = SWEState2D::new(1.0, 5.0, 0.0);
        apply_wet_dry_correction(&mut state, &config);
        assert!((state.hu - 5.0).abs() < 1e-12);

        // High velocity (capped, direction kept)
        let mut state = SWEState2D::new(1.0, 30.0, 40.0);
        apply_wet_dry_correction(&mut state, &config);
        assert!((state.hu.hypot(state.hv) - 10.0).abs() < 1e-12);
        assert!((state.hu / state.hv - 0.75).abs() < 1e-12);
    }

    #[test]
    fn test_thin_layer_rate() {
        let config = WetDryConfig::new(Depth::new(1e-3), 9.81).with_relaxation_time(2.0);
        assert!((config.thin_layer_rate(5e-4) - 0.5).abs() < 1e-15);
        assert!((config.thin_layer_rate(1e-4) - 40.5).abs() < 1e-12);
        // Wet water is not relaxed at all
        assert_eq!(config.thin_layer_rate(1e-3), 0.0);
        assert_eq!(config.thin_layer_rate(1.0), 0.0);
        assert_eq!(config.thin_layer_rate(0.0), f64::INFINITY);
    }

    #[test]
    fn test_apply_correction_single_node() {
        let config = WetDryConfig::new(Depth::new(0.01), 9.81);

        // Negative depth becomes dry with no momentum
        let mut state = SWEState2D::new(-0.1, 1.0, 0.5);
        apply_wet_dry_correction(&mut state, &config);
        assert_eq!(state, SWEState2D::zero());

        // Wet state unchanged
        let mut state = SWEState2D::new(1.0, 5.0, 3.0);
        apply_wet_dry_correction(&mut state, &config);
        assert_eq!(state, SWEState2D::new(1.0, 5.0, 3.0));
    }

    #[test]
    fn test_element_correction_preserves_nonnegative_mass() {
        let ops = DGOperators2D::new(2);
        let config = WetDryConfig::new(Depth::new(0.01), 9.81);
        let mut solution = SWESolution2D::new(1, ops.n_nodes);
        let k = ElementIndex::new(0);

        for i in 0..ops.n_nodes {
            let h = if i == 0 { -0.01 } else { 0.02 };
            solution.set_state(k, i, SWEState2D::new(h, 0.2, -0.1));
        }

        let initial_mass = weighted_mass(&solution, &ops).max(0.0);
        assert_eq!(
            apply_wet_dry_correction_all(&mut solution, &unit_geometry(&ops, 1), &config),
            0
        );
        let final_mass = weighted_mass(&solution, &ops);

        assert!(
            solution.element_h(k).iter().all(|&h| h >= 0.0),
            "Wet/dry correction must leave nonnegative nodal depths"
        );
        assert!(
            (final_mass - initial_mass).abs() < 1e-12,
            "Wet/dry correction changed element mass: initial={initial_mass:.16e}, final={final_mass:.16e}"
        );
    }

    #[test]
    fn test_correction_keeps_lake_at_rest_shoreline() {
        // The old correction rescaled every depth of an element with a clipped
        // node, lowering its wet nodes in every stage (notes/review-2026-09-25.md §1.5).
        let ops = DGOperators2D::new(3);
        let config = WetDryConfig::default();
        let mut solution = SWESolution2D::new(1, ops.n_nodes);
        for i in 0..ops.n_nodes {
            let bed = 0.4 * ops.nodes_r[i] - 0.2 * ops.nodes_s[i];
            let h = (0.1 - bed).max(0.0);
            solution.set_state(ElementIndex::new(0), i, SWEState2D::new(h, 0.0, 0.0));
        }
        let before = solution.data.clone();
        assert_eq!(
            apply_wet_dry_correction_all(&mut solution, &unit_geometry(&ops, 1), &config),
            0
        );
        assert_eq!(solution.data, before);
    }

    #[test]
    fn test_correction_does_not_touch_wet_currents() {
        // The old α(h) blending damped every node shallower than 10·h_min in
        // every stage; wet nodes must now keep their momentum exactly.
        let ops = DGOperators2D::new(2);
        let config = WetDryConfig::new(Depth::new(1e-3), 9.81);
        let mut solution = SWESolution2D::new(1, ops.n_nodes);
        for i in 0..ops.n_nodes {
            let h = 0.005 + 0.001 * i as f64;
            solution.set_state(
                ElementIndex::new(0),
                i,
                SWEState2D::new(h, 0.3 * h, -0.2 * h),
            );
        }
        let before = solution.data.clone();
        apply_wet_dry_correction_all(&mut solution, &unit_geometry(&ops, 1), &config);
        assert_eq!(solution.data, before);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_correction_matches_serial() {
        let ops = DGOperators2D::new(2);
        let config = WetDryConfig::new(Depth::new(0.01), 9.81).with_max_velocity(3.0);
        let n_elements = 16;
        let mut input = SWESolution2D::new(n_elements, ops.n_nodes);
        for k in ElementIndex::iter(n_elements) {
            for i in 0..ops.n_nodes {
                let x = (7 * k.as_usize() + 3 * i) as f64;
                let h = 0.02 * (0.37 * x).sin() + if k.as_usize() == 3 { -0.03 } else { 0.01 };
                input.set_state(k, i, SWEState2D::new(h, (0.5 * x).cos(), 0.1 * x.sin()));
            }
        }
        let (mut serial, mut parallel) = (input.clone(), input.clone());
        let n_serial =
            apply_wet_dry_correction_all(&mut serial, &unit_geometry(&ops, n_elements), &config);
        let n_parallel = apply_wet_dry_correction_all_parallel(
            &mut parallel,
            &unit_geometry(&ops, n_elements),
            &config,
        );
        assert_eq!(serial.data, parallel.data);
        assert_eq!(n_serial, n_parallel);
        assert_eq!(n_serial, 1);
    }

    #[test]
    fn test_implicit_damping_is_exact_quadratic_drag_and_sign_preserving() {
        let friction = ManningFriction2D::new(9.81, 0.03);
        let damping = ImplicitDamping2D {
            friction: Some(&friction),
            cages: None,
            wet_dry: None,
            h_min: Depth::new(1e-6),
        };
        let (h, u0) = (0.5, 2.0);
        let from = {
            let mut s = SWESolution2D::new(1, 1);
            s.set_state(ElementIndex::new(0), 0, SWEState2D::new(h, h * u0, 0.0));
            s
        };
        for dt in [0.1, 10.0, 1e5] {
            let mut stage = from.clone();
            apply_implicit_damping_2d(&mut stage, &from, dt, &damping);
            // d|u|/dt = −a|u|², a = g n²/h^{4/3}: |u| = u0 / (1 + a u0 dt)
            let a = 9.81 * 0.03f64.powi(2) / h.powf(4.0 / 3.0);
            let exact = u0 / (1.0 + a * u0 * dt);
            let u = stage.get_state(ElementIndex::new(0), 0).hu / h;
            assert!(
                u > 0.0 && (u - exact).abs() < 1e-14 * u0,
                "dt = {dt}: {u} vs {exact}"
            );
        }
    }

    /// The block path of wet nodes gives what the node-by-node path gives,
    /// bit for bit, in blocks that are all wet, at rest, or hold thin, dry and
    /// tiny depths (which fall back), for both friction laws.
    #[test]
    fn wet_blocks_are_damped_as_node_by_node_bit_for_bit() {
        let config = WetDryConfig::new(Depth::new(1e-3), 9.81);
        let manning = ManningFriction2D::new(9.81, 0.025);
        let chezy = crate::source::ChezyFriction2D::new(2.5e-3);
        let laws: [&dyn BottomFriction2D; 2] = [&manning, &chezy];
        let n = 5 * WET_BLOCK + 7;
        let x = |i: usize, a: f64| (a * i as f64).sin();
        let depth = |i: usize| match i {
            // One block with a thin node, a dry one and a tiny depth
            70 => 4e-4,
            71 => 0.0,
            72 => 1e-300,
            _ => 3.0 + 2.9 * x(i, 0.7) + 1e3 * (x(i, 0.01) + 1.0),
        };
        let fill = |shift: usize| {
            let h: Vec<f64> = (0..n).map(|i| depth(i + shift) + 1e-3).collect();
            // Still water in one block
            let moving = |i: usize| if (200..264).contains(&i) { 0.0 } else { 1.0 };
            let hu: Vec<f64> = (0..n).map(|i| moving(i) * h[i] * 1.3 * x(i, 1.9)).collect();
            let hv: Vec<f64> = (0..n).map(|i| moving(i) * h[i] * 0.4 * x(i, 2.3)).collect();
            (h, hu, hv)
        };
        let (fh, fhu, fhv) = fill(0);
        let (h, hu, hv) = fill(3);
        for friction in laws {
            let damping = ImplicitDamping2D {
                friction: Some(friction),
                cages: None,
                wet_dry: Some(&config),
                h_min: Depth::new(1e-6),
            };
            for dt in [0.3, 40.0] {
                let (mut hu_a, mut hv_a) = (hu.clone(), hv.clone());
                let (mut hu_b, mut hv_b) = (hu.clone(), hv.clone());
                damp_nodes(&h, &mut hu_a, &mut hv_a, (&fh, &fhu, &fhv), 0, dt, &damping);
                damp_nodes_one_by_one(&h, &mut hu_b, &mut hv_b, (&fh, &fhu, &fhv), 0, dt, &damping);
                let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&hu_a), bits(&hu_b));
                assert_eq!(bits(&hv_a), bits(&hv_b));
                // The friction did something
                assert!(hu_a.iter().zip(&hu).any(|(a, b)| a != b));
            }
        }
    }

    #[test]
    fn test_implicit_thin_layer_relaxation() {
        let config = WetDryConfig::new(Depth::new(1e-3), 9.81);
        let damping = ImplicitDamping2D {
            friction: None,
            cages: None,
            wet_dry: Some(&config),
            h_min: Depth::new(1e-6),
        };
        let mut from = SWESolution2D::new(1, 3);
        let k = ElementIndex::new(0);
        from.set_state(k, 0, SWEState2D::new(5e-4, 1e-4, 0.0)); // rate 1/s
        from.set_state(k, 1, SWEState2D::new(10.0, 1.0, 1.0)); // wet: rate 0
        from.set_state(k, 2, SWEState2D::new(0.0, 0.0, 0.0));
        let mut stage = from.clone();
        stage.set_state(k, 2, SWEState2D::new(0.0, 0.3, 0.3));

        apply_implicit_damping_2d(&mut stage, &from, 1.0, &damping);
        assert!((stage.get_state(k, 0).hu - 0.5e-4).abs() < 1e-18);
        assert_eq!(stage.get_state(k, 1).hu, 1.0);
        assert_eq!(stage.get_state(k, 2), SWEState2D::zero());
    }
}
