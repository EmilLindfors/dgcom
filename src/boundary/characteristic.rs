//! The characteristic open boundary condition.
//!
//! Flather, radiation, tidal and nesting boundaries differ only in the
//! external data that sets the *incoming* Riemann invariant; the *outgoing*
//! one always comes from the interior. [`CharacteristicOBC`] builds the state
//! on the boundary from the two invariants and evaluates the boundary flux as
//! its physical flux `F(q_b)·n`, so the result is the same for every Riemann
//! solver and spatial formulation. The external data comes from an
//! [`ExternalStateProvider`]: still water, a harmonic tide, a boundary tide
//! atlas, parent-model output, or any closure.
//!
//! # Boundary state
//!
//! With outward unit normal `n`, `u_n = u·n` and `c = √(g h)`, the Riemann
//! invariants of the SWE normal to the boundary are
//!
//! ```text
//! w+ = u_n + 2c   (outgoing, speed u_n + c)
//! w− = u_n − 2c   (incoming, speed u_n − c)
//! ```
//!
//! At a subcritical boundary (`|u_n| < c`) the boundary state takes `w+` from
//! the interior trace and `w−` from the external state `(h_e, u_n,e)`,
//! `h_e = η_e − B`:
//!
//! ```text
//! u_n,b = ½ (w+_int + w−_ext),     c_b = ¼ (w+_int − w−_ext),     h_b = c_b² / g
//! ```
//!
//! and the tangential velocity from the upwind side (interior on outflow). This
//! is the nonlinear form of the Flather (1976) condition
//! `u_n = u_n,e + √(g/h)(η − η_e)`, to which it reduces for small amplitude.
//! At supercritical outflow the interior state is used as is, at supercritical
//! inflow the external state. When one side is dry, the state is the sonic
//! point of the rarefaction into the dry side (exact Riemann solution at the
//! boundary): `u_n,b = c_b = w+_int / 3` draining into a dry exterior,
//! `u_n,b = −c_b = w−_ext / 3` flooding a dry interior.
//!
//! The flux is then `F(q_b)·n`. With the older ghost-state formulation the
//! Riemann solver received `(q_int, q_ext)`; that reproduces the invariant
//! relation exactly only for Roe (Lax–Friedrichs only at `u = 0`).
//!
//! # Elevation-only data
//!
//! If the provider gives no velocity, [`ElevationOnly`] chooses `u_n,e`:
//! - [`ElevationOnly::AtRest`]: `u_n,e = 0`. The boundary elevation matches
//!   `η_e` only at an antinode of a standing (co-oscillating) tide; a
//!   progressive wave entering through the boundary arrives at half amplitude.
//! - [`ElevationOnly::IncomingWave`]: the external state is a simple wave
//!   entering the domain from still water at mean sea level,
//!   `u_n,e = 2(√(g H) − √(g h_e))` with `H = −B` (linearised
//!   `u_n,e = −√(g/H) η_e`). A progressive tide is delivered at full
//!   amplitude, and outgoing waves still leave.
//!
//! The tangential velocity of elevation-only data is the interior one.
//!
//! # Oblique incidence
//!
//! The condition is one-dimensional (normal to the boundary). A plane wave
//! reaching it at angle θ from the normal is partly reflected, with linear
//! reflection coefficient `(1 − cos θ) / (1 + cos θ)`: 0 at normal
//! incidence, 17 % at 45° (`tests/open_boundary_flather_test.rs`).

use std::sync::atomic::AtomicBool;

use super::bathymetry_validation::warn_once_if_misconfigured;
use super::{BCContext2D, BoundaryState, SWEBoundaryCondition2D};
use crate::io::BoundaryTimeSeries;
use crate::mesh::BoundaryTag;
use crate::solver::SWEState2D;
use crate::types::Depth;

/// External (far-field) state at a boundary point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExternalState {
    /// Surface elevation η (m); the depth is `η − B`.
    pub eta: f64,
    /// Depth-averaged velocity (u, v), if known.
    pub velocity: Option<(f64, f64)>,
}

impl ExternalState {
    /// Elevation and velocity.
    pub fn new(eta: f64, u: f64, v: f64) -> Self {
        Self {
            eta,
            velocity: Some((u, v)),
        }
    }

    /// Elevation only; see [`ElevationOnly`] for the velocity used.
    pub fn elevation(eta: f64) -> Self {
        Self {
            eta,
            velocity: None,
        }
    }
}

/// Source of the external state at the boundary points of a
/// [`CharacteristicOBC`].
pub trait ExternalStateProvider: Send + Sync {
    /// External state at the boundary point and time of `ctx`.
    fn external_state(&self, ctx: &BCContext2D) -> ExternalState;
}

/// Any closure `(x, y, t) → ExternalState`.
impl<F> ExternalStateProvider for F
where
    F: Fn(f64, f64, f64) -> ExternalState + Send + Sync,
{
    fn external_state(&self, ctx: &BCContext2D) -> ExternalState {
        let (x, y) = ctx.position;
        self(x, y, ctx.time)
    }
}

/// Still water at surface elevation `eta` (m): a radiation condition.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StillWater {
    /// Far-field surface elevation (m).
    pub eta: f64,
}

impl StillWater {
    /// Still water at elevation `eta`.
    pub fn at(eta: f64) -> Self {
        Self { eta }
    }
}

impl ExternalStateProvider for StillWater {
    fn external_state(&self, _ctx: &BCContext2D) -> ExternalState {
        ExternalState::new(self.eta, 0.0, 0.0)
    }
}

/// Parent-model output at one point, applied along the whole boundary: the
/// time series gives depth `h` and momentum; the external elevation is
/// `h + B_parent`, with the parent's bed `B_parent = −h_ref`.
#[derive(Clone, Debug)]
pub struct ParentTimeSeries {
    series: BoundaryTimeSeries,
    /// Parent still-water depth (m): η = h_parent − h_ref.
    pub h_ref: f64,
}

impl ParentTimeSeries {
    /// Parent depth/momentum series; `h_ref` is the parent's still-water depth.
    pub fn new(series: BoundaryTimeSeries, h_ref: f64) -> Self {
        Self { series, h_ref }
    }

    /// The underlying series.
    pub fn series(&self) -> &BoundaryTimeSeries {
        &self.series
    }
}

impl ExternalStateProvider for ParentTimeSeries {
    fn external_state(&self, ctx: &BCContext2D) -> ExternalState {
        let parent = self.series.interpolate(ctx.time);
        let (u, v) = parent.velocity_simple(Depth::new(ctx.h_min));
        ExternalState::new(parent.h - self.h_ref, u, v)
    }
}

/// A surface level (m) at boundary points, e.g. the inverse-barometer
/// level of an atmospheric pressure field.
pub trait BoundaryLevel: Send + Sync {
    /// Level at the boundary point and time of `ctx`.
    fn level(&self, ctx: &BCContext2D) -> f64;
}

/// Any closure `(x, y, t) → level`.
impl<F> BoundaryLevel for F
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    fn level(&self, ctx: &BCContext2D) -> f64 {
        let (x, y) = ctx.position;
        self(x, y, ctx.time)
    }
}

/// Adds the inverse-barometer response of the sea to another provider's
/// elevation.
///
/// With atmospheric pressure forcing (`AtmosphericPressure2D`,
/// `GriddedAtmosphere2D`) the interior sea level settles at
/// `η_IB = −(p − p₀)/(ρ g)` (≈ −1 cm/hPa), which the pressure-gradient force
/// balances exactly. External data without that response (tides, still
/// water, a parent model forced without pressure) pins the boundary to the
/// wrong level and drives a spurious current through it. `level` is the
/// inverse-barometer elevation: a [`BoundaryLevel`] such as
/// `GriddedAtmosphere2D`, or a closure like
/// `|x, y, t| pressure.inverse_barometer(x, y, t)`.
///
/// A parent that carries part of the response already takes a fraction of
/// it ([`InverseBarometer::with_factor`]): NorKyst-800's ζ responds to
/// pressure at ≈ −0.22 cm/hPa, a quarter of the static −0.99, because its
/// Shchepetkin open boundary holds the level to its own parent's without the
/// inverse barometer (ROMS applies `PRESS_COMPENSATE` in Flather only;
/// `notes/norkyst-issues.md` §6).
#[derive(Clone, Debug)]
pub struct InverseBarometer<P, F> {
    provider: P,
    level: F,
    factor: f64,
}

impl<P, F> InverseBarometer<P, F>
where
    P: ExternalStateProvider,
    F: BoundaryLevel,
{
    /// `provider`'s external state, raised by `level`.
    pub fn new(provider: P, level: F) -> Self {
        Self {
            provider,
            level,
            factor: 1.0,
        }
    }

    /// Raise by `factor` times `level` instead (1: the full response).
    pub fn with_factor(mut self, factor: f64) -> Self {
        self.factor = factor;
        self
    }
}

impl<P, F> ExternalStateProvider for InverseBarometer<P, F>
where
    P: ExternalStateProvider,
    F: BoundaryLevel,
{
    fn external_state(&self, ctx: &BCContext2D) -> ExternalState {
        let mut state = self.provider.external_state(ctx);
        state.eta += self.factor * self.level.level(ctx);
        state
    }
}

/// External normal velocity for elevation-only data; see the module docs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ElevationOnly {
    /// `u_n,e = 0`: standing-tide assumption.
    #[default]
    AtRest,
    /// A simple wave entering from still water at mean sea level.
    IncomingWave,
}

/// Characteristic open boundary: `w+` from the interior, `w−` from the
/// provider's external state, flux `F(q_b)·n`. See the module docs.
#[derive(Clone, Debug)]
pub struct CharacteristicOBC<P> {
    /// External data.
    pub provider: P,
    /// Velocity of elevation-only external data.
    pub elevation_only: ElevationOnly,
}

impl<P: ExternalStateProvider> CharacteristicOBC<P> {
    /// Characteristic OBC with external data from `provider`
    /// (elevation-only data taken at rest).
    pub fn new(provider: P) -> Self {
        Self {
            provider,
            elevation_only: ElevationOnly::AtRest,
        }
    }

    /// Treat elevation-only data as a wave entering the domain.
    pub fn with_incoming_wave(mut self) -> Self {
        self.elevation_only = ElevationOnly::IncomingWave;
        self
    }

    /// Set the velocity of elevation-only data.
    pub fn with_elevation_only(mut self, mode: ElevationOnly) -> Self {
        self.elevation_only = mode;
        self
    }

    /// The boundary state `q_b` at the point of `ctx`.
    pub fn state(&self, ctx: &BCContext2D) -> SWEState2D {
        static WARNED: AtomicBool = AtomicBool::new(false);
        let external = self.provider.external_state(ctx);
        warn_once_if_misconfigured(
            &WARNED,
            "CharacteristicOBC",
            ctx.interior_state.h,
            ctx.bathymetry,
            external.eta,
        );
        let (nx, ny) = ctx.normal;
        let g = ctx.g;
        let h_e = (external.eta - ctx.bathymetry).max(0.0);
        let (un_e, ut_e) = match external.velocity {
            Some((u, v)) => (u * nx + v * ny, -u * ny + v * nx),
            None => {
                let un = match self.elevation_only {
                    ElevationOnly::AtRest => 0.0,
                    ElevationOnly::IncomingWave => {
                        2.0 * ((g * (-ctx.bathymetry).max(0.0)).sqrt() - (g * h_e).sqrt())
                    }
                };
                (un, ctx.interior_tangential_velocity())
            }
        };
        characteristic_state(
            &ctx.interior_state,
            (h_e, un_e, ut_e),
            ctx.normal,
            g,
            ctx.h_min,
        )
    }
}

impl CharacteristicOBC<StillWater> {
    /// Radiation towards still water at mean sea level.
    pub fn still_water() -> Self {
        Self::new(StillWater::default())
    }
}

impl<P: ExternalStateProvider> SWEBoundaryCondition2D for CharacteristicOBC<P> {
    fn is_wall(&self, _tag: Option<BoundaryTag>) -> Option<bool> {
        Some(false)
    }

    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        self.state(ctx)
    }

    fn boundary_state(&self, ctx: &BCContext2D) -> BoundaryState {
        BoundaryState::Exact(self.state(ctx))
    }

    fn name(&self) -> &'static str {
        "characteristic_obc"
    }
}

/// State on the boundary from the interior trace `interior` and the external
/// `(h_e, u_n,e, u_t,e)` (depth, normal and tangential velocity), for outward
/// unit normal `normal`. See the module docs; nodes with `h ≤ h_min` count as
/// dry.
pub fn characteristic_state(
    interior: &SWEState2D,
    (h_e, un_e, ut_e): (f64, f64, f64),
    (nx, ny): (f64, f64),
    g: f64,
    h_min: f64,
) -> SWEState2D {
    let h_i = interior.h.max(0.0);
    let (un_i, ut_i) = if h_i > h_min {
        let (u, v) = (interior.hu / h_i, interior.hv / h_i);
        (u * nx + v * ny, -u * ny + v * nx)
    } else {
        (0.0, 0.0)
    };
    let (c_i, c_e) = ((g * h_i).sqrt(), (g * h_e).sqrt());
    let (w_out, w_in) = (un_i + 2.0 * c_i, un_e - 2.0 * c_e);

    // (c_b, u_n,b) of the invariant states, or a whole side
    let (h_b, un_b) = match (h_i > h_min, h_e > h_min) {
        (false, false) => (0.0, 0.0),
        _ if h_i > h_min && un_i >= c_i => (h_i, un_i),
        _ if h_e > h_min && un_e <= -c_e => (h_e, un_e),
        // Drains into a dry exterior: sonic point of the rarefaction
        (true, false) => {
            let c = (w_out / 3.0).max(0.0);
            (c * c / g, c)
        }
        // Floods a dry interior
        (false, true) => {
            let c = (-w_in / 3.0).max(0.0);
            (c * c / g, -c)
        }
        (true, true) => {
            let c = (0.25 * (w_out - w_in)).max(0.0);
            (c * c / g, 0.5 * (w_out + w_in))
        }
    };
    let ut_b = if un_b > 0.0 { ut_i } else { ut_e };
    SWEState2D::from_primitives(h_b, un_b * nx - ut_b * ny, un_b * ny + ut_b * nx)
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 9.81;
    const H_MIN: f64 = 1e-6;

    fn ctx(state: SWEState2D, bed: f64, normal: (f64, f64)) -> BCContext2D {
        BCContext2D::new(0.0, (0.0, 0.0), state, bed, normal, G, H_MIN)
    }

    fn normal_velocity(q: &SWEState2D, (nx, ny): (f64, f64)) -> f64 {
        (q.hu * nx + q.hv * ny) / q.h
    }

    /// Still water on both sides is kept exactly: lake at rest.
    #[test]
    fn lake_at_rest_is_kept() {
        let bc = CharacteristicOBC::still_water();
        let n = (0.6, 0.8);
        let q = bc.state(&ctx(SWEState2D::new(40.0, 0.0, 0.0), -40.0, n));
        assert!((q.h - 40.0).abs() < 1e-12);
        assert_eq!((q.hu, q.hv), (0.0, 0.0));
    }

    /// The inverse barometer raises the provider's level by its level,
    /// scaled by the factor (a parent carrying part of the response).
    #[test]
    fn the_inverse_barometer_adds_a_fraction_of_its_level() {
        let tide = |_: f64, _: f64, _: f64| ExternalState::new(0.5, 0.1, 0.0);
        let low = |_: f64, _: f64, _: f64| 0.2;
        let c = ctx(SWEState2D::new(40.0, 0.0, 0.0), -40.0, (1.0, 0.0));
        let full = InverseBarometer::new(tide, low).external_state(&c);
        assert!((full.eta - 0.7).abs() < 1e-12);
        let part = InverseBarometer::new(tide, low)
            .with_factor(0.6)
            .external_state(&c);
        assert!((part.eta - 0.62).abs() < 1e-12);
        assert_eq!(part.velocity, full.velocity);
    }

    /// The boundary state carries the interior's outgoing and the external
    /// incoming invariant.
    #[test]
    fn invariants_come_from_the_right_sides() {
        let n = (0.0, -1.0);
        let interior = SWEState2D::from_primitives(12.0, 0.3, -0.4);
        let bc = CharacteristicOBC::new(|_, _, _| ExternalState::new(0.5, -0.2, 0.1));
        let q = bc.state(&ctx(interior, -10.0, n));

        let invariants = |q: &SWEState2D| {
            let (un, c) = (normal_velocity(q, n), (G * q.h).sqrt());
            (un + 2.0 * c, un - 2.0 * c)
        };
        let (w_out, _) = invariants(&interior);
        let (_, w_in) = invariants(&SWEState2D::from_primitives(10.5, -0.2, 0.1));
        let (b_out, b_in) = invariants(&q);
        assert!((b_out - w_out).abs() < 1e-12);
        assert!((b_in - w_in).abs() < 1e-12);
        // The interior stands 1.5 m higher: outflow (u_n,b > 0), so the
        // tangential velocity is the interior one, u_t = −u ny + v nx = 0.3
        assert!(normal_velocity(&q, n) > 0.0);
        let ut = (-q.hu * n.1 + q.hv * n.0) / q.h;
        assert!((ut - 0.3).abs() < 1e-12);

        // Deeper outside: inflow, tangential velocity from outside, −(−0.2)(−1) = −0.2
        let q = bc.state(&ctx(SWEState2D::from_primitives(10.0, 0.3, 0.0), -10.0, n));
        assert!(normal_velocity(&q, n) < 0.0);
        let ut = (-q.hu * n.1 + q.hv * n.0) / q.h;
        assert!((ut + 0.2).abs() < 1e-12);
    }

    /// For small amplitude the state satisfies Flather's relation
    /// u_n = u_n,e + √(g/H)(η − η_e) to second order.
    #[test]
    fn small_amplitude_is_flather() {
        let (depth, n) = (20.0, (1.0, 0.0));
        let (eta_i, un_i, eta_e, un_e) = (1e-3, 2e-4, -5e-4, 1e-4);
        let bc = CharacteristicOBC::new(move |_, _, _| ExternalState::new(eta_e, un_e, 0.0));
        let q = bc.state(&ctx(
            SWEState2D::from_primitives(depth + eta_i, un_i, 0.0),
            -depth,
            n,
        ));
        let eta_b = q.h - depth;
        let flather = un_e + (G / depth).sqrt() * (eta_b - eta_e);
        assert!((normal_velocity(&q, n) - flather).abs() < 1e-6);
    }

    #[test]
    fn supercritical_sides_are_taken_whole() {
        let n = (1.0, 0.0);
        let bc = CharacteristicOBC::new(|_, _, _| ExternalState::new(0.0, -30.0, 0.0));
        // Supercritical outflow: interior
        let out = SWEState2D::from_primitives(1.0, 5.0, 0.3);
        assert_eq!(bc.state(&ctx(out, -1.0, n)), out);
        // Supercritical inflow (u_n,e = −30 < −c_e ≈ −3.1): exterior
        let q = bc.state(&ctx(SWEState2D::from_primitives(1.0, 0.0, 0.0), -1.0, n));
        assert!((q.h - 1.0).abs() < 1e-12 && (q.hu + 30.0).abs() < 1e-12);
    }

    /// Wet interior next to dry land outside: the sonic Ritter state, which
    /// drains mass out of the domain; and nothing flows between two dry sides.
    #[test]
    fn dry_sides_give_sonic_or_empty_states() {
        let n = (1.0, 0.0);
        let bc = CharacteristicOBC::still_water();
        let h0 = 2.0;
        // Exterior dry: η_e = 0 above a bed at +1 m
        let q = bc.state(&ctx(SWEState2D::new(h0, 0.0, 0.0), 1.0, n));
        let c = 2.0 * (G * h0).sqrt() / 3.0;
        assert!((q.h - c * c / G).abs() < 1e-12);
        assert!((normal_velocity(&q, n) - c).abs() < 1e-12);

        let q = bc.state(&ctx(SWEState2D::new(0.0, 0.0, 0.0), 1.0, n));
        assert_eq!(q, SWEState2D::new(0.0, 0.0, 0.0));

        // Interior dry, sea outside: floods inward
        let q = bc.state(&ctx(SWEState2D::new(0.0, 0.0, 0.0), -h0, n));
        assert!((q.h - c * c / G).abs() < 1e-12);
        assert!((normal_velocity(&q, n) + c).abs() < 1e-12);
    }

    #[test]
    fn incoming_wave_is_a_simple_wave_from_still_water() {
        let (depth, n) = (10.0, (-1.0, 0.0));
        let bc =
            CharacteristicOBC::new(|_, _, _| ExternalState::elevation(0.1)).with_incoming_wave();
        // Interior already carries the same simple wave: the boundary state is it
        let h = depth + 0.1;
        let un = 2.0 * ((G * depth).sqrt() - (G * h).sqrt());
        let interior = SWEState2D::from_primitives(h, un * n.0, 0.0);
        let q = bc.state(&ctx(interior, -depth, n));
        assert!((q.h - h).abs() < 1e-12);
        assert!((normal_velocity(&q, n) - un).abs() < 1e-12);
    }
}
