//! Boundary conditions for shallow water equations.
//!
//! Boundary conditions supply the state outside (or on) the domain boundary
//! for the face flux.
//!
//! # 2D boundary conditions
//!
//! | BC Type | Description |
//! |---------|-------------|
//! | `Reflective2D` | Wall (no-flux), tangential velocity preserved |
//! | `CharacteristicOBC` | Open boundary: incoming Riemann invariant from external data |
//! | `HarmonicTidal2D`, `Tidal2D` | Clamped elevation (Dirichlet; reflects) |
//! | `Discharge2D`, `ConstantDischarge2D` | Prescribed flow rate |
//! | `MultiBoundaryCondition2D` | Per-tag dispatch |
//!
//! # Open boundaries
//!
//! Radiation, tidal (Flather) and nesting boundaries differ only in the
//! external data that fixes the incoming Riemann invariant, so they are one
//! [`CharacteristicOBC`] with an [`ExternalStateProvider`]:
//!
//! | External data | Provider |
//! |---------------|----------|
//! | Still water (radiation) | [`StillWater`] |
//! | Uniform harmonic tide | [`HarmonicTide`] |
//! | Spatially varying tides (NorKyst, TPXO) | [`BoundaryTides`] from a [`TidalAtlas`] |
//! | Parent model at one point | [`ParentTimeSeries`] |
//! | Parent model output (NetCDF) | `OceanModelState` (feature `netcdf`) |
//! | Anything else | a closure `(x, y, t) → ExternalState` |
//!
//! The boundary state is built from the invariants and the face flux is its
//! physical flux, so it does not depend on the Riemann solver; see
//! [`characteristic`](CharacteristicOBC) for the formulation, elevation-only
//! data and oblique incidence. Time-dependent data refers to a
//! [`ModelClock`](crate::time::ModelClock).
//!
//! # Bathymetry convention
//!
//! Depth is `h = η − B` with `B` the bed elevation (negative under water), so
//! external elevations are levels, not depths. For a basin of depth h₀ set
//! `B = −h₀`:
//!
//! ```text
//! bathymetry = Bathymetry2D::constant(n_elements, n_nodes, -h0);
//! ```
//!
//! With B = 0 by mistake, the interior surface sits h₀ above the external
//! level and the boundary drains the domain like a dam break;
//! [`CharacteristicOBC`] warns once when it sees that configuration.
//!
//! # Sponge layers
//!
//! Clamped-elevation boundaries (`HarmonicTidal2D`) reflect outgoing waves; in
//! closed basins add a sponge (`SpongeLayer2D`, or
//! [`TidalSimulationBuilder::closed_basin_stable`]) or use a
//! [`CharacteristicOBC`].

pub mod band;
pub mod bathymetry_validation;
mod boundary_2d;
mod characteristic;
mod coverage;
mod harmonic_tide;
mod multi_bc_2d;
mod nesting_3d;
mod ocean_columns;
mod ocean_nesting;
mod radiation;
mod reflective;
mod tidal;
mod tidal_atlas;
mod tidal_config;

// 1D boundary conditions
pub use radiation::{FlatherBC, RadiationBC, SpongeLayer};
pub use reflective::{PartialSlipBC, ReflectiveBC};
pub use tidal::{InterpolatedTidalBC, TidalBC, TidalConstituent};

// 2D boundary conditions
pub use boundary_2d::{
    BCContext2D, BoundaryState, ConstantDischarge2D, Discharge2D, Extrapolation2D, FixedState2D,
    HarmonicTidal2D, Reflective2D, SWEBoundaryCondition2D, Tidal2D,
};
pub use characteristic::{
    BoundaryLevel, CharacteristicOBC, ElevationOnly, ExternalState, ExternalStateProvider,
    InverseBarometer, ParentTimeSeries, StillWater, characteristic_state,
};
pub use coverage::{BoundaryCoverage, CoverageError};
pub use harmonic_tide::{HarmonicTide, tidal_ramp};
pub use multi_bc_2d::MultiBoundaryCondition2D;
pub use nesting_3d::{
    ColumnContext3D, Nesting3D, NestingBand3D, NestingColumns, ParentColumn, ParentColumns3D,
    ReferenceColumns, Supplied,
};
pub use ocean_columns::{DeepReference, OceanColumnsOptions, OceanModelColumns};
pub use ocean_nesting::{NestingError, NestingOptions, NestingRelaxation2D, OceanModelState};
pub use tidal_atlas::{AtlasConstituent, AtlasPoint, BoundaryTides, TidalAtlas, TidalAtlasError};
pub use tidal_config::{SpongeConfig, TidalBCType, TidalSimulationBuilder};

// Re-export validation types
pub use bathymetry_validation::{
    BathymetryValidationConfig, BathymetryValidationResult, format_bathymetry_warning,
    validate_bathymetry_convention, warn_once_if_misconfigured,
};

use crate::solver::SWEState;

/// Reflect a velocity `(u, v)` across a solid wall with outward unit normal
/// `(nx, ny)`.
///
/// The normal component is reversed (`u_ext·n = −u_int·n`) and the tangential
/// component is preserved:
///
/// ```text
/// u_ext = u − 2 (u·n) nx
/// v_ext = v − 2 (u·n) ny
/// ```
///
/// This is the free-slip solid-wall ghost state: pairing it with an unchanged
/// exterior depth makes the normal mass/momentum flux through the wall vanish.
/// Shared by the 2D `Reflective2D` boundary and the 3D continuity / momentum
/// advection land-boundary treatment so all three use one convention.
#[inline]
pub fn reflect_velocity(u: f64, v: f64, nx: f64, ny: f64) -> (f64, f64) {
    let un = u * nx + v * ny;
    (u - 2.0 * un * nx, v - 2.0 * un * ny)
}

/// Context for boundary condition evaluation.
///
/// Provides all information needed to compute the ghost state at a boundary.
#[derive(Clone, Copy, Debug)]
pub struct BCContext {
    /// Current simulation time
    pub time: f64,
    /// Physical position of the boundary face
    pub position: f64,
    /// Interior state at the boundary
    pub interior_state: SWEState,
    /// Bathymetry at the boundary
    pub bathymetry: f64,
    /// Outward normal direction (-1 for left boundary, +1 for right boundary)
    pub normal: f64,
}

impl BCContext {
    /// Create a new boundary condition context.
    pub fn new(
        time: f64,
        position: f64,
        interior_state: SWEState,
        bathymetry: f64,
        normal: f64,
    ) -> Self {
        Self {
            time,
            position,
            interior_state,
            bathymetry,
            normal,
        }
    }

    /// Water surface elevation at the interior: η = h + B
    pub fn interior_surface_elevation(&self) -> f64 {
        self.interior_state.h + self.bathymetry
    }

    /// Interior velocity (with protection for dry cells).
    pub fn interior_velocity(&self, h_min: f64) -> f64 {
        self.interior_state.velocity_simple(h_min)
    }

    /// Check if this is a left boundary (normal = -1).
    pub fn is_left_boundary(&self) -> bool {
        self.normal < 0.0
    }

    /// Check if this is a right boundary (normal = +1).
    pub fn is_right_boundary(&self) -> bool {
        self.normal > 0.0
    }
}

/// Trait for shallow water boundary conditions.
///
/// Implementations compute a "ghost" state that represents the exterior
/// state at a boundary face. This ghost state is then used in the
/// numerical flux computation.
pub trait SWEBoundaryCondition: Send + Sync {
    /// Compute the ghost state for flux evaluation.
    ///
    /// # Arguments
    /// * `ctx` - Boundary condition context with interior state and metadata
    ///
    /// # Returns
    /// The ghost (exterior) state to use in flux computation
    fn ghost_state(&self, ctx: &BCContext) -> SWEState;

    /// Name of this boundary condition for debugging/logging.
    fn name(&self) -> &'static str;

    /// Check if this BC allows inflow (used for validation).
    fn allows_inflow(&self) -> bool {
        true
    }

    /// Check if this BC allows outflow (used for validation).
    fn allows_outflow(&self) -> bool {
        true
    }
}

/// Extrapolation boundary condition (zero-gradient).
///
/// Simply copies the interior state to the exterior.
/// This is equivalent to a zero-gradient condition.
#[derive(Clone, Debug, Default)]
pub struct ExtrapolationBC;

impl SWEBoundaryCondition for ExtrapolationBC {
    fn ghost_state(&self, ctx: &BCContext) -> SWEState {
        ctx.interior_state
    }

    fn name(&self) -> &'static str {
        "extrapolation"
    }
}

/// Fixed state boundary condition (Dirichlet).
///
/// Sets a fixed state at the boundary, regardless of interior state.
#[derive(Clone, Debug)]
pub struct FixedStateBC {
    /// Fixed state to impose
    pub state: SWEState,
}

impl FixedStateBC {
    /// Create a new fixed state BC.
    pub fn new(h: f64, hu: f64) -> Self {
        Self {
            state: SWEState::new(h, hu),
        }
    }

    /// Create from primitive variables.
    pub fn from_primitives(h: f64, u: f64) -> Self {
        Self {
            state: SWEState::from_primitives(h, u),
        }
    }
}

impl SWEBoundaryCondition for FixedStateBC {
    fn ghost_state(&self, _ctx: &BCContext) -> SWEState {
        self.state
    }

    fn name(&self) -> &'static str {
        "fixed_state"
    }
}

/// Discharge boundary condition.
///
/// Prescribes the flow rate Q = h * u at the boundary.
/// The depth is determined from interior information.
#[derive(Clone, Debug)]
pub struct DischargeBC {
    /// Prescribed discharge (m²/s in 1D)
    pub discharge: f64,
    /// Minimum depth
    pub h_min: f64,
}

impl DischargeBC {
    /// Create a new discharge BC.
    pub fn new(discharge: f64) -> Self {
        Self {
            discharge,
            h_min: 1e-6,
        }
    }
}

impl SWEBoundaryCondition for DischargeBC {
    fn ghost_state(&self, ctx: &BCContext) -> SWEState {
        // Use interior depth, prescribed discharge
        let h = ctx.interior_state.h.max(self.h_min);
        SWEState::new(h, self.discharge)
    }

    fn name(&self) -> &'static str {
        "discharge"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bc_context() {
        let ctx = BCContext::new(1.0, 0.0, SWEState::new(2.0, 3.0), 0.5, -1.0);

        assert!(ctx.is_left_boundary());
        assert!(!ctx.is_right_boundary());
        assert!((ctx.interior_surface_elevation() - 2.5).abs() < 1e-14);
    }

    #[test]
    fn test_extrapolation_bc() {
        let bc = ExtrapolationBC;
        let ctx = BCContext::new(0.0, 0.0, SWEState::new(2.0, 3.0), 0.0, 1.0);

        let ghost = bc.ghost_state(&ctx);
        assert!((ghost.h - 2.0).abs() < 1e-14);
        assert!((ghost.hu - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_fixed_state_bc() {
        let bc = FixedStateBC::new(1.5, 2.0);
        let ctx = BCContext::new(0.0, 0.0, SWEState::new(2.0, 3.0), 0.0, 1.0);

        let ghost = bc.ghost_state(&ctx);
        assert!((ghost.h - 1.5).abs() < 1e-14);
        assert!((ghost.hu - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_discharge_bc() {
        let bc = DischargeBC::new(5.0);
        let ctx = BCContext::new(0.0, 0.0, SWEState::new(2.0, 3.0), 0.0, 1.0);

        let ghost = bc.ghost_state(&ctx);
        assert!((ghost.h - 2.0).abs() < 1e-14); // Uses interior depth
        assert!((ghost.hu - 5.0).abs() < 1e-14); // Prescribed discharge
    }

    #[test]
    fn reflect_velocity_reverses_normal_preserves_tangential() {
        // Axis-aligned wall (normal +x): normal component flips, tangential kept.
        let (u, v) = reflect_velocity(1.0, 0.5, 1.0, 0.0);
        assert!((u - (-1.0)).abs() < 1e-14);
        assert!((v - 0.5).abs() < 1e-14);

        // Purely tangential flow is untouched.
        let (u, v) = reflect_velocity(0.0, 2.0, 1.0, 0.0);
        assert!(u.abs() < 1e-14);
        assert!((v - 2.0).abs() < 1e-14);
    }

    #[test]
    fn reflect_velocity_zero_normal_flux_at_wall() {
        // For any incident velocity and unit normal, the exterior normal velocity
        // must equal minus the interior normal velocity (=> zero Rusanov mass flux
        // when the exterior depth equals the interior depth). This is the property
        // that stops mass leaking through coastal walls (TODO P0.7).
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        for &(u, v, nx, ny) in &[
            (1.3, -0.7, 1.0, 0.0),
            (1.3, -0.7, 0.0, 1.0),
            (1.3, -0.7, inv_sqrt2, inv_sqrt2),
            (-2.1, 0.4, -inv_sqrt2, inv_sqrt2),
        ] {
            let un_int = u * nx + v * ny;
            let (ue, ve) = reflect_velocity(u, v, nx, ny);
            let un_ext = ue * nx + ve * ny;
            assert!(
                (un_ext + un_int).abs() < 1e-14,
                "normal velocity not reflected: un_int={un_int}, un_ext={un_ext}"
            );
        }
    }
}
