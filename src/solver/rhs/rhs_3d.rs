//! 3D RHS computation.
//!
//! Assembles the semi-discrete right-hand side of the 3D primitive equations
//! in two parts, because under mode splitting they need different inputs:
//!
//! - [`compute_momentum_rhs_3d`]: the pointwise momentum terms (baroclinic
//!   pressure gradient, Coriolis), as velocity tendencies.
//! - [`compute_transport_rhs_3d`]: everything that moves with the layer volume
//!   fluxes (momentum and tracer advection), as inventory tendencies
//!   `∂(H_z u)/∂t`, `∂(H_z C)/∂t`. Under mode splitting the fluxes are
//!   corrected to the barotropic transport of the step
//!   ([`crate::solver::rhs::LayerTransport`]), so they exist only after the
//!   barotropic pass; the slow forcing `G` of the pass uses the state's own
//!   layer transports instead (see [`crate::physics::Hydrostatic3D`]).
//!
//! Vertical mixing is handled implicitly in the time stepper.

use crate::mesh::Mesh2D;
use crate::mesh::data::Bathymetry2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::rhs::advection_3d::TracerBoundaryCondition3D;
use crate::solver::rhs::baroclinic::{
    BalancedReference, PressureGradientForm, compute_pressure_gradient,
};
use crate::solver::rhs::boundary_3d::{Boundaries3D, Exterior3D};
use crate::solver::rhs::coriolis_3d::apply_coriolis_3d;
use crate::solver::rhs::transport_3d::{
    AdvectedTracer, LayerTransport, MetricForm, MomentumAdvectionForm, VerticalAdvection,
    apply_momentum_transport_3d, apply_tracer_transport_3d,
};
use crate::solver::state::Solution3D;
use crate::source::CoriolisSource2D;
use crate::vertical::SigmaGrid;

/// Configuration for 3D RHS.
pub struct Rhs3DConfig<'a> {
    pub mesh: &'a Mesh2D,
    pub ops: &'a DGOperators2D,
    pub geom: &'a GeometricFactors2D,
    pub bathymetry: &'a Bathymetry2D,
    pub sigma: &'a SigmaGrid,
    pub coriolis: &'a CoriolisSource2D,
    pub temp_bc: &'a dyn TracerBoundaryCondition3D,
    pub salt_bc: &'a dyn TracerBoundaryCondition3D,
    /// Walls and open faces of the domain.
    pub boundaries: &'a Boundaries3D,
    /// Values outside the open faces (a nesting parent's), where not the
    /// interior's.
    pub exterior: Exterior3D<'a>,
    pub g: f64,
    pub rho0: f64,
    /// Columns shallower than this (m) are thin (3D wetting and drying).
    pub min_column_depth: f64,
    /// Reconstruction of the tracers at the σ-surfaces.
    pub vertical_advection: VerticalAdvection,
    /// Form of the horizontal momentum advection within the elements.
    pub momentum_advection_form: MomentumAdvectionForm,
    /// Reconstruction of the velocity at the σ-surfaces.
    pub momentum_vertical_advection: VerticalAdvection,
    /// How the baroclinic pressure gradient differences the columns.
    pub pressure_gradient: PressureGradientForm,
    /// A reference state balanced by the constant-depth form, if any (with
    /// [`PressureGradientForm::SigmaPairs`]).
    pub balanced_reference: Option<&'a BalancedReference>,
    /// The volume form of the horizontal divergences, the 2D module's (the
    /// pressure gradient takes its adjoint).
    pub metric: MetricForm,
    /// Temperature and salinity at rest (`[element][node][level]`), about
    /// which the vertical advection reconstructs, if any (experimental).
    pub tracer_reference: [Option<&'a [f64]>; 2],
    /// The temperature's and the salinity's physical ranges `(min, max)`, which the
    /// vertical advection keeps the water leaving an end layer within
    /// ([`UNBOUNDED`](super::transport_3d::UNBOUNDED) for none).
    pub tracer_range: [(f64, f64); 2],
}

/// Overwrite `rhs.u` and `rhs.v` with the velocity tendency of the pointwise
/// momentum terms: baroclinic pressure gradient and Coriolis. The advection
/// is [`compute_transport_rhs_3d`]'s. The other fields of `rhs` are left
/// alone.
///
/// `state.rho` must be current (density is refreshed by the caller, so that
/// `state` stays immutable here).
pub fn compute_momentum_rhs_3d(rhs: &mut Solution3D, state: &Solution3D, config: &Rhs3DConfig) {
    // The BAROCLINIC-ONLY PGF (rho_ref = ρ₀), which overwrites rhs.u/rhs.v.
    // Under mode splitting the barotropic term −g∇η is supplied by the 2D
    // sub-model's ½gh² flux; including it here too would double-count it in
    // the depth-averaged G-term coupling (see ModeSplitIntegrator). The ρ₀
    // part of the full PGF integrates to exactly −g∇η, so subtracting ρ₀
    // leaves only the density-driven baroclinic force.
    compute_pressure_gradient(
        state,
        config.mesh,
        config.bathymetry,
        config.sigma,
        config.ops,
        config.geom,
        config.g,
        config.rho0,
        config.rho0, // baroclinic-only PGF
        config.min_column_depth,
        config.pressure_gradient,
        config.metric,
        &mut rhs.u,
        &mut rhs.v,
    );
    if let Some(reference) = config.balanced_reference {
        reference.add_to(&mut rhs.u, &mut rhs.v);
    }
    apply_coriolis_3d(rhs, state, config.mesh, config.ops, config.coriolis);
    // The horizontal viscosity needs scratch: see `Hydrostatic3D`
}

/// Add the inventory tendency `∂(H_z u)/∂t` of the momentum advection by the
/// layer transports `transport` (computed from `state` for this stage) to
/// `rhs.u` and `rhs.v`, and overwrite `rhs.temp` and `rhs.salt` with the
/// tracers' inventory tendencies `∂(H_z C)/∂t`. The advected fields are
/// `state`'s.
pub fn compute_transport_rhs_3d(
    rhs: &mut Solution3D,
    state: &Solution3D,
    transport: &LayerTransport,
    config: &Rhs3DConfig,
) {
    apply_momentum_transport_3d(
        &mut rhs.u,
        &mut rhs.v,
        &state.u,
        &state.v,
        transport,
        config.mesh,
        config.ops,
        config.geom,
        config.boundaries,
        config.exterior.velocity,
        config.momentum_advection_form,
        config.momentum_vertical_advection,
    );
    let tracers = [
        AdvectedTracer {
            rhs: &mut rhs.temp,
            tracer: &state.temp,
            bc: config.temp_bc,
            exterior: config.exterior.temp,
            reference: config.tracer_reference[0],
            range: config.tracer_range[0],
        },
        AdvectedTracer {
            rhs: &mut rhs.salt,
            tracer: &state.salt,
            bc: config.salt_bc,
            exterior: config.exterior.salt,
            reference: config.tracer_reference[1],
            range: config.tracer_range[1],
        },
    ];
    apply_tracer_transport_3d(
        tracers,
        transport,
        config.mesh,
        config.ops,
        config.geom,
        config.boundaries,
        config.vertical_advection,
    );
}
