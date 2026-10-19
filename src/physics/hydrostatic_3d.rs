//! Hydrostatic 3D Physics Module.
//!
//! Handles the full 3D primitive equations with hydrostatic approximation.
//! Contains the 2D barotropic physics module as a sub-component.
//!
//! # Barotropic coupling
//!
//! Under mode splitting ([`crate::time::ModeSplitIntegrator`]) the 2D module
//! owns the depth-mean flow: the barotropic pressure gradient (with the DG face
//! coupling of `η`), advection of `ū`, Coriolis on `ū`, and any bottom friction
//! on `ū`. Configure it with the same Coriolis parameter as the 3D model
//! (`with_source(CoriolisSource2D::…)`), and without wind or friction sources
//! that duplicate [`Forcing`], [`Hydrostatic3D::with_surface_stress`] or
//! [`Hydrostatic3D::with_bottom_drag`]: the wind belongs to the 3D columns
//! (for a weather model's fields, [`crate::source::GriddedAtmosphere2D::split_for_3d`]
//! keeps the pressure gradient, a depth-uniform force, in the 2D module and
//! hands the wind stress to the columns). The slow forcing it receives
//! ([`ModeSplitPhysics::slow_forcing_into`]) is
//!
//! ```text
//!     G = D·⟨R_PGF+Cor(u)⟩ + Σ_l A_l(u) − A(ū) − D·R_Cor(ū) + (τ_s − τ_b)/ρ₀
//!         − r·(u_b − ū) − D Σ_l Δσ_l λ_l (u_l − ū)
//! ```
//!
//! `⟨R_PGF+Cor(u)⟩` is the depth mean of the 3D momentum tendency that does
//! not move with the layer fluxes (baroclinic PGF, Coriolis, and the
//! horizontal viscosity of the shear, [`Hydrostatic3D::with_horizontal_viscosity`],
//! whose column integral is zero for a constant ν: the depth mean's viscosity
//! is the 2D module's; with [`Hydrostatic3D::with_smagorinsky_viscosity`] it
//! is the stress of the shear on the mean flow). `A_l(u)` is the momentum
//! advection of layer `l` in inventory form, `−∇·(Q_l u_l) − δ(Ω u)_l`
//! ([`crate::solver::rhs::apply_momentum_transport_3d`]), with the state's own
//! layer transports `Q_l = H_z u_l` (the barotropic transport of the step
//! does not exist yet when `G` is built). `A(ū)` is the same operator on one
//! layer carrying `ū` with the transport `Dū`: with `R_Cor(ū)`, the advection
//! and Coriolis of the mean flow, which the 2D module already has. What
//! remains is the depth-mean baroclinic PGF and the momentum dispersion of
//! the vertical shear, `−∇·(Σ_l Q_l u_l − Dūū)`. Without shear
//! `Q_l = Δσ_l Dū`, so `Σ_l A_l(u) = A(ū)` exactly, Coriolis cancels (it is
//! pointwise and linear), and `G` reduces to the stresses. The vertical
//! fluxes sum to zero over the column.

//! The last two terms are the vertical-shear parts of the quadratic bottom
//! drag ([`Hydrostatic3D::with_bottom_drag`], rate `r = C_d|u_b|`) and of
//! the net cages ([`Hydrostatic3D::with_cage_drag`], rate `λ_l` per layer);
//! the splitter applies their depth-mean parts `−r·ū` and `−Λ̄ū` implicitly
//! in the barotropic pass (see [`crate::physics::bottom_drag`] and
//! [`crate::physics::cage_drag`]). `τ_s` is the stress of [`Forcing`]
//! plus the column's of [`Hydrostatic3D::with_surface_stress`], if any (see
//! [`crate::physics::surface_stress`] for when it is evaluated), `τ_b` the
//! prescribed stress of [`Forcing`], if any. Heat and fresh water enter
//! the columns' vertical diffusion only: the uniform heat flux of
//! [`Forcing`] and the per-column fluxes of
//! [`Hydrostatic3D::with_surface_fluxes`] (see
//! [`crate::physics::surface_fluxes`]). Thin columns
//! ([`Hydrostatic3D::with_min_column_depth`]) get no `G`: their depth mean
//! is the 2D module's alone.
//!
//! The 3D PGF is baroclinic-only (`ρ − ρ₀`), so the barotropic pressure
//! gradient comes from the 2D module. (Since P4.3 the 3D PGF lifts pressure
//! jumps at element faces and could carry `−g∇η` too; the division of labour
//! is kept, see TODO P4.1.)

use std::sync::{Arc, Mutex, Once};

use crate::boundary::{Nesting3D, SWEBoundaryCondition2D};
use crate::mesh::Mesh2D;
use crate::mesh::data::Bathymetry2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::physics::SWEPhysics2D;
use crate::physics::air_sea::JerlovWater;
use crate::physics::bottom_drag::BottomDrag3D;
use crate::physics::cage_drag::{for_each_caged_node, layer_coefficient};
use crate::physics::eos::EquationOfState;
use crate::physics::implicit_advection::{
    ImplicitVerticalAdvection, apply_implicit_vertical_advection,
};
use crate::physics::surface_fluxes::{SeaSurface, SurfaceFluxFields, SurfaceFluxes3D};
use crate::physics::surface_stress::SurfaceStress3D;
use crate::physics::traits::PhysicsModule; // For SWEPhysics2D
use crate::physics::vertical_diffusion::{
    SurfaceFields, SurfaceTracerFluxes, apply_vertical_diffusion,
};
use crate::physics::vertical_mixing::{Forcing, VerticalMixing};
use crate::solver::SWESolution2D;
use crate::solver::core::blocks::{Pooled, for_each_block, reduce_blocks, update_with};
use crate::solver::rhs::transport_3d::layer_thickness_of;
use crate::solver::rhs::{
    AdvectedTracer, BalancedReference, BarotropicFlux, Boundaries3D, Exterior3D,
    ExtrapolationTracerBC3D, HorizontalViscosity3D, LayerTransport, MetricForm,
    MomentumAdvectionForm, PressureGradientForm, Rhs3DConfig, TracerBoundaryCondition3D,
    VerticalAdvection, ViscosityScratch3D, apply_coriolis_3d, apply_horizontal_viscosity_3d,
    apply_momentum_transport_3d, apply_tracer_transport_3d, compute_momentum_rhs_3d,
    compute_transport_rhs_3d, element_dt_viscous_swe_2d, largest_horizontal_viscosity_3d,
    w_cell_thicknesses,
};
use crate::solver::state::SWE_VAR_H;
use crate::solver::state::Solution3D;
use crate::solver::{TracerLimiter3DConfig, TracerLimiter3DStats, apply_tracer_limiters_3d};
use crate::source::{CageDrag2D, CageNode, CoriolisSource2D, RiverSources};
use crate::time::{ModeSplitPhysics, StageStash, StepDrag};
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

/// Hydrostatic 3D Physics Module.
///
/// Bundles all configuration and static data for the 3D solver.
pub struct Hydrostatic3D<EOS, MIX, BC>
where
    EOS: EquationOfState,
    MIX: VerticalMixing,
    BC: SWEBoundaryCondition2D,
{
    pub mesh: Arc<Mesh2D>,
    pub ops: Arc<DGOperators2D>,
    pub geom: Arc<GeometricFactors2D>,
    pub sigma: Arc<SigmaGrid>,
    pub bathymetry: Arc<Bathymetry2D>,
    pub coriolis: Arc<CoriolisSource2D>,
    pub eos: EOS,
    pub mixing: MIX,
    pub swe_physics: SWEPhysics2D<BC>, // 2D sub-model
    /// Stresses and heat flux, the same on every column.
    pub forcing: Forcing,
    /// A surface stress field added to `forcing.surface_stress` column by
    /// column, if any (see [`Self::with_surface_stress`]).
    pub surface_stress: Option<Arc<dyn SurfaceStress3D>>,
    /// The surface roughness `z₀ₛ` (m) of every column (`[element][node]`),
    /// e.g. from the waves' height, for the closure (see
    /// [`Self::with_surface_roughness`]).
    pub surface_roughness: Option<Vec<f64>>,
    /// Heat and freshwater fluxes of every column, if any (see
    /// [`Self::with_surface_fluxes`]).
    pub surface_fluxes: Option<Arc<dyn SurfaceFluxes3D>>,
    /// The optical water type that absorbs the shortwave of
    /// `surface_fluxes` (see [`Self::with_shortwave_absorption`]).
    pub shortwave_absorption: JerlovWater,
    pub g: f64,
    pub rho0: f64,
    /// Temperature of water flowing in through a physical boundary (walls
    /// carry none; see `TracerBoundaryCondition3D`).
    pub temp_bc: Arc<dyn TracerBoundaryCondition3D>,
    /// Salinity of water flowing in through a physical boundary (see `temp_bc`).
    pub salt_bc: Arc<dyn TracerBoundaryCondition3D>,
    /// Walls and open faces of the 3D kernels (see
    /// [`Self::with_wall_tags`]).
    pub boundaries: Boundaries3D,
    pub tracer_limiter: TracerLimiter3DConfig,
    /// Columns shallower than this (m) are thin (3D wetting and drying; see
    /// [`Self::with_min_column_depth`]).
    pub min_column_depth: f64,
    /// Reconstruction of the tracers at the σ-surfaces (see
    /// [`Self::with_vertical_advection`]).
    pub vertical_advection: VerticalAdvection,
    /// Form of the horizontal momentum advection within the elements (see
    /// [`Self::with_momentum_advection_form`]).
    pub momentum_advection_form: MomentumAdvectionForm,
    /// Reconstruction of the velocity at the σ-surfaces (see
    /// [`Self::with_momentum_vertical_advection`]).
    pub momentum_vertical_advection: VerticalAdvection,
    /// How the baroclinic pressure gradient differences the columns (see
    /// [`Self::with_pressure_gradient`]).
    pub pressure_gradient: PressureGradientForm,
    /// A reference state whose pressure gradient is the constant-depth
    /// form's, if any (see [`Self::with_balanced_reference`]).
    pub balanced_reference: Option<Arc<BalancedReference>>,
    /// Advection of a prognostic closure's turbulence, with this vertical
    /// reconstruction; `None` keeps it in its column (see
    /// [`Self::with_turbulence_advection`]).
    pub turbulence_advection: Option<VerticalAdvection>,
    /// Quadratic drag of the bottom-layer velocity, if any (see
    /// [`Self::with_bottom_drag`]).
    pub bottom_drag: Option<BottomDrag3D>,
    /// Drag of fish-farm net cages on the layers they reach, if any (see
    /// [`Self::with_cage_drag`]).
    pub cage_drag: Option<CageDrag2D>,
    /// Horizontal eddy viscosity of the vertical shear (see
    /// [`Self::with_horizontal_viscosity`] and
    /// [`Self::with_smagorinsky_viscosity`]).
    pub horizontal_viscosity: HorizontalViscosity3D,
    /// A parent model's profiles at open boundaries and in a relaxation
    /// band, if nested (see [`Self::with_nesting`]).
    pub nesting: Option<Nesting3D>,
    /// Rivers, as volume sources with their own tracers (see
    /// [`Self::with_rivers`]).
    pub rivers: Option<Arc<RiverSources>>,
    /// The adaptive implicit part of the vertical advection, if any (see
    /// [`Self::with_implicit_vertical_advection`]).
    pub implicit_vertical_advection: Option<ImplicitVerticalAdvection>,
    /// Warns once about stratified open boundaries without nesting.
    open_boundary_check: Once,
    /// Layer transports (and their Ω) of the last 3D stage.
    transport_scratch: Mutex<LayerTransport>,
    /// The implicit part of `Ω` of the last 3D stage, and its statistics.
    implicit_split: Mutex<ImplicitSplit>,
    /// The bounds of the last [`Self::compute_dt`].
    last_time_step_limits: Mutex<Option<TimeStepLimits>>,
    /// Buffers of the turbulence's advection (allocated on first use).
    w_transport_scratch: Mutex<Option<TurbulenceScratch>>,
    /// Buffers of the slow forcing (allocated on the first step).
    slow_forcing_scratch: Mutex<Option<SlowForcingScratch>>,
    /// `state` with the velocity of thin columns zeroed, for the momentum
    /// advection (allocated on the first step with a thin column).
    masked_scratch: Mutex<Option<Solution3D>>,
    /// Buffers of the horizontal viscosity (allocated on first use).
    viscosity_scratch: Mutex<Option<ViscosityScratch3D>>,
    /// Whether the stages after a step's first take the horizontal
    /// viscosity's tendency of the step's start
    /// ([`Self::with_viscosity_once_per_step`]).
    viscosity_once_per_step: bool,
    /// That tendency, `[u, v]` (velocity tendencies, `[element][node][level]`).
    step_viscosity: Mutex<[Vec<f64>; 2]>,
    /// `[τ_x, τ_y]` of `surface_stress` on every column (allocated on first
    /// use).
    surface_stress_scratch: Mutex<[Vec<f64>; 2]>,
    /// `surface_fluxes`' surface temperature and salinity, and its
    /// non-solar, shortwave and freshwater fluxes on every column
    /// (allocated on first use).
    surface_flux_scratch: Mutex<[Vec<f64>; 5]>,
    /// The stratification about which the vertical tracer advection
    /// reconstructs, if any (see [`Self::with_vertical_reference`]).
    vertical_reference: Option<VerticalReference>,
    /// Whether [`Self::ensure_vertical_reference`] takes a stratified initial
    /// state as the reference (the default; see
    /// [`Self::without_vertical_reference`]).
    automatic_vertical_reference: bool,
}

/// The implicit part of `Ω` of [`Hydrostatic3D`]'s last 3D stage (see
/// [`Hydrostatic3D::with_implicit_vertical_advection`]).
#[derive(Default)]
struct ImplicitSplit {
    /// `Ω_i` at the σ-surfaces, `[element][node][n_levels + 1]`.
    omega: Vec<f64>,
    /// Columns of the last stage with an implicit part.
    columns: usize,
    /// σ-thicknesses of the w-cells.
    d_sigma_w: Vec<f64>,
    /// Statistics since the last [`Hydrostatic3D::take_implicit_advection_stats`].
    stats: ImplicitAdvectionStats,
}

/// How much of the vertical advection was implicit (see
/// [`Hydrostatic3D::take_implicit_advection_stats`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImplicitAdvectionStats {
    /// Most columns with an implicit part in one 3D stage.
    pub columns: usize,
    /// Largest outflow Courant number `Δt Σ Ω_out / H_z` of a layer.
    pub largest_courant: f64,
}

/// One bound of [`Hydrostatic3D`]'s time step and the element that sets
/// it (see [`Hydrostatic3D::time_step_limits`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeStepLimit {
    /// The longest step the bound allows (s; infinite if nothing bounds it).
    pub dt: f64,
    /// The element with that step.
    pub element: usize,
}

impl TimeStepLimit {
    /// No bound.
    pub const NONE: Self = Self {
        dt: f64::INFINITY,
        element: usize::MAX,
    };

    /// The shorter of the two (the lower element on a tie: exact, so a
    /// reduction does not depend on its grouping).
    fn min(self, other: Self) -> Self {
        if (other.dt, other.element) < (self.dt, self.element) {
            other
        } else {
            self
        }
    }
}

/// The bounds of [`Hydrostatic3D::compute_dt`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeStepLimits {
    /// Horizontal advection and internal waves.
    pub advection: TimeStepLimit,
    /// Vertical advection, `|Ω|Δt/H_z ≤ 1` (not applied with
    /// [`Hydrostatic3D::with_implicit_vertical_advection`]).
    pub vertical: TimeStepLimit,
    /// Coriolis, `|f|Δt ≤ 1`.
    pub coriolis: TimeStepLimit,
    /// Horizontal viscosity.
    pub viscosity: TimeStepLimit,
}

/// An element's own step from horizontal advection and internal waves
/// ([`Hydrostatic3D::element_advection_steps`]) and what sets it at its
/// binding node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElementStep {
    /// The step (s).
    pub dt: f64,
    /// The largest layer speed `|u|` of the binding column (m/s).
    pub speed: f64,
    /// Its first-mode internal-wave speed `c₁` (m/s).
    pub c1: f64,
    /// The node's length scale `h` (m).
    pub length: f64,
    /// The column's depth (m).
    pub depth: f64,
}

/// [`Hydrostatic3D::element_rates`]' result.
#[derive(Clone, Copy, Debug, Default)]
struct ElementRates {
    wave: f64,
    vertical: f64,
    speed: f64,
    c1: f64,
    length: f64,
    depth: f64,
}

/// The temperature and salinity about which [`Hydrostatic3D`]'s vertical
/// tracer advection reconstructs (see [`Hydrostatic3D::with_vertical_reference`]),
/// per node and level, and how it follows the state.
struct VerticalReference {
    /// `[temperature, salinity]`, `[element][node][level]`.
    fields: Mutex<[Vec<f64>; 2]>,
    /// The time scale (s) of the relaxation towards the state, if any.
    timescale: Option<f64>,
}

impl<EOS, MIX, BC> Hydrostatic3D<EOS, MIX, BC>
where
    EOS: EquationOfState,
    MIX: VerticalMixing,
    SWEPhysics2D<BC>: PhysicsModule<crate::solver::SWESolution2D>,
    BC: Clone + Send + Sync + SWEBoundaryCondition2D,
{
    pub fn new(
        mesh: Arc<Mesh2D>,
        ops: Arc<DGOperators2D>,
        geom: Arc<GeometricFactors2D>,
        sigma: Arc<SigmaGrid>,
        bathymetry: Arc<Bathymetry2D>,
        coriolis: Arc<CoriolisSource2D>,
        eos: EOS,
        mixing: MIX,
        mut swe_physics: SWEPhysics2D<BC>,
        forcing: Forcing,
        g: f64,
        rho0: f64,
    ) -> Self {
        // The 2D wet/dry kernel's subcells where the layers take them
        swe_physics.subcell_depth = Some(Self::DEFAULT_MIN_COLUMN_DEPTH);
        let transport_scratch = Mutex::new(
            LayerTransport::new(mesh.n_elements, &ops, sigma.n_levels())
                .with_metric(MetricForm::of(swe_physics.formulation)),
        );
        // The 3D walls are the 2D boundary condition's
        let boundaries = Boundaries3D::matching(&mesh, &swe_physics.bc);
        Self {
            mesh,
            ops,
            geom,
            sigma,
            bathymetry,
            coriolis,
            eos,
            mixing,
            swe_physics,
            forcing,
            surface_stress: None,
            surface_roughness: None,
            surface_fluxes: None,
            shortwave_absorption: JerlovWater::default(),
            g,
            rho0,
            temp_bc: Arc::new(ExtrapolationTracerBC3D),
            salt_bc: Arc::new(ExtrapolationTracerBC3D),
            boundaries,
            tracer_limiter: TracerLimiter3DConfig::none(),
            min_column_depth: Self::DEFAULT_MIN_COLUMN_DEPTH,
            vertical_advection: VerticalAdvection::default(),
            momentum_advection_form: MomentumAdvectionForm::default(),
            momentum_vertical_advection: VerticalAdvection::Centred,
            pressure_gradient: PressureGradientForm::default(),
            balanced_reference: None,
            turbulence_advection: Some(VerticalAdvection::LimitedAkima),
            bottom_drag: None,
            cage_drag: None,
            horizontal_viscosity: HorizontalViscosity3D::default(),
            nesting: None,
            rivers: None,
            implicit_vertical_advection: None,
            open_boundary_check: Once::new(),
            transport_scratch,
            implicit_split: Mutex::new(ImplicitSplit::default()),
            last_time_step_limits: Mutex::new(None),
            w_transport_scratch: Mutex::new(None),
            slow_forcing_scratch: Mutex::new(None),
            masked_scratch: Mutex::new(None),
            viscosity_scratch: Mutex::new(None),
            viscosity_once_per_step: false,
            step_viscosity: Mutex::new([Vec::new(), Vec::new()]),
            surface_stress_scratch: Mutex::new([Vec::new(), Vec::new()]),
            surface_flux_scratch: Mutex::new(Default::default()),
            vertical_reference: None,
            automatic_vertical_reference: true,
        }
    }

    /// Reconstruct the tracers at the σ-surfaces about `state`'s temperature
    /// and salinity, its stratification (at rest, or a parent model's): the
    /// vertical scheme ([`Self::with_vertical_advection`]) keeps its limiting
    /// for departures from it, while the stratification itself is advected
    /// with the centred value (see
    /// [`crate::solver::rhs::apply_tracer_transport_3d`]).
    ///
    /// Use it for stratified runs: about a stratified rest state, a
    /// dissipative or higher-order vertical scheme mixes the background at a
    /// rate set by a perturbation's own vertical flow, which feeds the
    /// perturbation. On the Frøya coastline mesh at rest that grew with an
    /// e-folding of 35 min for the default `LimitedAkima` (26 min upwind,
    /// 50 min Akima) at terrace edges beside shore elements; centred, or any
    /// scheme about the reference, holds round-off (TODO P1.3,
    /// `docs/stratified-rest-over-steep-beds.md`).
    ///
    /// The reference is per node and level, so it follows the σ-levels as a
    /// tide stretches the columns. A uniform tracer stays uniform whatever
    /// the reference, and a column stratified like it at a fraction of its
    /// strength is still advected centred. Refresh it with
    /// [`Self::set_vertical_reference`], or let it follow the state with
    /// [`Self::with_vertical_reference_timescale`].
    ///
    /// It is the default: [`crate::simulation::Simulation3D`] takes a
    /// stratified initial state as the reference when none was given
    /// ([`Self::ensure_vertical_reference`]; opt out with
    /// [`Self::without_vertical_reference`]). Loops that step the model
    /// themselves call this, or [`Self::ensure_vertical_reference`].
    pub fn with_vertical_reference(mut self, state: &Solution3D) -> Self {
        self.vertical_reference = Some(VerticalReference {
            fields: Mutex::new([state.temp.clone(), state.salt.clone()]),
            timescale: None,
        });
        self
    }

    /// [`Self::with_vertical_reference`] of a horizontally uniform profile
    /// `profile(z) = (T, S)`, sampled at the levels of `state` (its `η`).
    pub fn with_vertical_reference_profile(
        self,
        state: &Solution3D,
        profile: impl Fn(f64) -> (f64, f64),
    ) -> Self {
        let mut reference = state.clone();
        let nl = state.n_levels;
        for (idx, (temp, salt)) in reference
            .temp
            .chunks_exact_mut(nl)
            .zip(reference.salt.chunks_exact_mut(nl))
            .enumerate()
        {
            let eta = state.eta.data[idx];
            let depth = eta - self.bathymetry.data[idx];
            for (l, &s) in self.sigma.sigma_rho().iter().enumerate() {
                (temp[l], salt[l]) = profile(eta + s * depth);
            }
        }
        self.with_vertical_reference(&reference)
    }

    /// Relax the reference of [`Self::with_vertical_reference`] towards the
    /// state with the time scale `timescale` (s) at every
    /// [`Self::relax_vertical_reference`] (each step of [`crate::simulation::Simulation3D`]),
    /// so that it follows a stratification that changes slowly (mixing,
    /// rivers, a parent model) while departures faster than `timescale`
    /// are still limited. A day or more keeps the tide's departures.
    ///
    /// # Panics
    /// Without a reference, or if `timescale` is not positive.
    pub fn with_vertical_reference_timescale(mut self, timescale: f64) -> Self {
        assert!(
            timescale > 0.0,
            "the reference's time scale must be positive, got {timescale}"
        );
        self.vertical_reference
            .as_mut()
            .expect("with_vertical_reference_timescale needs with_vertical_reference first")
            .timescale = Some(timescale);
        self
    }

    /// The plain vertical scheme, without a reference, and no automatic one
    /// from [`Self::ensure_vertical_reference`].
    pub fn without_vertical_reference(mut self) -> Self {
        self.vertical_reference = None;
        self.automatic_vertical_reference = false;
        self
    }

    /// Take `state`'s temperature and salinity as the reference of
    /// [`Self::with_vertical_reference`] if none is set, the state is
    /// stratified (a column's T or S varies by more than 1e-9) and
    /// [`Self::without_vertical_reference`] did not opt out.
    /// [`crate::simulation::Simulation3D`] calls it with its initial state.
    /// The first call decides: a later one (a run continued by another
    /// `run` call, whose state may have stratified since) takes no reference
    /// mid-run.
    pub fn ensure_vertical_reference(&mut self, state: &Solution3D) {
        if self.vertical_reference.is_some() || !self.automatic_vertical_reference {
            return;
        }
        self.automatic_vertical_reference = false;
        let nl = state.n_levels;
        let stratified = [&state.temp, &state.salt].iter().any(|field| {
            field.chunks_exact(nl).any(|column| {
                let (lo, hi) = column
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &c| {
                        (lo.min(c), hi.max(c))
                    });
                hi - lo > 1e-9
            })
        });
        if stratified {
            self.set_vertical_reference(state);
        }
    }

    /// Replace the reference of [`Self::with_vertical_reference`] by
    /// `state`'s temperature and salinity (none set: sets one).
    pub fn set_vertical_reference(&mut self, state: &Solution3D) {
        match &self.vertical_reference {
            Some(reference) => {
                let mut fields = reference
                    .fields
                    .lock()
                    .expect("Failed to lock the vertical reference");
                fields[0].copy_from_slice(&state.temp);
                fields[1].copy_from_slice(&state.salt);
            }
            None => {
                self.vertical_reference = Some(VerticalReference {
                    fields: Mutex::new([state.temp.clone(), state.salt.clone()]),
                    timescale: None,
                });
            }
        }
    }

    /// Continue a run's reference (from [`Self::vertical_reference`] of the
    /// run being resumed, `[temperature, salinity]` per node and level):
    /// its fields replace this physics' (keeping the relaxation time scale
    /// it was built with), and `None`, a run without one, removes any
    /// reference and the automatic one of [`Self::ensure_vertical_reference`].
    ///
    /// # Panics
    /// If the fields do not have the size of the reference's.
    pub fn restore_vertical_reference(&mut self, reference: Option<[Vec<f64>; 2]>) {
        let Some(fields) = reference else {
            self.vertical_reference = None;
            self.automatic_vertical_reference = false;
            return;
        };
        let n = self.mesh.n_elements * self.ops.n_nodes * self.sigma.n_levels();
        assert!(
            fields.iter().all(|field| field.len() == n),
            "the vertical reference has {} and {} values, expected {n}",
            fields[0].len(),
            fields[1].len()
        );
        match &mut self.vertical_reference {
            Some(reference) => {
                *reference
                    .fields
                    .get_mut()
                    .expect("Failed to lock the vertical reference") = fields;
            }
            None => {
                self.vertical_reference = Some(VerticalReference {
                    fields: Mutex::new(fields),
                    timescale: None,
                });
            }
        }
        self.automatic_vertical_reference = false;
    }

    /// Relax the reference towards `state` over a step `dt` (s):
    /// `C_ref += (1 − e^{−dt/τ})(C − C_ref)` with the time scale of
    /// [`Self::with_vertical_reference_timescale`]; nothing without one.
    pub fn relax_vertical_reference(&self, state: &Solution3D, dt: f64) {
        let Some(VerticalReference {
            fields,
            timescale: Some(timescale),
        }) = &self.vertical_reference
        else {
            return;
        };
        let weight = 1.0 - (-dt / timescale).exp();
        let mut fields = fields
            .lock()
            .expect("Failed to lock the vertical reference");
        for (reference, field) in fields.iter_mut().zip([&state.temp, &state.salt]) {
            for (r, &c) in reference.iter_mut().zip(field.iter()) {
                *r += weight * (c - *r);
            }
        }
    }

    /// The reference's temperature and salinity, if any (see
    /// [`Self::with_vertical_reference`]).
    pub fn vertical_reference(&self) -> Option<[Vec<f64>; 2]> {
        self.vertical_reference.as_ref().map(|reference| {
            reference
                .fields
                .lock()
                .expect("Failed to lock the vertical reference")
                .clone()
        })
    }

    /// Rivers as volume sources (see [`crate::source::river`]). The mode
    /// splitter adds their discharge to every barotropic stage and to the
    /// barotropic transport of the step; the 3D layers get it shared out by
    /// each river's [`crate::source::RiverProfile`], with the river water's
    /// temperature and salinity, and no momentum. Do not also give them to
    /// the 2D module.
    ///
    /// # Panics
    /// If a river's profile does not fit the σ-levels.
    pub fn with_rivers(mut self, mut rivers: RiverSources) -> Self {
        if let Err(error) = rivers.set_levels(&self.sigma) {
            panic!("Hydrostatic3D::with_rivers: {error}");
        }
        self.rivers = Some(Arc::new(rivers));
        self
    }

    /// Floor (m/s) of the internal-wave speed in [`Self::compute_dt`]: keeps
    /// the step finite in unstratified water at rest, where nothing else
    /// bounds it (a run starting from rest would otherwise take its first
    /// step unbounded). 0.1 m/s is the first-mode speed of weak
    /// stratification, `g′ ≈ 10⁻³ m/s²` over 10 m. A heuristic, not a
    /// stability limit.
    pub const MIN_INTERNAL_WAVE_SPEED: f64 = 0.1;

    /// Default of [`Self::with_min_column_depth`] (m), ROMS's usual `Dcrit`.
    pub const DEFAULT_MIN_COLUMN_DEPTH: f64 = 0.1;

    /// 3D wetting and drying: columns shallower than `depth` (m), down to dry
    /// nodes, are thin (Warner et al. 2013, ROMS's `Dcrit` masks):
    /// - they carry no vertical shear: their 3D velocity is the depth mean ū,
    ///   their momentum tendency is zero, and they get no vertical diffusion;
    /// - the surface and bottom stresses do not reach the depth mean through
    ///   them (masked in G);
    /// - they exert and feel no baroclinic pressure difference, and exchange
    ///   no baroclinic transport (only their share of the barotropic flux);
    /// - an element with a thin column moves its layers, tracers and momentum
    ///   through the GLL subcell interfaces of the 2D `WetDry` kernel, which
    ///   this sets to take its subcells in the same elements
    ///   (`SWEPhysics2D::subcell_depth`): the layers then carry exactly the
    ///   water the 2D pass moved node by node, a dry bank blocked by its
    ///   hydrostatic reconstruction (see
    ///   [`crate::solver::rhs::LayerTransport::compute`]).
    ///
    /// Where the pass still kept an element's balance only as a whole (it
    /// switched to the subcells during the step, or the positivity limiter
    /// changed it), the mode splitter carries its tracers as element means
    /// per level ([`crate::solver::rhs::from_inventory`]).
    ///
    /// Everything else stays 3D, including the wet nodes of shoreline
    /// elements.
    pub fn with_min_column_depth(mut self, depth: f64) -> Self {
        assert!(
            depth > 0.0,
            "minimum column depth must be positive, got {depth}"
        );
        self.min_column_depth = depth;
        self.swe_physics.subcell_depth = Some(depth);
        self
    }

    /// The vertical advection of the tracers: fourth-order Akima under a TVD
    /// limiter by default ([`VerticalAdvection::LimitedAkima`]; see
    /// [`VerticalAdvection`]).
    pub fn with_vertical_advection(mut self, scheme: VerticalAdvection) -> Self {
        self.vertical_advection = scheme;
        self
    }

    /// The form of the horizontal momentum advection within the elements, in
    /// the 3D stages and in the slow forcing `G`: split (kinetic-energy
    /// preserving) by default, which holds a sharp interface's grid-scale
    /// shear instability without horizontal viscosity (see
    /// [`MomentumAdvectionForm`]).
    pub fn with_momentum_advection_form(mut self, form: MomentumAdvectionForm) -> Self {
        self.momentum_advection_form = form;
        self
    }

    /// The vertical advection of the velocity, in the 3D stages and in the
    /// slow forcing `G`: second-order centred by default (see
    /// [`VerticalAdvection`]). Centred is the one that keeps the kinetic
    /// energy exactly with the split form
    /// ([`Self::with_momentum_advection_form`]).
    pub fn with_momentum_vertical_advection(mut self, scheme: VerticalAdvection) -> Self {
        self.momentum_vertical_advection = scheme;
        self
    }

    /// How the baroclinic pressure gradient differences two columns:
    /// σ-pairs by default, the form consistent in energy with the advection
    /// (a stratified fluid at rest over a slope stays at rest; the
    /// constant-depth form let round-off grow tenfold every 6 h over a
    /// seamount). See [`crate::solver::rhs::baroclinic`].
    pub fn with_pressure_gradient(mut self, form: PressureGradientForm) -> Self {
        self.pressure_gradient = form;
        self
    }

    /// Balance `state`'s stratification (its `rho` must be current, see
    /// [`Self::update_density`]) as the constant-depth form does, while the
    /// dynamics keep the σ-pairs form ([`BalancedReference`]): a fluid
    /// resting in that state stays at rest to the constant-depth form's
    /// interpolation error, not the σ form's. Use the initial (or a parent
    /// model's) stratification; [`Self::set_balanced_reference`] refreshes
    /// it. No effect with [`PressureGradientForm::ConstantDepth`].
    pub fn with_balanced_reference(mut self, state: &Solution3D) -> Self {
        self.set_balanced_reference(state);
        self
    }

    /// Balance a horizontally uniform reference density `profile(z)`
    /// (kg/m³) exactly: a fluid resting in it feels no pressure gradient
    /// force at all (Mellor et al. 1998's subtraction of a reference
    /// profile; [`BalancedReference::from_profile`]), at the levels of
    /// `state`'s `η`. Prefer it to [`Self::with_balanced_reference`] where
    /// the stratification starts horizontally uniform, or has a typical
    /// profile: over cliff coasts the constant-depth form's own error at rest
    /// is not small. No effect with [`PressureGradientForm::ConstantDepth`].
    pub fn with_reference_profile(
        mut self,
        state: &Solution3D,
        profile: impl Fn(f64) -> f64,
    ) -> Self {
        self.balanced_reference = Some(Arc::new(BalancedReference::from_profile(
            state,
            &self.mesh,
            &self.bathymetry,
            &self.sigma,
            &self.ops,
            &self.geom,
            self.g,
            self.rho0,
            self.rho0,
            self.min_column_depth,
            self.metric_form(),
            profile,
        )));
        self
    }

    /// Replace the balanced reference state (see
    /// [`Self::with_balanced_reference`]).
    pub fn set_balanced_reference(&mut self, state: &Solution3D) {
        self.balanced_reference = Some(Arc::new(BalancedReference::new(
            state,
            &self.mesh,
            &self.bathymetry,
            &self.sigma,
            &self.ops,
            &self.geom,
            self.g,
            self.rho0,
            self.rho0,
            self.min_column_depth,
            self.metric_form(),
        )));
    }

    /// The volume form of the 3D horizontal divergences: the 2D module's
    /// mass equation's, so that the layers carry exactly the barotropic
    /// pass's water on general quadrilaterals too ([`MetricForm`]).
    pub fn metric_form(&self) -> MetricForm {
        MetricForm::of(self.swe_physics.formulation)
    }

    /// Advection of the turbulence `k`, `ψ` of a prognostic closure
    /// ([`crate::physics::GlsMixing`]) by the 3D flow, as ROMS does
    /// (`gls_corstep`), with the vertical reconstruction `scheme`; by default
    /// the tracers' fourth-order Akima under a TVD limiter
    /// ([`VerticalAdvection::LimitedAkima`]). `None` keeps the turbulence in
    /// its column, as a one-dimensional model (GOTM) does.
    ///
    /// The turbulence is carried through the 3D stages as inventories over
    /// the control volumes of the w-points, with the layer transports
    /// averaged to them ([`LayerTransport::stagger_from`]): a uniform `k`
    /// stays uniform under any flow, and its inventory changes only through
    /// open faces. Upwind horizontally; at open faces the inflow carries the
    /// interior's value (zero gradient), and river water the column's own.
    /// The bed and surface w-points, which hold the closure's boundary
    /// values, are advected at their neighbours' values. The closure then
    /// steps production, dissipation and vertical diffusion in each column
    /// (operator splitting, as in ROMS), taking a point that the advection
    /// left without positive `k` or `ψ` as having no turbulence.
    pub fn with_turbulence_advection(mut self, scheme: Option<VerticalAdvection>) -> Self {
        self.turbulence_advection = scheme;
        self
    }

    /// Advect the part of the vertical flux `Ω` beyond an explicit Courant
    /// number implicitly (first-order upwind, backward Euler, after every
    /// 3D stage), and drop the vertical bound `|Ω|Δt/H_z ≤ 1` from
    /// [`Self::compute_dt`]: see [`crate::physics::implicit_advection`].
    ///
    /// Columns below `split.courant_min` are advected as without it, bit for
    /// bit. It lifts the step where thin layers carry fast vertical flow: the
    /// columns just deeper than the thin depth on a flooding shore
    /// ([`Self::with_min_column_depth`]), which held the Frøya tide in 3D at
    /// 0.9–2.2 s (TODO P1.3).
    pub fn with_implicit_vertical_advection(mut self, split: ImplicitVerticalAdvection) -> Self {
        self.implicit_vertical_advection = Some(split);
        self
    }

    /// The most columns with an implicit part in one 3D stage, and the
    /// largest outflow Courant number of a layer, since the last call (see
    /// [`Self::with_implicit_vertical_advection`]).
    pub fn take_implicit_advection_stats(&self) -> ImplicitAdvectionStats {
        std::mem::take(
            &mut self
                .implicit_split
                .lock()
                .expect("Failed to lock implicit_split")
                .stats,
        )
    }

    /// Split `transport.omega` of `state` into the explicit part, left in
    /// `transport`, and the implicit part for a step `dt`, kept for
    /// [`ModeSplitPhysics::implicit_vertical_advection`]. Thin columns keep
    /// theirs explicit.
    fn split_vertical_velocity(
        &self,
        split: &ImplicitVerticalAdvection,
        state: &Solution3D,
        transport: &mut LayerTransport,
        dt: f64,
    ) {
        let (ne, nn, nl) = (state.n_elements, state.n_nodes, state.n_levels);
        let mut guard = self
            .implicit_split
            .lock()
            .expect("Failed to lock implicit_split");
        let ImplicitSplit {
            omega: implicit,
            columns,
            d_sigma_w,
            stats,
        } = &mut *guard;
        implicit.resize(transport.omega.len(), 0.0);
        let d_sigma = self.sigma.d_sigma();
        if d_sigma_w.len() != nl + 1 {
            d_sigma_w.resize(nl + 1, 0.0);
            w_cell_thicknesses(d_sigma, d_sigma_w);
        }
        let (count, courant) = reduce_blocks(
            ne,
            [&mut transport.omega[..], &mut implicit[..]],
            || Pooled::take(|t: &Vec<f64>| t.len() == nl, || vec![0.0; nl]),
            |thickness, k, [omega_k, implicit_k]| {
                let (mut count, mut largest) = (0, 0.0_f64);
                for i in 0..nn {
                    let idx = k * nn + i;
                    let column = i * (nl + 1)..(i + 1) * (nl + 1);
                    let depth = state.eta.data[idx] - self.bathymetry.data[idx];
                    let implicit = &mut implicit_k[column.clone()];
                    if depth < self.min_column_depth {
                        implicit.fill(0.0);
                        continue;
                    }
                    for (h, &ds) in thickness.iter_mut().zip(d_sigma) {
                        *h = layer_thickness_of(depth, ds);
                    }
                    let omega = &mut omega_k[column];
                    largest = largest.max(split.split_column(omega, thickness, dt, implicit));
                    if implicit.iter().any(|&w| w != 0.0) {
                        count += 1;
                        for (w, &w_i) in omega.iter_mut().zip(implicit.iter()) {
                            *w -= w_i;
                        }
                    }
                }
                (count, largest)
            },
            || (0, 0.0),
            |a, b| (a.0 + b.0, a.1.max(b.1)),
        );
        *columns = count;
        stats.columns = stats.columns.max(count);
        stats.largest_courant = stats.largest_courant.max(courant);
    }

    /// Quadratic bottom drag `τ_b/ρ₀ = C_d|u_b|u_b` of the bottom-layer
    /// velocity, with a constant or log-layer `C_d` (see
    /// [`crate::physics::bottom_drag`] for the time discretisation). It adds
    /// to any prescribed `Forcing::bottom_stress`.
    ///
    /// Thin columns ([`Self::with_min_column_depth`]) have no shear, so their
    /// drag is `C_d|ū|ū` (with the `C_d` of their thin bottom layer, which a
    /// log layer puts at its upper bound). The 2D module should then carry no
    /// bottom friction of its own: it would count the drag twice.
    pub fn with_bottom_drag(mut self, drag: BottomDrag3D) -> Self {
        self.bottom_drag = Some(drag);
        self
    }

    /// The drag of fish-farm net cages, `−½ C_d a |u_l| u_l` on every layer
    /// a net reaches, from the surface down to its net depth, weighted by
    /// each node's share of the footprint (`CageDrag2D::new` on this model's
    /// mesh and operators; see [`crate::physics::cage_drag`] for the time
    /// discretisation). Without vertical shear it is the 2D cage drag. The
    /// 2D module must not carry the cages as well: it would count the drag
    /// twice.
    ///
    /// # Panics
    /// If `cages` was built for another mesh or order.
    pub fn with_cage_drag(mut self, cages: CageDrag2D) -> Self {
        assert_eq!(
            cages.n_total_nodes(),
            self.mesh.n_elements * self.ops.n_nodes,
            "CageDrag2D was built for a different mesh or order"
        );
        self.cage_drag = (!cages.is_empty()).then_some(cages);
        self
    }

    /// A surface stress that varies from column to column, such as a
    /// weather model's wind ([`crate::source::GriddedAtmosphere2D::split_for_3d`]),
    /// added to the uniform `Forcing::surface_stress`. It reaches the depth
    /// mean through `G`, and the shear, and with it the turbulence closure's
    /// shear production, through the vertical diffusion's surface flux (see
    /// [`crate::physics::surface_stress`]). The 2D module must not carry
    /// the same wind.
    pub fn with_surface_stress(mut self, stress: impl SurfaceStress3D + 'static) -> Self {
        self.surface_stress = Some(Arc::new(stress));
        self
    }

    /// Heat and freshwater fluxes that vary from column to column and with
    /// each column's surface temperature, such as the bulk formulae on a
    /// weather model's fields ([`crate::source::GriddedSurfaceFluxes`]),
    /// added to the uniform `Forcing::surface_heat_flux`: the non-solar
    /// heat at the surface, the shortwave absorbed with depth (see
    /// [`Self::with_shortwave_absorption`]) and E − P as a virtual salt flux,
    /// evaluated at the middle of each step from the surface layer at its
    /// start (see [`crate::physics::surface_fluxes`]).
    pub fn with_surface_fluxes(mut self, fluxes: impl SurfaceFluxes3D + 'static) -> Self {
        self.surface_fluxes = Some(Arc::new(fluxes));
        self
    }

    /// The optical water type that absorbs the shortwave of
    /// [`Self::with_surface_fluxes`] (default Jerlov III, turbid coastal
    /// water, as NorKyst v3).
    pub fn with_shortwave_absorption(mut self, water: JerlovWater) -> Self {
        self.shortwave_absorption = water;
        self
    }

    /// A surface roughness `z₀ₛ` (m) per column (`[element][node]`), for a
    /// closure with one (GLS: the larger of it and its own, then Charnock's).
    /// From waves, Terray et al.'s (1996) `z₀ₛ ≈ 0.6 H_s` (ROMS `ZOS_HSIG`):
    /// [`crate::waves::WaveModel2D::surface_roughness`].
    pub fn with_surface_roughness(mut self, z0: Vec<f64>) -> Self {
        self.set_surface_roughness(Some(z0));
        self
    }

    /// Replace the per-column surface roughness (none: the closure's own), as
    /// a coupling to a wave model does every coupling interval.
    pub fn set_surface_roughness(&mut self, z0: Option<Vec<f64>>) {
        if let Some(z0) = &z0 {
            assert_eq!(
                z0.len(),
                self.bathymetry.data.len(),
                "one roughness per column"
            );
            assert!(
                z0.iter().all(|&z| z >= 0.0),
                "roughness must be non-negative"
            );
        }
        self.surface_roughness = z0;
    }

    /// Call `f` with the field of [`Self::with_surface_stress`] at time `t`
    /// on every column, `[τ_x, τ_y]`, or with `None` if there is none.
    fn surface_stress_at<R>(&self, t: f64, f: impl FnOnce(Option<[&[f64]; 2]>) -> R) -> R {
        let Some(stress) = &self.surface_stress else {
            return f(None);
        };
        let mut guard = self
            .surface_stress_scratch
            .lock()
            .expect("Failed to lock surface_stress_scratch");
        let [tau_x, tau_y] = &mut *guard;
        let n = self.mesh.n_elements * self.ops.n_nodes;
        tau_x.resize(n, 0.0);
        tau_y.resize(n, 0.0);
        stress.surface_stress_into(t, tau_x, tau_y);
        f(Some([tau_x, tau_y]))
    }

    /// The fluxes of [`Self::with_surface_fluxes`] at time `t` over
    /// `state`'s surface layer, as `[SST, SSS, non-solar, shortwave, E − P]`
    /// on every column, or `None` if there are none.
    fn surface_fluxes_at(
        &self,
        state: &Solution3D,
        t: f64,
    ) -> Option<std::sync::MutexGuard<'_, [Vec<f64>; 5]>> {
        let fluxes = self.surface_fluxes.as_ref()?;
        let mut guard = self
            .surface_flux_scratch
            .lock()
            .expect("Failed to lock surface_flux_scratch");
        let [sst, sss, nonsolar, shortwave, freshwater] = &mut *guard;
        let (n, nl) = (state.eta.data.len(), state.n_levels);
        for buffer in [
            &mut *sst,
            &mut *sss,
            &mut *nonsolar,
            &mut *shortwave,
            &mut *freshwater,
        ] {
            buffer.resize(n, 0.0);
        }
        for (idx, (t, s)) in sst.iter_mut().zip(sss.iter_mut()).enumerate() {
            *t = state.temp[(idx + 1) * nl - 1];
            *s = state.salt[(idx + 1) * nl - 1];
        }
        fluxes.surface_fluxes_into(
            t,
            SeaSurface {
                temperature: sst,
                salinity: sss,
            },
            SurfaceFluxFields {
                nonsolar,
                shortwave,
                freshwater,
            },
        );
        Some(guard)
    }

    /// Horizontal eddy viscosity `nu` (m²/s, constant) of the 3D momentum,
    /// along σ-surfaces, on the vertical shear `u − ū` only (BR1; see
    /// [`crate::solver::rhs::viscosity_3d`]). The depth mean is the 2D
    /// module's: give it its own `HorizontalViscosity2D` for a viscous mean
    /// flow. With [`Self::with_smagorinsky_viscosity`] it is the background
    /// added to Smagorinsky's.
    ///
    /// Higher orders need it in sheared, stratified flow: without it the
    /// shear instability of an interface grows at the grid scale (a P2 lock
    /// exchange blows up).
    pub fn with_horizontal_viscosity(mut self, nu: f64) -> Self {
        assert!(
            nu >= 0.0 && nu.is_finite(),
            "horizontal viscosity must be finite and non-negative, got {nu}"
        );
        self.horizontal_viscosity.background = nu;
        self
    }

    /// Smagorinsky's (1963) horizontal viscosity `(C_s Δ)²|S|` of the 3D
    /// shear, with coefficient `cs`, the strain rate `|S|` of each layer's
    /// velocity and the node spacing `Δ` (see
    /// [`crate::solver::rhs::viscosity_3d`]); added to the constant
    /// background of [`Self::with_horizontal_viscosity`], if any. Its column
    /// integral, the stress of the shear on the mean flow, reaches the depth
    /// mean through `G`.
    pub fn with_smagorinsky_viscosity(mut self, cs: f64) -> Self {
        assert!(
            cs >= 0.0 && cs.is_finite(),
            "Smagorinsky coefficient must be finite and non-negative, got {cs}"
        );
        self.horizontal_viscosity.smagorinsky = cs;
        self
    }

    /// Evaluate the horizontal viscosity once per mode-split step, at its
    /// start, and give the later stages that tendency, as ROMS applies its
    /// horizontal mixing of momentum once per baroclinic step. Its own error
    /// is first order in time, `O(Δt ∂ₜ(∇·ν∇u))`, against the third order of
    /// the stages; the viscosity's cost falls from one evaluation per stage
    /// to one per step. Off by default.
    pub fn with_viscosity_once_per_step(mut self) -> Self {
        self.viscosity_once_per_step = true;
        self
    }

    /// Nest the 3D fields in a parent model: its velocity and tracer profiles
    /// at the open faces of the nesting's tags, and relaxation within its
    /// band (see [`Nesting3D`]). The depth mean stays the
    /// 2D module's: nest it there too (its open-boundary condition and
    /// [`crate::boundary::NestingRelaxation2D`]).
    ///
    /// Stratified runs with open boundaries need it, with a relaxation band:
    /// without, the extrapolated open faces let the boundary columns'
    /// stratification run away (see [`crate::boundary::Nesting3D`]'s module
    /// docs; [`crate::boundary::ReferenceColumns`] relaxes to a fixed state
    /// where there is no parent model).
    ///
    /// # Panics
    /// If a nested tag is a wall for the 3D kernels.
    pub fn with_nesting(mut self, nesting: Nesting3D) -> Self {
        for tag in nesting.tags() {
            assert!(
                !self.boundaries.wall_tags.contains(tag),
                "nested boundary tag {tag:?} is a wall for the 3D kernels"
            );
        }
        self.nesting = Some(nesting);
        self
    }

    /// Drag rate `r = C_d(z_b)·|u_b|` (m/s) of the column at node `idx`
    /// (`[element][node]`), with `z_b` the height of the bottom-layer centre
    /// above the bed; zero where the column is dry.
    #[inline]
    fn drag_rate(&self, drag: &BottomDrag3D, state: &Solution3D, idx: usize) -> f64 {
        let depth = state.eta.data[idx] - self.bathymetry.data[idx];
        if depth <= 0.0 {
            return 0.0;
        }
        let bottom = idx * state.n_levels;
        let (u, v) = if depth < self.min_column_depth {
            (state.ubar.data[idx], state.vbar.data[idx])
        } else {
            (state.u[bottom], state.v[bottom])
        };
        let z_b = (1.0 + self.sigma.sigma_rho()[0]) * depth;
        drag.rate(z_b, (u * u + v * v).sqrt())
    }

    /// Cage-drag rate `λ_l = c_l·|u_l|` (1/s) of every layer of the column at
    /// node `idx` (`[element][node]`) with cage entries `entries`, into
    /// `rate`: with the depth mean's speed in thin columns, zero where dry.
    fn cage_rates_into(
        &self,
        entries: &[CageNode],
        state: &Solution3D,
        idx: usize,
        rate: &mut [f64],
    ) {
        let depth = state.eta.data[idx] - self.bathymetry.data[idx];
        if depth <= 0.0 {
            rate.fill(0.0);
            return;
        }
        let nl = state.n_levels;
        let thin = depth < self.min_column_depth;
        for (l, r) in rate.iter_mut().enumerate() {
            let (u, v) = if thin {
                (state.ubar.data[idx], state.vbar.data[idx])
            } else {
                (state.u[idx * nl + l], state.v[idx * nl + l])
            };
            *r = layer_coefficient(entries, &self.sigma, l, depth) * u.hypot(v);
        }
    }

    /// Whether the column at node `idx` (`[element][node]`) is thin.
    #[inline]
    fn is_thin(&self, state: &Solution3D, idx: usize) -> bool {
        state.eta.data[idx] - self.bathymetry.data[idx] < self.min_column_depth
    }

    /// Call `f` with `state`, or with a copy of it whose thin columns have no
    /// velocity: the momentum advection sees them at rest (their momentum is
    /// the 2D module's; a film at the 2D velocity cap would otherwise carry
    /// its speed into the 3D shear), consistently in the stages and in `G`.
    /// The copy holds η, ū, v̄, u, v, and T, S, k and ψ if `tracers`: `f`
    /// must not read the others. No copy where the thin columns
    /// are at rest already (the mode splitter's stage values, see
    /// [`ModeSplitPhysics::mask_stage`]).
    fn with_thin_columns_at_rest<R>(
        &self,
        state: &Solution3D,
        tracers: bool,
        f: impl FnOnce(&Solution3D) -> R,
    ) -> R {
        let nl = state.n_levels;
        let n_columns = state.eta.data.len();
        let moving = |idx: usize| {
            self.is_thin(state, idx)
                && (state.ubar.data[idx] != 0.0
                    || state.vbar.data[idx] != 0.0
                    || state.u[idx * nl..(idx + 1) * nl].iter().any(|&u| u != 0.0)
                    || state.v[idx * nl..(idx + 1) * nl].iter().any(|&v| v != 0.0))
        };
        if !(0..n_columns).any(moving) {
            return f(state);
        }
        let mut guard = self
            .masked_scratch
            .lock()
            .expect("Failed to lock masked_scratch");
        let copy = crate::profiling::start();
        let masked = guard.get_or_insert_with(|| {
            Solution3D::new(state.n_elements, state.n_nodes, state.n_levels)
        });
        // Only the fields the callers read under the mask (the layer
        // transports η, u, v; the advection T, S; the turbulence k, ψ):
        // w, ρ and the eddy coefficients stay stale, and the tracers unless
        // asked for
        masked.eta.copy_from(&state.eta);
        masked.ubar.copy_from(&state.ubar);
        masked.vbar.copy_from(&state.vbar);
        let fields = [
            (&mut masked.u, &state.u),
            (&mut masked.v, &state.v),
            (&mut masked.temp, &state.temp),
            (&mut masked.salt, &state.salt),
            (&mut masked.tke, &state.tke),
            (&mut masked.gls, &state.gls),
        ];
        for (x, y) in fields.into_iter().take(if tracers { 6 } else { 2 }) {
            x.resize(y.len(), 0.0);
            update_with(x, y, |x, y| *x = y);
        }
        for idx in 0..n_columns {
            if self.is_thin(state, idx) {
                masked.u[idx * nl..(idx + 1) * nl].fill(0.0);
                masked.v[idx * nl..(idx + 1) * nl].fill(0.0);
                masked.ubar.data[idx] = 0.0;
                masked.vbar.data[idx] = 0.0;
            }
        }
        crate::profiling::stop("3d: thin-column copy", copy);
        f(masked)
    }

    /// Zero the momentum tendency of thin columns: they carry no shear, and
    /// the splitter adds the depth-mean rate.
    fn zero_thin_momentum(&self, state: &Solution3D, rhs: &mut Solution3D) {
        let nl = state.n_levels;
        for idx in 0..state.eta.data.len() {
            if self.is_thin(state, idx) {
                rhs.u[idx * nl..(idx + 1) * nl].fill(0.0);
                rhs.v[idx * nl..(idx + 1) * nl].fill(0.0);
            }
        }
    }

    /// Override the boundary tags that are walls for the 3D kernels (untagged
    /// faces are then walls too). Every other boundary is open: its layers
    /// carry the 2D open-boundary flux in the interior's vertical profile,
    /// and the 3D velocity is extrapolated there (see
    /// [`crate::solver::rhs::boundary_3d`]).
    ///
    /// [`Self::new`] already derives the walls from the 2D module's boundary
    /// condition ([`Boundaries3D::matching`]); override only for a condition
    /// that cannot tell (`SWEBoundaryCondition2D::is_wall` is `None`). A tag
    /// the 2D condition treats as a wall must be a wall here.
    pub fn with_wall_tags(
        mut self,
        tags: impl IntoIterator<Item = crate::mesh::BoundaryTag>,
    ) -> Self {
        self.boundaries = Boundaries3D::with_walls(tags);
        self
    }

    /// Override scalar tracer boundary conditions used by the high-level RHS path.
    pub fn with_tracer_boundary_conditions(
        mut self,
        temp_bc: Arc<dyn TracerBoundaryCondition3D>,
        salt_bc: Arc<dyn TracerBoundaryCondition3D>,
    ) -> Self {
        self.temp_bc = temp_bc;
        self.salt_bc = salt_bc;
        self
    }

    /// Set scalar tracer boundary conditions used by the high-level RHS path.
    pub fn set_tracer_boundary_conditions(
        &mut self,
        temp_bc: Arc<dyn TracerBoundaryCondition3D>,
        salt_bc: Arc<dyn TracerBoundaryCondition3D>,
    ) {
        self.temp_bc = temp_bc;
        self.salt_bc = salt_bc;
    }

    /// Override 3D tracer limiter configuration.
    pub fn with_tracer_limiter(mut self, tracer_limiter: TracerLimiter3DConfig) -> Self {
        self.tracer_limiter = tracer_limiter;
        self
    }

    /// Set 3D tracer limiter configuration.
    pub fn set_tracer_limiter(&mut self, tracer_limiter: TracerLimiter3DConfig) {
        self.tracer_limiter = tracer_limiter;
    }

    /// Apply configured 3D tracer limiters and refresh density if tracers changed.
    pub fn apply_tracer_limiters(&self, state: &mut Solution3D) -> TracerLimiter3DStats {
        let limiter = crate::profiling::start();
        let stats = apply_tracer_limiters_3d(
            state,
            &self.mesh,
            &self.ops,
            &self.geom,
            &self.bathymetry,
            &self.sigma,
            &self.tracer_limiter,
        );
        crate::profiling::stop("3d: tracer limiter", limiter);

        if stats.changed() {
            crate::profiling::scope("3d: density", || self.eos.update_density(state));
        }

        stats
    }

    fn rhs_config(&self) -> Rhs3DConfig<'_> {
        Rhs3DConfig {
            mesh: &self.mesh,
            ops: &self.ops,
            geom: &self.geom,
            bathymetry: &self.bathymetry,
            sigma: &self.sigma,
            coriolis: &self.coriolis,
            temp_bc: &*self.temp_bc,
            salt_bc: &*self.salt_bc,
            boundaries: &self.boundaries,
            exterior: Exterior3D::default(),
            g: self.g,
            rho0: self.rho0,
            min_column_depth: self.min_column_depth,
            vertical_advection: self.vertical_advection,
            momentum_advection_form: self.momentum_advection_form,
            momentum_vertical_advection: self.momentum_vertical_advection,
            pressure_gradient: self.pressure_gradient,
            balanced_reference: match self.pressure_gradient {
                PressureGradientForm::SigmaPairs => self.balanced_reference.as_deref(),
                PressureGradientForm::ConstantDepth => None,
            },
            metric: self.metric_form(),
            // The vertical reference is locked where the transport needs it
            tracer_reference: [None, None],
        }
    }

    /// Overwrite `rhs.u` and `rhs.v` with the velocity tendency of the
    /// momentum terms of `state` at time `t` that do not move with the layer
    /// fluxes (baroclinic PGF, Coriolis; see [`compute_momentum_rhs_3d`]; the
    /// horizontal viscosity of the shear; and the nesting's relaxation of
    /// the shear, which has no depth mean), zero in thin columns.
    /// `state.rho` must be current.
    pub fn compute_momentum_rhs_into(&self, state: &Solution3D, t: f64, rhs: &mut Solution3D) {
        self.momentum_rhs(state, t, rhs, false);
    }

    /// [`Self::compute_momentum_rhs_into`]; at a later stage of a step
    /// (`stage`) with [`Self::with_viscosity_once_per_step`], the viscosity's
    /// tendency of the step's start instead of the state's.
    fn momentum_rhs(&self, state: &Solution3D, t: f64, rhs: &mut Solution3D, stage: bool) {
        crate::profiling::scope("3d: pgf, coriolis", || {
            compute_momentum_rhs_3d(rhs, state, &self.rhs_config())
        });
        let viscosity = crate::profiling::start();
        if !self.horizontal_viscosity.is_zero() && self.viscosity_once_per_step {
            let mut kept = self
                .step_viscosity
                .lock()
                .expect("Failed to lock step_viscosity");
            if !stage {
                let mut guard = self
                    .viscosity_scratch
                    .lock()
                    .expect("Failed to lock viscosity_scratch");
                let scratch = guard.get_or_insert_with(|| {
                    ViscosityScratch3D::new(self.mesh.n_elements, &self.ops)
                });
                let [u, v] = &mut *kept;
                for field in [&mut *u, &mut *v] {
                    field.clear();
                    field.resize(rhs.u.len(), 0.0);
                }
                apply_horizontal_viscosity_3d(
                    u,
                    v,
                    state,
                    self.horizontal_viscosity,
                    &self.mesh,
                    &self.ops,
                    &self.geom,
                    &self.bathymetry,
                    &self.sigma,
                    &self.boundaries,
                    self.min_column_depth,
                    scratch,
                );
            }
            let [u, v] = &*kept;
            update_with(&mut rhs.u, u, |r, x| *r += x);
            update_with(&mut rhs.v, v, |r, x| *r += x);
        } else if !self.horizontal_viscosity.is_zero() {
            let mut guard = self
                .viscosity_scratch
                .lock()
                .expect("Failed to lock viscosity_scratch");
            let scratch = guard
                .get_or_insert_with(|| ViscosityScratch3D::new(self.mesh.n_elements, &self.ops));
            apply_horizontal_viscosity_3d(
                &mut rhs.u,
                &mut rhs.v,
                state,
                self.horizontal_viscosity,
                &self.mesh,
                &self.ops,
                &self.geom,
                &self.bathymetry,
                &self.sigma,
                &self.boundaries,
                self.min_column_depth,
                scratch,
            );
        }
        crate::profiling::stop("3d: viscosity", viscosity);
        if let Some(nesting) = &self.nesting {
            nesting
                .columns(
                    state,
                    &self.bathymetry,
                    &self.sigma,
                    t,
                    self.min_column_depth,
                )
                .relax_shear(state, &self.sigma, rhs, |idx| self.is_thin(state, idx));
        }
        self.zero_thin_momentum(state, rhs);
    }

    /// The layer transports of `state` corrected to `barotropic`, then the
    /// inventory tendency of the momentum advection added to `rhs.u`, `rhs.v`
    /// (zero in thin columns, which the advection sees at rest) and the tracer
    /// inventory tendencies written to `rhs.temp`, `rhs.salt` (see
    /// [`compute_transport_rhs_3d`]), with the nesting parent's values at
    /// time `t` outside its open faces and its relaxation of the tracers.
    pub fn compute_transport_rhs_into(
        &self,
        state: &Solution3D,
        t: f64,
        barotropic: BarotropicFlux,
        rhs: &mut Solution3D,
    ) {
        let mut guard = self
            .transport_scratch
            .lock()
            .expect("Failed to lock transport_scratch");
        let transport = &mut *guard;
        self.open_boundary_check
            .call_once(|| self.warn_if_stratified_open_boundaries(state));
        let columns = self.nesting.as_ref().map(|nesting| {
            nesting.columns(
                state,
                &self.bathymetry,
                &self.sigma,
                t,
                self.min_column_depth,
            )
        });
        let exterior = columns
            .as_ref()
            .map_or_else(Exterior3D::default, |c| c.exterior());
        // Thin columns at rest change no layer transport: theirs are uniform
        // in σ, which the correction to the barotropic transport replaces
        self.with_thin_columns_at_rest(state, true, |state| {
            let layer = crate::profiling::start();
            transport.metric = self.metric_form();
            transport.compute(
                state,
                Some(barotropic),
                &self.mesh,
                &self.ops,
                &self.geom,
                &self.sigma,
                &self.bathymetry,
                &self.boundaries,
                &exterior,
                self.min_column_depth,
            );
            // The stages advect with the explicit part of Ω; the rest is
            // the implicit relaxation's
            if let Some(split) = &self.implicit_vertical_advection {
                self.split_vertical_velocity(split, state, transport, barotropic.euler_dt);
            }
            crate::profiling::stop("3d: layer transports", layer);
            let reference = self.vertical_reference.as_ref().map(|reference| {
                reference
                    .fields
                    .lock()
                    .expect("Failed to lock the vertical reference")
            });
            let config = Rhs3DConfig {
                exterior,
                tracer_reference: match &reference {
                    Some(fields) => [Some(fields[0].as_slice()), Some(fields[1].as_slice())],
                    None => [None, None],
                },
                ..self.rhs_config()
            };
            crate::profiling::scope("3d: advection kernels", || {
                compute_transport_rhs_3d(rhs, state, transport, &config)
            });
            // The water no transport carries (the 2D module's own mass
            // sources) at the concentrations and velocities it joins
            let means = barotropic.element_means;
            for (out, field) in [
                (&mut rhs.temp, &state.temp),
                (&mut rhs.salt, &state.salt),
                (&mut rhs.u, &state.u),
                (&mut rhs.v, &state.v),
            ] {
                transport.add_volume_sources(
                    out,
                    field,
                    self.sigma.d_sigma(),
                    &state.eta.data,
                    &self.bathymetry,
                    &self.geom,
                    means,
                );
            }
            crate::profiling::scope("3d: turbulence advection", || {
                self.turbulence_transport_rhs(state, transport, barotropic, rhs)
            });
        });
        // The river water's tracers, with the volume the layer transports
        // gave the layers
        if let Some(rivers) = barotropic.rivers {
            let nn = state.n_nodes;
            rivers.add_tracer_sources(&mut rhs.temp, nn, |r| r.temperature.at(t));
            rivers.add_tracer_sources(&mut rhs.salt, nn, |r| r.salinity.at(t));
        }
        if let Some(columns) = &columns {
            let thin = |idx| self.is_thin(state, idx);
            columns.relax_tracers(state, &self.bathymetry, &self.sigma, rhs, thin);
        }
        self.zero_thin_momentum(state, rhs);
    }

    /// Overwrite `rhs.tke` and `rhs.gls` with the inventory tendencies of the
    /// turbulence advected by the w-cells of `transport` (zero without
    /// [`Self::with_turbulence_advection`]); nothing without a prognostic
    /// closure.
    ///
    /// The bed and surface w-points hold the closure's boundary values
    /// (`k = u*²/(c_μ⁰)²` and the length `κ z₀` of the wall itself), which
    /// nothing reads and the closure overwrites every step; they are not
    /// means over the half-layers of their w-cells. They are advected as
    /// their interior neighbours' values (zero gradient), so that what flows
    /// through the end layers' centres is the interior's turbulence. Carried
    /// in instead, the surface's centimetre length cut the top interior
    /// w-point's viscosity to the background wherever Ω pointed down, and
    /// decoupled the surface layer from the water below (`profile_3d`: the
    /// wind-driven surface current 0.10 against 0.087 m/s after one step).
    fn turbulence_transport_rhs(
        &self,
        state: &Solution3D,
        transport: &LayerTransport,
        barotropic: BarotropicFlux,
        rhs: &mut Solution3D,
    ) {
        if state.tke.is_empty() {
            return;
        }
        let Some(scheme) = self.turbulence_advection else {
            rhs.tke.fill(0.0);
            rhs.gls.fill(0.0);
            return;
        };
        let mut guard = self
            .w_transport_scratch
            .lock()
            .expect("Failed to lock w_transport_scratch");
        let TurbulenceScratch { w_cells, advected } =
            guard.get_or_insert_with(|| TurbulenceScratch {
                w_cells: LayerTransport::new(self.mesh.n_elements, &self.ops, state.n_levels + 1)
                    .with_metric(self.metric_form()),
                advected: [vec![0.0; state.tke.len()], vec![0.0; state.tke.len()]],
            });
        crate::profiling::scope("turb: stagger", || w_cells.stagger_from(transport));
        let nl = state.n_levels;
        let n_elements = self.mesh.n_elements;
        let len = state.tke.len() / n_elements;
        let [tke, gls] = advected;
        let copy = crate::profiling::start();
        for_each_block(
            n_elements,
            [&mut tke[..], &mut gls[..]],
            || (),
            |_, k, advected| {
                for (advected, field) in advected.into_iter().zip([&state.tke, &state.gls]) {
                    advected.copy_from_slice(&field[k * len..(k + 1) * len]);
                    if nl >= 2 {
                        for column in advected.chunks_exact_mut(nl + 1) {
                            column[0] = column[1];
                            column[nl] = column[nl - 1];
                        }
                    }
                }
            },
        );
        crate::profiling::stop("turb: copy", copy);
        crate::profiling::scope("turb: kernel", || {
            apply_tracer_transport_3d(
                [
                    AdvectedTracer::new(&mut rhs.tke, tke, &ExtrapolationTracerBC3D),
                    AdvectedTracer::new(&mut rhs.gls, gls, &ExtrapolationTracerBC3D),
                ],
                w_cells,
                &self.mesh,
                &self.ops,
                &self.geom,
                &self.boundaries,
                scheme,
            )
        });
        let sources = crate::profiling::start();
        for (out, advected) in [(&mut rhs.tke, &*tke), (&mut rhs.gls, &*gls)] {
            // River water brings the column's own turbulence: its volume
            // source must not dilute it; nor the 2D module's
            if let Some(rivers) = barotropic.rivers {
                rivers.add_w_point_sources_at_own_value(out, advected, state.n_nodes);
            }
            w_cells.add_volume_sources(
                out,
                advected,
                &w_cells.d_sigma,
                &state.eta.data,
                &self.bathymetry,
                &self.geom,
                barotropic.element_means,
            );
        }
        crate::profiling::stop("turb: volume sources", sources);
    }

    /// Warn if `state` is stratified and has open 3D faces that no nesting
    /// relaxes (see [`Self::with_nesting`]).
    fn warn_if_stratified_open_boundaries(&self, state: &Solution3D) {
        let nested = |tag| {
            self.nesting
                .as_ref()
                .is_some_and(|n| n.tags().contains(&tag))
        };
        let open_tag = (0..self.mesh.n_elements).find_map(|k| {
            (0..4).find_map(|f| {
                match self
                    .boundaries
                    .exterior(&self.mesh, ElementIndex::new(k), f)
                {
                    crate::solver::rhs::FaceExterior::Open(tag) if !nested(tag) => Some(tag),
                    _ => None,
                }
            })
        });
        let Some(tag) = open_tag else {
            return;
        };
        let nl = state.n_levels;
        let stratified = state.rho.chunks_exact(nl).any(|column| {
            let (lo, hi) = column
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &r| {
                    (lo.min(r), hi.max(r))
                });
            hi - lo > 1e-6 * self.rho0
        });
        if stratified {
            eprintln!(
                "warning: Hydrostatic3D: the open boundary {tag:?} is not nested, and the \
                 water is stratified. Extrapolated open faces let the boundary columns' \
                 stratification run away; relax it with `with_nesting` (a parent model, or \
                 `ReferenceColumns` of the state at rest) and a relaxation band."
            );
        }
    }

    /// Largest surface residual of `Ω` in the last 3D stage before it was
    /// spread over the column (m/s): round-off where the barotropic pass keeps
    /// its nodal identity (see [`LayerTransport::surface_residual`]).
    pub fn last_surface_residual(&self) -> f64 {
        self.transport_scratch
            .lock()
            .expect("Failed to lock transport_scratch")
            .surface_residual
    }

    /// Update density field based on current temperature and salinity.
    pub fn update_density(&self, state: &mut Solution3D) {
        self.eos.update_density(state);
    }

    /// The baroclinic time step at Courant number `cfl` (the barotropic pass
    /// takes its own substeps).
    ///
    /// The explicit 3D terms bound it:
    /// - horizontal advection and internal waves, `cfl·h/(|u| + c₁)/(N+1)²`
    ///   at every node, with `h = 2/(|∇r| + |∇s|)` from the node's metric
    ///   (`√J`, half the side, on a square; the narrow side's scale on an
    ///   elongated element, as in the 2D step), `|u|` the largest layer speed
    ///   of the column and `c₁` its first-mode internal-wave speed in the WKB
    ///   estimate `c₁ = (1/π) ∫ N dz` (exact `NH/π` for a constant `N`),
    ///   `N² = −(g/ρ₀) ∂ρ/∂z` between the layer centres, unstable parts
    ///   counted as zero, and at least [`Self::MIN_INTERNAL_WAVE_SPEED`];
    /// - vertical advection, `|Ω| Δt/H_z ≤ 1` on every layer (`Ω` from the
    ///   last `post_process`; the limit of upwind, below Akima's ≈ 1.2,
    ///   see [`VerticalAdvection`]), unless the excess is advected implicitly
    ///   ([`Self::with_implicit_vertical_advection`]);
    /// - Coriolis, `|f| Δt ≤ 1` (SSP-RK3 is stable on the imaginary axis up
    ///   to √3, SSP-RK(4,3) up to 2.16);
    /// - the horizontal viscosity, by the BR1 spectral radius.
    ///
    /// Thin columns are left to the 2D module (a domain of thin columns
    /// only gets 1 s).
    pub fn compute_dt(&self, state: &Solution3D, cfl: f64) -> f64 {
        self.compute_dt_ssp(state, cfl, 1.0)
    }

    /// [`Self::compute_dt`] for 3D stages of an SSP scheme with coefficient
    /// `ssp` (1 for SSP-RK3, 2 for SSP-RK(4,3),
    /// [`crate::time::SspScheme::ssp_coefficient`]):
    /// the vertical advection's bound is a forward-Euler one (the stages'
    /// upwind-limited reconstruction), so it holds for `ssp` times its step.
    /// `cfl` sets the horizontal one, which is linear in the DG elements and
    /// a forward-Euler (positivity) bound in the subcell elements: raise it
    /// with the scheme (see `ModeSplitIntegrator::with_baroclinic_scheme`).
    pub fn compute_dt_ssp(&self, state: &Solution3D, cfl: f64, ssp: f64) -> f64 {
        let limits = self.time_step_limits(state, cfl);
        let mut dt = limits
            .advection
            .dt
            .min(limits.coriolis.dt)
            .min(limits.viscosity.dt);
        if self.implicit_vertical_advection.is_none() {
            dt = dt.min(ssp * limits.vertical.dt);
        }
        *self
            .last_time_step_limits
            .lock()
            .expect("Failed to lock last_time_step_limits") = Some(limits);
        if dt == f64::INFINITY { 1.0 } else { dt }
    }

    /// The bounds of the last [`Self::compute_dt`] (see
    /// [`Self::time_step_limits`]), if any.
    pub fn last_time_step_limits(&self) -> Option<TimeStepLimits> {
        *self
            .last_time_step_limits
            .lock()
            .expect("Failed to lock last_time_step_limits")
    }

    /// Element `k`'s rates of [`Self::compute_dt`]: the largest
    /// `(|u| + c₁)/h` and `|Ω|/H_z` over its nodes that are not thin, and
    /// what makes up the first at the node that sets it.
    fn element_rates(&self, state: &Solution3D, k: usize) -> ElementRates {
        let (nn, nl) = (state.n_nodes, state.n_levels);
        let (sigma_rho, d_sigma) = (self.sigma.sigma_rho(), self.sigma.d_sigma());
        let mut rates = ElementRates::default();
        for i in 0..nn {
            let idx = k * nn + i;
            // Thin films are the 2D module's (and its own CFL's) business
            if self.is_thin(state, idx) {
                continue;
            }
            let depth = state.eta.data[idx] - self.bathymetry.data[idx];
            let column = idx * nl..(idx + 1) * nl;
            let (u, v, w, rho) = (
                &state.u[column.clone()],
                &state.v[column.clone()],
                &state.w[column.clone()],
                &state.rho[column],
            );
            let mut speed = 0.0_f64;
            for l in 0..nl {
                speed = speed.max(u[l].hypot(v[l]));
                rates.vertical = rates.vertical.max(w[l].abs() / (d_sigma[l] * depth));
            }
            // WKB: Σ N Δz = Σ √(−(g/ρ₀) Δρ Δz) over the interior w-points
            let mut integral = 0.0;
            for l in 1..nl {
                let dz = (sigma_rho[l] - sigma_rho[l - 1]) * depth;
                let n2_dz2 = -self.g / self.rho0 * (rho[l] - rho[l - 1]) * dz;
                integral += n2_dz2.max(0.0).sqrt();
            }
            let c1 = (integral / std::f64::consts::PI).max(Self::MIN_INTERNAL_WAVE_SPEED);
            // 1/h = (|∇r| + |∇s|)/2: 1/√J on a square
            let ((rx, ry), (sx, sy)) = (self.geom.grad_r(k, i), self.geom.grad_s(k, i));
            let inverse_length = 0.5 * (rx.hypot(ry) + sx.hypot(sy));
            let wave = (speed + c1) * inverse_length;
            if wave > rates.wave {
                rates.wave = wave;
                rates.speed = speed;
                rates.c1 = c1;
                rates.length = 1.0 / inverse_length;
                rates.depth = depth;
            }
        }
        rates
    }

    /// Every element's own step from horizontal advection and internal
    /// waves at Courant number `cfl` (the bound of [`Self::compute_dt`] that
    /// usually sets it; infinite for an element of thin columns only), with
    /// what sets it: the layer speed `|u|`, `c₁`, the length `h` and the
    /// depth of the node that binds.
    pub fn element_advection_steps(&self, state: &Solution3D, cfl: f64) -> Vec<ElementStep> {
        let order_factor = (self.ops.order as f64 + 1.0).powi(2);
        (0..self.mesh.n_elements)
            .map(|k| {
                let r = self.element_rates(state, k);
                ElementStep {
                    dt: cfl / r.wave / order_factor,
                    speed: r.speed,
                    c1: r.c1,
                    length: r.length,
                    depth: r.depth,
                }
            })
            .collect()
    }

    /// Each bound of [`Self::compute_dt`] at Courant number `cfl`, and the
    /// element that sets it: horizontal advection and internal waves,
    /// vertical advection (reported also where it is implicit, when
    /// [`Self::compute_dt`] does not apply it), Coriolis, and the horizontal
    /// viscosity. A bound nothing sets is infinite.
    pub fn time_step_limits(&self, state: &Solution3D, cfl: f64) -> TimeStepLimits {
        let [advection, vertical, coriolis] = reduce_blocks::<f64, _, _, 0>(
            self.mesh.n_elements,
            [],
            || (),
            |_, k, []| {
                let ElementRates {
                    wave: wave_rate,
                    vertical: vertical_rate,
                    ..
                } = self.element_rates(state, k);
                let f = self
                    .mesh
                    .element_vertices(ElementIndex::new(k))
                    .iter()
                    .map(|&[_, y]| self.coriolis.f_at(y).abs())
                    .fold(0.0, f64::max);
                let bound = |dt: f64| TimeStepLimit { dt, element: k };
                [
                    bound(cfl / wave_rate / (self.ops.order as f64 + 1.0).powi(2)),
                    bound(1.0 / vertical_rate),
                    bound(1.0 / f),
                ]
            },
            || [TimeStepLimit::NONE; 3],
            |a, b| [0, 1, 2].map(|j| a[j].min(b[j])),
        );

        // The horizontal viscosity, explicit in the SSP stages: BR1
        // couples an element to its face neighbours' gradients, so each
        // element is bounded by the largest ν of its own and theirs
        let mut viscosity = TimeStepLimit::NONE;
        if !self.horizontal_viscosity.is_zero() {
            let mut guard = self
                .viscosity_scratch
                .lock()
                .expect("Failed to lock viscosity_scratch");
            let scratch = guard
                .get_or_insert_with(|| ViscosityScratch3D::new(self.mesh.n_elements, &self.ops));
            let mut largest = vec![0.0; self.mesh.n_elements];
            largest_horizontal_viscosity_3d(
                &mut largest,
                state,
                self.horizontal_viscosity,
                &self.mesh,
                &self.ops,
                &self.geom,
                &self.bathymetry,
                &self.sigma,
                &self.boundaries,
                self.min_column_depth,
                scratch,
            );
            for k in 0..self.mesh.n_elements {
                let nu = (0..4)
                    .filter_map(|face| self.mesh.neighbor(ElementIndex::new(k), face))
                    .map(|nb| largest[nb.element])
                    .fold(largest[k], f64::max);
                let dt = element_dt_viscous_swe_2d(
                    &self.mesh,
                    &self.ops,
                    &self.geom,
                    nu,
                    self.ops.order,
                    cfl,
                    k,
                );
                viscosity = viscosity.min(TimeStepLimit { dt, element: k });
            }
        }
        TimeStepLimits {
            advection,
            vertical,
            coriolis,
            viscosity,
        }
    }

    /// The end state of a step: its tracers limited (the mode splitter's
    /// last stage leaves them to this, after the vertical terms), their
    /// density, and `Ω` for output ([`Self::update_vertical_velocity`]).
    pub fn post_process(&self, state: &mut Solution3D) {
        if !self.apply_tracer_limiters(state).changed() {
            self.update_density(state);
        }

        crate::profiling::scope("post: omega", || self.update_vertical_velocity(state));
    }

    /// Write `Ω` (m/s) at the layer centres to `state.w`, for output: the
    /// mean of the two bounding σ-surfaces, from the state's own layer
    /// transports ([`LayerTransport::compute`] without a barotropic flux, thin
    /// columns at rest as in the stages). Their column sum is `Dū`, which
    /// after a mode-split step is the filtered barotropic transport, and `Ω`
    /// closes with the free-surface rate its divergence implies.
    ///
    /// The same transports advect the momentum of `G` at the start of the
    /// next step ([`ModeSplitPhysics::slow_forcing_into`]): that advection is
    /// kept for it, and taken again only if the state's velocities or
    /// free surface have changed in between. The physics' configuration
    /// (bed, minimum column depth, boundaries, advection forms) must not
    /// change between the two.
    pub fn update_vertical_velocity(&self, state: &mut Solution3D) {
        let mut guard = self
            .slow_forcing_scratch
            .lock()
            .expect("Failed to lock slow_forcing_scratch");
        let scratch = guard.get_or_insert_with(|| {
            SlowForcingScratch::new(state.n_elements, &self.ops, state.n_levels)
        });
        let mut transport_guard = self
            .transport_scratch
            .lock()
            .expect("Failed to lock transport_scratch");
        let transport = &mut *transport_guard;
        self.layer_advection(state, transport, scratch);
        let nl = state.n_levels;
        for (w_col, omega_col) in state
            .w
            .chunks_exact_mut(nl)
            .zip(transport.omega.chunks_exact(nl + 1))
        {
            for (l, w) in w_col.iter_mut().enumerate() {
                *w = 0.5 * (omega_col[l] + omega_col[l + 1]);
            }
        }
    }

    /// `Σ_l A_l(u)` of `state` into `scratch` (the columns advected with
    /// their own layer transports, thin columns at rest as in the stages),
    /// and the transports, without a barotropic flux, into `transport`.
    fn layer_advection(
        &self,
        state: &Solution3D,
        transport: &mut LayerTransport,
        scratch: &mut SlowForcingScratch,
    ) {
        let SlowForcingScratch {
            advection_u,
            advection_v,
            advection_of,
            ..
        } = scratch;
        self.with_thin_columns_at_rest(state, false, |state| {
            transport.metric = self.metric_form();
            transport.compute(
                state,
                None,
                &self.mesh,
                &self.ops,
                &self.geom,
                &self.sigma,
                &self.bathymetry,
                &self.boundaries,
                &Exterior3D::default(),
                self.min_column_depth,
            );
            advection_u.fill(0.0);
            advection_v.fill(0.0);
            apply_momentum_transport_3d(
                advection_u,
                advection_v,
                &state.u,
                &state.v,
                transport,
                &self.mesh,
                &self.ops,
                &self.geom,
                &self.boundaries,
                None,
                self.momentum_advection_form,
                self.momentum_vertical_advection,
            );
        });
        *advection_of = Some(flow_fingerprint(state));
    }
}

impl<EOS, MIX, BC> ModeSplitPhysics for Hydrostatic3D<EOS, MIX, BC>
where
    EOS: EquationOfState,
    MIX: VerticalMixing,
    SWEPhysics2D<BC>: PhysicsModule<SWESolution2D>,
    BC: Clone + Send + Sync + SWEBoundaryCondition2D,
{
    type Barotropic = SWEPhysics2D<BC>;

    fn barotropic(&self) -> &SWEPhysics2D<BC> {
        &self.swe_physics
    }

    fn sigma(&self) -> &SigmaGrid {
        &self.sigma
    }

    fn bathymetry(&self) -> &Bathymetry2D {
        &self.bathymetry
    }

    fn rivers(&self) -> Option<&RiverSources> {
        self.rivers.as_deref()
    }

    fn momentum_rhs_into(&self, state: &Solution3D, t: f64, out: &mut Solution3D) {
        self.compute_momentum_rhs_into(state, t, out);
    }

    fn stage_momentum_rhs_into(&self, state: &Solution3D, t: f64, out: &mut Solution3D) {
        self.momentum_rhs(state, t, out, true);
    }

    fn transport_rhs_into(
        &self,
        state: &Solution3D,
        t: f64,
        barotropic: BarotropicFlux,
        out: &mut Solution3D,
    ) {
        self.compute_transport_rhs_into(state, t, barotropic, out);
    }

    /// `G = D·⟨R_PGF+Cor(u)⟩ + Σ_l A_l(u) − A(ū) − D·R_Cor(ū) + (τ_s − τ_b)/ρ₀
    /// − r·(u_b − ū)`, zero in thin columns (see the module docs), with the
    /// surface stress at `t`.
    fn slow_forcing_into(
        &self,
        state: &Solution3D,
        rhs: &Solution3D,
        t: f64,
        g: &mut SWESolution2D,
    ) {
        let (ne, nn, nl) = (state.n_elements, state.n_nodes, state.n_levels);
        let mut guard = self
            .slow_forcing_scratch
            .lock()
            .expect("Failed to lock slow_forcing_scratch");
        let scratch = guard.get_or_insert_with(|| SlowForcingScratch::new(ne, &self.ops, nl));
        let mut transport_guard = self
            .transport_scratch
            .lock()
            .expect("Failed to lock transport_scratch");
        let transport = &mut *transport_guard;
        // Σ_l A_l(u), unless the last `update_vertical_velocity` left this
        // state's
        if scratch.advection_of != Some(flow_fingerprint(state)) {
            self.layer_advection(state, transport, scratch);
        }
        let SlowForcingScratch {
            advection_u,
            advection_v,
            bar,
            bar_rhs,
            bar_transport,
            bar_sigma,
            ..
        } = scratch;

        // A(ū) + D·R_Cor(ū): the mean flow as one layer, thin columns at rest
        bar.eta.copy_from(&state.eta);
        for (idx, (u, v)) in bar.u.iter_mut().zip(&mut bar.v).enumerate() {
            (*u, *v) = if self.is_thin(state, idx) {
                (0.0, 0.0)
            } else {
                (state.ubar.data[idx], state.vbar.data[idx])
            };
        }
        bar_transport.metric = self.metric_form();
        bar_transport.compute(
            bar,
            None,
            &self.mesh,
            &self.ops,
            &self.geom,
            bar_sigma,
            &self.bathymetry,
            &self.boundaries,
            &Exterior3D::default(),
            self.min_column_depth,
        );
        bar_rhs.u.fill(0.0);
        bar_rhs.v.fill(0.0);
        apply_coriolis_3d(bar_rhs, bar, &self.mesh, &self.ops, &self.coriolis);
        for ((u, v), (&eta, &b)) in bar_rhs
            .u
            .iter_mut()
            .zip(&mut bar_rhs.v)
            .zip(state.eta.data.iter().zip(&self.bathymetry.data))
        {
            *u *= eta - b;
            *v *= eta - b;
        }
        apply_momentum_transport_3d(
            &mut bar_rhs.u,
            &mut bar_rhs.v,
            &bar.u,
            &bar.v,
            bar_transport,
            &self.mesh,
            &self.ops,
            &self.geom,
            &self.boundaries,
            None,
            self.momentum_advection_form,
            self.momentum_vertical_advection,
        );

        let [tau_sx, tau_sy] = self.forcing.surface_stress;
        let [tau_bx, tau_by] = self.forcing.bottom_stress;

        g.data[SWE_VAR_H].fill(0.0);
        let [_, g_hu, g_hv] = &mut g.data;
        let (advection_u, advection_v, bar_rhs): (&[f64], &[f64], &Solution3D) =
            (advection_u, advection_v, bar_rhs);
        self.surface_stress_at(t, |field| {
            for_each_block(
                ne,
                [&mut g_hu[..ne * nn], &mut g_hv[..ne * nn]],
                || (),
                |_, k, [g_hu, g_hv]| {
                    let bed = self.bathymetry.element(ElementIndex::new(k));
                    for (i, &b) in bed.iter().enumerate() {
                        let idx = k * nn + i;
                        let depth = state.eta.data[idx] - b;
                        // Thin columns: their depth mean is the 2D module's alone
                        if depth < self.min_column_depth {
                            g_hu[i] = 0.0;
                            g_hv[i] = 0.0;
                            continue;
                        }
                        let columns = idx * nl..(idx + 1) * nl;
                        let mean_u = self.sigma.depth_average(&rhs.u[columns.clone()]);
                        let mean_v = self.sigma.depth_average(&rhs.v[columns.clone()]);
                        let advection_x: f64 = advection_u[columns.clone()].iter().sum();
                        let advection_y: f64 = advection_v[columns].iter().sum();
                        let (field_x, field_y) =
                            field.map_or((0.0, 0.0), |[x, y]| (x[idx], y[idx]));
                        let stress_x = (tau_sx + field_x - tau_bx) / self.rho0;
                        let stress_y = (tau_sy + field_y - tau_by) / self.rho0;
                        // The shear part of the bottom drag; the pass applies −r·ū
                        let (drag_x, drag_y) = match &self.bottom_drag {
                            Some(drag) => {
                                let r = self.drag_rate(drag, state, idx);
                                (
                                    r * (state.u[idx * nl] - state.ubar.data[idx]),
                                    r * (state.v[idx * nl] - state.vbar.data[idx]),
                                )
                            }
                            None => (0.0, 0.0),
                        };
                        g_hu[i] = depth * mean_u + advection_x - bar_rhs.u[idx] + stress_x - drag_x;
                        g_hv[i] = depth * mean_v + advection_y - bar_rhs.v[idx] + stress_y - drag_y;
                    }
                    // The shear part of the cage drag, −D Σ_l Δσ_l λ_l (u_l − ū);
                    // the pass applies −Λ̄ū
                    if let Some(cages) = &self.cage_drag {
                        for_each_caged_node(cages, k, nn, |i, entries| {
                            let idx = k * nn + i;
                            let depth = state.eta.data[idx] - bed[i];
                            if depth < self.min_column_depth {
                                return;
                            }
                            let (ubar, vbar) = (state.ubar.data[idx], state.vbar.data[idx]);
                            let (mut drag_x, mut drag_y) = (0.0, 0.0);
                            for (l, &ds) in self.sigma.d_sigma().iter().enumerate() {
                                let (u, v) = (state.u[idx * nl + l], state.v[idx * nl + l]);
                                let rate =
                                    layer_coefficient(entries, &self.sigma, l, depth) * u.hypot(v);
                                drag_x += ds * rate * (u - ubar);
                                drag_y += ds * rate * (v - vbar);
                            }
                            g_hu[i] -= depth * drag_x;
                            g_hv[i] -= depth * drag_y;
                        });
                    }
                },
            )
        });
    }

    /// `r = C_d|u_b|` of every column from the bottom-layer velocity (the
    /// depth mean in thin columns), if a drag is set.
    fn bottom_drag_into(&self, state: &Solution3D, _t: f64, rate: &mut [f64]) -> bool {
        let Some(drag) = &self.bottom_drag else {
            return false;
        };
        for (idx, r) in rate.iter_mut().enumerate() {
            *r = self.drag_rate(drag, state, idx);
        }
        true
    }

    /// `λ_l` of every layer under a cage (zero elsewhere), if cages are set.
    fn layer_drag_into(&self, state: &Solution3D, _t: f64, rate: &mut [f64]) -> bool {
        let Some(cages) = &self.cage_drag else {
            return false;
        };
        let (nn, nl) = (state.n_nodes, state.n_levels);
        rate.fill(0.0);
        for k in 0..state.n_elements {
            for_each_caged_node(cages, k, nn, |i, entries| {
                let idx = k * nn + i;
                self.cage_rates_into(entries, state, idx, &mut rate[idx * nl..(idx + 1) * nl]);
            });
        }
        true
    }

    /// The implicit part of `Ω` of the last [`Self::transport_rhs_into`],
    /// if any ([`Self::with_implicit_vertical_advection`]).
    fn implicit_vertical_advection(&self, stage: &mut Solution3D, dt: f64, element_means: &[bool]) {
        if self.implicit_vertical_advection.is_none() {
            return;
        }
        let split = self
            .implicit_split
            .lock()
            .expect("Failed to lock implicit_split");
        if split.columns == 0 {
            return;
        }
        apply_implicit_vertical_advection(
            stage,
            &split.omega,
            self.sigma.d_sigma(),
            &split.d_sigma_w,
            &self.bathymetry,
            &self.geom,
            element_means,
            dt,
        );
    }

    /// The vertical diffusion with the surface stress and fluxes at the
    /// middle of the step, `t + dt/2` (the fluxes over the surface layer at
    /// `t`).
    fn vertical_implicit(&self, state: &mut Solution3D, t: f64, dt: f64, drag: StepDrag<'_>) {
        let fluxes = self.surface_fluxes_at(state, t + 0.5 * dt);
        let tracers =
            fluxes.as_deref().map(
                |[_, _, nonsolar, shortwave, freshwater]| SurfaceTracerFluxes {
                    nonsolar,
                    shortwave,
                    freshwater,
                    water: self.shortwave_absorption,
                },
            );
        self.surface_stress_at(t + 0.5 * dt, |field| {
            apply_vertical_diffusion(
                state,
                &self.sigma,
                &self.bathymetry,
                dt,
                &self.mixing,
                &self.forcing,
                SurfaceFields {
                    stress: field,
                    roughness: self.surface_roughness.as_deref(),
                    tracers,
                },
                self.g,
                self.rho0,
                self.min_column_depth,
                drag,
            )
        });
        // Thin columns carry the depth mean only
        let nl = state.n_levels;
        for idx in 0..state.eta.data.len() {
            if self.is_thin(state, idx) {
                state.u[idx * nl..(idx + 1) * nl].fill(state.ubar.data[idx]);
                state.v[idx * nl..(idx + 1) * nl].fill(state.vbar.data[idx]);
            }
        }
    }

    fn end_density(&self, state: &mut Solution3D) {
        self.update_density(state);
    }

    /// Tracer limiters, then the density of the limited tracers.
    fn post_stage(&self, state: &mut Solution3D) {
        if !self.apply_tracer_limiters(state).changed() {
            self.update_density(state);
        }
    }

    /// The thin columns at rest, as every 3D kernel sees them (so the
    /// stage's RHS need not copy the state to mask them,
    /// [`Self::with_thin_columns_at_rest`]); their values into `stash`.
    fn mask_stage(&self, state: &mut Solution3D, stash: &mut StageStash) {
        let nl = state.n_levels;
        stash.columns.clear();
        stash.values.clear();
        for idx in 0..state.eta.data.len() {
            if self.is_thin(state, idx) {
                let column = idx * nl..(idx + 1) * nl;
                stash.columns.push(idx);
                stash.values.extend_from_slice(&state.u[column.clone()]);
                stash.values.extend_from_slice(&state.v[column.clone()]);
                stash.values.push(state.ubar.data[idx]);
                stash.values.push(state.vbar.data[idx]);
                state.u[column.clone()].fill(0.0);
                state.v[column].fill(0.0);
                state.ubar.data[idx] = 0.0;
                state.vbar.data[idx] = 0.0;
            }
        }
    }
}

/// Buffers of [`Hydrostatic3D`]'s advection of the turbulence.
struct TurbulenceScratch {
    /// The w-cells of the layer transports.
    w_cells: LayerTransport,
    /// `k` and `ψ` as advected: the end w-points at their neighbours' values.
    advected: [Vec<f64>; 2],
}

/// Buffers of [`Hydrostatic3D`]'s slow forcing.
struct SlowForcingScratch {
    /// `A_l(u)` of every layer, `[element][node][level]`.
    advection_u: Vec<f64>,
    advection_v: Vec<f64>,
    /// The [`flow_fingerprint`] of the state they are of, if any.
    advection_of: Option<u64>,
    /// One layer carrying `ū`, and its advection + Coriolis tendency
    /// (inventory form).
    bar: Solution3D,
    bar_rhs: Solution3D,
    bar_transport: LayerTransport,
    bar_sigma: SigmaGrid,
}

impl SlowForcingScratch {
    fn new(n_elements: usize, ops: &DGOperators2D, n_levels: usize) -> Self {
        let nn = ops.n_nodes;
        Self {
            advection_u: vec![0.0; n_elements * nn * n_levels],
            advection_v: vec![0.0; n_elements * nn * n_levels],
            advection_of: None,
            bar: Solution3D::new(n_elements, nn, 1),
            bar_rhs: Solution3D::new(n_elements, nn, 1),
            bar_transport: LayerTransport::new(n_elements, ops, 1),
            bar_sigma: SigmaGrid::uniform(1),
        }
    }
}

/// A fingerprint of what the layer transports and the momentum advection
/// of `state` read: `η`, `ū`, `v̄`, `u` and `v`. Each element's values are
/// hashed in a chain of bijections, so a change of one value always changes
/// it (other changes collide with a chance of about 2⁻⁶⁴); the elements'
/// hashes are summed, so it does not depend on the thread count.
fn flow_fingerprint(state: &Solution3D) -> u64 {
    let (nn, nl) = (state.n_nodes, state.n_levels);
    let [hash] = reduce_blocks::<u64, _, _, 0>(
        state.n_elements,
        [],
        || (),
        |_, k, []| {
            let nodes = k * nn..(k + 1) * nn;
            let levels = k * nn * nl..(k + 1) * nn * nl;
            let mut h = (k as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
            for &x in [
                &state.eta.data[nodes.clone()],
                &state.ubar.data[nodes.clone()],
                &state.vbar.data[nodes],
                &state.u[levels.clone()],
                &state.v[levels],
            ]
            .into_iter()
            .flatten()
            {
                h = (h.rotate_left(5) ^ x.to_bits()).wrapping_mul(0x517c_c1b7_2722_0a95);
            }
            [h]
        },
        || [0],
        |[a], [b]| [a.wrapping_add(b)],
    );
    hash
}
