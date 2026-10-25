//! Right-hand side computation for 2D DG shallow water equations.
//!
//! For the 2D SWE:
//!   ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0
//!   ∂(hu)/∂t + ∂(hu² + gh²/2)/∂x + ∂(huv)/∂y = fhv + S_x
//!   ∂(hv)/∂t + ∂(huv)/∂x + ∂(hv² + gh²/2)/∂y = -fhu + S_y
//!
//! The DG semi-discrete form uses the weak formulation:
//!   dq/dt = L(q) = -(volume terms) + (surface terms) + (source terms)
//!
//! Two spatial formulations are available ([`SWEFormulation2D`]): the standard
//! collocated form below, and the entropy-stable, well-balanced split form of
//! Wintermeyer et al. (2017) in `swe_2d_split_form.rs`.

use crate::boundary::{BCContext2D, BoundaryState, SWEBoundaryCondition2D};
use crate::equations::ShallowWater2D;
use crate::flux::{SWEFluxType2D, compute_flux_swe_2d};
use crate::mesh::{Bathymetry2D, Mesh2D};
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::core::disjoint::{DisjointChunks, all_distinct};
use crate::solver::rhs::subcells::subcell_interfaces;
use crate::solver::{SWESolution2D, SWEState2D};
use crate::source::swe_2d::viscosity::HorizontalViscosity2D;
use crate::source::{ElementSources, HydrostaticReconstruction2D, SourceTerm2D};
use crate::time::SspScheme;
use crate::types::ElementIndex;

use super::diffusion_2d::ScalarGradient2D;
use super::swe_2d_split_form::{ELEMENT_CHUNK, SplitFormSWE2D, SplitFormWorkspace, VOLUME_BATCH};
use super::swe_2d_viscosity::{ViscousScratch, ViscousTerm, ViscousWorkspace};
#[cfg(feature = "simd")]
use crate::solver::simd::{apply_diff_matrix, apply_lift, coriolis_source};
#[cfg(not(feature = "simd"))]
use crate::solver::simd::{
    apply_diff_matrix_scalar as apply_diff_matrix, apply_lift_scalar as apply_lift,
    coriolis_source_scalar as coriolis_source,
};
use std::cell::RefCell;

/// Spatial discretization of the 2D SWE flux divergence and bathymetry terms.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SWEFormulation2D {
    /// Collocated nodal DG: the nodal flux is differentiated directly, faces use
    /// the `flux_type` Riemann solver (optionally on hydrostatically reconstructed
    /// states, `well_balanced`), and the bed slope comes from `BathymetrySource2D`
    /// in `source_terms`. Lake-at-rest is exact only for deg B ≤ p/2 per element.
    #[default]
    Standard,
    /// Wintermeyer et al. (2017) split-form DGSEM with entropy-conservative
    /// fluxes in the volume *and* at faces (no interface dissipation).
    ///
    /// Conserves mass exactly and the total energy up to round-off on periodic
    /// domains; meant for verification. Use [`Self::EntropyStable`] for runs.
    EntropyConservative,
    /// Wintermeyer et al. (2017) split-form DGSEM with an entropy-stable
    /// interface flux (entropy-conservative flux plus Lax–Friedrichs dissipation
    /// in entropy variables).
    ///
    /// Well-balanced for *any* nodal bathymetry, including bathymetry that jumps
    /// across faces, with exact mass conservation and a discrete entropy
    /// (energy) inequality. The bed slope is part of the operator: `flux_type`
    /// and `well_balanced` are ignored, and `source_terms` must not contain
    /// `BathymetrySource2D` (the RHS panics if it does). Assumes wet nodes; there
    /// is no wetting/drying treatment (use [`Self::WetDry`]). See
    /// `solver/rhs/swe_2d_split_form.rs`.
    EntropyStable,
    /// Split form for wetting and drying:
    /// - fully wet elements use the flux-differencing volume term of
    ///   [`Self::EntropyStable`];
    /// - elements with a node shallower than `SWE2DRhsConfig::h_dry` (or
    ///   `subcell_depth`) use a second-order (limited linear reconstruction)
    ///   finite-volume update on their GLL subcells;
    /// - every interface, faces and subcell faces alike, uses HLL on
    ///   Audusse et al. (2004) hydrostatically reconstructed states.
    ///
    /// Well-balanced at lake at rest also across wet/dry shorelines,
    /// mass-conservative, and positivity preserving for the element means under
    /// `positivity_cfl_swe_2d` (with the positivity limiter). Like the other
    /// split forms the bed slope is part of the operator: no
    /// `BathymetrySource2D`; `flux_type` and `well_balanced` are ignored.
    ///
    /// Compared with [`Self::Standard`] with hydrostatic reconstruction, which
    /// leaves a stationary spurious circulation at a lake-at-rest shoreline
    /// (≈ 5 cm/s where h > 1 cm on a 2 % beach), it is also more accurate on a
    /// moving shoreline (Thacker's paraboloid at 40², L1 depth error: P1
    /// 3.2 % vs 10 %, P2 1.2 % vs 1.3 %, P3 0.9 % vs 1.8 %, P4 0.7 % vs 2.3 %;
    /// `tests/wet_dry_2d_test.rs`). `SWEPhysics2DBuilder` uses it by default
    /// for runs with wetting/drying.
    WetDry,
}

/// Configuration for 2D SWE RHS computation.
pub struct SWE2DRhsConfig<'a, BC: SWEBoundaryCondition2D> {
    /// The shallow water equation parameters
    pub equation: &'a ShallowWater2D,
    /// Spatial formulation (collocated or split-form entropy-stable)
    pub formulation: SWEFormulation2D,
    /// Numerical flux type (`SWEFormulation2D::Standard` only)
    pub flux_type: SWEFluxType2D,
    /// Boundary condition handler
    pub bc: &'a BC,
    /// Whether to include Coriolis source term (legacy, use source_terms instead)
    pub include_coriolis: bool,
    /// Optional trait-based source terms (preferred over include_coriolis)
    ///
    /// When set, these source terms are evaluated at each node and added to the RHS.
    /// Multiple sources can be combined using `CombinedSource2D`.
    pub source_terms: Option<&'a dyn SourceTerm2D>,
    /// Optional bathymetry data for well-balanced schemes and source terms.
    ///
    /// When set, bathymetry values and gradients are passed to boundary conditions
    /// and source terms for proper handling of variable bottom topography.
    pub bathymetry: Option<&'a Bathymetry2D>,
    /// Enable hydrostatic reconstruction for well-balanced treatment of bathymetry.
    ///
    /// When enabled, the numerical flux is evaluated on Audusse et al. (2004)
    /// hydrostatically reconstructed interface states and the interior side gets
    /// the momentum correction ½g(h*² − h²)·n. This balances the bathymetry
    /// *jumps* across faces; the interior flux stays that of the nodal state, so
    /// mass is conserved exactly.
    ///
    /// Requires `bathymetry` to be set. The in-element slope ∂B still has to be
    /// balanced by `BathymetrySource2D` in `source_terms`:
    /// - cell-constant B (`Bathymetry2D::to_cell_average`): the source is zero and
    ///   may be omitted; lake-at-rest is preserved to round-off;
    /// - nodal (non-constant) B: **include** `BathymetrySource2D`, otherwise the
    ///   volume term leaves an unbalanced `g h ∂B/∂x` (O(1) m/s² accelerations).
    ///   Even with the source, lake-at-rest is exact only if `½gh²` is resolved by
    ///   the element polynomials (deg B ≤ p/2).
    pub well_balanced: bool,
    /// Optional horizontal viscosity for momentum diffusion.
    ///
    /// When set, adds ∇·(ν h ∇u) and ∇·(ν h ∇v) to the momentum equations.
    /// This is computed via a double-derivative Laplacian (differentiate velocity
    /// gradients, then differentiate viscous fluxes), following the same pattern
    /// as tracer diffusion.
    pub viscosity: Option<&'a HorizontalViscosity2D>,
    /// Depth (m) below which a node counts as dry for
    /// [`SWEFormulation2D::WetDry`]: elements with such a node use the subcell
    /// finite-volume update, and the subcells do not reconstruct the surface
    /// through it. Default 1 mm (`WetDryConfig::DEFAULT_H_DRY`).
    pub h_dry: f64,
    /// Depth (m) below which a node puts its element on the subcell update
    /// (default: `h_dry`). A larger depth runs the subcells in wet elements
    /// too, e.g. to measure their accuracy on wet flow; which nodes are dry
    /// is still decided by `h_dry` alone.
    pub subcell_depth: Option<f64>,
}

impl<'a, BC: SWEBoundaryCondition2D> SWE2DRhsConfig<'a, BC> {
    /// Create a new RHS configuration.
    pub fn new(equation: &'a ShallowWater2D, bc: &'a BC) -> Self {
        Self {
            equation,
            formulation: SWEFormulation2D::Standard,
            flux_type: SWEFluxType2D::Roe,
            bc,
            include_coriolis: true,
            source_terms: None,
            bathymetry: None,
            well_balanced: false,
            viscosity: None,
            h_dry: crate::solver::WetDryConfig::DEFAULT_H_DRY,
            subcell_depth: None,
        }
    }

    /// Set the spatial formulation.
    ///
    /// # Example
    /// ```ignore
    /// // Well-balanced for arbitrary (also face-discontinuous) nodal bathymetry.
    /// // The bed slope is part of the operator: no BathymetrySource2D here.
    /// let config = SWE2DRhsConfig::new(&equation, &bc)
    ///     .with_formulation(SWEFormulation2D::EntropyStable)
    ///     .with_bathymetry(&bathymetry)
    ///     .with_source_terms(&coriolis);
    /// ```
    pub fn with_formulation(mut self, formulation: SWEFormulation2D) -> Self {
        self.formulation = formulation;
        self
    }

    /// Set the numerical flux type.
    pub fn with_flux_type(mut self, flux_type: SWEFluxType2D) -> Self {
        self.flux_type = flux_type;
        self
    }

    /// Set whether to include Coriolis (legacy method).
    ///
    /// For new code, prefer using `with_source_terms` with `CoriolisSource2D`.
    pub fn with_coriolis(mut self, include: bool) -> Self {
        self.include_coriolis = include;
        self
    }

    /// Set trait-based source terms.
    ///
    /// Source terms are evaluated at each quadrature node and added to the RHS.
    /// Use `CombinedSource2D` to compose multiple source terms.
    ///
    /// # Example
    /// ```ignore
    /// let coriolis = CoriolisSource2D::norwegian_coast();
    /// let config = SWE2DRhsConfig::new(&equation, &bc)
    ///     .with_coriolis(false)  // Disable legacy Coriolis
    ///     .with_source_terms(&coriolis);
    /// ```
    pub fn with_source_terms(mut self, sources: &'a dyn SourceTerm2D) -> Self {
        self.source_terms = Some(sources);
        self
    }

    /// Set bathymetry data for well-balanced schemes.
    ///
    /// When bathymetry is provided, the values and gradients are passed to:
    /// - Boundary condition contexts (for correct ghost state computation)
    /// - Source term contexts (for bathymetry-dependent physics)
    ///
    /// # Example
    /// ```ignore
    /// let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
    ///     // Sill profile
    ///     0.5 * (-((x - 5.0).powi(2)) / 2.0).exp()
    /// });
    /// let config = SWE2DRhsConfig::new(&equation, &bc)
    ///     .with_bathymetry(&bathymetry);
    /// ```
    pub fn with_bathymetry(mut self, bathymetry: &'a Bathymetry2D) -> Self {
        self.bathymetry = Some(bathymetry);
        self
    }

    /// Enable hydrostatic reconstruction for well-balanced treatment of bathymetry.
    ///
    /// Uses the Audusse et al. (2004) hydrostatic reconstruction at element faces,
    /// which balances bathymetry *jumps* across faces while keeping discrete mass
    /// conservation exact. Lake-at-rest (η = h + B = const, u = v = 0) is preserved
    /// to machine precision when the in-element bathymetry is also balanced, see
    /// the requirements below.
    ///
    /// # Requirements
    /// - `bathymetry` must be set via `with_bathymetry()`
    /// - For nodal (non-constant within elements) bathymetry, **include**
    ///   `BathymetrySource2D` in `source_terms`; it balances the volume pressure
    ///   term. Only for cell-constant bathymetry (`Bathymetry2D::to_cell_average`),
    ///   whose gradient is zero, is the source term a no-op that may be omitted.
    /// - Even with the source term, nodal bathymetry is balanced exactly only when
    ///   deg B ≤ p/2 in each element.
    ///
    /// # Example
    /// ```ignore
    /// // Well-balanced scheme for steep Norwegian bathymetry
    /// let bathy_source = BathymetrySource2D::new(g);
    /// let config = SWE2DRhsConfig::new(&equation, &bc)
    ///     .with_bathymetry(&bathymetry)
    ///     .with_source_terms(&bathy_source)
    ///     .with_well_balanced(true);
    /// ```
    pub fn with_well_balanced(mut self, enable: bool) -> Self {
        self.well_balanced = enable;
        self
    }

    /// Set horizontal viscosity for momentum diffusion.
    ///
    /// When enabled, adds ∇·(ν h ∇u) and ∇·(ν h ∇v) to the momentum RHS.
    /// Use `compute_dt_viscosity` to compute the diffusive time step restriction,
    /// and take `dt = dt_advective.min(dt_viscous)`.
    pub fn with_viscosity(mut self, visc: &'a HorizontalViscosity2D) -> Self {
        self.viscosity = Some(visc);
        self
    }

    /// Set the dry-node depth of [`SWEFormulation2D::WetDry`].
    pub fn with_dry_threshold(mut self, h_dry: f64) -> Self {
        assert!(h_dry >= 0.0, "dry threshold must be non-negative");
        self.h_dry = h_dry;
        self
    }

    /// Run the `WetDry` subcell update in every element with a node shallower
    /// than `depth` (default: `h_dry`; see [`Self::subcell_depth`]).
    pub fn with_subcell_depth(mut self, depth: f64) -> Self {
        assert!(depth >= 0.0, "subcell depth must be non-negative");
        self.subcell_depth = Some(depth);
        self
    }
}

/// Boundary state from the configured boundary condition at `node` of
/// boundary face `face` of element `k`: a ghost state for the Riemann solver,
/// or the state on the boundary whose physical flux is the face flux. Shared
/// by the standard and split-form kernels and the viscous boundary terms.
pub(super) fn boundary_state<BC: SWEBoundaryCondition2D>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
    k: ElementIndex,
    face: usize,
    node: usize,
    normal: (f64, f64),
) -> BoundaryState {
    let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[node], ops.nodes_s[node]);
    let state = q.get_state(k, node);
    let bathy_value = config.bathymetry.map_or(0.0, |b| b.get(k, node));
    let g = config.equation.g;
    let h_min = config.equation.h_min.meters();

    let ctx = match mesh.boundary_tag(k, face) {
        Some(tag) => BCContext2D::with_tag(time, (x, y), state, bathy_value, normal, g, h_min, tag),
        None => BCContext2D::new(time, (x, y), state, bathy_value, normal, g, h_min),
    }
    .with_node_index(k.as_usize() * ops.n_nodes + node);
    config.bc.boundary_state(&ctx)
}

/// Scratch space for one element of the RHS kernel (SoA, one `Vec` per
/// variable).
struct ElementWorkspace {
    n_nodes: usize,
    n_face_nodes: usize,
    /// Contravariant fluxes F·J∇r and F·J∇s
    flux_r: [Vec<f64>; 3],
    flux_s: [Vec<f64>; 3],
    /// Dr·F̃r and Ds·F̃s
    dfr_dr: [Vec<f64>; 3],
    dfs_ds: [Vec<f64>; 3],
    /// hu, hv of the element (f-plane Coriolis kernel)
    momentum: [Vec<f64>; 2],
    /// F(q⁻)·n − F* at the face nodes
    flux_diff: [Vec<f64>; 3],
    /// Exterior (neighbour or ghost) states at the face nodes
    ext: [Vec<f64>; 3],
    /// Face nodes whose exterior state is a boundary state `q_b` with flux
    /// `F(q_b)·n` ([`BoundaryState::Exact`])
    ext_exact: Vec<bool>,
    int_bathy: Vec<f64>,
    ext_bathy: Vec<f64>,
    split_form: SplitFormWorkspace,
    viscous: ViscousScratch,
}

/// `n` zeros with two cache lines of unused capacity after them.
///
/// Workspaces of different threads are small and allocated at about the same
/// time, so without the slack their buffers can share cache lines, and every
/// write then invalidates another core's copy (false sharing): the cached
/// workspaces made the parallel RHS ~25 % slower than fresh ones.
fn padded(n: usize) -> Vec<f64> {
    const SLACK: usize = 16; // 128 bytes
    let mut v = Vec::with_capacity(n + SLACK);
    v.resize(n, 0.0);
    v
}

impl ElementWorkspace {
    fn new(n_nodes: usize, n_face_nodes: usize) -> Self {
        let nodes = || [padded(n_nodes), padded(n_nodes), padded(n_nodes)];
        let face = || {
            [
                padded(n_face_nodes),
                padded(n_face_nodes),
                padded(n_face_nodes),
            ]
        };
        Self {
            n_nodes,
            n_face_nodes,
            flux_r: nodes(),
            flux_s: nodes(),
            dfr_dr: nodes(),
            dfs_ds: nodes(),
            momentum: [padded(n_nodes), padded(n_nodes)],
            flux_diff: face(),
            ext: face(),
            ext_exact: vec![false; n_face_nodes],
            int_bathy: padded(n_face_nodes),
            ext_bathy: padded(n_face_nodes),
            split_form: SplitFormWorkspace::new(n_nodes),
            viscous: ViscousScratch::new(n_nodes),
        }
    }
}

thread_local! {
    /// Cached element workspace of this thread (rayon workers persist, so after
    /// the first RHS evaluation nothing is allocated).
    static ELEMENT_WORKSPACE: RefCell<Option<ElementWorkspace>> = const { RefCell::new(None) };
}

/// An element workspace taken from this thread's cache for one serial loop or
/// one rayon job, and returned to the cache on drop.
///
/// Taking it once per job instead of looking it up per element matters: a
/// per-element lookup (lock or thread-local) made the parallel RHS ~25 %
/// slower. A new workspace is allocated only if the cache is empty (first
/// use, or a nested job on the same thread) or sized for other operators.
struct WorkspaceGuard(Option<ElementWorkspace>);

impl WorkspaceGuard {
    fn take(ops: &DGOperators2D) -> Self {
        let (n_nodes, n_face_nodes) = (ops.n_nodes, ops.n_face_nodes);
        let cached = ELEMENT_WORKSPACE
            .with(|cell| cell.try_borrow_mut().ok().and_then(|mut slot| slot.take()))
            .filter(|ws| ws.n_nodes == n_nodes && ws.n_face_nodes == n_face_nodes);
        Self(Some(cached.unwrap_or_else(|| {
            ElementWorkspace::new(n_nodes, n_face_nodes)
        })))
    }
}

impl std::ops::Deref for WorkspaceGuard {
    type Target = ElementWorkspace;
    fn deref(&self) -> &ElementWorkspace {
        self.0.as_ref().expect("workspace present until drop")
    }
}

impl std::ops::DerefMut for WorkspaceGuard {
    fn deref_mut(&mut self) -> &mut ElementWorkspace {
        self.0.as_mut().expect("workspace present until drop")
    }
}

impl Drop for WorkspaceGuard {
    fn drop(&mut self) {
        let ws = self.0.take();
        // `try_with`: the thread-local may already be gone at thread exit
        let _ = ELEMENT_WORKSPACE.try_with(|cell| {
            if let Ok(mut slot) = cell.try_borrow_mut()
                && slot.is_none()
            {
                *slot = ws;
            }
        });
    }
}

thread_local! {
    /// Interior-face flux buffer of the split forms, cached on the thread
    /// that drives the RHS (reused by every evaluation after the first).
    static FACE_FLUXES: RefCell<Vec<SWEState2D>> = const { RefCell::new(Vec::new()) };
}

/// The face-flux buffer, taken from this thread's cache for one RHS
/// evaluation and returned on drop (a nested evaluation on the same thread
/// gets a fresh one).
struct FaceFluxGuard(Vec<SWEState2D>);

impl FaceFluxGuard {
    fn take() -> Self {
        Self(
            FACE_FLUXES
                .with(|cell| cell.try_borrow_mut().map(|mut v| std::mem::take(&mut *v)))
                .unwrap_or_default(),
        )
    }
}

impl std::ops::Deref for FaceFluxGuard {
    type Target = Vec<SWEState2D>;
    fn deref(&self) -> &Vec<SWEState2D> {
        &self.0
    }
}

impl std::ops::DerefMut for FaceFluxGuard {
    fn deref_mut(&mut self) -> &mut Vec<SWEState2D> {
        &mut self.0
    }
}

impl Drop for FaceFluxGuard {
    fn drop(&mut self) {
        let faces = std::mem::take(&mut self.0);
        // Keep the larger buffer; `try_with`: the thread-local may be gone at
        // thread exit
        let _ = FACE_FLUXES.try_with(|cell| {
            if let Ok(mut slot) = cell.try_borrow_mut()
                && slot.capacity() < faces.capacity()
            {
                *slot = faces;
            }
        });
    }
}

thread_local! {
    /// Viscous-term workspace (velocity gradients of every node), cached on
    /// the thread that drives the RHS.
    static VISCOUS_WORKSPACE: RefCell<ViscousWorkspace> = RefCell::new(ViscousWorkspace::default());
}

/// The viscous workspace, taken from this thread's cache for one RHS
/// evaluation and returned on drop (a nested evaluation on the same thread
/// gets a fresh one).
struct ViscousGuard(ViscousWorkspace);

impl ViscousGuard {
    /// The gradients of u and v (whole-mesh arrays).
    fn gradients(&self) -> [&[ScalarGradient2D]; 2] {
        [&self.0.grad_u, &self.0.grad_v]
    }

    /// With gradient storage for every node of `n_elements × n_nodes`.
    fn take(n_total: usize) -> Self {
        let mut ws = VISCOUS_WORKSPACE
            .with(|cell| cell.try_borrow_mut().map(|mut ws| std::mem::take(&mut *ws)))
            .unwrap_or_default();
        ws.resize(n_total);
        Self(ws)
    }
}

impl Drop for ViscousGuard {
    fn drop(&mut self) {
        let ws = std::mem::take(&mut self.0);
        // Keep the larger buffers; `try_with`: the thread-local may be gone
        // at thread exit
        let _ = VISCOUS_WORKSPACE.try_with(|cell| {
            if let Ok(mut slot) = cell.try_borrow_mut()
                && slot.grad_u.capacity() < ws.grad_u.capacity()
            {
                *slot = ws;
            }
        });
    }
}

/// Pass 1 of the viscous term (its gradients) over every element at `time`,
/// in parallel with `parallel`.
fn viscous_gradients_all<BC: SWEBoundaryCondition2D>(
    viscous: &ViscousTerm<BC>,
    ops: &DGOperators2D,
    time: f64,
    ws: &mut ViscousWorkspace,
    parallel: bool,
) {
    let n = ops.n_nodes;
    let ViscousWorkspace { grad_u, grad_v, .. } = ws;
    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        grad_u
            .par_chunks_exact_mut(n)
            .zip(grad_v.par_chunks_exact_mut(n))
            .enumerate()
            .for_each_init(
                || WorkspaceGuard::take(ops),
                |ws, (k, (gu, gv))| viscous.gradients(k, time, &mut ws.viscous, gu, gv),
            );
        return;
    }
    let _ = parallel;
    let mut ws = WorkspaceGuard::take(ops);
    for (k, (gu, gv)) in grad_u
        .chunks_exact_mut(n)
        .zip(grad_v.chunks_exact_mut(n))
        .enumerate()
    {
        viscous.gradients(k, time, &mut ws.viscous, gu, gv);
    }
}

/// The face fluxes of the consecutive edges `e0..` whose slots are `batch`
/// (`per_edge` values each, at most [`VOLUME_BATCH`] edges), batched.
fn edge_batch<BC: SWEBoundaryCondition2D>(
    split_form: &SplitFormSWE2D<BC>,
    e0: usize,
    batch: &mut [SWEState2D],
    per_edge: usize,
) {
    let count = batch.len() / per_edge;
    let edges: [usize; VOLUME_BATCH] = std::array::from_fn(|l| e0 + l);
    let mut chunks = batch.chunks_exact_mut(per_edge);
    let mut slots: [&mut [SWEState2D]; VOLUME_BATCH] =
        std::array::from_fn(|_| chunks.next().unwrap_or_default());
    split_form.edge_fluxes_batch(&edges[..count], &mut slots[..count]);
}

/// One 2D SWE RHS evaluation. `element` is the only place the per-element
/// terms are computed; the serial and parallel drivers just iterate over it.
struct SWE2DRhsKernel<'a, 'c, BC: SWEBoundaryCondition2D> {
    q: &'a SWESolution2D,
    mesh: &'a Mesh2D,
    ops: &'a DGOperators2D,
    geom: &'a GeometricFactors2D,
    config: &'a SWE2DRhsConfig<'c, BC>,
    time: f64,
    /// Split-form (Wintermeyer et al. 2017) operator, if selected
    split_form: Option<SplitFormSWE2D<'a, 'c, BC>>,
}

impl<'a, 'c, BC: SWEBoundaryCondition2D> SWE2DRhsKernel<'a, 'c, BC> {
    fn new(
        q: &'a SWESolution2D,
        mesh: &'a Mesh2D,
        ops: &'a DGOperators2D,
        geom: &'a GeometricFactors2D,
        config: &'a SWE2DRhsConfig<'c, BC>,
        time: f64,
    ) -> Self {
        Self {
            q,
            mesh,
            ops,
            geom,
            config,
            time,
            split_form: SplitFormSWE2D::new(q, mesh, ops, geom, config, time),
        }
    }

    /// Interior-face fluxes of the split forms, once per face, into `faces`
    /// (resized as needed; left empty for the collocated formulation).
    fn face_fluxes(&self, faces: &mut Vec<SWEState2D>) {
        let Some(split_form) = &self.split_form else {
            faces.clear();
            return;
        };
        faces.resize(split_form.face_buffer_len(), SWEState2D::zero());
        let per_edge = 2 * self.ops.n_face_nodes;
        for (b, batch) in faces.chunks_mut(VOLUME_BATCH * per_edge).enumerate() {
            edge_batch(split_form, b * VOLUME_BATCH, batch, per_edge);
        }
    }

    /// [`Self::face_fluxes`] over the edges in parallel (identical result).
    #[cfg(feature = "parallel")]
    fn face_fluxes_parallel(&self, faces: &mut Vec<SWEState2D>) {
        use rayon::prelude::*;

        let Some(split_form) = &self.split_form else {
            faces.clear();
            return;
        };
        faces.resize(split_form.face_buffer_len(), SWEState2D::zero());
        let per_edge = 2 * self.ops.n_face_nodes;
        faces
            .par_chunks_mut(VOLUME_BATCH * per_edge)
            .enumerate()
            .for_each(|(b, batch)| edge_batch(split_form, b * VOLUME_BATCH, batch, per_edge));
    }

    /// Volume, surface and source terms of element `k`, written (not added)
    /// to `out = [h, hu, hv]`; `faces` from [`Self::face_fluxes`].
    ///
    /// `face_mass`, if given (`4 · n_face_nodes` values), receives the mass
    /// component of `F*` at every face node of the element, along its outward
    /// normal (see [`compute_rhs_swe_2d_face_mass_into`]), and `subcell_mass`,
    /// if given, the mass flux through every subcell interface (NaN where the
    /// element has no subcells; see [`compute_rhs_swe_2d_mass_fluxes_into`]).
    ///
    /// `volume`: the slot of the workspace's batch holding the element's
    /// split-form volume term, from [`Self::volume_chunk`].
    fn element(
        &self,
        k: usize,
        ws: &mut ElementWorkspace,
        faces: &[SWEState2D],
        out: [&mut [f64]; 3],
        face_mass: Option<&mut [f64]>,
        subcell_mass: Option<&mut [f64]>,
        volume: Option<usize>,
    ) {
        let k_idx = ElementIndex::new(k);
        let [out_h, out_hu, out_hv] = out;

        // 1–2. Volume and surface terms
        match &self.split_form {
            Some(split_form) => split_form.element_rhs(
                k_idx,
                &mut ws.split_form,
                faces,
                [out_h, out_hu, out_hv],
                face_mass,
                subcell_mass,
                volume,
            ),
            None => {
                if let Some(mass) = subcell_mass {
                    mass.fill(f64::NAN);
                }
                self.collocated_terms(k_idx, ws, [out_h, out_hu, out_hv], face_mass)
            }
        }

        // 3–4. Source terms
        self.source_terms(k_idx, ws, [out_h, out_hu, out_hv]);
    }

    /// Split-form volume terms of the elements `ks` (at most
    /// [`ELEMENT_CHUNK`]), vectorised across them (`simd` feature), into
    /// the workspace for the next [`Self::element`] calls: bit `j` of the
    /// result is set if `ks[j]`'s is there (slot `j`).
    fn volume_chunk(&self, ks: &[usize], ws: &mut ElementWorkspace) -> u64 {
        self.split_form.as_ref().map_or(0, |split_form| {
            split_form.volume_chunk(ks, &mut ws.split_form)
        })
    }

    /// [`Self::element`] for the consecutive elements `k0..` whose rows are
    /// `out` (at most [`ELEMENT_CHUNK`] elements; `face_mass` and
    /// `subcell_mass` likewise), their volume terms batched, then
    /// `after(k, ws, hu, hv)` for each.
    fn element_batch(
        &self,
        k0: usize,
        ws: &mut ElementWorkspace,
        faces: &[SWEState2D],
        [out_h, out_hu, out_hv]: [&mut [f64]; 3],
        face_mass: Option<&mut [f64]>,
        subcell_mass: Option<&mut [f64]>,
        after: impl Fn(usize, &mut ElementWorkspace, &mut [f64], &mut [f64]),
    ) {
        let ops = self.ops;
        let n = ops.n_nodes;
        let ks: [usize; ELEMENT_CHUNK] = std::array::from_fn(|l| k0 + l);
        let volumes = self.volume_chunk(&ks[..out_h.len() / n], ws);
        let mut face_mass = face_mass.map(|fm| fm.chunks_exact_mut(4 * ops.n_face_nodes));
        let mut subcell_mass =
            subcell_mass.map(|sm| sm.chunks_exact_mut(subcell_interfaces(ops.n_1d)));
        for (l, ((h, hu), hv)) in out_h
            .chunks_exact_mut(n)
            .zip(out_hu.chunks_exact_mut(n))
            .zip(out_hv.chunks_exact_mut(n))
            .enumerate()
        {
            let k = k0 + l;
            let fm = face_mass.as_mut().and_then(Iterator::next);
            let sm = subcell_mass.as_mut().and_then(Iterator::next);
            let volume = ((volumes >> l) & 1 == 1).then_some(l);
            self.element(k, ws, faces, [h, &mut *hu, &mut *hv], fm, sm, volume);
            after(k, ws, hu, hv);
        }
    }

    /// Collocated nodal DG volume and surface terms in conservative
    /// (divergence) form, with the metric at every node (Kopriva 2009, ch. 6):
    ///   −J⁻¹[Dr·F̃r + Ds·F̃s] + J⁻¹ Σ_f LIFT_f sJ_f (F(q⁻)·n − F*)
    /// with the contravariant fluxes F̃r = F·J∇r, F̃s = F·J∇s. With the
    /// discrete metric identities (`GeometricFactors2D`) it preserves a uniform
    /// state on non-affine elements, and it conserves mass on any element
    /// (the volume term telescopes to the faces, `D` being SBP).
    fn collocated_terms(
        &self,
        k: ElementIndex,
        ws: &mut ElementWorkspace,
        out: [&mut [f64]; 3],
        mut face_mass: Option<&mut [f64]>,
    ) {
        let (q, mesh, ops, geom, config) = (self.q, self.mesh, self.ops, self.geom, self.config);
        let ki = k.as_usize();
        let n_nodes = ops.n_nodes;
        let n_face_nodes = ops.n_face_nodes;
        let g = config.equation.g;
        let h_min = config.equation.h_min.meters();
        let [out_h, out_hu, out_hv] = out;

        // 1. Volume term: −J⁻¹[Dr·F̃r + Ds·F̃s], from the contravariant fluxes
        //    (into the workspace flux_r, flux_s)
        {
            let [fr_h, fr_hu, fr_hv] = &mut ws.flux_r;
            let [fs_h, fs_hu, fs_hv] = &mut ws.flux_s;
            for i in 0..n_nodes {
                let state = q.get_state(k, i);
                let fx = config.equation.flux_x(&state);
                let fy = config.equation.flux_y(&state);
                let ((ar_x, ar_y), (as_x, as_y)) = geom.contravariant(ki, i);
                let fr = ar_x * fx + ar_y * fy;
                let fs = as_x * fx + as_y * fy;
                (fr_h[i], fr_hu[i], fr_hv[i]) = (fr.h, fr.hu, fr.hv);
                (fs_h[i], fs_hu[i], fs_hv[i]) = (fs.h, fs.hu, fs.hv);
            }
        }
        for (d, flux, out) in [
            (&ops.dr_row_major, &ws.flux_r, &mut ws.dfr_dr),
            (&ops.ds_row_major, &ws.flux_s, &mut ws.dfs_ds),
        ] {
            let [o_h, o_hu, o_hv] = out;
            apply_diff_matrix(d, &flux[0], &flux[1], &flux[2], o_h, o_hu, o_hv, n_nodes);
        }
        for i in 0..n_nodes {
            let j_inv = geom.jacobian_inv(ki, i);
            out_h[i] = -j_inv * (ws.dfr_dr[0][i] + ws.dfs_ds[0][i]);
            out_hu[i] = -j_inv * (ws.dfr_dr[1][i] + ws.dfs_ds[1][i]);
            out_hv[i] = -j_inv * (ws.dfr_dr[2][i] + ws.dfs_ds[2][i]);
        }

        // 2. Surface terms
        let well_balanced = config.bathymetry.filter(|_| config.well_balanced);
        let hr = well_balanced.map(|_| HydrostaticReconstruction2D::new(g, h_min));

        for face in 0..4 {
            let face_nodes = &ops.face_nodes[face];
            let [ext_h, ext_hu, ext_hv] = &mut ws.ext;

            // Interior bathymetry at the face (well-balanced reconstruction only)
            match well_balanced {
                Some(b) => {
                    for (fi, &node) in face_nodes.iter().enumerate() {
                        ws.int_bathy[fi] = b.get(k, node);
                    }
                }
                None => ws.int_bathy.fill(0.0),
            }

            // Exterior states and bathymetry: neighbour (face nodes reversed) or ghost
            ws.ext_exact.fill(false);
            if let Some(neighbor) = mesh.neighbor(k, face) {
                let nb = ElementIndex::new(neighbor.element);
                let nb_face_nodes = &ops.face_nodes[neighbor.face];
                for fi in 0..n_face_nodes {
                    let nb_node = nb_face_nodes[n_face_nodes - 1 - fi];
                    let state = q.get_state(nb, nb_node);
                    (ext_h[fi], ext_hu[fi], ext_hv[fi]) = (state.h, state.hu, state.hv);
                    ws.ext_bathy[fi] = well_balanced.map_or(0.0, |b| b.get(nb, nb_node));
                }
            } else {
                for (fi, &node) in face_nodes.iter().enumerate() {
                    let normal = geom.normal(ki, face, fi);
                    let ghost = match boundary_state(
                        q, mesh, ops, config, self.time, k, face, node, normal,
                    ) {
                        BoundaryState::Ghost(ghost) => ghost,
                        BoundaryState::Exact(q_b) => {
                            ws.ext_exact[fi] = true;
                            q_b
                        }
                    };
                    (ext_h[fi], ext_hu[fi], ext_hv[fi]) = (ghost.h, ghost.hu, ghost.hv);
                }
                // Boundary faces mirror the interior bathymetry
                ws.ext_bathy.copy_from_slice(&ws.int_bathy);
            }

            // Numerical flux and flux difference at the face nodes
            let [diff_h, diff_hu, diff_hv] = &mut ws.flux_diff;
            for (fi, &node) in face_nodes.iter().enumerate() {
                let normal = geom.normal(ki, face, fi);
                // Per-node GLL lift scale sJ/J (LIFT couples a face node only
                // to the volume node it coincides with)
                let scale = geom.lift_scale(ki, face, fi, node);
                let q_int = q.get_state(k, node);
                let q_ext = SWEState2D::new(ext_h[fi], ext_hu[fi], ext_hv[fi]);

                // Boundary state q_b: the face flux is its physical flux F(q_b)·n
                if ws.ext_exact[fi] {
                    let f_b = config.equation.normal_flux(&q_ext, normal);
                    if let Some(face_mass) = face_mass.as_deref_mut() {
                        face_mass[face * n_face_nodes + fi] = f_b.h;
                    }
                    let diff = scale * (config.equation.normal_flux(&q_int, normal) - f_b);
                    (diff_h[fi], diff_hu[fi], diff_hv[fi]) = (diff.h, diff.hu, diff.hv);
                    continue;
                }

                // Hydrostatically reconstructed states for F* (if enabled)
                let (q_int_flux, q_ext_flux) = match &hr {
                    Some(r) => r.reconstruct(&q_int, &q_ext, ws.int_bathy[fi], ws.ext_bathy[fi]),
                    None => (q_int, q_ext),
                };
                let f_star = compute_flux_swe_2d(
                    &q_int_flux,
                    &q_ext_flux,
                    normal,
                    g,
                    h_min,
                    config.flux_type,
                );
                if let Some(face_mass) = face_mass.as_deref_mut() {
                    face_mass[face * n_face_nodes + fi] = f_star.h;
                }

                // Interior flux F(q⁻)·n of the actual nodal state. It must match the
                // volume term for the surface/volume pair to telescope (SBP); using the
                // reconstructed state here adds −∮(h⁻ − h*⁻)u⁻·n to the element mass.
                let mut diff = config.equation.normal_flux(&q_int, normal) - f_star;

                // Well-balancing enters as the Audusse momentum correction ½g(h*⁻² − h⁻²)·n
                if let Some(r) = &hr {
                    diff = diff + r.pressure_correction(q_int.h, q_int_flux.h, normal);
                }
                let diff = scale * diff;
                (diff_h[fi], diff_hu[fi], diff_hv[fi]) = (diff.h, diff.hu, diff.hv);
            }

            // out += LIFT_f (sJ/J)(F(q⁻)·n − F*), the scale already applied
            apply_lift(
                &ops.lift_row_major[face],
                diff_h,
                diff_hu,
                diff_hv,
                out_h,
                out_hu,
                out_hv,
                n_nodes,
                n_face_nodes,
                1.0,
            );
        }
    }

    /// Legacy built-in Coriolis and the trait-based source terms, added to `out`.
    fn source_terms(&self, k: ElementIndex, ws: &mut ElementWorkspace, out: [&mut [f64]; 3]) {
        let (q, mesh, ops, config) = (self.q, self.mesh, self.ops, self.config);
        let n_nodes = ops.n_nodes;
        let g = config.equation.g;
        let h_min = config.equation.h_min.meters();
        let [out_h, out_hu, out_hv] = out;

        // 3. Legacy Coriolis (only without trait-based source terms)
        if config.include_coriolis
            && config.source_terms.is_none()
            && (config.equation.f0.abs() > 1e-14 || config.equation.beta.abs() > 1e-14)
        {
            if config.equation.beta.abs() < 1e-14 {
                let [hu, hv] = &mut ws.momentum;
                for i in 0..n_nodes {
                    let state = q.get_state(k, i);
                    (hu[i], hv[i]) = (state.hu, state.hv);
                }
                coriolis_source(hu, hv, out_hu, out_hv, config.equation.f0, n_nodes);
            } else {
                for i in 0..n_nodes {
                    let [_x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                    let source = config.equation.coriolis_source(&q.get_state(k, i), y);
                    out_hu[i] += source.hu;
                    out_hv[i] += source.hv;
                }
            }
        }

        // 4. Trait-based source terms (preferred), one call per element: each
        //    source reads only the node data it needs
        if let Some(sources) = config.source_terms {
            let element = ElementSources {
                element: k,
                time: self.time,
                solution: q,
                mesh,
                ops,
                bathymetry: config.bathymetry,
                g,
                h_min,
            };
            sources.add_element(&element, out_h, out_hu, out_hv);
        }
    }
}

/// Compute the right-hand side for 2D SWE.
///
/// Implements the DG weak form for a system:
///   dq/dt = -1/J * [Dr * (Fr) + Ds * (Fs)] + 1/J * Σ_f LIFT_f * sJ_f * (F(q⁻)·n - F*) + S
///
/// where:
///   Fr = F · ∇r = F_x * rx + F_y * ry
///   Fs = F · ∇s = F_x * sx + F_y * sy
///
/// Allocates the result; time steppers should reuse an output buffer with
/// [`compute_rhs_swe_2d_into`] instead.
pub fn compute_rhs_swe_2d<BC: SWEBoundaryCondition2D>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
) -> SWESolution2D {
    let mut rhs = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    compute_rhs_swe_2d_into(q, mesh, ops, geom, config, time, &mut rhs);
    rhs
}

/// [`compute_rhs_swe_2d`] into `out`, which is overwritten.
///
/// Allocation-free after the first call on a thread (the element workspace
/// is cached per thread), unless horizontal viscosity is enabled. Gives the same result, bit for bit, as
/// [`compute_rhs_swe_2d_parallel_into`].
pub fn compute_rhs_swe_2d_into<BC: SWEBoundaryCondition2D>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
    out: &mut SWESolution2D,
) {
    rhs_serial(q, mesh, ops, geom, config, time, out, None, None);
}

/// [`compute_rhs_swe_2d_into`] that also writes the numerical mass flux of
/// every element face into `face_mass`.
///
/// Layout: `face_mass[(k · 4 + face) · n_face_nodes + fi]` is the mass
/// component of `F*` at face node `fi` of face `face` of element `k`, along
/// the element's outward normal (m²/s; [`face_mass_len`] values). With the
/// nodal `(hu, hv)` it is the transport whose DG divergence is the mass
/// tendency: `dh/dt = −(∇·(hu, hv) − J⁻¹ Σ_f LIFT_f sJ_f ((hu, hv)·n − F*_h))`
/// for the collocated and flux-differencing volume terms (the mass part of
/// the two-point flux is the mean of `(hu, hv)`). In `WetDry` elements with a
/// dry node the volume term is a subcell finite-volume update; there only the
/// element balance `∫ dh/dt = −∮ F*_h` holds.
pub fn compute_rhs_swe_2d_face_mass_into<BC: SWEBoundaryCondition2D>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
    out: &mut SWESolution2D,
    face_mass: &mut [f64],
) {
    rhs_serial(q, mesh, ops, geom, config, time, out, Some(face_mass), None);
}

/// [`compute_rhs_swe_2d_face_mass_into`] that also writes the mass flux
/// through every subcell interface into `subcell_mass`
/// ([`subcell_mass_len`] values; layout `k · n_sub + slot`, `n_sub =`
/// [`crate::solver::rhs::subcells::subcell_interfaces`], slots as in
/// [`crate::solver::rhs::subcells`]).
///
/// In `WetDry` elements whose volume term is the subcell finite-volume
/// update, the face and subcell mass fluxes give the mass tendency node by
/// node: `J_i dh_i/dt = −Σ_dir (F̂_{a+½} − F̂_{a−½})/w_a` with the end
/// interfaces replaced by the face flux `F*` (the 3D layers rely on it). In
/// every other element all its slots are NaN.
pub fn compute_rhs_swe_2d_mass_fluxes_into<BC: SWEBoundaryCondition2D>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
    out: &mut SWESolution2D,
    face_mass: &mut [f64],
    subcell_mass: &mut [f64],
) {
    rhs_serial(
        q,
        mesh,
        ops,
        geom,
        config,
        time,
        out,
        Some(face_mass),
        Some(subcell_mass),
    );
}

/// Length of the face mass flux buffer of
/// [`compute_rhs_swe_2d_face_mass_into`]: four faces of `n_face_nodes` per
/// element.
pub fn face_mass_len(mesh: &Mesh2D, ops: &DGOperators2D) -> usize {
    mesh.n_elements * 4 * ops.n_face_nodes
}

/// Length of the subcell mass flux buffer of
/// [`compute_rhs_swe_2d_mass_fluxes_into`].
pub fn subcell_mass_len(mesh: &Mesh2D, ops: &DGOperators2D) -> usize {
    mesh.n_elements * subcell_interfaces(ops.n_1d)
}

fn rhs_serial<BC: SWEBoundaryCondition2D>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
    out: &mut SWESolution2D,
    face_mass: Option<&mut [f64]>,
    subcell_mass: Option<&mut [f64]>,
) {
    check_rhs_output(out, mesh, ops);
    let kernel = SWE2DRhsKernel::new(q, mesh, ops, geom, config, time);
    let mut faces = FaceFluxGuard::take();
    kernel.face_fluxes(&mut faces);
    let rows = ELEMENT_CHUNK * ops.n_nodes;
    let mut face_mass = face_mass.map(|fm| {
        assert_eq!(
            fm.len(),
            face_mass_len(mesh, ops),
            "face mass flux buffer length"
        );
        fm.chunks_mut(ELEMENT_CHUNK * 4 * ops.n_face_nodes)
    });
    let mut subcell_mass = subcell_mass.map(|sm| {
        assert_eq!(
            sm.len(),
            subcell_mass_len(mesh, ops),
            "subcell mass flux buffer length"
        );
        sm.chunks_mut(ELEMENT_CHUNK * subcell_interfaces(ops.n_1d))
    });
    let viscous = viscous_all(q, mesh, ops, geom, config, time, false);
    let [out_h, out_hu, out_hv] = &mut out.data;
    let mut ws = WorkspaceGuard::take(ops);
    for (b, ((h, hu), hv)) in out_h
        .chunks_mut(rows)
        .zip(out_hu.chunks_mut(rows))
        .zip(out_hv.chunks_mut(rows))
        .enumerate()
    {
        let fm = face_mass.as_mut().and_then(Iterator::next);
        let sm = subcell_mass.as_mut().and_then(Iterator::next);
        kernel.element_batch(
            b * ELEMENT_CHUNK,
            &mut ws,
            &faces,
            [h, hu, hv],
            fm,
            sm,
            |k, ws, hu, hv| {
                if let Some((term, grads)) = &viscous {
                    term.add(k, grads.gradients(), &mut ws.viscous, hu, hv);
                }
            },
        );
    }
}

/// The viscous term of one whole-mesh RHS evaluation with its gradients
/// (pass 1) over every element at `time`, if `config` has viscosity.
fn viscous_all<'a, 'c, BC: SWEBoundaryCondition2D>(
    q: &'a SWESolution2D,
    mesh: &'a Mesh2D,
    ops: &'a DGOperators2D,
    geom: &'a GeometricFactors2D,
    config: &'a SWE2DRhsConfig<'c, BC>,
    time: f64,
    parallel: bool,
) -> Option<(ViscousTerm<'a, 'c, BC>, ViscousGuard)> {
    let term = ViscousTerm::new(q, mesh, ops, geom, config)?;
    let mut guard = ViscousGuard::take(mesh.n_elements * ops.n_nodes);
    viscous_gradients_all(&term, ops, time, &mut guard.0, parallel);
    Some((term, guard))
}

fn check_rhs_output(out: &SWESolution2D, mesh: &Mesh2D, ops: &DGOperators2D) {
    assert!(
        out.n_elements == mesh.n_elements && out.n_nodes == ops.n_nodes,
        "RHS output is {}×{}, mesh and operators need {}×{}",
        out.n_elements,
        out.n_nodes,
        mesh.n_elements,
        ops.n_nodes
    );
}

/// Compute the stable time step for 2D SWE.
///
/// Each node pairs its own wave speeds with its element's metric:
///
///   Δt = CFL/(2N+1) · min over elements k and nodes i of 4 / (λ_r + λ_s),
///   λ_r = |u·∇r| + c|∇r|,  λ_s = |u·∇s| + c|∇s|,
///
/// the spectral radii of the flux Jacobian along the reference directions
/// (reference length 2). On a square element of side Δx at rest this is
/// CFL·Δx/((2N+1)c), the classic form, so CFL values keep their meaning. On an
/// element of width w ≪ length L the limit follows w, not √(wL).
///
/// Summing the two directions is the 2D tensor-product bound (as in Trixi.jl's
/// `max_dt`); with it the SSP-RK3 + Zhang–Shu positivity bound is a single
/// CFL per order for every element shape, see [`positivity_cfl_swe_2d`].
///
/// Returns `f64::INFINITY` when every node is dry or at rest in zero depth.
pub fn compute_dt_swe_2d(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    geom: &GeometricFactors2D,
    equation: &ShallowWater2D,
    order: usize,
    cfl: f64,
) -> f64 {
    let max_rate = ElementIndex::iter(mesh.n_elements)
        .map(|k| element_max_reference_rate(q, geom, equation, k))
        .fold(0.0_f64, f64::max);
    dt_from_reference_rate(max_rate, order, cfl)
}

/// Parallel version of [`compute_dt_swe_2d`] (identical result).
#[cfg(feature = "parallel")]
pub fn compute_dt_swe_2d_parallel(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    geom: &GeometricFactors2D,
    equation: &ShallowWater2D,
    order: usize,
    cfl: f64,
) -> f64 {
    use rayon::prelude::*;

    let max_rate = (0..mesh.n_elements)
        .into_par_iter()
        .map(|k| element_max_reference_rate(q, geom, equation, ElementIndex::new(k)))
        .reduce(|| 0.0_f64, f64::max);
    dt_from_reference_rate(max_rate, order, cfl)
}

/// Largest CFL for which [`compute_dt_swe_2d`] satisfies the DGSEM positivity
/// bound of Zhang & Shu (2010) for SSP-RK3 with a positivity-preserving flux
/// (HLL, Rusanov/Lax–Friedrichs; not Roe).
///
/// The cell mean stays non-negative under forward Euler when
/// Δt·(λ_x/Δx + λ_y/Δy) ≤ ŵ₀ = 1/(N(N+1)), the first GLL weight on [0, 1].
/// In this module's units that is CFL ≤ (2N+1)/(2N(N+1)): 0.75, 0.42, 0.29,
/// 0.23 for N = 1–4, independent of element shape. An SSP integrator with
/// SSP coefficient C keeps it up to C times that: SSP-RK3 (C = 1) inherits
/// it, SSP-RK(4,3) (C = 2) doubles it. Wet/dry runs should use
/// `cfl.min(C·positivity_cfl_swe_2d(N))`, as `Simulation` does.
pub const fn positivity_cfl_swe_2d(order: usize) -> f64 {
    assert!(order >= 1, "positivity bound needs N ≥ 1");
    let n = order as f64;
    (2.0 * n + 1.0) / (2.0 * n * (n + 1.0))
}

/// Linear stability limit of the DGSEM SWE under `scheme`, in the CFL units
/// of [`compute_dt_swe_2d`], for N = 1–4 (`None` above).
///
/// The largest CFL at which every eigenvalue of the linearised RHS, times the
/// step, lies in the scheme's stability region, measured on periodic meshes
/// (`tests/linear_stability_test.rs`, which checks these values stay below
/// the measured limits): flat and sinusoidal beds at rest, a diagonal flow at
/// Froude 0.5, square, 3:1 and 10:1 rectangles and distorted quadrilaterals,
/// `WetDry` and `Standard` alike. The per-node metric of the time step makes
/// distortion and a varying depth raise the limit, not lower it; elongated
/// elements at rest set it.
///
/// This is above the Zhang–Shu positivity bound [`positivity_cfl_swe_2d`]
/// times the SSP coefficient by 2.1–2.5 (SSP-RK3) and 1.5–2.0 (SSP-RK(4,3)):
/// the room [`PositivityBound`] steps into where elements cannot run dry.
///
/// It holds for the flux-differencing volume term of fully wet elements.
/// Elements with a dry node run the `WetDry` subcells, whose limit is lower:
/// [`linear_cfl_subcells_swe_2d`].
pub const fn linear_cfl_swe_2d(order: usize, scheme: SspScheme) -> Option<f64> {
    let limits = match scheme {
        SspScheme::Rk3 => [1.6, 1.04, 0.73, 0.56],
        SspScheme::Rk43 => [2.2, 1.51, 1.15, 0.93],
    };
    if order >= 1 && order <= limits.len() {
        Some(limits[order - 1])
    } else {
        None
    }
}

/// Stability limit of the `WetDry` subcell finite volumes (the volume term of
/// elements with a node shallower than `h_dry`) under `scheme`, in the CFL
/// units of [`compute_dt_swe_2d`], for N = 1–4 (`None` above).
///
/// Measured with every element on the subcells (`tests/linear_stability_test.rs`):
/// the linear limit of the first-order scheme (a one-node perturbation of a
/// uniform state leaves every limited slope zero) and the largest CFL at
/// which a random perturbation stays bounded under the limited second-order
/// reconstruction, on square to 30:1 rectangles at rest and with a flow at
/// Froude 0.5. Elongated elements set it, and at N = 2 under SSP-RK(4,3) the
/// reconstruction does (1.11 at 30:1 against 1.28 at first order).
///
/// The smallest subcells, at the element faces, are as wide as the first
/// GLL weight, so the limit follows the Zhang–Shu positivity bound, 1.25–1.6
/// times it with SSP-RK3 and 1.0–1.7 times it with SSP-RK(4,3). (At N = 1
/// under SSP-RK(4,3) the plain bound, 1.5, is at the limit.) It is 0.6–0.85
/// of [`linear_cfl_swe_2d`].
pub const fn linear_cfl_subcells_swe_2d(order: usize, scheme: SspScheme) -> Option<f64> {
    let limits = match scheme {
        SspScheme::Rk3 => [0.94, 0.65, 0.47, 0.36],
        SspScheme::Rk43 => [1.5, 1.1, 0.97, 0.75],
    };
    if order >= 1 && order <= limits.len() {
        Some(limits[order - 1])
    } else {
        None
    }
}

/// max over the nodes of element `k` of (λ_r + λ_s)/4 (1/s).
#[inline]
fn element_max_reference_rate(
    q: &SWESolution2D,
    geom: &GeometricFactors2D,
    equation: &ShallowWater2D,
    k: ElementIndex,
) -> f64 {
    let ki = k.as_usize();
    let h_min = equation.h_min.meters();
    let norms = |(grad_r, grad_s): ((f64, f64), (f64, f64))| {
        let norm_r = (grad_r.0 * grad_r.0 + grad_r.1 * grad_r.1).sqrt();
        let norm_s = (grad_s.0 * grad_s.0 + grad_s.1 * grad_s.1).sqrt();
        (grad_r, grad_s, norm_r, norm_s)
    };
    // A parallelogram has one metric: evaluate it once, from the dense
    // per-element geometry
    let element = geom.element_geometry(ki);
    let affine = element.affine.then(|| {
        norms((
            (element.metric.rx, element.metric.ry),
            (element.metric.sx, element.metric.sy),
        ))
    });
    (0..q.n_nodes)
        .map(|i| {
            let state = q.get_state(k, i);
            if state.h <= h_min {
                return 0.0;
            }
            let metric = affine.unwrap_or_else(|| norms((geom.grad_r(ki, i), geom.grad_s(ki, i))));
            node_reference_rate(equation, &state, metric)
        })
        .fold(0.0_f64, f64::max)
}

/// ¼(λ_r + λ_s) of `state` with the metric `(∇r, ∇s, |∇r|, |∇s|)` of a node:
/// λ_r = |u·∇r| + c|∇r|, |∇r| times the wave speed along ∇r.
#[inline]
fn node_reference_rate(
    equation: &ShallowWater2D,
    state: &SWEState2D,
    (grad_r, grad_s, norm_r, norm_s): ((f64, f64), (f64, f64), f64, f64),
) -> f64 {
    let (u, v) = equation.velocity_simple(state);
    let c = equation.celerity(state.h);
    let lambda_r = (u * grad_r.0 + v * grad_r.1).abs() + c * norm_r;
    let lambda_s = (u * grad_s.0 + v * grad_s.1).abs() + c * norm_s;
    0.25 * (lambda_r + lambda_s)
}

#[inline]
fn dt_from_reference_rate(max_rate: f64, order: usize, cfl: f64) -> f64 {
    if max_rate < 1e-14 {
        return f64::INFINITY;
    }
    cfl / ((2.0 * order as f64 + 1.0) * max_rate)
}

/// Largest stable SSP-RK3 step of the BR1 viscous term alone, `cfl` times
/// the real-axis stability limit, for a constant viscosity `nu`, on element
/// `k` (the largest metric over its own and its face neighbours' nodes: the
/// neighbours' gradients enter its term).
///
/// The spectral radius of the BR1 operator (`solver/rhs/swe_2d_viscosity.rs`)
/// is bounded by `0.0825 (N+1)⁴ ν (|∇r|² + |∇s|²)`, fitted to power
/// iteration on periodic meshes of square and 4:1 elements (0.080, 0.080,
/// 0.080, 0.081 for N = 1–4 on squares; the same on 4:1 elements), and
/// SSP-RK3 is stable on the negative real axis up to `Δt·ρ = 2.5127`.
/// `f64::INFINITY` for `nu = 0`.
pub fn element_dt_viscous_swe_2d(
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    nu: f64,
    order: usize,
    cfl: f64,
    k: usize,
) -> f64 {
    /// Real-axis stability limit of the three-stage, third-order RK methods
    const SSP_RK3_REAL_AXIS: f64 = 2.5127;
    if nu <= 0.0 {
        return f64::INFINITY;
    }
    let metric = |j: usize| {
        (0..ops.n_nodes)
            .map(|i| {
                let ((rx, ry), (sx, sy)) = (geom.grad_r(j, i), geom.grad_s(j, i));
                rx * rx + ry * ry + sx * sx + sy * sy
            })
            .fold(0.0_f64, f64::max)
    };
    let k_idx = ElementIndex::new(k);
    let largest = (0..4)
        .filter_map(|face| mesh.neighbor(k_idx, face))
        .map(|nb| metric(nb.element))
        .fold(metric(k), f64::max);
    let rate = 0.0825 * ((order + 1) as f64).powi(4) * nu * largest;
    cfl * SSP_RK3_REAL_AXIS / rate
}

/// Compute the diffusive time step restriction for horizontal viscosity.
///
/// Its `(2N+1)²` underestimates the BR1 spectral radius at N ≥ 2 (by 2.0,
/// 3.4, 5.0× at N = 2, 3, 4), so at `cfl = 1` the step exceeds SSP-RK3's
/// stability limit from N = 3; `SWEPhysics2D` uses the measured bound of
/// [`element_dt_viscous_swe_2d`].
///
/// The diffusive CFL condition is:
///   Δt ≤ CFL × Δx² / (ν × (2N+1)²)
///
/// where N is the polynomial order and (2N+1)² accounts for the DG
/// spectral radius of the diffusion operator.
///
/// Users should combine with the advective time step:
///   `dt = compute_dt_swe_2d(...).min(compute_dt_viscosity(...))`
///
/// # Arguments
/// * `nu_max` - Maximum viscosity coefficient [m²/s]
/// * `min_h_elem` - Minimum element size (e.g. from `geom.element_size(k)`)
/// * `order` - Polynomial order of the DG scheme
/// * `cfl` - CFL number (typically 0.1–0.5 for diffusion)
pub fn compute_dt_viscosity(nu_max: f64, min_h_elem: f64, order: usize, cfl: f64) -> f64 {
    if nu_max < 1e-14 {
        return f64::INFINITY;
    }
    let dg_factor = (2.0 * order as f64 + 1.0).powi(2);
    cfl * min_h_elem * min_h_elem / (nu_max * dg_factor)
}

/// Parallel version of [`compute_rhs_swe_2d`] (identical result).
#[cfg(feature = "parallel")]
pub fn compute_rhs_swe_2d_parallel<BC: SWEBoundaryCondition2D + Sync>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
) -> SWESolution2D {
    let mut rhs = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    compute_rhs_swe_2d_parallel_into(q, mesh, ops, geom, config, time, &mut rhs);
    rhs
}

/// Parallel version of [`compute_rhs_swe_2d_into`]: the same element kernel
/// over `par_chunks_mut`, so the result is identical bit for bit.
#[cfg(feature = "parallel")]
pub fn compute_rhs_swe_2d_parallel_into<BC: SWEBoundaryCondition2D + Sync>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
    out: &mut SWESolution2D,
) {
    rhs_parallel(q, mesh, ops, geom, config, time, out, None, None);
}

/// Parallel version of [`compute_rhs_swe_2d_face_mass_into`] (identical
/// result).
#[cfg(feature = "parallel")]
pub fn compute_rhs_swe_2d_parallel_face_mass_into<BC: SWEBoundaryCondition2D + Sync>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
    out: &mut SWESolution2D,
    face_mass: &mut [f64],
) {
    rhs_parallel(q, mesh, ops, geom, config, time, out, Some(face_mass), None);
}

/// Parallel version of [`compute_rhs_swe_2d_mass_fluxes_into`] (identical
/// result).
#[cfg(feature = "parallel")]
pub fn compute_rhs_swe_2d_parallel_mass_fluxes_into<BC: SWEBoundaryCondition2D + Sync>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
    out: &mut SWESolution2D,
    face_mass: &mut [f64],
    subcell_mass: &mut [f64],
) {
    rhs_parallel(
        q,
        mesh,
        ops,
        geom,
        config,
        time,
        out,
        Some(face_mass),
        Some(subcell_mass),
    );
}

/// `subcell_mass` is only taken together with `face_mass`.
#[cfg(feature = "parallel")]
fn rhs_parallel<BC: SWEBoundaryCondition2D + Sync>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
    out: &mut SWESolution2D,
    face_mass: Option<&mut [f64]>,
    subcell_mass: Option<&mut [f64]>,
) {
    use rayon::prelude::*;

    check_rhs_output(out, mesh, ops);
    let kernel = SWE2DRhsKernel::new(q, mesh, ops, geom, config, time);
    let mut faces = FaceFluxGuard::take();
    kernel.face_fluxes_parallel(&mut faces);
    let faces: &[SWEState2D] = &faces;
    let viscous = viscous_all(q, mesh, ops, geom, config, time, true);
    // Chunks of ELEMENT_CHUNK consecutive elements (the last one shorter)
    let batch = |ws: &mut WorkspaceGuard,
                 b: usize,
                 [h, hu, hv]: [&mut [f64]; 3],
                 fm: Option<&mut [f64]>,
                 sm: Option<&mut [f64]>| {
        kernel.element_batch(
            b * ELEMENT_CHUNK,
            ws,
            faces,
            [h, hu, hv],
            fm,
            sm,
            |k, ws, hu, hv| {
                if let Some((term, grads)) = &viscous {
                    term.add(k, grads.gradients(), &mut ws.viscous, hu, hv);
                }
            },
        );
    };
    let rows = ELEMENT_CHUNK * ops.n_nodes;
    let [out_h, out_hu, out_hv] = &mut out.data;
    let batches = out_h
        .par_chunks_mut(rows)
        .zip(out_hu.par_chunks_mut(rows))
        .zip(out_hv.par_chunks_mut(rows))
        .enumerate();
    assert!(
        face_mass.is_some() || subcell_mass.is_none(),
        "subcell mass fluxes come with the face mass fluxes"
    );
    match (face_mass, subcell_mass) {
        (Some(face_mass), subcell_mass) => {
            assert_eq!(
                face_mass.len(),
                face_mass_len(mesh, ops),
                "face mass flux buffer length"
            );
            let batches =
                batches.zip(face_mass.par_chunks_mut(ELEMENT_CHUNK * 4 * ops.n_face_nodes));
            match subcell_mass {
                Some(subcell_mass) => {
                    assert_eq!(
                        subcell_mass.len(),
                        subcell_mass_len(mesh, ops),
                        "subcell mass flux buffer length"
                    );
                    batches
                        .zip(
                            subcell_mass
                                .par_chunks_mut(ELEMENT_CHUNK * subcell_interfaces(ops.n_1d)),
                        )
                        .for_each_init(
                            || WorkspaceGuard::take(ops),
                            |ws, (((b, ((h, hu), hv)), fm), sm)| {
                                batch(ws, b, [h, hu, hv], Some(fm), Some(sm))
                            },
                        );
                }
                None => batches.for_each_init(
                    || WorkspaceGuard::take(ops),
                    |ws, ((b, ((h, hu), hv)), fm)| batch(ws, b, [h, hu, hv], Some(fm), None),
                ),
            }
        }
        (None, _) => batches.for_each_init(
            || WorkspaceGuard::take(ops),
            |ws, (b, ((h, hu), hv))| batch(ws, b, [h, hu, hv], None, None),
        ),
    }
}

/// Per-element continuation of [`compute_rhs_swe_2d_subset_then`]: element
/// k's rows of `out` (holding its RHS) and of the extra buffer, if any.
pub type ElementRhsThen<'a> = &'a (dyn Fn(usize, [&mut [f64]; 3], Option<[&mut [f64]; 3]>) + Sync);

/// One element's mass fluxes of an RHS evaluation, in the layouts of
/// [`compute_rhs_swe_2d_mass_fluxes_into`] restricted to the element: its
/// `4 · n_face_nodes` face values and its subcell interfaces (NaN unless its
/// volume term was the subcell update).
#[derive(Clone, Copy, Debug)]
pub struct ElementMassFluxes<'a> {
    pub face: &'a [f64],
    pub subcell: &'a [f64],
}

/// Per-element continuation of
/// [`compute_rhs_swe_2d_subset_mass_fluxes_then`]: element k's rows of
/// `out` (holding its RHS) and its mass fluxes.
pub type ElementRhsMassThen<'a> =
    &'a (dyn Fn(usize, [&mut [f64]; 3], ElementMassFluxes<'_>) + Sync);

/// Most face plus subcell mass flux values of one element (order 8).
const MAX_ELEMENT_MASS_FLUXES: usize = 4 * 9 + 2 * 9 * 8;

/// Subsets smaller than this run serially: a parallel dispatch costs more
/// than their work (local time stepping has many tiny subsets).
#[cfg(feature = "parallel")]
const SUBSET_PARALLEL_MIN: usize = 64;

/// The distinct edges of a list of elements, in order of first appearance,
/// without sorting: an edge is taken when its stamp is not the current pass.
#[derive(Default)]
struct SubsetEdges {
    list: Vec<u32>,
    stamp: Vec<u32>,
    pass: u32,
}

impl SubsetEdges {
    fn collect(&mut self, mesh: &Mesh2D, elements: &[u32]) {
        if self.stamp.len() != mesh.n_edges || self.pass == u32::MAX {
            self.stamp.clear();
            self.stamp.resize(mesh.n_edges, 0);
            self.pass = 0;
        }
        self.pass += 1;
        self.list.clear();
        for &k in elements {
            for e in mesh.element_edges[k as usize] {
                if self.stamp[e] != self.pass {
                    self.stamp[e] = self.pass;
                    self.list.push(e as u32);
                }
            }
        }
    }
}

thread_local! {
    /// Edge scratch of the subset RHS on this thread (reused, not reallocated)
    static SUBSET_EDGES: RefCell<SubsetEdges> = RefCell::new(SubsetEdges::default());
}

/// [`compute_rhs_swe_2d_into`] for the elements with `time_of(k) = Some(t)`
/// only, each at its own time `t`, for local time stepping; the other rows
/// of `out` are left as they are. A wrapper of
/// [`compute_rhs_swe_2d_subset_then`] (it collects the element list, a pass
/// over all elements). With horizontal viscosity, unselected face
/// neighbours take the time of the first selected element.
pub fn compute_rhs_swe_2d_where_into<BC: SWEBoundaryCondition2D>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time_of: &(dyn Fn(usize) -> Option<f64> + Sync),
    out: &mut SWESolution2D,
) {
    let elements: Vec<u32> = (0..mesh.n_elements)
        .filter(|&k| time_of(k).is_some())
        .map(|k| k as u32)
        .collect();
    let fallback = elements
        .first()
        .and_then(|&k| time_of(k as usize))
        .unwrap_or(0.0);
    compute_rhs_swe_2d_subset_then(
        q,
        mesh,
        ops,
        geom,
        config,
        &elements,
        &|k| time_of(k).unwrap_or(fallback),
        out,
        None,
        &|_, _, _| {},
    );
}

/// [`compute_rhs_swe_2d_into`] for the listed `elements` only (distinct),
/// element k at time `time_of(k)`, then `then(k, out rows, extra rows)` for
/// each of them in the same pass, while its rows are in cache: local time
/// stepping turns the RHS into the next stage value there (combination,
/// damping, limiting). The other rows are left as they are.
///
/// A listed element gets bit for bit the RHS that the full evaluation at its
/// time gives it: the split-form interior-face fluxes come from the same
/// face kernel, run on the faces of the listed elements only. The work is
/// proportional to the list, not to the mesh, so the many small subsets of
/// a multirate step cost only their elements. Parallel with `parallel` from
/// [`SUBSET_PARALLEL_MIN`] elements; allocation-free after the first calls
/// on a thread.
///
/// With horizontal viscosity the RHS of an element also reads its face
/// neighbours' velocity gradients (the state of the elements at their far
/// corners too). They are
/// computed here for the listed elements and their neighbours, with each
/// neighbour's boundary values at `time_of(neighbour)`, so `time_of` must
/// be defined for those too. The viscous flux through a face is then the
/// same whichever side evaluates it.
///
/// # Panics
/// If `extra` has another shape than `out`.
pub fn compute_rhs_swe_2d_subset_then<BC: SWEBoundaryCondition2D>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    elements: &[u32],
    time_of: &(dyn Fn(usize) -> f64 + Sync),
    out: &mut SWESolution2D,
    extra: Option<&mut SWESolution2D>,
    then: ElementRhsThen,
) {
    subset_then(
        q,
        mesh,
        ops,
        geom,
        config,
        elements,
        time_of,
        out,
        extra,
        false,
        &|k, rows, extra, _| then(k, rows, extra),
    );
}

/// [`compute_rhs_swe_2d_subset_then`] that also hands each element's mass
/// fluxes to `then` (bit for bit what
/// [`compute_rhs_swe_2d_mass_fluxes_into`] writes for it): the mode
/// splitter's barotropic pass with local time stepping sums them into
/// DU_avg2.
pub fn compute_rhs_swe_2d_subset_mass_fluxes_then<BC: SWEBoundaryCondition2D>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    elements: &[u32],
    time_of: &(dyn Fn(usize) -> f64 + Sync),
    out: &mut SWESolution2D,
    then: ElementRhsMassThen,
) {
    subset_then(
        q,
        mesh,
        ops,
        geom,
        config,
        elements,
        time_of,
        out,
        None,
        true,
        &|k, rows, _, mass| then(k, rows, mass.expect("mass fluxes were asked for")),
    );
}

/// The subset RHS of [`compute_rhs_swe_2d_subset_then`], with the elements'
/// mass fluxes if `mass_fluxes`.
#[allow(clippy::type_complexity)]
fn subset_then<BC: SWEBoundaryCondition2D>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    elements: &[u32],
    time_of: &(dyn Fn(usize) -> f64 + Sync),
    out: &mut SWESolution2D,
    extra: Option<&mut SWESolution2D>,
    mass_fluxes: bool,
    then: &(
         dyn Fn(usize, [&mut [f64]; 3], Option<[&mut [f64]; 3]>, Option<ElementMassFluxes<'_>>)
             + Sync
     ),
) {
    check_rhs_output(out, mesh, ops);
    let (n_face_mass, n_subcell_mass) = (4 * ops.n_face_nodes, subcell_interfaces(ops.n_1d));
    assert!(
        n_face_mass + n_subcell_mass <= MAX_ELEMENT_MASS_FLUXES,
        "order {} has more mass fluxes per element than the subset RHS holds",
        ops.order
    );
    if let Some(extra) = &extra {
        check_rhs_output(extra, mesh, ops);
    }
    debug_assert!(all_distinct(elements, mesh.n_elements), "repeated elements");
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;
    #[cfg(feature = "parallel")]
    let parallel = elements.len() >= SUBSET_PARALLEL_MIN;

    // Interior-face fluxes of the faces of the listed elements (they do not
    // depend on time)
    let faces_kernel = SWE2DRhsKernel::new(q, mesh, ops, geom, config, 0.0);
    let mut faces = FaceFluxGuard::take();
    match &faces_kernel.split_form {
        Some(split_form) => {
            faces.resize(split_form.face_buffer_len(), SWEState2D::zero());
            let mut scratch = SUBSET_EDGES.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
            scratch.collect(mesh, elements);
            let edges = &scratch.list;
            let per_edge = 2 * ops.n_face_nodes;
            let slots = DisjointChunks::new(&mut faces[..], per_edge);
            let batch = |batch: &[u32]| {
                let mut edges = [0; VOLUME_BATCH];
                for (edge, &e) in edges.iter_mut().zip(batch) {
                    *edge = e as usize;
                }
                let mut edge_slots = edges.map(|_| -> &mut [SWEState2D] { &mut [] });
                for (slot, &e) in edge_slots.iter_mut().zip(batch) {
                    // SAFETY: `edges` is distinct, so each edge's slots once
                    *slot = unsafe { slots.chunk(e as usize) };
                }
                let n = batch.len();
                split_form.edge_fluxes_batch(&edges[..n], &mut edge_slots[..n]);
            };
            #[cfg(feature = "parallel")]
            if parallel {
                edges.par_chunks(VOLUME_BATCH).for_each(batch);
            } else {
                edges.chunks(VOLUME_BATCH).for_each(batch);
            }
            #[cfg(not(feature = "parallel"))]
            edges.chunks(VOLUME_BATCH).for_each(batch);
            SUBSET_EDGES.with(|cell| *cell.borrow_mut() = scratch);
        }
        None => faces.clear(),
    }
    let faces: &[SWEState2D] = &faces;

    // Viscous gradients of the listed elements and their face neighbours,
    // each at its own time
    let n = ops.n_nodes;
    let viscous = ViscousTerm::new(q, mesh, ops, geom, config).map(|term| {
        let mut guard = ViscousGuard::take(mesh.n_elements * n);
        let ViscousWorkspace {
            grad_u,
            grad_v,
            neighbourhood,
            ..
        } = {
            guard.0.collect_neighbourhood(mesh, elements);
            &mut guard.0
        };
        let rows = [
            DisjointChunks::new(grad_u, n),
            DisjointChunks::new(grad_v, n),
        ];
        let gradients = |ws: &mut WorkspaceGuard, &k: &u32| {
            let k = k as usize;
            // SAFETY: the neighbourhood is distinct, so each element's rows once
            let [gu, gv] = rows.each_ref().map(|rows| unsafe { rows.chunk(k) });
            term.gradients(k, time_of(k), &mut ws.viscous, gu, gv);
        };
        #[cfg(feature = "parallel")]
        if neighbourhood.len() >= SUBSET_PARALLEL_MIN {
            neighbourhood
                .par_iter()
                .for_each_init(|| WorkspaceGuard::take(ops), gradients);
            return (term, guard);
        }
        let mut ws = WorkspaceGuard::take(ops);
        for k in neighbourhood.iter() {
            gradients(&mut ws, k);
        }
        (term, guard)
    });

    let [out_h, out_hu, out_hv] = &mut out.data;
    let out_rows = [
        DisjointChunks::new(out_h, n),
        DisjointChunks::new(out_hu, n),
        DisjointChunks::new(out_hv, n),
    ];
    let extra_rows = extra.map(|extra| {
        let [h, hu, hv] = &mut extra.data;
        [
            DisjointChunks::new(h, n),
            DisjointChunks::new(hu, n),
            DisjointChunks::new(hv, n),
        ]
    });
    let element = |ws: &mut WorkspaceGuard, k: usize, volume: Option<usize>| {
        // SAFETY: the elements are distinct, so each element's rows once
        let [h, hu, hv] = out_rows.each_ref().map(|rows| unsafe { rows.chunk(k) });
        let extra = extra_rows
            .as_ref()
            .map(|x| x.each_ref().map(|rows| unsafe { rows.chunk(k) }));
        let mut mass = [0.0; MAX_ELEMENT_MASS_FLUXES];
        let (face, subcell) = mass[..n_face_mass + n_subcell_mass].split_at_mut(n_face_mass);
        SWE2DRhsKernel::new(q, mesh, ops, geom, config, time_of(k)).element(
            k,
            ws,
            faces,
            [&mut *h, &mut *hu, &mut *hv],
            mass_fluxes.then_some(&mut *face),
            mass_fluxes.then_some(&mut *subcell),
            volume,
        );
        if let Some((term, grads)) = &viscous {
            term.add(k, grads.gradients(), &mut ws.viscous, hu, hv);
        }
        let mass = mass_fluxes.then_some(ElementMassFluxes { face, subcell });
        then(k, [h, hu, hv], extra, mass);
    };
    // The volume terms (independent of time) ELEMENT_CHUNK elements at a time
    let chunk = |ws: &mut WorkspaceGuard, ks: &[u32]| {
        let mut slots = [0; ELEMENT_CHUNK];
        for (slot, &k) in slots.iter_mut().zip(ks) {
            *slot = k as usize;
        }
        let volumes = faces_kernel.volume_chunk(&slots[..ks.len()], ws);
        for (j, &k) in slots[..ks.len()].iter().enumerate() {
            element(ws, k, ((volumes >> j) & 1 == 1).then_some(j));
        }
    };
    #[cfg(feature = "parallel")]
    if parallel {
        elements
            .par_chunks(ELEMENT_CHUNK)
            .for_each_init(|| WorkspaceGuard::take(ops), chunk);
        return;
    }
    let mut ws = WorkspaceGuard::take(ops);
    for ks in elements.chunks(ELEMENT_CHUNK) {
        chunk(&mut ws, ks);
    }
}

/// Largest stable time step of every element, into `out`
/// (`mesh.n_elements` values), for local time stepping.
///
/// The bound of [`compute_dt_swe_2d`] with the element's own metric, over its
/// own nodes and the neighbours' nodes on its faces. The interface flux sees
/// the wave speeds of both sides: a nearly dry element next to deep water, or
/// a small element next to a fast one, must not step with its own speeds
/// alone. Each neighbour face node is taken with the metric at the element's
/// coincident node, which is the speed the flux dissipation and the Zhang–Shu
/// bound of the element's mean see there. (Every neighbour node with the
/// element's largest metric, as before, took 37 % more steps with one level
/// on the Mausund coastline mesh.) The minimum is at most
/// [`compute_dt_swe_2d`] at the same CFL.
///
/// `cfl` is the linear stability bound. With `positivity` (wet/dry runs) the
/// element's CFL is further capped by its Zhang–Shu bound,
/// [`PositivityBound::element_cfl`].
///
/// `f64::INFINITY` where everything the element sees is dry or at rest in
/// zero depth. Parallel with `parallel`.
pub fn element_dt_swe_2d(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    equation: &ShallowWater2D,
    order: usize,
    cfl: f64,
    positivity: Option<PositivityBound>,
    out: &mut [f64],
) {
    assert_eq!(out.len(), mesh.n_elements, "one time step per element");
    let element_dt = |(k, dt): (usize, &mut f64)| {
        *dt = element_dt(q, mesh, ops, geom, equation, order, cfl, positivity, k);
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_iter_mut().enumerate().for_each(element_dt);
    }
    #[cfg(not(feature = "parallel"))]
    out.iter_mut().enumerate().for_each(element_dt);
}

/// The smallest [`element_dt_swe_2d`]: the global time step of a wet/dry run
/// whose positivity bound is relaxed per element. Parallel with `parallel`;
/// the same result either way.
pub fn min_element_dt_swe_2d(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    equation: &ShallowWater2D,
    order: usize,
    cfl: f64,
    positivity: Option<PositivityBound>,
) -> f64 {
    let element_dt = |k| element_dt(q, mesh, ops, geom, equation, order, cfl, positivity, k);
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..mesh.n_elements)
            .into_par_iter()
            .map(element_dt)
            .reduce(|| f64::INFINITY, f64::min)
    }
    #[cfg(not(feature = "parallel"))]
    (0..mesh.n_elements)
        .map(element_dt)
        .fold(f64::INFINITY, f64::min)
}

/// One element's [`element_dt_swe_2d`].
#[inline]
fn element_dt(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    equation: &ShallowWater2D,
    order: usize,
    cfl: f64,
    positivity: Option<PositivityBound>,
    k: usize,
) -> f64 {
    let h_min = equation.h_min.meters();
    let norm = |g: (f64, f64)| (g.0 * g.0 + g.1 * g.1).sqrt();
    let k_idx = ElementIndex::new(k);
    let mut rate = element_max_reference_rate(q, geom, equation, k_idx);
    let n_face = ops.n_face_nodes;
    for face in 0..4 {
        let Some(nb) = mesh.neighbor(k_idx, face) else {
            continue;
        };
        let nb_k = ElementIndex::new(nb.element);
        // Neighbours list the shared face nodes in reverse order
        for (fi, &i) in ops.face_nodes[face].iter().enumerate() {
            let j = ops.face_nodes[nb.face][n_face - 1 - fi];
            let state = q.get_state(nb_k, j);
            if state.h <= h_min {
                continue;
            }
            let (grad_r, grad_s) = (geom.grad_r(k, i), geom.grad_s(k, i));
            let metric = (grad_r, grad_s, norm(grad_r), norm(grad_s));
            rate = rate.max(node_reference_rate(equation, &state, metric));
        }
    }
    let cfl = positivity.map_or(cfl, |bound| {
        cfl.min(bound.element_cfl(q.element_h(k_idx), geom, ops.n_1d, k))
    });
    dt_from_reference_rate(rate, order, cfl)
}

/// The Zhang–Shu positivity bound of a wet/dry run, relaxed per element by
/// how its water is distributed.
///
/// The bound [`positivity_cfl_swe_2d`] protects the worst case: all of an
/// element's water on its boundary nodes, flowing out through its faces.
/// Zhang & Shu (2010) write the forward-Euler update of the cell mean as a
/// convex combination of the interior nodes' depths and first-order updates
/// of the boundary nodes, each holding a share of its node's mass `m_i =
/// w_i J_i h_i` (the corner nodes' split between their two faces). At the
/// bound every such update is non-negative. At ρ times the bound it is at
/// least `(1 − ρ)` times its share (the flux part scales with Δt and was at
/// most the share), so
///
/// ```text
/// |K| h̄ⁿ⁺¹ ≥ M − ρ M_∂,   M = Σ_i m_i,   M_∂ = Σ_{i on ∂K} m_i,
/// ```
///
/// and the mean stays non-negative for ρ ≤ M/M_∂. For uniform depth that is
/// the weight ratio W/W_∂ (W = Σ w_i J_i): 1, 1.8, 3.3, 5.3 for N = 1–4 on a
/// parallelogram (the interior nodes hold (1 − 2ŵ₀)² of it), so a well-filled
/// element can step 1.8× (P2) to 3.3× (P3) past the bound. The relaxed CFL is
/// capped at [`LINEAR_CFL_SAFETY`] times the linear stability limit of the
/// element's volume term: [`linear_cfl_swe_2d`], 1.5–2.5× the bound, for
/// the flux-differencing term of fully wet elements, and
/// [`linear_cfl_subcells_swe_2d`], 1.0–1.7× the bound, for the `WetDry`
/// subcells of elements with a node shallower than `h_dry`. The positivity
/// bound had kept every wet/dry run inside both, whatever CFL was asked for.
///
/// An element switches to the subcells within a step once a stage takes a
/// node below `h_dry`, and then runs them at the step chosen at its start.
/// So every element with a node at or below [`SUBCELL_CAP_DEPTH_FACTOR`]
/// times `h_dry` (10 cm for the 1 mm default) takes the subcell cap: far more
/// than a node falls in one step as the tide drains a shore (a fraction of a
/// mm), and free, since such elements hardly ever set the step (on the
/// Mausund coastline mesh the global step is unchanged, and the ideal local
/// time stepping work rises by 0.1 %).
///
/// The stages of an SSP step start from later states than the one ρ is
/// computed from, so ρ keeps a margin: [`POSITIVITY_RELAXATION_SAFETY`] times
/// `min(M/M_∂, W/W_∂)`. The cap at the weight ratio keeps an element whose
/// boundary nodes are nearly empty (a shoreline with its dry nodes on the
/// faces) from stepping at a ρ that a stage filling them would void.
///
/// The elements that set the global step of a coastal run are steep
/// shorelines: a dry node beside 25–90 m of water (on the Mausund coastline
/// mesh), whose deep nodes set the element's rate. Thin films on tidal flats
/// are relaxed too but never bind: their wave speeds are small.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositivityBound {
    /// The forward-Euler bound for elements that may run dry:
    /// [`positivity_cfl_swe_2d`] times the integrator's SSP coefficient.
    pub cfl: f64,
    /// Largest relaxed CFL where every node is deeper than `h_dry`:
    /// [`LINEAR_CFL_SAFETY`] times the linear stability limit (at least
    /// `cfl`; `cfl` itself, no relaxation, where the limit is not known).
    pub max_cfl: f64,
    /// Largest relaxed CFL of an element with a node at or below
    /// [`SUBCELL_CAP_DEPTH_FACTOR`] times `h_dry`, from the subcells' limit in
    /// the same way.
    pub max_cfl_subcells: f64,
    /// The `WetDry` subcell threshold: elements with a node shallower than
    /// this run the subcells.
    pub h_dry: f64,
}

/// Elements with a node at or below this multiple of the dry threshold take
/// [`PositivityBound`]'s subcell cap: they may switch to the subcells during
/// the step.
pub const SUBCELL_CAP_DEPTH_FACTOR: f64 = 100.0;

/// Margin of [`PositivityBound`]'s relaxation against the change of the
/// water distribution between the stage values of one step.
pub const POSITIVITY_RELAXATION_SAFETY: f64 = 0.9;

/// Margin of [`PositivityBound`]'s relaxed CFL below the linear stability
/// limit [`linear_cfl_swe_2d`] (measured on a few mesh families only).
pub const LINEAR_CFL_SAFETY: f64 = 0.9;

impl PositivityBound {
    /// The bound at order `order` for the SSP scheme `scheme` (`None`: an
    /// SSP coefficient of 1 and no relaxation), with the subcell cap where a
    /// node is at or below [`SUBCELL_CAP_DEPTH_FACTOR`] times `h_dry`.
    pub fn new(order: usize, scheme: Option<SspScheme>, h_dry: f64) -> Self {
        let ssp_coefficient = scheme.map_or(1.0, SspScheme::ssp_coefficient);
        let cfl = ssp_coefficient * positivity_cfl_swe_2d(order);
        let cap =
            |limit: Option<f64>| limit.map_or(cfl, |limit| cfl.max(LINEAR_CFL_SAFETY * limit));
        Self {
            cfl,
            max_cfl: cap(scheme.and_then(|scheme| linear_cfl_swe_2d(order, scheme))),
            max_cfl_subcells: cap(
                scheme.and_then(|scheme| linear_cfl_subcells_swe_2d(order, scheme))
            ),
            h_dry,
        }
    }

    /// The CFL of element `k` with nodal depths `h` (`n1` GLL nodes per
    /// direction): `cfl` times the relaxation factor ρ ≥ 1, at most
    /// `max_cfl`, or `max_cfl_subcells` if a node is at or below
    /// [`SUBCELL_CAP_DEPTH_FACTOR`] times `h_dry`.
    #[inline]
    pub fn element_cfl(&self, h: &[f64], geom: &GeometricFactors2D, n1: usize, k: usize) -> f64 {
        let mass = &geom.mass[geom.node_index(k, 0)..][..h.len()];
        let shallow = SUBCELL_CAP_DEPTH_FACTOR * self.h_dry;
        let (rho, subcells) = positivity_relaxation(h, mass, n1, shallow);
        let cap = if subcells {
            self.max_cfl_subcells
        } else {
            self.max_cfl
        };
        (self.cfl * rho).min(cap)
    }
}

/// The relaxation factor ρ of [`PositivityBound`] for nodal depths `h` with
/// nodal masses `mass` (GLL weight × J), `n1` nodes per direction (node
/// `i = j·n1 + a`), and whether a node is at or below `shallow`.
#[inline]
fn positivity_relaxation(h: &[f64], mass: &[f64], n1: usize, shallow: f64) -> (f64, bool) {
    let last = n1 - 1;
    let (mut water, mut water_boundary) = (0.0, 0.0);
    let (mut weight, mut weight_boundary) = (0.0, 0.0);
    let mut is_shallow = false;
    for (i, (&h, &m)) in h.iter().zip(mass).enumerate() {
        is_shallow |= h <= shallow;
        // Negative round-off depths count as empty
        let water_i = m * h.max(0.0);
        water += water_i;
        weight += m;
        let (a, j) = (i % n1, i / n1);
        if a == 0 || a == last || j == 0 || j == last {
            water_boundary += water_i;
            weight_boundary += m;
        }
    }
    // No water on the boundary nodes: nothing can leave in the first stage,
    // and the weight ratio caps ρ (`f64::min` drops the NaN of an empty
    // element)
    let ratio = (water / water_boundary).min(weight / weight_boundary);
    ((POSITIVITY_RELAXATION_SAFETY * ratio).max(1.0), is_shallow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::Reflective2D;
    use crate::source::CoriolisSource2D;

    const G: f64 = 10.0;

    /// The face mass flux output (mode splitting, TODO P4.1) leaves the RHS
    /// unchanged bit for bit, is the same serial and parallel, and balances
    /// every element, `∫_K dh/dt = −∮_K F*_h`, for every formulation, with
    /// a dry region (the `WetDry` subcells included).
    #[test]
    fn face_mass_flux_balances_each_element() {
        let (mesh, ops, geom) = create_test_setup(2);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _| -1.0 + 2.0 * x);
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let eta = 0.2 + 0.05 * (2.0 * std::f64::consts::PI * x).sin();
                let h = (eta - bathymetry.get(k, i)).max(0.0);
                let (u, v) = (
                    0.2 * (std::f64::consts::PI * y).cos(),
                    0.1 * (std::f64::consts::PI * x).sin(),
                );
                q.set_state(k, i, SWEState2D::new(h, h * u, h * v));
            }
        }
        let n_face = ops.n_face_nodes;

        for formulation in [
            SWEFormulation2D::Standard,
            SWEFormulation2D::EntropyStable,
            SWEFormulation2D::WetDry,
        ] {
            let config = SWE2DRhsConfig::new(&equation, &bc)
                .with_coriolis(false)
                .with_formulation(formulation)
                .with_bathymetry(&bathymetry)
                .with_flux_type(SWEFluxType2D::HLL);
            let plain = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
            let mut rhs = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
            let mut face_mass = vec![f64::NAN; face_mass_len(&mesh, &ops)];
            compute_rhs_swe_2d_face_mass_into(
                &q,
                &mesh,
                &ops,
                &geom,
                &config,
                0.0,
                &mut rhs,
                &mut face_mass,
            );
            assert_eq!(
                rhs.data, plain.data,
                "{formulation:?}: the face output changed the RHS"
            );
            assert!(
                face_mass.iter().all(|f| f.is_finite()),
                "{formulation:?}: face slots not all written"
            );

            #[cfg(feature = "parallel")]
            {
                let mut rhs_par = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
                let mut face_par = vec![f64::NAN; face_mass.len()];
                compute_rhs_swe_2d_parallel_face_mass_into(
                    &q,
                    &mesh,
                    &ops,
                    &geom,
                    &config,
                    0.0,
                    &mut rhs_par,
                    &mut face_par,
                );
                assert_eq!(rhs_par.data, rhs.data, "{formulation:?}: parallel RHS");
                assert_eq!(face_par, face_mass, "{formulation:?}: parallel face fluxes");
            }

            let mut max_inflow: f64 = 0.0;
            for k in 0..mesh.n_elements {
                let dh = &rhs.data[0][k * ops.n_nodes..(k + 1) * ops.n_nodes];
                let volume = geom.integrate_element(k, dh);
                let outflow: f64 = (0..4)
                    .map(|face| {
                        let f = &face_mass[(k * 4 + face) * n_face..][..n_face];
                        f.iter()
                            .zip(&ops.weights_1d)
                            .enumerate()
                            .map(|(fi, (f, w))| geom.surface_jacobian(k, face, fi) * f * w)
                            .sum::<f64>()
                    })
                    .sum();
                max_inflow = max_inflow.max(outflow.abs());
                assert!(
                    (volume + outflow).abs() < 1e-12 * max_inflow.max(1e-3),
                    "{formulation:?}, element {k}: ∫dh/dt = {volume:.3e}, ∮F* = {outflow:.3e}"
                );
            }
        }
    }

    /// The whole-mesh BR1 viscosity that the per-element viscous term
    /// replaced (whole-mesh velocity, gradient, coefficient and diffusion
    /// arrays), added to `rhs`: on the elements in the viscous domain, with
    /// a zero-gradient, no-flux edge toward the others.
    fn reference_viscosity<BC: SWEBoundaryCondition2D>(
        rhs: &mut SWESolution2D,
        q: &SWESolution2D,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        config: &SWE2DRhsConfig<BC>,
        time: f64,
    ) {
        use super::super::diffusion_2d::{
            DiffusionScratch, br1_diffusion_element, compute_br1_gradient_2d,
        };
        use super::super::swe_2d_viscosity::in_viscous_domain;

        let visc = config.viscosity.expect("viscosity");
        let subcell_depth = match config.formulation {
            SWEFormulation2D::WetDry => config.subcell_depth.unwrap_or(config.h_dry),
            _ => 0.0,
        };
        let in_water = |k: ElementIndex| in_viscous_domain(q, visc, subcell_depth, k);
        let n_nodes = ops.n_nodes;
        let total_nodes = mesh.n_elements * n_nodes;
        let mut u = vec![0.0; total_nodes];
        let mut v = vec![0.0; total_nodes];
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..n_nodes {
                let flat = k.as_usize() * n_nodes + i;
                let state = q.get_state(k, i);
                if state.h > visc.h_min {
                    let h_safe = state.h.max(visc.h_min);
                    u[flat] = state.hu / h_safe;
                    v[flat] = state.hv / h_safe;
                }
            }
        }
        let boundary = |component: usize| {
            move |k: ElementIndex, face: usize, fi: usize, node: usize, interior: f64| {
                if mesh.neighbor(k, face).is_some() {
                    return interior;
                }
                let normal = geom.normal(k.as_usize(), face, fi);
                let ghost =
                    boundary_state(q, mesh, ops, config, time, k, face, node, normal).state();
                if ghost.h <= visc.h_min {
                    return 0.0;
                }
                let h_safe = ghost.h.max(visc.h_min);
                if component == 0 {
                    ghost.hu / h_safe
                } else {
                    ghost.hv / h_safe
                }
            }
        };
        let grad_u = compute_br1_gradient_2d(&u, mesh, ops, geom, boundary(0), in_water);
        let grad_v = compute_br1_gradient_2d(&v, mesh, ops, geom, boundary(1), in_water);

        let mut coeff = vec![0.0; total_nodes];
        for k in ElementIndex::iter(mesh.n_elements).filter(|&k| in_water(k)) {
            let delta = HorizontalViscosity2D::filter_width(geom.area[k.as_usize()], ops.order);
            for i in 0..n_nodes {
                let flat = k.as_usize() * n_nodes + i;
                let state = q.get_state(k, i);
                if state.h > visc.h_min {
                    let nu = visc.compute_viscosity(
                        grad_u[flat].dx,
                        grad_u[flat].dy,
                        grad_v[flat].dx,
                        grad_v[flat].dy,
                        delta,
                    );
                    coeff[flat] = nu * state.h;
                }
            }
        }
        // Both components at once: a wall (every face of `Reflective2D`)
        // mirrors the velocity and passes the normal stress only
        let (mut diff_u, mut diff_v) = (vec![0.0; total_nodes], vec![0.0; total_nodes]);
        let mut scratch = DiffusionScratch::<2>::new(n_nodes);
        for k in ElementIndex::iter(mesh.n_elements).filter(|&k| in_water(k)) {
            let rows = k.as_usize() * n_nodes..(k.as_usize() + 1) * n_nodes;
            br1_diffusion_element(
                k,
                mesh,
                ops,
                geom,
                |j, node| {
                    let flat = j.as_usize() * n_nodes + node;
                    let c = coeff[flat].max(0.0);
                    [
                        (c * grad_u[flat].dx, c * grad_u[flat].dy),
                        (c * grad_v[flat].dx, c * grad_v[flat].dy),
                    ]
                },
                |k, face, _, (nx, ny), [fu, fv]| {
                    if mesh.neighbor(k, face).is_some() {
                        return [0.0; 2];
                    }
                    let normal = nx * fu + ny * fv;
                    [nx * normal, ny * normal]
                },
                in_water,
                &mut scratch,
                [&mut diff_u[rows.clone()], &mut diff_v[rows]],
            );
        }
        for flat in 0..total_nodes {
            rhs.data[1][flat] += diff_u[flat];
            rhs.data[2][flat] += diff_v[flat];
        }
    }

    /// The viscous term keeps off the shore: on a beach with an along-shore
    /// shear current, it changes nothing in the shoreline (subcell) and dry
    /// elements, dissipates kinetic energy, and leaves the mass alone.
    /// Before, the divergence of the wet nodes' stress put momentum into the
    /// dry nodes of the shoreline elements, and at ν = 10 m²/s that pushed
    /// films 4 m up a cliff shore at Frøya (TODO F.1).
    #[test]
    fn viscosity_keeps_off_the_shore() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 800.0, 0.0, 400.0, 8, 4);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        // Shoreline at x = 500 m: the bed rises 1 m per 100 m
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _| -5.0 + 0.01 * x);
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let h = (-bathymetry.get(k, i)).max(0.0);
                let (u, v) = (0.05 * (y / 60.0).sin(), 0.3 * (1.0 - x / 500.0));
                q.set_state(k, i, SWEState2D::new(h, h * u, h * v));
            }
        }
        let visc = HorizontalViscosity2D::constant(10.0);
        let inviscid = || {
            SWE2DRhsConfig::new(&equation, &bc)
                .with_coriolis(false)
                .with_formulation(SWEFormulation2D::WetDry)
                .with_bathymetry(&bathymetry)
        };
        let without = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &inviscid(), 0.0);
        let config = inviscid().with_viscosity(&visc);
        let with = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
        assert_eq!(with.data[0], without.data[0], "no mass from viscosity");

        let (mut shore, mut water, mut work) = (0, 0, 0.0);
        for k in ElementIndex::iter(mesh.n_elements) {
            let wet = q.element_h(k).iter().all(|&h| h >= config.h_dry);
            let mut changed = false;
            for i in 0..ops.n_nodes {
                let flat = k.as_usize() * ops.n_nodes + i;
                let (dhu, dhv) = (
                    with.data[1][flat] - without.data[1][flat],
                    with.data[2][flat] - without.data[2][flat],
                );
                changed |= dhu != 0.0 || dhv != 0.0;
                let state = q.get_state(k, i);
                if state.h > 0.0 {
                    let w = ops.weights[i] * geom.jacobian(k.as_usize(), i);
                    work += w * (state.hu * dhu + state.hv * dhv) / state.h;
                }
            }
            if wet {
                water += usize::from(changed);
            } else {
                shore += 1;
                assert!(!changed, "element {k:?} has a node shallower than h_dry");
            }
        }
        assert!(
            shore >= 8,
            "test regime: {shore} shoreline and dry elements"
        );
        assert!(
            water >= 16,
            "the viscous term must act in the water: {water}"
        );
        assert!(work < 0.0, "viscosity dissipates: {work:.3e}");
    }

    /// The per-element viscous term (whole mesh serial and parallel, and the
    /// element-subset RHS of local time stepping) gives bit for bit the
    /// whole-mesh BR1 viscosity it replaced: on curved-sided quads, with
    /// walls, a dry region, constant and Smagorinsky viscosity.
    #[test]
    fn viscous_term_matches_the_whole_mesh_reference() {
        let mut mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 5, 4);
        for v in &mut mesh.vertices {
            let [x, y] = *v;
            let bump = x * (1.0 - x) * y * (1.0 - y);
            *v = [
                x + 0.4 * bump * (7.0 * y).sin(),
                y + 0.5 * bump * (5.0 * x).cos(),
            ];
        }
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
            -0.3 + 0.6 * x + 0.1 * (3.0 * y).sin()
        });
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let h = (0.02 * (4.0 * x).cos() - bathymetry.get(k, i)).max(0.0);
                let (u, v) = (0.3 * (3.0 * y).sin() + 0.1 * x, 0.2 * (2.0 * x).cos() * y);
                q.set_state(k, i, SWEState2D::new(h, h * u, h * v));
            }
        }
        assert!(q.h_data().contains(&0.0), "part of the domain must be dry");
        let t = 3.0;

        for visc in [
            HorizontalViscosity2D::constant(0.7),
            HorizontalViscosity2D::smagorinsky(0.2),
            HorizontalViscosity2D::smagorinsky(0.2).with_background(0.7),
        ] {
            let inviscid = || {
                SWE2DRhsConfig::new(&equation, &bc)
                    .with_coriolis(false)
                    .with_formulation(SWEFormulation2D::WetDry)
                    .with_bathymetry(&bathymetry)
            };
            let without = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &inviscid(), t);
            let config = inviscid().with_viscosity(&visc);
            let mut expected = without.clone();
            reference_viscosity(&mut expected, &q, &mesh, &ops, &geom, &config, t);

            let serial = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, t);
            assert_eq!(serial.data, expected.data, "{:?}: serial", visc);
            assert_ne!(
                serial.data[1], without.data[1],
                "the viscous term must do something"
            );
            #[cfg(feature = "parallel")]
            {
                let parallel = compute_rhs_swe_2d_parallel(&q, &mesh, &ops, &geom, &config, t);
                assert_eq!(parallel.data, expected.data, "{:?}: parallel", visc);
            }

            let mut subset = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
            let selected = |k: usize| k % 3 != 1;
            compute_rhs_swe_2d_where_into(
                &q,
                &mesh,
                &ops,
                &geom,
                &config,
                &|k| selected(k).then_some(t),
                &mut subset,
            );
            for k in ElementIndex::iter(mesh.n_elements).filter(|k| selected(k.as_usize())) {
                for var in 0..3 {
                    assert_eq!(
                        subset.element_var(k, var),
                        expected.element_var(k, var),
                        "{:?}: subset, element {k:?}, variable {var}",
                        visc
                    );
                }
            }
        }
    }

    /// The subcell mass flux output (3D wetting and drying, TODO P1.3):
    /// in every `WetDry` element that takes the subcells, the subcell and face
    /// mass fluxes give the mass tendency node by node through
    /// `subcell_divergence_element`, the operator the 3D layers use; every
    /// other element's slots are NaN. On rectangles and on general
    /// quadrilaterals (the telescoped interface metrics), serial and
    /// parallel, and the RHS is unchanged.
    #[test]
    fn subcell_mass_fluxes_give_the_wet_dry_mass_tendency_node_by_node() {
        use crate::solver::rhs::subcell_divergence_element;
        use crate::solver::rhs::subcells::subcell_interfaces;
        let tau = 2.0 * std::f64::consts::PI;
        let distorted = || {
            let mut mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 4, 4);
            for v in &mut mesh.vertices {
                let [x, y] = *v;
                let bump = (tau * x).sin() * (tau * y).sin();
                *v = [x + 0.06 * bump, y - 0.04 * bump];
            }
            mesh
        };
        for (name, mesh) in [
            (
                "rectangles",
                Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 4, 4),
            ),
            ("distorted", distorted()),
        ] {
            for order in [1, 2, 3] {
                let ops = DGOperators2D::new(order);
                let geom = GeometricFactors2D::compute(&mesh, &ops);
                let equation = ShallowWater2D::new(G);
                let bc = Reflective2D::new();
                let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
                    -1.0 + 2.0 * x + 0.2 * y
                });
                let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
                for k in ElementIndex::iter(mesh.n_elements) {
                    for i in 0..ops.n_nodes {
                        let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                        let eta = 0.2 + 0.05 * (tau * x).sin();
                        let h = (eta - bathymetry.get(k, i)).max(0.0);
                        let (u, v) = (0.2 * (0.5 * tau * y).cos(), 0.1 * (0.5 * tau * x).sin());
                        q.set_state(k, i, SWEState2D::new(h, h * u, h * v));
                    }
                }
                let config = SWE2DRhsConfig::new(&equation, &bc)
                    .with_coriolis(false)
                    .with_formulation(SWEFormulation2D::WetDry)
                    .with_bathymetry(&bathymetry)
                    .with_subcell_depth(0.1)
                    .with_flux_type(SWEFluxType2D::HLL);
                let plain = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
                let mut rhs = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
                let mut face_mass = vec![f64::NAN; face_mass_len(&mesh, &ops)];
                let mut subcell_mass = vec![0.0; subcell_mass_len(&mesh, &ops)];
                compute_rhs_swe_2d_mass_fluxes_into(
                    &q,
                    &mesh,
                    &ops,
                    &geom,
                    &config,
                    0.0,
                    &mut rhs,
                    &mut face_mass,
                    &mut subcell_mass,
                );
                assert_eq!(rhs.data, plain.data, "{name} N = {order}: the RHS changed");

                #[cfg(feature = "parallel")]
                {
                    let mut rhs_par = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
                    let mut face_par = vec![f64::NAN; face_mass.len()];
                    let mut sub_par = vec![0.0; subcell_mass.len()];
                    compute_rhs_swe_2d_parallel_mass_fluxes_into(
                        &q,
                        &mesh,
                        &ops,
                        &geom,
                        &config,
                        0.0,
                        &mut rhs_par,
                        &mut face_par,
                        &mut sub_par,
                    );
                    assert_eq!(rhs_par.data, rhs.data);
                    assert_eq!(face_par, face_mass);
                    let same = sub_par
                        .iter()
                        .zip(&subcell_mass)
                        .all(|(a, b)| a == b || (a.is_nan() && b.is_nan()));
                    assert!(same, "{name} N = {order}: parallel subcell fluxes");
                }

                let (nn, nfn) = (ops.n_nodes, ops.n_face_nodes);
                let n_sub = subcell_interfaces(ops.n_1d);
                let (mut subcell_elements, mut largest) = (0, 0.0_f64);
                let mut div = vec![0.0; nn];
                for k in 0..mesh.n_elements {
                    let slots = &subcell_mass[k * n_sub..(k + 1) * n_sub];
                    let shallow = (0..nn).any(|i| q.data[0][k * nn + i] < 0.1);
                    if !shallow {
                        assert!(slots.iter().all(|f| f.is_nan()), "{name}: element {k}");
                        continue;
                    }
                    assert!(slots.iter().all(|f| f.is_finite()), "{name}: element {k}");
                    subcell_elements += 1;
                    let nodes = k * nn..(k + 1) * nn;
                    subcell_divergence_element(
                        &ops,
                        &geom,
                        k,
                        &q.data[1][nodes.clone()],
                        &q.data[2][nodes.clone()],
                        slots,
                        &face_mass[k * 4 * nfn..(k + 1) * 4 * nfn],
                        &mut div,
                    );
                    let dh = &rhs.data[0][nodes];
                    let scale = div.iter().chain(dh).fold(1e-3_f64, |m, x| m.max(x.abs()));
                    for (d, r) in div.iter().zip(dh) {
                        largest = largest.max((d + r).abs() / scale);
                    }
                }
                assert!(subcell_elements >= 4, "{name} N = {order}: test regime");
                assert!(
                    largest < 1e-12,
                    "{name} N = {order}: dh/dt + div = {largest:.2e}"
                );
            }
        }
    }

    fn create_test_setup(order: usize) -> (Mesh2D, DGOperators2D, GeometricFactors2D) {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        (mesh, ops, geom)
    }

    #[test]
    fn test_rhs_still_water() {
        // Lake at rest: h = const, u = v = 0, bathymetry = 0
        // RHS should be zero (well-balanced test)
        let (mesh, ops, geom) = create_test_setup(2);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        // Set h = 2.0, u = v = 0 everywhere
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(2.0, 0.0, 0.0));
            }
        }

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

        // Check RHS is zero (lake at rest)
        let mut max_rhs: f64 = 0.0;
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let state = rhs.get_state(k, i);
                max_rhs = max_rhs.max(state.h.abs());
                max_rhs = max_rhs.max(state.hu.abs());
                max_rhs = max_rhs.max(state.hv.abs());
            }
        }

        assert!(
            max_rhs < 1e-10,
            "Lake at rest RHS should be zero, got {}",
            max_rhs
        );
    }

    #[test]
    fn test_rhs_uniform_flow() {
        // Uniform flow: h = const, u = const, v = 0
        // RHS should be zero (steady state)
        let (_mesh, ops, _geom) = create_test_setup(2);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

        // Use periodic mesh to avoid boundary effects
        let mesh_periodic = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let geom_periodic = GeometricFactors2D::compute(&mesh_periodic, &ops);

        let mut q = SWESolution2D::new(mesh_periodic.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh_periodic.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::from_primitives(2.0, 1.0, 0.0));
            }
        }

        let rhs = compute_rhs_swe_2d(&q, &mesh_periodic, &ops, &geom_periodic, &config, 0.0);

        // Uniform flow should give zero RHS (for periodic domain)
        let mut max_rhs: f64 = 0.0;
        for k in ElementIndex::iter(mesh_periodic.n_elements) {
            for i in 0..ops.n_nodes {
                let state = rhs.get_state(k, i);
                max_rhs = max_rhs.max(state.h.abs());
                max_rhs = max_rhs.max(state.hu.abs());
                max_rhs = max_rhs.max(state.hv.abs());
            }
        }

        assert!(
            max_rhs < 1e-10,
            "Uniform flow RHS should be zero, got {}",
            max_rhs
        );
    }

    #[test]
    fn test_mass_conservation() {
        // Test that total mass is conserved (RHS of h integrates to zero)
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

        // Non-uniform initial condition
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        q.set_from_functions(
            &mesh,
            &ops,
            |x, y| {
                1.0 + 0.5
                    * (2.0 * std::f64::consts::PI * x).sin()
                    * (2.0 * std::f64::consts::PI * y).sin()
            },
            |_, _| 0.0,
            |_, _| 0.0,
        );

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

        // Integrate RHS of h
        let mut integral = 0.0;
        for k in ElementIndex::iter(mesh.n_elements) {
            let ki = k.as_usize();
            for (i, &w) in ops.weights.iter().enumerate() {
                integral += w * rhs.get_var(k, i, 0) * geom.jacobian(ki, i);
            }
        }

        assert!(
            integral.abs() < 1e-10,
            "Mass should be conserved: d(mass)/dt = {:.2e}",
            integral
        );
    }

    #[test]
    fn test_dt_computation() {
        let (mesh, ops, geom) = create_test_setup(2);
        let equation = ShallowWater2D::new(G);

        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        // h = 1, u = 2 -> wave speed ≈ 2 + sqrt(10) ≈ 5.16
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::from_primitives(1.0, 2.0, 0.0));
            }
        }

        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, 0.5);

        assert!(dt > 0.0);
        assert!(dt < f64::INFINITY);
        assert!(dt < 0.1); // Should be small for this test case
    }

    /// Depth `depth(k)` in element k, velocity (u, 0) everywhere.
    fn element_wise_state(
        mesh: &Mesh2D,
        n_nodes: usize,
        depth: impl Fn(usize) -> f64,
        u: f64,
    ) -> SWESolution2D {
        let mut q = SWESolution2D::new(mesh.n_elements, n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..n_nodes {
                q.set_state(
                    k,
                    i,
                    SWEState2D::from_primitives(depth(k.as_usize()), u, 0.0),
                );
            }
        }
        q
    }

    #[test]
    fn test_dt_square_elements_match_classic_formula() {
        // At rest on squares the per-direction bound reduces to CFL·Δx/((2N+1)c),
        // so CFL numbers keep the meaning they had before P0.21.
        let order = 2;
        let (cfl, dx, depth) = (0.4, 250.0, 100.0);
        let mesh = Mesh2D::uniform_rectangle(0.0, 4.0 * dx, 0.0, 3.0 * dx, 4, 3);
        let geom = GeometricFactors2D::compute(&mesh, &DGOperators2D::new(order));
        let n_nodes = (order + 1) * (order + 1);
        let equation = ShallowWater2D::new(G);
        let c = (G * depth).sqrt();
        let dg = 2.0 * order as f64 + 1.0;

        let q = element_wise_state(&mesh, n_nodes, |_| depth, 0.0);
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, order, cfl);
        let expected = cfl * dx / (dg * c);
        assert!(
            (dt / expected - 1.0).abs() < 1e-12,
            "dt = {dt}, expected {expected}"
        );

        // Flow along x: λ_r = 2(|u| + c)/Δx, λ_s = 2c/Δx.
        let u = 3.0;
        let q = element_wise_state(&mesh, n_nodes, |_| depth, u);
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, order, cfl);
        let expected = 2.0 * cfl * dx / (dg * (u + 2.0 * c));
        assert!(
            (dt / expected - 1.0).abs() < 1e-12,
            "dt = {dt}, expected {expected}"
        );
    }

    /// Two elements side by side, 50 m × 1000 m and 1000 m × 1000 m.
    fn graded_anisotropic_mesh() -> Mesh2D {
        let mut mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 1);
        for v in mesh.vertices.iter_mut() {
            v[0] = if v[0] < 0.75 {
                100.0 * v[0]
            } else {
                50.0 + 1000.0 * (v[0] - 0.5) * 2.0
            };
            v[1] *= 1000.0;
        }
        mesh
    }

    #[test]
    fn test_dt_pairs_local_wave_speed_with_local_anisotropic_size() {
        // P0.21 regression. The old bound used min_k 2√detJ_k with max over all
        // nodes of |u| + c: here √(50·1000) = 224 m paired with the deep-water
        // celerity, ≈ 3.8× too small, while for the thin element alone √detJ
        // overestimates the 50 m width.
        let order = 3;
        let cfl = 0.25;
        let mesh = graded_anisotropic_mesh();
        let geom = GeometricFactors2D::compute(&mesh, &DGOperators2D::new(order));
        let equation = ShallowWater2D::new(G);
        let n_nodes = (order + 1) * (order + 1);
        let depths = [5.0, 400.0];
        let q = element_wise_state(&mesh, n_nodes, |k| depths[k], 0.0);

        let dg = 2.0 * order as f64 + 1.0;
        let sizes = [(50.0, 1000.0), (1000.0, 1000.0)];
        let expected = depths
            .iter()
            .zip(sizes)
            .map(|(&h, (dx, dy))| 2.0 * cfl / (dg * (G * h).sqrt() * (1.0 / dx + 1.0 / dy)))
            .fold(f64::INFINITY, f64::min);

        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, order, cfl);
        assert!(
            (dt / expected - 1.0).abs() < 1e-12,
            "dt = {dt}, expected {expected}"
        );

        let old = cfl * geom.element_size(0) / (dg * (G * 400.0).sqrt());
        assert!(
            dt > 3.0 * old,
            "dt = {dt} should exceed the old global pairing {old}"
        );

        // The thin element alone: limited by its 50 m width, below the √detJ size.
        let q = element_wise_state(&mesh, n_nodes, |_| 5.0, 0.0);
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, order, cfl);
        let sqrt_det_j = cfl * geom.element_size(0) / (dg * (G * 5.0).sqrt());
        assert!(
            dt < 0.5 * sqrt_det_j,
            "dt = {dt} vs √detJ-based {sqrt_det_j}"
        );
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_dt_parallel_matches_serial() {
        let order = 2;
        let mesh = graded_anisotropic_mesh();
        let geom = GeometricFactors2D::compute(&mesh, &DGOperators2D::new(order));
        let equation = ShallowWater2D::new(G);
        let q = element_wise_state(&mesh, (order + 1) * (order + 1), |k| [3.0, 250.0][k], 1.5);
        let serial = compute_dt_swe_2d(&q, &mesh, &geom, &equation, order, 0.3);
        let parallel = compute_dt_swe_2d_parallel(&q, &mesh, &geom, &equation, order, 0.3);
        assert_eq!(serial, parallel);
    }

    #[test]
    fn test_positivity_cfl_values() {
        // (2N+1)/(2N(N+1)): notes/review-2026-09-25.md §1.7 quotes 0.75 / 0.42 / 0.29 / 0.23.
        let expected = [0.75, 5.0 / 12.0, 7.0 / 24.0, 9.0 / 40.0];
        for (n, &e) in (1..=4).zip(&expected) {
            assert!((positivity_cfl_swe_2d(n) - e).abs() < 1e-15);
        }
    }

    #[test]
    fn test_dt_dry_domain_is_unbounded() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);
        let geom = GeometricFactors2D::compute(&mesh, &DGOperators2D::new(2));
        let equation = ShallowWater2D::new(G);
        let q = SWESolution2D::new(mesh.n_elements, 9);
        assert_eq!(
            compute_dt_swe_2d(&q, &mesh, &geom, &equation, 2, 0.5),
            f64::INFINITY
        );
    }

    #[test]
    fn test_rhs_into_overwrites_and_reuses_workspace() {
        // P1.1: `_into` must overwrite a reused (stale) output buffer and give the
        // allocating result, for both formulations and repeated evaluations (the
        // thread-local workspace is reused across them and across orders).
        let (mesh, ops, geom, bathymetry) = sloped_periodic_setup(2, false);
        let q = sloped_state(&mesh, &ops, &bathymetry, 0.5);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let bathy_source = crate::source::BathymetrySource2D::new(G);
        let standard = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&bathy_source)
            .with_well_balanced(true);
        let split = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_formulation(SWEFormulation2D::EntropyStable)
            .with_bathymetry(&bathymetry);

        let mut out = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for config in [&standard, &split, &standard] {
            let expected = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, config, 0.0);
            for v in out.data.iter_mut().flatten() {
                *v = 1.0e30;
            }
            compute_rhs_swe_2d_into(&q, &mesh, &ops, &geom, config, 0.0, &mut out);
            assert_eq!(out.data, expected.data);
            #[cfg(feature = "parallel")]
            {
                for v in out.data.iter_mut().flatten() {
                    *v = -1.0e30;
                }
                compute_rhs_swe_2d_parallel_into(&q, &mesh, &ops, &geom, config, 0.0, &mut out);
                assert_eq!(out.data, expected.data);
            }
        }
    }

    #[test]
    #[should_panic(expected = "RHS output is")]
    fn test_rhs_into_rejects_wrong_output_size() {
        let (mesh, ops, geom) = create_test_setup(2);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc);
        let q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        let mut out = SWESolution2D::new(mesh.n_elements + 1, ops.n_nodes);
        compute_rhs_swe_2d_into(&q, &mesh, &ops, &geom, &config, 0.0, &mut out);
    }

    #[test]
    fn test_coriolis_source() {
        // Test that Coriolis source term is included correctly
        let ops = DGOperators2D::new(2);
        let equation = ShallowWater2D::with_coriolis(G, 1.0e-4);
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(true);

        // Use periodic mesh to eliminate surface terms
        let mesh_periodic = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let geom_periodic = GeometricFactors2D::compute(&mesh_periodic, &ops);

        let mut q = SWESolution2D::new(mesh_periodic.n_elements, ops.n_nodes);
        // h = 10, hu = 100 (so v-momentum source = f*hu = 1e-4 * 100 = 0.01)
        for k in ElementIndex::iter(mesh_periodic.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(10.0, 100.0, 0.0));
            }
        }

        let rhs_with = compute_rhs_swe_2d(&q, &mesh_periodic, &ops, &geom_periodic, &config, 0.0);

        let config_no_coriolis = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);
        let rhs_without = compute_rhs_swe_2d(
            &q,
            &mesh_periodic,
            &ops,
            &geom_periodic,
            &config_no_coriolis,
            0.0,
        );

        // The difference should be the Coriolis source
        // Source: d(hu)/dt += f*hv = 0, d(hv)/dt -= f*hu = -0.01
        let k0 = ElementIndex::new(0);
        let diff_hv = rhs_with.get_var(k0, 0, 2) - rhs_without.get_var(k0, 0, 2);
        let expected = -1.0e-4 * 100.0; // -f*hu

        assert!(
            (diff_hv - expected).abs() < 1e-10,
            "Coriolis source difference: got {}, expected {}",
            diff_hv,
            expected
        );
    }

    #[test]
    fn test_stability_single_step() {
        // Test that a single time step doesn't blow up
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

        // Smooth initial condition
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        q.set_from_functions(
            &mesh,
            &ops,
            |x, y| {
                1.0 + 0.1
                    * (2.0 * std::f64::consts::PI * x).sin()
                    * (2.0 * std::f64::consts::PI * y).sin()
            },
            |_, _| 0.0,
            |_, _| 0.0,
        );

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

        // RHS should be bounded
        let max_rhs = rhs.max_abs();
        assert!(max_rhs < 100.0, "RHS should be bounded, got {}", max_rhs);

        // Do one forward Euler step
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, 0.1);
        q.axpy(dt, &rhs);

        // Solution should still be bounded
        assert!(!q.has_negative_depth(), "Depth should remain positive");
        assert!(
            q.max_abs() < 100.0,
            "Solution should be bounded after one step"
        );
    }

    #[test]
    fn test_flux_types() {
        // Test that different flux types work
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();

        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::from_primitives(2.0, 0.5, 0.2));
            }
        }

        for flux_type in [
            SWEFluxType2D::Roe,
            SWEFluxType2D::HLL,
            SWEFluxType2D::Rusanov,
        ] {
            let config = SWE2DRhsConfig::new(&equation, &bc)
                .with_flux_type(flux_type)
                .with_coriolis(false);

            let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

            // For uniform state, all fluxes should give zero RHS
            let max_rhs = rhs.max_abs();
            assert!(
                max_rhs < 1e-10,
                "Uniform state should give zero RHS for {:?}, got {}",
                flux_type,
                max_rhs
            );
        }
    }

    #[test]
    fn test_trait_based_coriolis() {
        // Test that trait-based CoriolisSource2D gives same result as legacy Coriolis
        let ops = DGOperators2D::new(2);
        let equation = ShallowWater2D::with_coriolis(G, 1.0e-4);
        let bc = Reflective2D::new();

        // Use periodic mesh to eliminate surface terms
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let geom = GeometricFactors2D::compute(&mesh, &ops);

        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(10.0, 100.0, 50.0));
            }
        }

        // Legacy Coriolis
        let config_legacy = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(true);
        let rhs_legacy = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config_legacy, 0.0);

        // Trait-based Coriolis
        let coriolis_source = CoriolisSource2D::f_plane(1.0e-4);
        let config_trait = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false) // Disable legacy
            .with_source_terms(&coriolis_source);
        let rhs_trait = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config_trait, 0.0);

        // Compare results
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let legacy = rhs_legacy.get_state(k, i);
                let trait_based = rhs_trait.get_state(k, i);

                assert!(
                    (legacy.h - trait_based.h).abs() < 1e-12,
                    "h mismatch at ({}, {}): {} vs {}",
                    k,
                    i,
                    legacy.h,
                    trait_based.h
                );
                assert!(
                    (legacy.hu - trait_based.hu).abs() < 1e-12,
                    "hu mismatch at ({}, {}): {} vs {}",
                    k,
                    i,
                    legacy.hu,
                    trait_based.hu
                );
                assert!(
                    (legacy.hv - trait_based.hv).abs() < 1e-12,
                    "hv mismatch at ({}, {}): {} vs {}",
                    k,
                    i,
                    legacy.hv,
                    trait_based.hv
                );
            }
        }
    }

    #[test]
    fn test_well_balanced_steep_slope() {
        // Test lake-at-rest preservation with steep bathymetry gradient
        // This tests the hydrostatic reconstruction implementation
        // Note: We need BOTH reconstruction (for surface terms) AND source term (for volume balance)
        use crate::mesh::Bathymetry2D;
        use crate::source::BathymetrySource2D;

        let mesh = Mesh2D::uniform_rectangle(0.0, 10.0, 0.0, 10.0, 4, 4);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();

        // Steep bathymetry: B = 0.5 * x (Norwegian-like gradient)
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _y| 0.5 * x);

        // Lake-at-rest: η = h + B = 10.0
        let eta_ref = 10.0;
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let b = bathymetry.get(k, i);
                let h = (eta_ref - b).max(0.0);
                q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
            }
        }

        // Config WITH hydrostatic reconstruction AND bathymetry source term
        // Reconstruction: handles surface flux well-balancing
        // Source term: balances volume term pressure gradient
        let bathy_source = BathymetrySource2D::new(G);
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&bathy_source)
            .with_well_balanced(true);

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

        // RHS should be zero to machine precision
        let max_rhs = rhs.max_abs();
        assert!(
            max_rhs < 1e-10,
            "Lake at rest with hydrostatic reconstruction: max RHS = {:.2e}, expected < 1e-10",
            max_rhs
        );
    }

    #[test]
    fn test_well_balanced_bilinear_slope() {
        // Test with bilinear bathymetry (x + y direction slope)
        // This is still linear within elements, so well-balanced should work perfectly
        use crate::mesh::Bathymetry2D;
        use crate::source::BathymetrySource2D;

        let mesh = Mesh2D::uniform_rectangle(0.0, 10.0, 0.0, 10.0, 6, 6);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();

        // Bilinear bathymetry: B = 0.3*x + 0.2*y
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| 0.3 * x + 0.2 * y);

        // Lake-at-rest: η = 10.0
        let eta_ref = 10.0;
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let b = bathymetry.get(k, i);
                let h = (eta_ref - b).max(0.0);
                q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
            }
        }

        let bathy_source = BathymetrySource2D::new(G);
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&bathy_source)
            .with_well_balanced(true);

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

        let max_rhs = rhs.max_abs();
        assert!(
            max_rhs < 1e-10,
            "Lake at rest (bilinear slope): max RHS = {:.2e}",
            max_rhs
        );
    }

    #[test]
    fn test_well_balanced_all_flux_types() {
        // Verify all flux types work with hydrostatic reconstruction
        use crate::flux::SWEFluxType2D;
        use crate::mesh::Bathymetry2D;
        use crate::source::BathymetrySource2D;

        let mesh = Mesh2D::uniform_rectangle(0.0, 10.0, 0.0, 10.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();

        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| 0.3 * x + 0.4 * y);

        let eta_ref = 10.0;
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let b = bathymetry.get(k, i);
                q.set_state(k, i, SWEState2D::new((eta_ref - b).max(0.0), 0.0, 0.0));
            }
        }

        let bathy_source = BathymetrySource2D::new(G);
        for flux_type in [
            SWEFluxType2D::Roe,
            SWEFluxType2D::HLL,
            SWEFluxType2D::Rusanov,
        ] {
            let config = SWE2DRhsConfig::new(&equation, &bc)
                .with_flux_type(flux_type)
                .with_coriolis(false)
                .with_bathymetry(&bathymetry)
                .with_source_terms(&bathy_source)
                .with_well_balanced(true);

            let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
            let max_rhs = rhs.max_abs();

            assert!(
                max_rhs < 1e-10,
                "{:?} flux with well-balanced: max RHS = {:.2e}",
                flux_type,
                max_rhs
            );
        }
    }

    #[test]
    fn test_source_term_alone_well_balanced_linear() {
        // For LINEAR bathymetry, the standard DG scheme with bathymetry source term
        // is already well-balanced because the volume gradient and source term
        // are discretized consistently using the same DG differentiation operators.
        // This test verifies this property.
        use crate::mesh::Bathymetry2D;
        use crate::source::BathymetrySource2D;

        let mesh = Mesh2D::uniform_rectangle(0.0, 10.0, 0.0, 10.0, 4, 4);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();

        // Linear bathymetry
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _y| 0.5 * x);

        let eta_ref = 10.0;
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let b = bathymetry.get(k, i);
                q.set_state(k, i, SWEState2D::new((eta_ref - b).max(0.0), 0.0, 0.0));
            }
        }

        // Config with source term only (no reconstruction)
        let bathy_source = BathymetrySource2D::new(G);
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&bathy_source)
            .with_well_balanced(false); // No reconstruction

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

        let max_rhs = rhs.max_abs();

        // For linear bathymetry, source term alone should be well-balanced
        // (DG computes gradients exactly for linear functions)
        assert!(
            max_rhs < 1e-10,
            "Source term alone should be well-balanced for linear bathymetry: {:.2e}",
            max_rhs
        );
    }

    #[test]
    fn test_reconstruction_improves_accuracy() {
        // Verify that hydrostatic reconstruction is correctly applied
        // by checking that reconstructed states match expected values.
        use crate::mesh::Bathymetry2D;
        use crate::source::BathymetrySource2D;

        let mesh = Mesh2D::uniform_rectangle(0.0, 10.0, 0.0, 10.0, 4, 4);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();

        // Linear bathymetry
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _y| 0.5 * x);

        let eta_ref = 10.0;
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let b = bathymetry.get(k, i);
                q.set_state(k, i, SWEState2D::new((eta_ref - b).max(0.0), 0.0, 0.0));
            }
        }

        // Config WITH reconstruction AND source term
        let bathy_source = BathymetrySource2D::new(G);
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&bathy_source)
            .with_well_balanced(true);

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

        let max_rhs = rhs.max_abs();

        // With reconstruction, should also be well-balanced
        assert!(
            max_rhs < 1e-10,
            "With reconstruction should be well-balanced: {:.2e}",
            max_rhs
        );
    }

    /// Periodic 20 km domain with smooth bathymetry of large amplitude.
    ///
    /// B = −200 + 150·sin(2πx/L)·cos(2πy/L), optionally cell-averaged so that B
    /// jumps across every element face (as `examples/froya_real_data.rs` does).
    fn sloped_periodic_setup(
        order: usize,
        cell_average: bool,
    ) -> (Mesh2D, DGOperators2D, GeometricFactors2D, Bathymetry2D) {
        const L: f64 = 20_000.0;
        let mesh = Mesh2D::uniform_periodic(0.0, L, 0.0, L, 12, 12);
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let tau = 2.0 * std::f64::consts::PI / L;
        let mut bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
            -200.0 + 150.0 * (tau * x).sin() * (tau * y).cos()
        });
        if cell_average {
            bathymetry.to_cell_average();
        }
        (mesh, ops, geom, bathymetry)
    }

    /// State with free surface η = 0.3 m over `bathymetry` and momentum
    /// hu = velocity_scale·h·cos(2πx/L)·cos(2πy/L), hv = 0.
    ///
    /// The velocity is correlated with the bathymetry slope on purpose: a
    /// uniform velocity hides reconstruction-induced mass errors by symmetry.
    fn sloped_state(
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        bathymetry: &Bathymetry2D,
        velocity_scale: f64,
    ) -> SWESolution2D {
        let tau = 2.0 * std::f64::consts::PI / 20_000.0;
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let h = 0.3 - bathymetry.get(k, i);
                let hu = velocity_scale * h * (tau * x).cos() * (tau * y).cos();
                q.set_state(k, i, SWEState2D::new(h, hu, 0.0));
            }
        }
        q
    }

    /// ∫ q_var dA using the GLL quadrature.
    fn integrate_var(
        q: &SWESolution2D,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        var: usize,
    ) -> f64 {
        let mut integral = 0.0;
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                integral += geom.mass[geom.node_index(k.as_usize(), i)] * q.get_var(k, i, var);
            }
        }
        integral
    }

    #[test]
    fn test_mass_conservation_hydrostatic_reconstruction_discontinuous_bathymetry() {
        // Regression (notes/review-2026-09-25.md §1.3): with cell-averaged B the
        // reconstruction is active at every face. Evaluating the interior flux on the reconstructed
        // state broke the SBP telescoping and leaked ≈5e-5 of the total mass per
        // second on this configuration.
        use crate::source::BathymetrySource2D;

        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let bathy_source = BathymetrySource2D::new(G);

        for order in 1..=3 {
            for cell_average in [false, true] {
                let (mesh, ops, geom, bathymetry) = sloped_periodic_setup(order, cell_average);
                let q = sloped_state(&mesh, &ops, &bathymetry, 1.0);
                let config = SWE2DRhsConfig::new(&equation, &bc)
                    .with_coriolis(false)
                    .with_bathymetry(&bathymetry)
                    .with_source_terms(&bathy_source)
                    .with_well_balanced(true);

                let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
                let mass = integrate_var(&q, &mesh, &ops, &geom, 0);
                let mass_rate = integrate_var(&rhs, &mesh, &ops, &geom, 0);

                assert!(
                    mass_rate.abs() / mass < 1e-12,
                    "p={order}, cell_average={cell_average}: \
                     d(mass)/dt / mass = {:.3e} (d(mass)/dt = {:.3e} m³/s)",
                    mass_rate / mass,
                    mass_rate
                );
            }
        }
    }

    #[test]
    fn test_lake_at_rest_cell_averaged_bathymetry() {
        // Cell-constant B: zero volume gradient, every bathymetry effect comes from
        // the hydrostatic interface correction. Must stay balanced to round-off.
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();

        for order in 1..=3 {
            let (mesh, ops, geom, bathymetry) = sloped_periodic_setup(order, true);
            let q = sloped_state(&mesh, &ops, &bathymetry, 0.0);
            let config = SWE2DRhsConfig::new(&equation, &bc)
                .with_coriolis(false)
                .with_bathymetry(&bathymetry)
                .with_well_balanced(true);

            let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
            let max_rhs = rhs.max_abs();
            assert!(
                max_rhs < 1e-10,
                "p={order}: lake at rest over cell-averaged B, max RHS = {max_rhs:.3e}"
            );

            #[cfg(all(feature = "parallel", feature = "simd"))]
            {
                let rhs_par = compute_rhs_swe_2d_parallel(&q, &mesh, &ops, &geom, &config, 0.0);
                let max_rhs_par = rhs_par.max_abs();
                assert!(
                    max_rhs_par < 1e-10,
                    "p={order}: parallel lake at rest over cell-averaged B, \
                     max RHS = {max_rhs_par:.3e}"
                );
            }
        }
    }

    #[test]
    fn test_lake_at_rest_linearized_bathymetry() {
        // P0.14: `Bathymetry2D::linearize` used to claim planar B is well-balanced.
        // With the collocated scheme that holds only for p ≥ 2 (½gh² resolved);
        // at p = 1 the interpolation of ½gh² leaves g(h̄ − hᵢ)∂B. The split form
        // is balanced at every order.
        use crate::source::BathymetrySource2D;

        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let bathy_source = BathymetrySource2D::new(G);

        for order in 1..=3 {
            let (mesh, ops, geom, mut bathymetry) = sloped_periodic_setup(order, false);
            bathymetry.linearize(&mesh, &ops, &geom);
            let q = sloped_state(&mesh, &ops, &bathymetry, 0.0);

            let standard = SWE2DRhsConfig::new(&equation, &bc)
                .with_coriolis(false)
                .with_bathymetry(&bathymetry)
                .with_source_terms(&bathy_source)
                .with_well_balanced(true);
            let residual = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &standard, 0.0).max_abs();
            if order == 1 {
                assert!(
                    residual > 1e-3,
                    "p=1: expected an aliasing residual for planar B, got {residual:.3e}"
                );
            } else {
                assert!(
                    residual < 1e-10,
                    "p={order}: lake at rest over planar B, max RHS = {residual:.3e}"
                );
            }

            let split = SWE2DRhsConfig::new(&equation, &bc)
                .with_coriolis(false)
                .with_formulation(SWEFormulation2D::EntropyStable)
                .with_bathymetry(&bathymetry);
            let residual = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &split, 0.0).max_abs();
            assert!(
                residual < 1e-9,
                "p={order}: split form over planar B, max RHS = {residual:.3e}"
            );
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_matches_serial() {
        use crate::source::BathymetrySource2D;

        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();

        // Case 1: flat bottom, non-uniform flow.
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let [x, y] = mesh.reference_to_physical(k, r, s);
                let h =
                    2.0 + 0.1 * (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).cos();
                let u = 0.5 * (std::f64::consts::PI * y).sin();
                let v = 0.3 * (std::f64::consts::PI * x).cos();
                q.set_state(k, i, SWEState2D::from_primitives(h, u, v));
            }
        }
        let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);
        assert_parallel_matches_serial(&q, &mesh, &ops, &geom, &config, "flat bottom");

        // Case 2: cell-averaged (face-discontinuous) bathymetry with hydrostatic
        // reconstruction active at every face and a slope-correlated flow.
        let (mesh, ops, geom, bathymetry) = sloped_periodic_setup(3, true);
        let q = sloped_state(&mesh, &ops, &bathymetry, 1.0);
        let bathy_source = BathymetrySource2D::new(G);
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&bathy_source)
            .with_well_balanced(true);
        assert_parallel_matches_serial(
            &q,
            &mesh,
            &ops,
            &geom,
            &config,
            "cell-averaged bathymetry + reconstruction",
        );

        // Case 3: reflective walls (ghost states), nodal bathymetry, trait sources
        // including Coriolis, both formulations.
        let mesh = Mesh2D::uniform_rectangle(0.0, 20_000.0, 0.0, 10_000.0, 6, 3);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
            -150.0 + 0.004 * x - 50.0 * (y / 10_000.0).powi(2)
        });
        let q = sloped_state(&mesh, &ops, &bathymetry, 0.8);
        let coriolis = CoriolisSource2D::f_plane(1.2e-4);
        let sources = crate::source::CombinedSource2D::new(vec![&coriolis, &bathy_source]);
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&sources)
            .with_well_balanced(true);
        assert_parallel_matches_serial(&q, &mesh, &ops, &geom, &config, "walls + sources");
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_formulation(SWEFormulation2D::EntropyStable)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&coriolis);
        assert_parallel_matches_serial(&q, &mesh, &ops, &geom, &config, "walls, split form");

        // Case 4: a tidal open boundary (characteristic OBC, exact boundary
        // flux) on the west side, walls elsewhere, all three formulations
        use crate::boundary::{CharacteristicOBC, HarmonicTide, MultiBoundaryCondition2D};
        use crate::mesh::BoundaryTag;
        let mesh = Mesh2D::uniform_rectangle_with_sides(
            0.0,
            20_000.0,
            0.0,
            10_000.0,
            6,
            3,
            [
                BoundaryTag::Wall,
                BoundaryTag::Wall,
                BoundaryTag::Wall,
                BoundaryTag::Open,
            ],
        );
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
            -150.0 + 0.004 * x - 50.0 * (y / 10_000.0).powi(2)
        });
        let q = sloped_state(&mesh, &ops, &bathymetry, 0.8);
        let sea = CharacteristicOBC::new(HarmonicTide::m2(0.5, 0.3));
        let open = MultiBoundaryCondition2D::new(&bc).with_open(&sea);
        for formulation in [
            SWEFormulation2D::Standard,
            SWEFormulation2D::EntropyStable,
            SWEFormulation2D::WetDry,
        ] {
            let config = SWE2DRhsConfig::new(&equation, &open)
                .with_formulation(formulation)
                .with_bathymetry(&bathymetry);
            // The split forms carry the bed slope in the operator
            let config = if formulation == SWEFormulation2D::Standard {
                config.with_source_terms(&sources).with_well_balanced(true)
            } else {
                config.with_source_terms(&coriolis)
            };
            let serial = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 1000.0);
            let parallel = compute_rhs_swe_2d_parallel(&q, &mesh, &ops, &geom, &config, 1000.0);
            assert!(serial.max_abs() > 0.0);
            assert_eq!(serial.data, parallel.data, "open boundary, {formulation:?}");
        }
    }

    #[cfg(feature = "parallel")]
    fn assert_parallel_matches_serial<BC: SWEBoundaryCondition2D + Sync>(
        q: &SWESolution2D,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        config: &SWE2DRhsConfig<BC>,
        case: &str,
    ) {
        // P1.1: one element kernel for both drivers, so the results are identical.
        let rhs_serial = compute_rhs_swe_2d(q, mesh, ops, geom, config, 0.0);
        let rhs_parallel = compute_rhs_swe_2d_parallel(q, mesh, ops, geom, config, 0.0);
        assert!(rhs_serial.max_abs() > 0.0, "{case}: trivial RHS");
        assert_eq!(
            rhs_serial.data, rhs_parallel.data,
            "{case}: serial != parallel"
        );
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_with_coriolis() {
        use super::compute_rhs_swe_2d_parallel;

        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let equation = ShallowWater2D::with_coriolis(G, 1.0e-4); // f-plane Coriolis
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(true);

        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(10.0, 100.0, 50.0));
            }
        }

        let rhs_serial = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
        let rhs_parallel = compute_rhs_swe_2d_parallel(&q, &mesh, &ops, &geom, &config, 0.0);

        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let serial = rhs_serial.get_state(k, i);
                let parallel = rhs_parallel.get_state(k, i);

                assert!(
                    (serial.h - parallel.h).abs() < 1e-10,
                    "h mismatch with Coriolis at ({}, {})",
                    k,
                    i
                );
                assert!(
                    (serial.hu - parallel.hu).abs() < 1e-10,
                    "hu mismatch with Coriolis at ({}, {})",
                    k,
                    i
                );
                assert!(
                    (serial.hv - parallel.hv).abs() < 1e-10,
                    "hv mismatch with Coriolis at ({}, {})",
                    k,
                    i
                );
            }
        }
    }

    /// An element's local time step sees its own nodes and the neighbours'
    /// nodes on its faces (the states the interface flux sees), not the
    /// neighbours' interiors; the smallest element step is at most the global
    /// one.
    #[test]
    fn element_dt_sees_the_neighbours_face_nodes_only() {
        let order = 2;
        let mesh = Mesh2D::uniform_rectangle(0.0, 3.0, 0.0, 1.0, 3, 1);
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let equation = ShallowWater2D::new(G);
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(1.0, 0.0, 0.0));
            }
        }
        // West element: 100 m deep inside, 1 m on its east face (towards the
        // middle one); east element: 25 m deep on its west face
        let (west, east) = (ElementIndex::new(0), ElementIndex::new(2));
        let deep_centre = (0..ops.n_nodes)
            .find(|&i| ops.nodes_r[i].abs() < 1e-12 && ops.nodes_s[i].abs() < 1e-12)
            .unwrap();
        q.set_state(west, deep_centre, SWEState2D::new(100.0, 0.0, 0.0));
        for i in 0..ops.n_nodes {
            if (ops.nodes_r[i] + 1.0).abs() < 1e-12 {
                q.set_state(east, i, SWEState2D::new(25.0, 0.0, 0.0));
            }
        }
        let cfl = 0.4;
        let mut dt = vec![0.0; mesh.n_elements];
        element_dt_swe_2d(&q, &mesh, &ops, &geom, &equation, order, cfl, None, &mut dt);
        // Unit squares at rest: ¼(λ_r + λ_s) = c, so Δt = CFL/((2N+1)c)
        let expected = |depth: f64| cfl / ((2 * order + 1) as f64 * (G * depth).sqrt());
        // The middle element sees the east element's 25 m face, not the
        // west element's 100 m centre
        assert!((dt[1] - expected(25.0)).abs() < 1e-14 * dt[1], "{dt:?}");
        assert!((dt[0] - expected(100.0)).abs() < 1e-14 * dt[0], "{dt:?}");
        assert!((dt[2] - expected(25.0)).abs() < 1e-14 * dt[2], "{dt:?}");
        let global = compute_dt_swe_2d(&q, &mesh, &geom, &equation, order, cfl);
        assert!(dt.iter().copied().fold(f64::INFINITY, f64::min) <= global);
    }

    /// ρ of a uniform depth on a parallelogram is the weight ratio
    /// 1/(1 − (1 − 2ŵ₀)²), ŵ₀ = 1/(N(N+1)); water piled on the boundary
    /// nodes lowers it (to 1 at least); the linear limit caps the relaxed
    /// CFL, the subcells' lower one where a node is dry, and without a known
    /// limit nothing is relaxed.
    #[test]
    fn positivity_relaxation_follows_the_water_on_interior_nodes() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 3.0, 0.0, 1.0, 1, 1);
        for order in 1..=4 {
            let ops = DGOperators2D::new(order);
            let geom = GeometricFactors2D::compute(&mesh, &ops);
            let mass = &geom.mass[..ops.n_nodes];
            let n = order as f64;
            let w0 = 1.0 / (n * (n + 1.0));
            let weight_ratio = 1.0 / (1.0 - (1.0 - 2.0 * w0).powi(2));
            let uniform = vec![5.0; ops.n_nodes];
            let (rho, dry) = positivity_relaxation(&uniform, mass, ops.n_1d, 1e-3);
            assert!(!dry);
            let expected = (POSITIVITY_RELAXATION_SAFETY * weight_ratio).max(1.0);
            assert!(
                (rho - expected).abs() < 1e-12,
                "N = {order}: {rho} against {expected}"
            );

            // A dry corner is flagged; it empties a boundary node, so M/M_∂
            // rises and the weight ratio still caps ρ
            let mut shore = uniform.clone();
            shore[0] = 0.0;
            let (rho_shore, dry) = positivity_relaxation(&shore, mass, ops.n_1d, 1e-3);
            assert!(dry);
            assert!((rho_shore - rho).abs() < 1e-12, "N = {order}: {rho_shore}");

            // Deeper on the boundary nodes: M/M_∂ below the weight ratio;
            // shallower there: capped at the weight ratio
            let boundary = |i: usize| ops.face_nodes.iter().any(|f| f.contains(&i));
            let piled: Vec<f64> = (0..ops.n_nodes)
                .map(|i| if boundary(i) { 5.0 } else { 1.0 })
                .collect();
            let thin: Vec<f64> = (0..ops.n_nodes)
                .map(|i| if boundary(i) { 0.01 } else { 5.0 })
                .collect();
            if order > 1 {
                assert!(positivity_relaxation(&piled, mass, ops.n_1d, 1e-3).0 < rho);
                let (capped, _) = positivity_relaxation(&thin, mass, ops.n_1d, 1e-3);
                assert!((capped - rho).abs() < 1e-12, "N = {order}: {capped}");
            }

            for scheme in [SspScheme::Rk3, SspScheme::Rk43] {
                let bound = PositivityBound::new(order, Some(scheme), 1e-3);
                let c = scheme.ssp_coefficient();
                assert_eq!(bound.cfl, c * positivity_cfl_swe_2d(order));
                let linear = LINEAR_CFL_SAFETY * linear_cfl_swe_2d(order, scheme).unwrap();
                assert_eq!(bound.max_cfl, linear);
                let cfl = bound.element_cfl(&uniform, &geom, ops.n_1d, 0);
                assert!(
                    (cfl - (bound.cfl * rho).min(linear)).abs() < 1e-14,
                    "N = {order}, {scheme:?}"
                );
                // The subcells' limit is lower: at least the plain bound
                let subcells = linear_cfl_subcells_swe_2d(order, scheme).unwrap();
                assert!(subcells < linear_cfl_swe_2d(order, scheme).unwrap());
                let subcells = bound.cfl.max(LINEAR_CFL_SAFETY * subcells);
                assert_eq!(bound.max_cfl_subcells, subcells);
                let cfl = bound.element_cfl(&shore, &geom, ops.n_1d, 0);
                assert!(
                    (cfl - (bound.cfl * rho_shore).min(subcells)).abs() < 1e-14,
                    "N = {order}, {scheme:?}"
                );
                // Still wet but within SUBCELL_CAP_DEPTH_FACTOR · h_dry of
                // drying (0.1 m here): the subcell cap; deeper: the DGSEM one
                for (corner, cap) in [(0.05, subcells), (0.2, linear)] {
                    let mut h = uniform.clone();
                    h[0] = corner;
                    let rho = positivity_relaxation(&h, mass, ops.n_1d, 0.0).0;
                    let cfl = bound.element_cfl(&h, &geom, ops.n_1d, 0);
                    assert!(
                        (cfl - (bound.cfl * rho).min(cap)).abs() < 1e-14,
                        "N = {order}, {scheme:?}, corner {corner}: {cfl}"
                    );
                }
            }
            let plain = PositivityBound::new(order, None, 1e-3);
            assert_eq!(plain.max_cfl, plain.cfl);
            assert_eq!(plain.max_cfl_subcells, plain.cfl);
            assert_eq!(plain.element_cfl(&uniform, &geom, ops.n_1d, 0), plain.cfl);
            assert_eq!(plain.element_cfl(&shore, &geom, ops.n_1d, 0), plain.cfl);
        }
    }
}
