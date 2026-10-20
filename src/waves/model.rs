//! The spectral wave model on the DG mesh: the wave action balance
//!
//! ```text
//! ∂N/∂t + ∇·((c_g e_θ + U) N) + ∂(c_σ N)/∂σ + ∂(c_θ N)/∂θ = S/σ
//! ```
//!
//! for the action density `N(x, y, σ, θ, t) = E/σ` (Komen et al. 1994; the SWAN
//! Scientific Documentation, §2.2), with
//!
//! - geographic propagation at the group velocity plus the current, discretised
//!   per spectral component by the strong-form DG of the rest of the crate
//!   (Hesthaven & Warburton 2008, §6): the volume term through the contravariant
//!   fluxes, an upwind flux on the faces (the neighbour's face nodes in reverse
//!   order), on the same mesh, order and quadrature;
//! - refraction by depth and current shear, `c_θ = −(1/k)(∂σ/∂d ∂d/∂m + k·∂U/∂m)`,
//!   `m` across the crest (θ + 90°): waves turn towards shallower water;
//! - frequency shifting by currents, `c_σ = ∂σ/∂d U·∇d − c_g k·∂U/∂s` (`s` along
//!   θ; the term of a changing depth `∂σ/∂d ∂d/∂t` is not included yet);
//! - both by finite volumes over the direction bins (periodic) and the frequency
//!   bins (energy leaves through the ends, none comes in): MUSCL, the upwind bin
//!   reconstructed linearly with van Leer's limiter (van Leer 1979; second order
//!   where the spectrum is smooth, TVD), or first-order upwind
//!   ([`SpectralAdvection`]). Every term is in flux form, so the total action
//!   `Σ Δσ Δθ ∫ N dA` is conserved up to what crosses the open boundaries and the
//!   spectral ends.
//!
//! Depth (`d = η − B`, floored at a minimum depth), its gradient, the current and
//! its gradient are nodal fields (gradients by the element's own derivative
//! matrices), set from a circulation run with [`WaveModel2D::set_water_level`] and
//! [`WaveModel2D::set_currents`]; the wavenumber, group velocity and `∂σ/∂d` of
//! every frequency at every node are kept with them.
//!
//! Boundaries: open faces let waves out and bring in the boundary spectrum,
//! one for the whole boundary ([`WaveModel2D::with_boundary_spectrum`]) or one
//! per node of the open faces, changed as often as wanted
//! ([`WaveModel2D::set_boundary_spectra`], e.g. a parent wave model's through
//! [`super::BoundarySpectra`]); every other face absorbs (nothing comes in),
//! as SWAN's default coast. Land inside the mesh is water of the minimum
//! depth, or with [`WaveModel2D::with_absorbing_land`] absorbs like the
//! coast: dry nodes lose their action every propagation step, and the faces
//! of a wholly dry element are coast to its wet neighbours.
//!
//! A step ([`WaveModel2D::step`]) is SSP-RK3 for the propagation, with a
//! positivity-preserving scaling of each component in each element towards its
//! mean after every stage (Zhang & Shu 2010; conservative), then the sources over
//! the whole step at every node ([`super::SourceTerms::integrate`]), split
//! first-order in time from the propagation.
//!
//! The propagation's own time error is small next to the sources' (measured on
//! a young wind sea: 8e-5 of H_s after an hour at the geographic step, against
//! 1e-2 with SWAN's sources). [`WaveModel2D::with_substeps`] runs the sources
//! once per several propagation steps, between their halves; the implicit
//! refraction and shifting and depth-induced breaking, which are fast on steep
//! coasts and in the surf zone, stay with each propagation step. With SWAN's
//! frozen rates and per-step limiter
//! that costs accuracy, and frozen rates without the limiter blow up at long
//! steps; WAM's integration ([`super::SourceIntegration::Implicit`] with
//! [`super::GrowthLimiter::Rate`]) keeps the growth curves at an outer step of
//! minutes. Under [`super::GrowthLimiter::RateBothSigns`], which caps the
//! losses too, breaking has its own pass after the other sources even in one
//! step, so that the cap leaves the surf zone's dissipation whole.
//! Against wave buoys (the E39 fjord buoys at Ålesund, `notes/waves.md`)
//! WAM's integration with 4 substeps at a CFL number of 1.0 does as well as
//! SWAN's with one at 0.5, for half the cost: it is `froya_real_data`'s
//! default.
//!
//! On steep slopes refraction turns the waves fast, and its explicit Courant
//! limit `|c_θ| Δt ≤ Δθ` sets a step far below the geographic one (100× at
//! Frøya). [`WaveModel2D::with_implicit_refraction`] takes the direction
//! advection out of the Runge–Kutta stages and steps it implicitly at every
//! node and frequency after them, as SWAN does: first-order upwind in θ,
//! backward Euler, a cyclic tridiagonal system per node and frequency. Its
//! matrix is an M-matrix whose columns sum to one, so the step is positive
//! and conserves the action to round-off at any Courant number. With
//! [`SpectralAdvection::VanLeer`] (the default) a deferred correction adds
//! MUSCL's flux less upwind's, of the state before the step, to the
//! right-hand side: the fixed point is MUSCL's steady state (second order in
//! the bins), and the correction is scaled down where it would make the
//! right-hand side negative (it sums to zero over the circle, so the action is
//! still kept). With [`SpectralAdvection::Upwind`] the steady state is the
//! explicit upwind scheme's (first order).
//!
//! Frequency shifting has the same trouble where currents cross steep, shallow
//! beds: `∂σ/∂d U·∇d` grows as the depth shrinks, and on a tidal flat it set
//! a step of 0.6 s against the geographic 7 s at Frøya.
//! [`WaveModel2D::with_implicit_frequency_shift`] steps it the same way, per
//! node and direction over the frequencies: first-order upwind, backward
//! Euler, a tridiagonal M-matrix (not periodic: action leaves through the
//! ends of the grid, none comes in), whose columns weighted by `Δσ` sum to
//! one but for the ends' outflow; and the same donor-limited MUSCL deferred
//! correction on the faces between the bins.

use std::borrow::Cow;
use std::sync::Arc;

use crate::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::profiling;
use crate::time::SspScheme;
use crate::types::ElementIndex;

use super::dispersion::{dsigma_ddepth, group_velocity, wavenumber};
#[cfg(feature = "simd")]
use super::nonlinear::LANES;
use super::sources::{Dissipation, Nodes, SourceScratch, SourceTerms, Wind};
use super::spectrum::{SpectralGrid, WaveParameters};
use super::state::WaveSolution;

#[cfg(feature = "simd")]
#[path = "implicit_lanes.rs"]
mod implicit_lanes;
#[path = "propagation.rs"]
mod propagation;

#[cfg(all(test, feature = "simd"))]
static SCALAR_ONLY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the vector kernels run (`simd` feature; tests can turn them off
/// to compare with the scalar ones, which give the same bits).
#[cfg(feature = "simd")]
#[inline]
fn vector_kernels() -> bool {
    #[cfg(all(test, feature = "simd"))]
    if SCALAR_ONLY.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    true
}

/// Default minimum depth (m): shallower water, and land, is taken this deep.
pub const DEFAULT_DEPTH_MIN: f64 = 0.1;

/// How refraction and frequency shifting move action between the spectral bins.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpectralAdvection {
    /// First-order upwind finite volumes
    Upwind,
    /// MUSCL: the upwind bin linearly reconstructed with van Leer's limiter,
    /// second order where the spectrum is smooth, TVD for Courant numbers ≤ ½
    #[default]
    VanLeer,
}

/// The action density entering through the open faces.
#[derive(Clone, Debug)]
enum Boundary {
    /// Nothing enters
    None,
    /// The same spectrum everywhere, per component
    Uniform(Vec<f64>),
    /// One spectrum per open-boundary node: `slot[p]` is node `p`'s index in
    /// `points` (`u32::MAX` off the boundary), `action[c · n + slot]`
    Nodal {
        points: Vec<usize>,
        slot: Vec<u32>,
        action: Vec<f64>,
    },
}

impl Boundary {
    /// The action density of component `c` entering at node `p`.
    #[inline]
    fn action(&self, c: usize, p: usize) -> f64 {
        match self {
            Boundary::None => 0.0,
            Boundary::Uniform(action) => action[c],
            Boundary::Nodal {
                points,
                slot,
                action,
            } => match slot[p] {
                u32::MAX => 0.0,
                s => action[c * points.len() + s as usize],
            },
        }
    }
}

/// One bound of the wave step ([`WaveModel2D::time_step_limits`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaveTimeStepLimit {
    /// The largest stable step (s) of this term; infinite if it is absent
    pub dt: f64,
    /// The node that sets it
    pub point: usize,
    /// The frequency bin that sets it
    pub frequency: usize,
}

/// The bounds of the wave step at a CFL number ([`WaveModel2D::compute_dt`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaveTimeStepLimits {
    /// Geographic propagation `c_g e_θ + U`
    pub propagation: WaveTimeStepLimit,
    /// Explicit refraction `c_θ` (infinite with implicit refraction)
    pub refraction: WaveTimeStepLimit,
    /// Frequency shifting `c_σ` by currents
    pub frequency_shift: WaveTimeStepLimit,
}

impl WaveTimeStepLimits {
    /// The step: the smallest bound.
    pub fn dt(&self) -> f64 {
        self.propagation
            .dt
            .min(self.refraction.dt)
            .min(self.frequency_shift.dt)
    }

    /// The bound that sets the step, and its name.
    pub fn binding(&self) -> (&'static str, WaveTimeStepLimit) {
        [
            ("propagation", self.propagation),
            ("refraction", self.refraction),
            ("frequency shift", self.frequency_shift),
        ]
        .into_iter()
        .min_by(|a, b| a.1.dt.total_cmp(&b.1.dt))
        .expect("three bounds")
    }
}

/// Reusable storage of [`WaveModel2D::step`].
#[derive(Default)]
pub struct WaveWorkspace {
    /// The Runge–Kutta stage values: the last and the next
    stage: Vec<f64>,
    next: Vec<f64>,
}

/// The spectral wave model (see the module docs).
pub struct WaveModel2D {
    pub mesh: Arc<Mesh2D>,
    pub ops: Arc<DGOperators2D>,
    pub geom: Arc<GeometricFactors2D>,
    pub grid: SpectralGrid,
    pub sources: SourceTerms,
    /// The wind where `winds` has none
    wind: Wind,
    /// The wind per node (empty: `wind` everywhere)
    winds: Vec<Wind>,
    g: f64,
    depth_min: f64,
    /// Bed elevation B per node (m)
    bed: Vec<f64>,
    /// Surface elevation per node (m), as set
    water_level: Vec<f64>,
    /// Depth d per node (m, ≥ `depth_min`), its gradient
    depth: Vec<f64>,
    depth_grad: Vec<[f64; 2]>,
    /// Current per node (m/s) and its gradient `[∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y]`
    current: Vec<[f64; 2]>,
    current_grad: Vec<[f64; 4]>,
    /// Per frequency and node (`[i · n_points + p]`): wavenumber, group velocity
    /// and ∂σ/∂d
    k: Vec<f64>,
    cg: Vec<f64>,
    sigma_d: Vec<f64>,
    /// Action density entering through open faces
    boundary: Boundary,
    /// Largest turning rate |c_θ| (rad/s) refraction may have, or none
    turning_limit: Option<f64>,
    spectral_advection: SpectralAdvection,
    /// Refraction stepped implicitly around the Runge–Kutta stages
    implicit_refraction: bool,
    /// Frequency shifting stepped implicitly around the Runge–Kutta stages
    implicit_frequency_shift: bool,
    /// The Runge–Kutta method of the propagation
    time_integrator: SspScheme,
    /// Propagation steps per step, each with its implicit refraction,
    /// shifting and breaking; the other sources run once per step
    substeps: usize,
    /// Land absorbs the waves ([`Self::with_absorbing_land`])
    absorbing_land: bool,
    /// With absorbing land, the dry nodes (`η − B` below the minimum depth)
    /// and the elements all of whose nodes are dry; empty without
    dry_nodes: Vec<bool>,
    dry_elements: Vec<bool>,
}

impl WaveModel2D {
    /// A model on `mesh` over the bed `bathymetry` (still water, no current, no
    /// wind, no sources) with the spectral grid `grid`.
    pub fn new(
        mesh: Arc<Mesh2D>,
        ops: Arc<DGOperators2D>,
        geom: Arc<GeometricFactors2D>,
        bathymetry: &Bathymetry2D,
        grid: SpectralGrid,
        g: f64,
    ) -> Self {
        let n_points = mesh.n_elements * ops.n_nodes;
        assert_eq!(bathymetry.data.len(), n_points, "the bed must be nodal");
        let n_freq = grid.n_freq();
        let mut model = Self {
            sources: SourceTerms::none(g),
            wind: Wind::default(),
            winds: Vec::new(),
            g,
            depth_min: DEFAULT_DEPTH_MIN,
            bed: bathymetry.data.clone(),
            water_level: vec![0.0; n_points],
            depth: vec![0.0; n_points],
            depth_grad: vec![[0.0; 2]; n_points],
            current: vec![[0.0; 2]; n_points],
            current_grad: vec![[0.0; 4]; n_points],
            k: vec![0.0; n_freq * n_points],
            cg: vec![0.0; n_freq * n_points],
            sigma_d: vec![0.0; n_freq * n_points],
            boundary: Boundary::None,
            turning_limit: None,
            spectral_advection: SpectralAdvection::default(),
            implicit_refraction: false,
            implicit_frequency_shift: false,
            time_integrator: SspScheme::Rk3,
            substeps: 1,
            absorbing_land: false,
            dry_nodes: Vec::new(),
            dry_elements: Vec::new(),
            mesh,
            ops,
            geom,
            grid,
        };
        model.set_water_level(&vec![0.0; n_points]);
        model
    }

    pub fn with_sources(mut self, sources: SourceTerms) -> Self {
        self.sources = sources;
        self
    }

    /// The Runge–Kutta method of the propagation (default SSP-RK3).
    /// [`SspScheme::Rk43`] has twice SSP-RK3's SSP coefficient, so its
    /// positivity scaling holds at twice the CFL number, for 4 stages
    /// against 3.
    pub fn with_time_integrator(mut self, integrator: SspScheme) -> Self {
        self.time_integrator = integrator;
        self
    }

    /// Run the propagation in `substeps` equal steps within each step, each
    /// with its implicit refraction and frequency shifting around it and
    /// depth-induced breaking after it, and the other sources once per step
    /// (default 1). [`Self::compute_dt`] gives `substeps` times the
    /// propagation's step.
    ///
    /// Refraction and breaking stay on the short step because the long one
    /// costs accuracy where they are fast (measured at Frøya, a 56 s step of
    /// 4 substeps against a 7 s step, H_s after 6 h): with refraction once
    /// per step, the backward Euler half-steps of 28 s smeared the focusing
    /// on the steep coast (the largest H_s 3.81 m against 3.96 m, 1.7 % RMS
    /// in water over 10 m deep); with breaking once per step, the surf zone
    /// kept up to a minute of incoming energy undissipated (+45 % in water
    /// under 1 m).
    pub fn with_substeps(mut self, substeps: usize) -> Self {
        assert!(substeps >= 1, "at least one propagation step per step");
        self.substeps = substeps;
        self
    }

    /// A uniform wind over the domain.
    pub fn with_wind(mut self, wind: Wind) -> Self {
        self.set_wind(wind);
        self
    }

    /// Replace the wind by a uniform one (e.g. a parent model's, as it
    /// changes).
    pub fn set_wind(&mut self, wind: Wind) {
        self.wind = wind;
        self.winds.clear();
    }

    /// Replace the wind by one per node: the 10 m wind vectors `[u, v]` (m/s,
    /// mesh axes), e.g. a weather model's from
    /// [`GriddedAtmosphere2D::wind_into`](crate::source::GriddedAtmosphere2D::wind_into)
    /// on this model's mesh.
    ///
    /// # Panics
    /// If `wind` is not one per node.
    pub fn set_winds(&mut self, wind: &[[f64; 2]]) {
        assert_eq!(wind.len(), self.n_points(), "one wind per node");
        self.winds.clear();
        self.winds.extend(wind.iter().map(|&[u, v]| Wind {
            u10: u.hypot(v),
            direction: v.atan2(u),
        }));
    }

    /// The uniform wind (what [`Self::set_wind`] set; see [`Self::wind_at`]).
    pub fn wind(&self) -> Wind {
        self.wind
    }

    /// The wind at node `p`.
    #[inline]
    pub fn wind_at(&self, p: usize) -> Wind {
        self.winds.get(p).copied().unwrap_or(self.wind)
    }

    /// The variance density `e[c]` (m²/(rad/s)/rad) of the waves entering through
    /// open faces.
    pub fn with_boundary_spectrum(mut self, e: &[f64]) -> Self {
        assert_eq!(e.len(), self.grid.n_components());
        let nd = self.grid.n_dir();
        self.boundary = Boundary::Uniform(
            e.iter()
                .enumerate()
                .map(|(c, e)| e / self.grid.sigma[c / nd])
                .collect(),
        );
        self
    }

    /// The nodes on open faces, ascending: where [`Self::set_boundary_spectra`]
    /// takes a spectrum each.
    pub fn open_boundary_points(&self) -> Vec<usize> {
        let (nn, mesh, ops) = (self.ops.n_nodes, &*self.mesh, &*self.ops);
        let mut points: Vec<usize> = ElementIndex::iter(mesh.n_elements)
            .flat_map(|k| {
                (0..4)
                    .filter(move |&face| {
                        mesh.neighbor(k, face).is_none()
                            && mesh.boundary_tag(k, face) == Some(BoundaryTag::Open)
                    })
                    .flat_map(move |face| {
                        ops.face_nodes[face]
                            .iter()
                            .map(move |&a| k.as_usize() * nn + a)
                    })
            })
            .collect();
        points.sort_unstable();
        points.dedup();
        points
    }

    /// One variance density spectrum (m²/(rad/s)/rad, `[component]`) per node
    /// of [`Self::open_boundary_points`], in its order (`e` is
    /// `[point][component]`): what enters through the open faces from now on.
    pub fn set_boundary_spectra(&mut self, e: &[f64]) {
        let (nc, nd) = (self.grid.n_components(), self.grid.n_dir());
        if !matches!(self.boundary, Boundary::Nodal { .. }) {
            let points = self.open_boundary_points();
            let mut slot = vec![u32::MAX; self.n_points()];
            for (s, &p) in points.iter().enumerate() {
                slot[p] = s as u32;
            }
            self.boundary = Boundary::Nodal {
                action: vec![0.0; nc * points.len()],
                points,
                slot,
            };
        }
        let Boundary::Nodal { points, action, .. } = &mut self.boundary else {
            unreachable!()
        };
        let n_slots = points.len();
        assert_eq!(
            e.len(),
            n_slots * nc,
            "one spectrum per open-boundary point"
        );
        // In place: a run sets them every step
        for (s, spectrum) in e.chunks_exact(nc).enumerate() {
            for (c, &x) in spectrum.iter().enumerate() {
                action[c * n_slots + s] = x / self.grid.sigma[c / nd];
            }
        }
    }

    /// Water shallower than `depth` (m) is taken this deep.
    pub fn with_depth_min(mut self, depth: f64) -> Self {
        assert!(depth > 0.0);
        self.depth_min = depth;
        let eta = std::mem::take(&mut self.water_level);
        self.set_water_level(&eta);
        self
    }

    /// Let land absorb the waves (off by default): at the nodes where the
    /// water is shallower than the minimum depth (dry nodes), the action is
    /// removed after every propagation step; an element whose nodes are all
    /// dry is not stepped, and its faces are coast to its wet neighbours
    /// (waves leave through them, none come in). Which nodes are dry follows
    /// [`Self::set_water_level`].
    ///
    /// Without it, land is water of the minimum depth: waves running onto it
    /// slow to its group velocity and pile up until the sinks there (bottom
    /// friction, breaking) remove them, so what reaches the land depends on
    /// the floor and on the sinks' details.
    pub fn with_absorbing_land(mut self, on: bool) -> Self {
        self.absorbing_land = on;
        self.update_dry_land();
        self
    }

    /// The dry nodes and elements of [`Self::with_absorbing_land`] from the
    /// water level `depth + B` (unfloored where it is above the floor).
    fn update_dry_land(&mut self) {
        self.dry_nodes.clear();
        self.dry_elements.clear();
        if !self.absorbing_land {
            return;
        }
        let (depth_min, nn) = (self.depth_min, self.ops.n_nodes);
        // `depth` is floored, so a node is dry where the floor applies; the
        // floor itself (exactly `depth_min` of water) counts as dry
        self.dry_nodes
            .extend(self.depth.iter().map(|&d| d <= depth_min));
        self.dry_elements.extend(
            self.dry_nodes
                .chunks_exact(nn)
                .map(|e| e.iter().all(|&d| d)),
        );
    }

    /// Whether node `p` is dry land that absorbs ([`Self::with_absorbing_land`]).
    #[inline]
    fn absorbs_at(&self, p: usize) -> bool {
        self.dry_nodes.get(p).copied().unwrap_or(false)
    }

    /// Whether element `k` is dry land that is not stepped.
    #[inline]
    fn dry_element(&self, k: usize) -> bool {
        self.dry_elements.get(k).copied().unwrap_or(false)
    }

    /// Remove the action at the dry nodes of the node-major state.
    fn absorb_on_land(&self, node_major: &mut [f64]) {
        if self.dry_nodes.is_empty() {
            return;
        }
        let nc = self.grid.n_components();
        for_each_chunk(
            node_major,
            nc,
            || (),
            |_, p, spectrum| {
                if self.dry_nodes[p] {
                    spectrum.fill(0.0);
                }
            },
        );
    }

    /// Cap the refraction rate |c_θ| at `rate` (rad/s): on steep, shallow slopes
    /// it can otherwise set a time step far below the geographic one. SWAN-type
    /// models limit it the same way (e.g. Dietrich et al. 2013). `None` (the
    /// default) leaves refraction exact.
    pub fn with_turning_limit(mut self, rate: Option<f64>) -> Self {
        self.turning_limit = rate;
        self
    }

    /// Step refraction implicitly (see the module docs): the step is no longer
    /// limited by the turning rate, only by the geographic propagation and the
    /// frequency shifting. Off by default.
    pub fn with_implicit_refraction(mut self, on: bool) -> Self {
        self.implicit_refraction = on;
        self
    }

    /// Step frequency shifting implicitly (see the module docs): the step is
    /// no longer limited by `c_σ`. Off by default.
    pub fn with_implicit_frequency_shift(mut self, on: bool) -> Self {
        self.implicit_frequency_shift = on;
        self
    }

    /// The scheme of the direction and frequency advection (van Leer's MUSCL by
    /// default), explicit or as the deferred correction of the implicit steps.
    pub fn with_spectral_advection(mut self, scheme: SpectralAdvection) -> Self {
        self.spectral_advection = scheme;
        self
    }

    /// Gravitational acceleration (m/s²).
    pub fn g(&self) -> f64 {
        self.g
    }

    pub fn n_points(&self) -> usize {
        self.depth.len()
    }

    /// A state with no energy.
    pub fn zero_state(&self) -> WaveSolution {
        WaveSolution::new(self.grid.n_components(), self.n_points())
    }

    /// A state with the variance density `e[c]` at every node.
    pub fn uniform_state(&self, e: &[f64]) -> WaveSolution {
        let nd = self.grid.n_dir();
        let action: Vec<f64> = e
            .iter()
            .enumerate()
            .map(|(c, e)| e / self.grid.sigma[c / nd])
            .collect();
        let mut state = self.zero_state();
        state.fill_uniform(&action);
        state
    }

    /// Depth per node (m), floored at the minimum depth.
    pub fn depth(&self) -> &[f64] {
        &self.depth
    }

    /// Current per node (m/s, mesh x and y).
    pub fn current(&self) -> &[[f64; 2]] {
        &self.current
    }

    /// Wavenumber of frequency `i` at node `p` (rad/m).
    pub fn wavenumber(&self, i: usize, p: usize) -> f64 {
        self.k[i * self.n_points() + p]
    }

    /// Set the surface elevation η (m, per node): the depth `η − B` (floored), its
    /// gradient, the wave kinematics and, with [`Self::with_absorbing_land`],
    /// the dry land follow.
    pub fn set_water_level(&mut self, eta: &[f64]) {
        let n_points = self.n_points();
        assert_eq!(eta.len(), n_points);
        self.water_level.clear();
        self.water_level.extend_from_slice(eta);
        for ((d, &eta), &bed) in self.depth.iter_mut().zip(eta).zip(&self.bed) {
            *d = (eta - bed).max(self.depth_min);
        }
        nodal_gradient(&self.ops, &self.geom, &self.depth, &mut self.depth_grad);
        self.update_dry_land();
        let (g, grid, depth) = (self.g, &self.grid, &self.depth);
        let rows = self
            .k
            .chunks_mut(n_points)
            .zip(self.cg.chunks_mut(n_points))
            .zip(self.sigma_d.chunks_mut(n_points));
        for (i, ((k, cg), sd)) in rows.enumerate() {
            let s = grid.sigma[i];
            for p in 0..n_points {
                let kp = wavenumber(s, depth[p], g);
                k[p] = kp;
                cg[p] = group_velocity(s, kp, depth[p]);
                sd[p] = dsigma_ddepth(s, kp, depth[p]);
            }
        }
    }

    /// Set the current (m/s, per node, mesh x and y).
    pub fn set_currents(&mut self, u: &[f64], v: &[f64]) {
        let n_points = self.n_points();
        assert!(u.len() == n_points && v.len() == n_points);
        for p in 0..n_points {
            self.current[p] = [u[p], v[p]];
        }
        let (mut gu, mut gv) = (vec![[0.0; 2]; n_points], vec![[0.0; 2]; n_points]);
        nodal_gradient(&self.ops, &self.geom, u, &mut gu);
        nodal_gradient(&self.ops, &self.geom, v, &mut gv);
        for p in 0..n_points {
            self.current_grad[p] = [gu[p][0], gu[p][1], gv[p][0], gv[p][1]];
        }
    }

    /// Turning rate `c_θ` (rad/s) of frequency `i` in direction `theta` at node `p`.
    #[inline]
    fn c_theta(&self, i: usize, theta: f64, p: usize) -> f64 {
        let [depth, current] = self.turning_terms(theta.sin_cos(), p);
        self.turning_rate(i, p, depth, current)
    }

    /// The parts of `c_θ` at node `p` that do not depend on the frequency,
    /// in the direction of `(sin θ, cos θ)`: the depth gradient across the
    /// ray and the current's turning, `c_θ = −(∂σ/∂d / k) depth − current`.
    #[inline]
    fn turning_terms(&self, (sn, cs): (f64, f64), p: usize) -> [f64; 2] {
        let [dx, dy] = self.depth_grad[p];
        let [ux, uy, vx, vy] = self.current_grad[p];
        [
            -sn * dx + cs * dy,
            cs * (-sn * ux + cs * uy) + sn * (-sn * vx + cs * vy),
        ]
    }

    /// `c_θ` of frequency `i` at node `p` from its [`Self::turning_terms`].
    #[inline]
    fn turning_rate(&self, i: usize, p: usize, depth: f64, current: f64) -> f64 {
        let ip = i * self.n_points() + p;
        let depth_term = self.sigma_d[ip] / self.k[ip] * depth;
        let c = -depth_term - current;
        match self.turning_limit {
            Some(limit) => c.clamp(-limit, limit),
            None => c,
        }
    }

    /// Frequency shift `c_σ` (rad/s²) of frequency `i` in direction `theta` at node `p`.
    #[inline]
    fn c_sigma(&self, i: usize, theta: f64, p: usize) -> f64 {
        let strain = self.strain_term(theta.sin_cos(), p);
        self.shift_rate(i, p, self.advection_term(p), strain)
    }

    /// The current along the depth gradient `U·∇d` at node `p`.
    #[inline]
    fn advection_term(&self, p: usize) -> f64 {
        let [dx, dy] = self.depth_grad[p];
        let [u, v] = self.current[p];
        u * dx + v * dy
    }

    /// The current's strain along the ray in the direction of `(sin θ,
    /// cos θ)` at node `p`.
    #[inline]
    fn strain_term(&self, (sn, cs): (f64, f64), p: usize) -> f64 {
        let [ux, uy, vx, vy] = self.current_grad[p];
        cs * (cs * ux + sn * uy) + sn * (cs * vx + sn * vy)
    }

    /// `c_σ = ∂σ/∂d U·∇d − c_g k strain` of frequency `i` at node `p`.
    #[inline]
    fn shift_rate(&self, i: usize, p: usize, advection: f64, strain: f64) -> f64 {
        let ip = i * self.n_points() + p;
        self.sigma_d[ip] * advection - self.cg[ip] * self.k[ip] * strain
    }

    /// The largest stable step (s) for Courant number `cfl` (≤ 1 for SSP-RK3): DG
    /// propagation `Δt ≤ cfl h / ((2N + 1) |c_g e_θ + U|)` per element (h = √area),
    /// and the spectral advection `|c_θ| Δt ≤ cfl Δθ`, `|c_σ| Δt ≤ cfl Δσ`, with
    /// half of that for MUSCL (van Leer's reconstruction is TVD, so positive, for
    /// Courant numbers ≤ ½).
    pub fn compute_dt(&self, cfl: f64) -> f64 {
        self.time_step_limits(cfl).dt() * self.substeps as f64
    }

    /// Each bound of [`Self::compute_dt`] and the node that sets it: what
    /// limits the step (e.g. frequency shifting by strong tidal currents).
    pub fn time_step_limits(&self, cfl: f64) -> WaveTimeStepLimits {
        let (n_nodes, n_points) = (self.ops.n_nodes, self.n_points());
        let order_factor = (2 * self.ops.order + 1) as f64;
        let none = WaveTimeStepLimit {
            dt: f64::INFINITY,
            point: 0,
            frequency: 0,
        };
        let mut limits = WaveTimeStepLimits {
            propagation: none,
            refraction: none,
            frequency_shift: none,
        };
        let tighten = |limit: &mut WaveTimeStepLimit, dt: f64, point: usize, frequency: usize| {
            if dt < limit.dt {
                *limit = WaveTimeStepLimit {
                    dt,
                    point,
                    frequency,
                };
            }
        };
        for k in 0..self.mesh.n_elements {
            let h = self.geom.element_size(k);
            for p in (k * n_nodes..(k + 1) * n_nodes).filter(|&p| !self.absorbs_at(p)) {
                let [u, v] = self.current[p];
                let (i, cg_max) = (0..self.grid.n_freq())
                    .map(|i| (i, self.cg[i * n_points + p]))
                    .fold((0, 0.0), |a, b| if b.1 > a.1 { b } else { a });
                let speed = cg_max + u.hypot(v);
                if speed > 0.0 {
                    let dt = cfl * h / (order_factor * speed);
                    tighten(&mut limits.propagation, dt, p, i);
                }
            }
        }
        if self.implicit_refraction && self.implicit_frequency_shift {
            return limits;
        }
        let (dtheta, nd) = (self.grid.d_theta, self.grid.n_dir());
        let cfl = match self.spectral_advection {
            SpectralAdvection::Upwind => cfl,
            SpectralAdvection::VanLeer => 0.5 * cfl,
        };
        for p in (0..n_points).filter(|&p| !self.absorbs_at(p)) {
            for i in 0..self.grid.n_freq() {
                for j in 0..nd {
                    if !self.implicit_refraction {
                        let theta = self.grid.theta[j] + 0.5 * dtheta;
                        let ct = self.c_theta(i, theta, p).abs();
                        if ct > 0.0 {
                            tighten(&mut limits.refraction, cfl * dtheta / ct, p, i);
                        }
                    }
                    if !self.implicit_frequency_shift {
                        let cs = self.c_sigma(i, self.grid.theta[j], p).abs();
                        if cs > 0.0 {
                            let dt = cfl * self.grid.d_sigma[i] / cs;
                            tighten(&mut limits.frequency_shift, dt, p, i);
                        }
                    }
                }
            }
        }
        limits
    }

    /// `out = −∇·((c_g e_θ + U) N) − ∂(c_θ N)/∂θ − ∂(c_σ N)/∂σ` for every
    /// component (no sources). The step does not store it: each Runge–Kutta
    /// stage combines it element by element ([`Self::propagation_stage`]).
    pub fn propagation_rhs_into(&self, n: &WaveSolution, out: &mut WaveSolution) {
        self.propagation_rhs_node_major(&n.data, &mut out.data);
    }

    /// Scale every component in every element towards its mean so that no node is
    /// negative (Zhang & Shu 2010): the mean, and so the total action, is kept; an
    /// element whose mean is negative is zeroed.
    pub fn limit_positivity(&self, n: &mut WaveSolution) {
        self.limit_positivity_node_major(&mut n.data);
    }

    /// Advance `n` by `dt`: the propagation by the Runge–Kutta method
    /// of [`Self::with_time_integrator`] (positivity limited every stage) and
    /// the sources over the step. Implicit refraction and frequency shifting
    /// take half the step before the propagation and half after (Strang).
    ///
    /// In one propagation step the sources follow it. With
    /// [`Self::with_substeps`] (`m ≥ 2`) each propagation step has its own
    /// halves of implicit refraction and shifting and is followed by
    /// breaking over it, and the other sources sit between the first `⌊m/2⌋`
    /// propagation steps and the rest, so for even `m` the step is symmetric
    /// (Strang), and its splitting second order in the step.
    ///
    /// With [`Self::with_absorbing_land`] the dry nodes lose their action at
    /// the start of the step and after every propagation step's passes.
    pub fn step(&self, n: &mut WaveSolution, dt: f64, ws: &mut WaveWorkspace) {
        let substeps = self.substeps;
        let h = dt / substeps as f64;
        // Strang: half of the implicit spectral advection on each side of
        // each propagation step's stages
        let half = |on: bool| on.then_some(0.5 * h);
        let (refraction, shift) = (
            half(self.implicit_refraction),
            half(self.implicit_frequency_shift),
        );
        let sources = self.sources.any().then_some(dt);
        // With substeps breaking follows every propagation step, and the
        // other sources the first `⌊m/2⌋`; in one step all follow it, but
        // breaking on a pass of its own when the limiter caps losses
        let caps_losses = self.sources.limiter.is_some_and(|l| l.caps_losses());
        let breaking =
            ((substeps > 1 || caps_losses) && self.sources.breaking.is_some()).then_some(h);
        let with_sources = (substeps / 2).max(1) - 1;
        // Land that has fallen dry since the last step takes its waves
        self.absorb_on_land(&mut n.data);
        for m in 0..substeps {
            if refraction.is_some() || shift.is_some() {
                profiling::scope("waves: node pass before", || {
                    self.node_pass_node_major(
                        &mut n.data,
                        NodePass::before_stages(refraction, shift),
                    )
                });
            }
            profiling::scope("waves: propagation", || self.propagate(&mut n.data, h, ws));
            let sources = sources.filter(|_| m == with_sources);
            if refraction.is_some() || shift.is_some() || sources.is_some() || breaking.is_some() {
                let pass = NodePass {
                    refraction,
                    shift,
                    sources,
                    breaking,
                    shift_first: false,
                };
                profiling::scope("waves: node pass after", || {
                    self.node_pass_node_major(&mut n.data, pass)
                });
            }
            self.absorb_on_land(&mut n.data);
        }
    }

    /// The propagation of the state `n` over `dt` by the Runge–Kutta method of
    /// [`Self::with_time_integrator`], one pass per stage
    /// ([`Self::propagation_stage`]), the stage values alternating between
    /// `ws.stage` and `ws.next` (the last swapped into `n`).
    fn propagate(&self, n: &mut Vec<f64>, dt: f64, ws: &mut WaveWorkspace) {
        let len = n.len();
        ws.stage.resize(len, 0.0);
        ws.next.resize(len, 0.0);
        for (s, &stage) in self.time_integrator.stages().iter().enumerate() {
            let previous = if s == 0 { &*n } else { &ws.stage };
            self.propagation_stage(n, previous, stage, dt, &mut ws.next);
            std::mem::swap(&mut ws.stage, &mut ws.next);
        }
        std::mem::swap(n, &mut ws.stage);
    }

    /// The sources over `dt` at every node.
    pub fn apply_sources(&self, n: &mut WaveSolution, dt: f64) {
        let pass = NodePass {
            sources: Some(dt),
            ..NodePass::default()
        };
        self.node_pass(n, pass);
    }

    /// Implicit refraction over `dt` at every node (see the module docs),
    /// whether or not the model steps it so.
    pub fn apply_implicit_refraction(&self, n: &mut WaveSolution, dt: f64) {
        let pass = NodePass {
            refraction: Some(dt),
            ..NodePass::default()
        };
        self.node_pass(n, pass);
    }

    /// Implicit frequency shifting over `dt` at every node (see the module
    /// docs), whether or not the model steps it so.
    pub fn apply_implicit_frequency_shift(&self, n: &mut WaveSolution, dt: f64) {
        let pass = NodePass {
            shift: Some(dt),
            ..NodePass::default()
        };
        self.node_pass(n, pass);
    }

    /// At every node, on its spectrum, what `pass` gives: implicit refraction
    /// and frequency shifting (in its order), then the sources; dry land's
    /// action goes after it, as in [`Self::step`].
    fn node_pass(&self, n: &mut WaveSolution, pass: NodePass) {
        self.node_pass_node_major(&mut n.data, pass);
        self.absorb_on_land(&mut n.data);
    }

    /// [`Self::node_pass`] on the node-major state `node_major`.
    fn node_pass_node_major(&self, node_major: &mut [f64], pass: NodePass) {
        let NodePass {
            refraction,
            shift,
            sources,
            breaking,
            shift_first,
        } = pass;
        // With breaking over an interval of its own, the other sources
        // without it
        let terms = match breaking {
            Some(_) => Cow::Owned(self.sources.clone().with_breaking(None)),
            None => Cow::Borrowed(&self.sources),
        };
        let sources = sources.filter(|_| terms.any());
        let breaking = breaking.map(|h| {
            let terms = SourceTerms::none(self.sources.g)
                .with_breaking(self.sources.breaking)
                .with_depth_limit(self.sources.depth_limit)
                .with_integration(self.sources.integration);
            (terms, h)
        });
        let (np, nc, nf) = (
            self.n_points(),
            self.grid.n_components(),
            self.grid.n_freq(),
        );
        let nd = self.grid.n_dir();
        // Nodes per chunk: the DIA's lanes take `LANES` at once
        #[cfg(feature = "simd")]
        let tile = if sources.is_some() && terms.quadruplets.is_some() && vector_kernels() {
            LANES
        } else {
            1
        };
        #[cfg(not(feature = "simd"))]
        let tile = 1;
        for_each_chunk(
            node_major,
            tile * nc,
            || {
                (
                    vec![0.0; tile * nf],
                    Vec::with_capacity(tile),
                    Vec::with_capacity(tile),
                    SourceScratch::new(nc),
                    CyclicScratch::new(nd),
                    CyclicScratch::new(nf),
                    vec![0.0; nf],
                    NodeRates::new(&self.grid),
                )
            },
            |(k, depths, winds, scratch, cyclic, banded, column, rates), chunk, nodes| {
                let first = chunk * tile;
                for (q, spectrum) in nodes.chunks_exact_mut(nc).enumerate() {
                    let p = first + q;
                    // Dry land's action goes after the pass (`absorb_on_land`)
                    if self.absorbs_at(p) {
                        continue;
                    }
                    rates.fill(self, p, refraction.is_some(), shift.is_some());
                    let rates = &*rates;
                    let mut shift_now = |spectrum: &mut [f64]| {
                        if let Some(dt) = shift {
                            #[cfg(feature = "simd")]
                            if vector_kernels()
                                && self.shift_lanes(p, spectrum, dt, rates.advection, &rates.strain)
                            {
                                return;
                            }
                            for j in 0..nd {
                                for (i, x) in column.iter_mut().enumerate() {
                                    *x = spectrum[i * nd + j];
                                }
                                let strain = rates.strain[j];
                                self.shift_implicitly(
                                    p,
                                    column,
                                    dt,
                                    rates.advection,
                                    strain,
                                    banded,
                                );
                                for (i, x) in column.iter().enumerate() {
                                    spectrum[i * nd + j] = *x;
                                }
                            }
                        }
                    };
                    if shift_first {
                        shift_now(spectrum);
                    }
                    if let Some(dt) = refraction {
                        #[cfg(feature = "simd")]
                        let done =
                            vector_kernels() && self.refract_lanes(p, spectrum, dt, &rates.turning);
                        #[cfg(not(feature = "simd"))]
                        let done = false;
                        if !done {
                            for (i, row) in spectrum.chunks_exact_mut(nd).enumerate() {
                                self.refract_implicitly(i, p, row, dt, &rates.turning, cyclic);
                            }
                        }
                    }
                    if !shift_first {
                        shift_now(spectrum);
                    }
                }
                // The sources at all the chunk's nodes (at once in the DIA's
                // lanes), then breaking over its own interval
                let width = nodes.len() / nc;
                let on_land = (first..first + width).all(|p| self.absorbs_at(p));
                if (sources.is_some() || breaking.is_some()) && !on_land {
                    for q in 0..width {
                        for (i, k) in k[q * nf..(q + 1) * nf].iter_mut().enumerate() {
                            *k = self.k[i * np + first + q];
                        }
                    }
                    depths.clear();
                    depths.extend((first..first + width).map(|p| self.depth[p]));
                    winds.clear();
                    winds.extend((first..first + width).map(|p| self.wind_at(p)));
                    if let Some(dt) = sources {
                        let at = Nodes { k, depths, winds };
                        terms.integrate_at(&self.grid, nodes, at, dt, scratch, tile > 1);
                    }
                    if let Some((terms, h)) = &breaking {
                        let at = Nodes { k, depths, winds };
                        terms.integrate_at(&self.grid, nodes, at, *h, scratch, false);
                    }
                }
            },
        );
    }

    /// Backward Euler over `dt` for the direction advection of frequency `i` at
    /// node `p`, on its action densities over the directions `row`: first-order
    /// upwind fluxes `c⁺ N_j + c⁻ N_{j+1}` through the faces `θ_j + Δθ/2`,
    /// periodic, so
    ///
    /// ```text
    /// N_j + λ (c⁺_{j+½} − c⁻_{j−½}) N_j + λ c⁻_{j+½} N_{j+1} − λ c⁺_{j−½} N_{j−1} = N*_j
    /// ```
    ///
    /// with `λ = Δt/Δθ`: a cyclic tridiagonal M-matrix whose columns sum to one.
    /// `turning` holds the node's [`Self::turning_terms`] at the faces.
    fn refract_implicitly(
        &self,
        i: usize,
        p: usize,
        row: &mut [f64],
        dt: f64,
        turning: &[[f64; 2]],
        s: &mut CyclicScratch,
    ) {
        let nd = row.len();
        let lambda = dt / self.grid.d_theta;
        // The turning rate at the face above each bin
        for (c, &[depth, current]) in s.faces.iter_mut().zip(turning) {
            *c = self.turning_rate(i, p, depth, current);
        }
        if s.faces.iter().all(|&c| c == 0.0) {
            return;
        }
        // The periodic neighbours (a branch, not a remainder: an integer
        // division per access was most of this kernel's cost)
        let previous = |j: usize| if j == 0 { nd - 1 } else { j - 1 };
        let next = |j: usize| if j + 1 == nd { 0 } else { j + 1 };
        for j in 0..nd {
            let (above, below) = (s.faces[j], s.faces[previous(j)]);
            s.diagonal[j] = 1.0 + lambda * (above.max(0.0) - below.min(0.0));
            s.upper[j] = lambda * above.min(0.0);
            s.lower[j] = -lambda * below.max(0.0);
        }
        if self.spectral_advection == SpectralAdvection::VanLeer {
            // Deferred correction: MUSCL's flux less upwind's, of the state now
            for j in 0..nd {
                let (c, left, right) = (s.faces[j], row[j], row[next(j)]);
                let (far_left, far_right) = (row[previous(j)], row[next(next(j))]);
                let muscl = face_flux(c, Some(far_left), left, right, Some(far_right));
                s.correction[j] = muscl - (c.max(0.0) * left + c.min(0.0) * right);
            }
            // Limit each face's correction by its donor (Zalesak 1979): what
            // the corrections take out of a bin is at most what it holds, so
            // the right-hand side stays non-negative, and as fluxes they keep
            // the action
            for j in 0..nd {
                let below = s.correction[previous(j)];
                let taken = lambda * (s.correction[j].max(0.0) + (-below).max(0.0));
                s.divergence[j] = if taken > row[j] { row[j] / taken } else { 1.0 };
            }
            for j in 0..nd {
                let c = s.correction[j];
                let donor = if c >= 0.0 { j } else { next(j) };
                s.correction[j] = c * s.divergence[donor];
            }
            for j in 0..nd {
                let d = lambda * (s.correction[j] - s.correction[previous(j)]);
                row[j] = (row[j] - d).max(0.0);
            }
        }
        solve_cyclic_tridiagonal(row, s);
    }

    /// Backward Euler over `dt` for the frequency advection of direction `j`
    /// at node `p`, on its action densities over the frequencies `column`:
    /// first-order upwind fluxes `c⁺ N_i + c⁻ N_{i+1}` through the faces
    /// between the bins (the speed the mean of the two bins'), and through
    /// the ends of the grid outflow only (`c⁻ N_0` below, `c⁺ N_{nf−1}`
    /// above, as the explicit scheme), so
    ///
    /// ```text
    /// N_i + λ_i (F_{i+½} − F_{i−½}) = N*_i,   λ_i = Δt/Δσ_i
    /// ```
    ///
    /// a tridiagonal M-matrix. With MUSCL, the deferred correction of
    /// [`Self::refract_implicitly`] on the faces between the bins. `advection`
    /// and `strain` are the node's [`Self::advection_term`] and the
    /// direction's [`Self::strain_term`].
    fn shift_implicitly(
        &self,
        p: usize,
        column: &mut [f64],
        dt: f64,
        advection: f64,
        strain: f64,
        s: &mut CyclicScratch,
    ) {
        let nf = column.len();
        // c_σ of each bin, then the faces above each bin but the last
        for (i, c) in s.bins.iter_mut().enumerate() {
            *c = self.shift_rate(i, p, advection, strain);
        }
        if s.bins.iter().all(|&c| c == 0.0) {
            return;
        }
        for i in 0..nf - 1 {
            s.faces[i] = 0.5 * (s.bins[i] + s.bins[i + 1]);
        }
        // Outflow only through the ends
        let (bottom, top) = (s.bins[0].min(0.0), s.bins[nf - 1].max(0.0));
        s.faces[nf - 1] = top;
        for i in 0..nf {
            let lambda = dt / self.grid.d_sigma[i];
            let above = s.faces[i];
            let below = if i > 0 { s.faces[i - 1] } else { bottom };
            s.diagonal[i] = 1.0 + lambda * (above.max(0.0) - below.min(0.0));
            s.upper[i] = if i + 1 < nf {
                lambda * above.min(0.0)
            } else {
                0.0
            };
            s.lower[i] = if i > 0 { -lambda * below.max(0.0) } else { 0.0 };
        }
        if self.spectral_advection == SpectralAdvection::VanLeer {
            // Deferred correction on the faces between the bins (first order at
            // the ends, as the explicit scheme)
            s.correction[nf - 1] = 0.0;
            for i in 0..nf - 1 {
                let (c, left, right) = (s.faces[i], column[i], column[i + 1]);
                let far_left = (i > 0).then(|| column[i - 1]);
                let far_right = (i + 2 < nf).then(|| column[i + 2]);
                let muscl = face_flux(c, far_left, left, right, far_right);
                s.correction[i] = muscl - (c.max(0.0) * left + c.min(0.0) * right);
            }
            // Each face limited by its donor (Zalesak 1979)
            for (i, (&x, &d_sigma)) in column.iter().zip(&self.grid.d_sigma).enumerate() {
                let below = if i > 0 { s.correction[i - 1] } else { 0.0 };
                let taken = dt / d_sigma * (s.correction[i].max(0.0) + (-below).max(0.0));
                s.divergence[i] = if taken > x { x / taken } else { 1.0 };
            }
            for i in 0..nf - 1 {
                let c = s.correction[i];
                let donor = if c >= 0.0 { i } else { i + 1 };
                s.correction[i] = c * s.divergence[donor];
            }
            for (i, (x, &d_sigma)) in column.iter_mut().zip(&self.grid.d_sigma).enumerate() {
                let below = if i > 0 { s.correction[i - 1] } else { 0.0 };
                let d = dt / d_sigma * (s.correction[i] - below);
                *x = (*x - d).max(0.0);
            }
        }
        solve_cyclic_tridiagonal(column, s);
    }

    /// The variance density `E = σ N` at node `p` into `e`.
    pub fn energy_spectrum_into(&self, n: &WaveSolution, p: usize, e: &mut [f64]) {
        e.copy_from_slice(n.spectrum(p));
        let nd = self.grid.n_dir();
        for (c, x) in e.iter_mut().enumerate() {
            *x *= self.grid.sigma[c / nd];
        }
    }

    /// Integrated parameters at every node.
    pub fn parameters(&self, n: &WaveSolution) -> Vec<WaveParameters> {
        let mut e = vec![0.0; self.grid.n_components()];
        (0..self.n_points())
            .map(|p| {
                self.energy_spectrum_into(n, p, &mut e);
                self.grid.parameters(&e)
            })
            .collect()
    }

    /// Stokes drift (m/s) at height `z` (m, ≤ 0) below the surface at every node.
    pub fn stokes_drift(&self, n: &WaveSolution, z: f64) -> Vec<[f64; 2]> {
        let (np, nf) = (self.n_points(), self.grid.n_freq());
        let mut e = vec![0.0; self.grid.n_components()];
        let mut k = vec![0.0; nf];
        (0..np)
            .map(|p| {
                self.energy_spectrum_into(n, p, &mut e);
                (0..nf).for_each(|i| k[i] = self.k[i * np + p]);
                self.grid
                    .stokes_drift(&e, &k, self.depth[p], z.max(-self.depth[p]))
            })
            .collect()
    }

    /// Surface roughness `alpha · H_s` (m) at every node, for the turbulence
    /// closure of the circulation (Terray et al. 1996: `alpha ≈ 0.6`; see
    /// [`crate::physics::Hydrostatic3D::with_surface_roughness`]).
    pub fn surface_roughness(&self, n: &WaveSolution, alpha: f64) -> Vec<f64> {
        self.parameters(n).iter().map(|p| alpha * p.hs).collect()
    }

    /// The waves' bed stress per ρ (m²/s²) at every node on a bed of roughness
    /// length `z0` (m), Soulsby's `½ f_w U_w²` ([`SpectralGrid::bed_wave_stress`]),
    /// for [`crate::source::WaveCurrentFriction2D`].
    pub fn bed_wave_stress(&self, n: &WaveSolution, z0: f64) -> Vec<f64> {
        let (np, nf) = (self.n_points(), self.grid.n_freq());
        let mut e = vec![0.0; self.grid.n_components()];
        let mut k = vec![0.0; nf];
        (0..np)
            .map(|p| {
                self.energy_spectrum_into(n, p, &mut e);
                (0..nf).for_each(|i| k[i] = self.k[i * np + p]);
                self.grid.bed_wave_stress(&e, &k, self.depth[p], z0)
            })
            .collect()
    }

    /// Radiation stress per ρg `[S_xx, S_xy, S_yy]` (m²) at every node.
    pub fn radiation_stress(&self, n: &WaveSolution) -> Vec<[f64; 3]> {
        let (np, nf) = (self.n_points(), self.grid.n_freq());
        let mut e = vec![0.0; self.grid.n_components()];
        let mut k = vec![0.0; nf];
        (0..np)
            .map(|p| {
                self.energy_spectrum_into(n, p, &mut e);
                (0..nf).for_each(|i| k[i] = self.k[i * np + p]);
                self.grid.radiation_stress(&e, &k, self.depth[p])
            })
            .collect()
    }

    /// What the model's sinks take from the wave state `n` per second at
    /// every node over a step `dt` (s; 0 for the instantaneous rate), by
    /// term, and the momentum they take with it
    /// ([`SourceTerms::dissipation`], [`Dissipation`]).
    pub fn dissipation(&self, n: &WaveSolution, dt: f64) -> Vec<Dissipation> {
        let (np, nf, nc) = (
            self.n_points(),
            self.grid.n_freq(),
            self.grid.n_components(),
        );
        let (mut e, mut a, mut b) = (vec![0.0; nc], vec![0.0; nc], vec![0.0; nc]);
        let mut k = vec![0.0; nf];
        (0..np)
            .map(|p| {
                self.energy_spectrum_into(n, p, &mut e);
                (0..nf).for_each(|i| k[i] = self.k[i * np + p]);
                self.sources
                    .dissipation(&self.grid, &e, &k, self.depth[p], dt, &mut a, &mut b)
            })
            .collect()
    }

    /// Total action `Σ_c Δσ Δθ ∫ N_c dA` (m⁴·s/rad), conserved by the propagation
    /// up to boundary and spectral-end fluxes.
    pub fn total_action(&self, n: &WaveSolution) -> f64 {
        let (nn, nd, nc) = (
            self.ops.n_nodes,
            self.grid.n_dir(),
            self.grid.n_components(),
        );
        let mut total = 0.0;
        for c in 0..nc {
            let mut integral = 0.0;
            for k in 0..self.mesh.n_elements {
                for a in 0..nn {
                    let x = n.data[(k * nn + a) * nc + c];
                    integral += self.ops.weights[a] * self.geom.jacobian(k, a) * x;
                }
            }
            total += integral * self.grid.d_sigma[c / nd] * self.grid.d_theta;
        }
        total
    }
}

/// The upwind flux `speed · N_face` through the face between the bins `left` and
/// `right`. With the bins beyond them (`far_left`, `far_right`) the face value
/// is the upwind bin's, reconstructed linearly with van Leer's limited slope
/// (`2ab/(a + b)` of the one-sided differences, 0 at an extremum): second
/// order where smooth, TVD; without, first-order upwind.
#[inline]
fn face_flux(
    speed: f64,
    far_left: Option<f64>,
    left: f64,
    right: f64,
    far_right: Option<f64>,
) -> f64 {
    // Half of van Leer's slope from the one-sided differences a and b
    let half_slope = |a: f64, b: f64| if a * b > 0.0 { a * b / (a + b) } else { 0.0 };
    if speed >= 0.0 {
        let value = far_left.map_or(left, |ll| left + half_slope(left - ll, right - left));
        speed * value
    } else {
        let value = far_right.map_or(right, |rr| right - half_slope(right - left, rr - right));
        speed * value
    }
}

/// What [`WaveModel2D::node_pass`] does at every node: implicit refraction
/// and frequency shifting over their intervals, the shift first or last, then
/// the sources.
#[derive(Clone, Copy, Debug, Default)]
struct NodePass {
    refraction: Option<f64>,
    shift: Option<f64>,
    sources: Option<f64>,
    /// Depth-induced breaking over this interval, after the other sources
    /// (which then leave it out), instead of with them
    breaking: Option<f64>,
    shift_first: bool,
}

impl NodePass {
    /// The half-steps before the stages: the reverse order of those after
    /// them (frequency shift, then refraction), for a symmetric splitting.
    fn before_stages(refraction: Option<f64>, shift: Option<f64>) -> Self {
        Self {
            refraction,
            shift,
            sources: None,
            breaking: None,
            shift_first: true,
        }
    }
}

/// What the implicit spectral advection at one node needs of each direction,
/// for all frequencies: `c_θ` and `c_σ` separate into a frequency's factors
/// (`∂σ/∂d / k`, `c_g k`) and terms of the node and the direction, which are
/// computed once per node instead of once per frequency, and the sines and
/// cosines of the directions once per pass.
struct NodeRates {
    /// `(sin, cos)` of the direction faces `θ_j + Δθ/2` and of the bins `θ_j`
    face_angles: Vec<(f64, f64)>,
    bin_angles: Vec<(f64, f64)>,
    /// [`WaveModel2D::turning_terms`] at each face
    turning: Vec<[f64; 2]>,
    /// [`WaveModel2D::strain_term`] at each bin
    strain: Vec<f64>,
    /// [`WaveModel2D::advection_term`]
    advection: f64,
}

impl NodeRates {
    fn new(grid: &SpectralGrid) -> Self {
        let nd = grid.n_dir();
        Self {
            face_angles: grid
                .theta
                .iter()
                .map(|&t| (t + 0.5 * grid.d_theta).sin_cos())
                .collect(),
            bin_angles: grid.theta.iter().map(|t| t.sin_cos()).collect(),
            turning: vec![[0.0; 2]; nd],
            strain: vec![0.0; nd],
            advection: 0.0,
        }
    }

    /// The terms of `model` at node `p` for refraction and for frequency
    /// shifting, as asked.
    fn fill(&mut self, model: &WaveModel2D, p: usize, refraction: bool, shift: bool) {
        if refraction {
            for (t, &angle) in self.turning.iter_mut().zip(&self.face_angles) {
                *t = model.turning_terms(angle, p);
            }
        }
        if shift {
            for (s, &angle) in self.strain.iter_mut().zip(&self.bin_angles) {
                *s = model.strain_term(angle, p);
            }
            self.advection = model.advection_term(p);
        }
    }
}

/// Storage of the implicit refraction at one node: the turning rates at the
/// faces, a cyclic tridiagonal system over the directions and its solve.
struct CyclicScratch {
    faces: Vec<f64>,
    /// The frequency shift of each bin (frequency columns only)
    bins: Vec<f64>,
    /// MUSCL's flux less upwind's through each face, then its divergence
    correction: Vec<f64>,
    divergence: Vec<f64>,
    /// `lower[j] x[j−1] + diagonal[j] x[j] + upper[j] x[j+1]`, indices periodic
    lower: Vec<f64>,
    diagonal: Vec<f64>,
    upper: Vec<f64>,
    modified: Vec<f64>,
    c_prime: Vec<f64>,
    x: Vec<f64>,
    u: Vec<f64>,
    z: Vec<f64>,
}

impl CyclicScratch {
    fn new(n: usize) -> Self {
        Self {
            faces: vec![0.0; n],
            bins: vec![0.0; n],
            correction: vec![0.0; n],
            divergence: vec![0.0; n],
            lower: vec![0.0; n],
            diagonal: vec![0.0; n],
            upper: vec![0.0; n],
            modified: vec![0.0; n],
            c_prime: vec![0.0; n],
            x: vec![0.0; n],
            u: vec![0.0; n],
            z: vec![0.0; n],
        }
    }
}

/// The Thomas algorithm: `lower[j] x[j−1] + diagonal[j] x[j] + upper[j] x[j+1]
/// = rhs[j]` (`lower[0]` and `upper[n−1]` ignored) into `x`. Stable without
/// pivoting for diagonally dominant rows or columns.
fn thomas(
    lower: &[f64],
    diagonal: &[f64],
    upper: &[f64],
    rhs: &[f64],
    x: &mut [f64],
    c_prime: &mut [f64],
) {
    let n = diagonal.len();
    c_prime[0] = upper[0] / diagonal[0];
    x[0] = rhs[0] / diagonal[0];
    for j in 1..n {
        let m = 1.0 / (diagonal[j] - lower[j] * c_prime[j - 1]);
        c_prime[j] = upper[j] * m;
        x[j] = (rhs[j] - lower[j] * x[j - 1]) * m;
    }
    for j in (0..n - 1).rev() {
        x[j] -= c_prime[j] * x[j + 1];
    }
}

/// [`thomas`] for two right-hand sides at once: one elimination, its
/// reciprocals shared, the same operations on each (so each solution is
/// [`thomas`]'s bit for bit).
fn thomas_pair(
    lower: &[f64],
    diagonal: &[f64],
    upper: &[f64],
    [ra, rb]: [&[f64]; 2],
    [xa, xb]: [&mut [f64]; 2],
    c_prime: &mut [f64],
) {
    let n = diagonal.len();
    c_prime[0] = upper[0] / diagonal[0];
    xa[0] = ra[0] / diagonal[0];
    xb[0] = rb[0] / diagonal[0];
    for j in 1..n {
        let m = 1.0 / (diagonal[j] - lower[j] * c_prime[j - 1]);
        c_prime[j] = upper[j] * m;
        xa[j] = (ra[j] - lower[j] * xa[j - 1]) * m;
        xb[j] = (rb[j] - lower[j] * xb[j - 1]) * m;
    }
    for j in (0..n - 1).rev() {
        xa[j] -= c_prime[j] * xa[j + 1];
        xb[j] -= c_prime[j] * xb[j + 1];
    }
}

/// Solve the cyclic tridiagonal system of `s` (`lower[0]` couples to
/// `x[n−1]`, `upper[n−1]` to `x[0]`) for the right-hand side `rhs`, in place:
/// the Thomas algorithm on the system without its corners, corrected by
/// Sherman–Morrison (Press et al., "Numerical Recipes", §2.7).
fn solve_cyclic_tridiagonal(rhs: &mut [f64], s: &mut CyclicScratch) {
    let n = rhs.len();
    // The corners A[n−1][0] and A[0][n−1]
    let (alpha, beta) = (s.upper[n - 1], s.lower[0]);
    if alpha == 0.0 && beta == 0.0 {
        thomas(
            &s.lower,
            &s.diagonal,
            &s.upper,
            rhs,
            &mut s.x,
            &mut s.c_prime,
        );
        rhs.copy_from_slice(&s.x);
        return;
    }
    let gamma = -s.diagonal[0];
    s.modified.copy_from_slice(&s.diagonal);
    s.modified[0] -= gamma;
    s.modified[n - 1] -= alpha * beta / gamma;
    s.u.fill(0.0);
    s.u[0] = gamma;
    s.u[n - 1] = alpha;
    thomas_pair(
        &s.lower,
        &s.modified,
        &s.upper,
        [rhs, &s.u],
        [&mut s.x, &mut s.z],
        &mut s.c_prime,
    );
    let fact = (s.x[0] + beta * s.x[n - 1] / gamma) / (1.0 + s.z[0] + beta * s.z[n - 1] / gamma);
    for (r, (x, z)) in rhs.iter_mut().zip(s.x.iter().zip(&s.z)) {
        *r = x - fact * z;
    }
}

/// Physical gradient of the nodal field `f` by each element's derivative matrices.
pub(crate) fn nodal_gradient(
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    f: &[f64],
    out: &mut [[f64; 2]],
) {
    let nn = ops.n_nodes;
    for k in 0..f.len() / nn {
        let fk = &f[k * nn..(k + 1) * nn];
        for a in 0..nn {
            let (dr, ds) = (
                &ops.dr_row_major[a * nn..(a + 1) * nn],
                &ops.ds_row_major[a * nn..(a + 1) * nn],
            );
            let fr: f64 = dr.iter().zip(fk).map(|(d, f)| d * f).sum();
            let fs: f64 = ds.iter().zip(fk).map(|(d, f)| d * f).sum();
            let ((rx, ry), (sx, sy)) = (geom.grad_r(k, a), geom.grad_s(k, a));
            out[k * nn + a] = [rx * fr + sx * fs, ry * fr + sy * fs];
        }
    }
}

/// `f(scratch, index, chunk)` for every `size`-long chunk of `data`, in parallel
/// with the `parallel` feature, with scratch from `init` per worker.
fn for_each_chunk<T, S, I, F>(data: &mut [T], size: usize, init: I, f: F)
where
    T: Send,
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize, &mut [T]) + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        data.par_chunks_mut(size)
            .enumerate()
            .for_each_init(init, |s, (i, chunk)| f(s, i, chunk));
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut s = init();
        for (i, chunk) in data.chunks_mut(size).enumerate() {
            f(&mut s, i, chunk);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh2D;

    /// The cyclic solve against the matrix it inverts: residual at round-off,
    /// with both corners set and with one (as the upwind refraction has).
    #[test]
    fn the_cyclic_solve_inverts_its_matrix() {
        let n = 9;
        for corners in [(0.3, -0.7), (0.0, -0.4), (0.25, 0.0)] {
            let mut s = CyclicScratch::new(n);
            for j in 0..n {
                let t = j as f64;
                s.lower[j] = -0.2 - 0.1 * (1.3 * t).sin().abs();
                s.upper[j] = -0.3 + 0.05 * (0.7 * t).cos();
                s.diagonal[j] = 1.5 + 0.2 * (0.4 * t).sin();
            }
            s.upper[n - 1] = corners.0;
            s.lower[0] = corners.1;
            let rhs: Vec<f64> = (0..n).map(|j| 1.0 + (j as f64).sqrt()).collect();
            let mut x = rhs.clone();
            let (lower, diagonal, upper) = (s.lower.clone(), s.diagonal.clone(), s.upper.clone());
            solve_cyclic_tridiagonal(&mut x, &mut s);
            for j in 0..n {
                let ax =
                    lower[j] * x[(j + n - 1) % n] + diagonal[j] * x[j] + upper[j] * x[(j + 1) % n];
                assert!(
                    (ax - rhs[j]).abs() < 1e-13 * rhs[j],
                    "row {j}: {ax} against {}",
                    rhs[j]
                );
            }
        }
    }

    /// Implicit refraction over a step 10⁴ times the geographic limit, on a
    /// steep slope: the action of every frequency at every node is kept to
    /// round-off, nothing goes negative, and the waves have turned.
    #[test]
    fn implicit_refraction_keeps_the_action_at_any_step() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 200.0, 0.0, 200.0, 2, 2);
        let ops = Arc::new(DGOperators2D::new(2));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bathymetry =
            Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| -(2.0 + 0.1 * y + 0.02 * x));
        let grid = SpectralGrid::new(0.06, 0.3, 6, 36);
        let model = WaveModel2D::new(Arc::new(mesh), ops, geom, &bathymetry, grid, 9.81)
            .with_implicit_refraction(true);
        let e = model.grid.jonswap(1.0, 8.0, 3.3, 0.3, 2.0);
        let mut n = model.uniform_state(&e);
        let before = n.clone();
        let explicit = model.compute_dt(1.0).min(1.0);
        model.apply_implicit_refraction(&mut n, 1e4 * explicit);
        let (nf, nd, np) = (model.grid.n_freq(), model.grid.n_dir(), model.n_points());
        let mut turned = 0.0f64;
        for p in 0..np {
            for i in 0..nf {
                let sum =
                    |s: &WaveSolution| -> f64 { (0..nd).map(|j| s.spectrum(p)[i * nd + j]).sum() };
                let (a, b) = (sum(&before), sum(&n));
                assert!(
                    (a - b).abs() <= 1e-13 * a.max(1e-300),
                    "node {p}, frequency {i}"
                );
                for j in 0..nd {
                    let c = i * nd + j;
                    assert!(n.spectrum(p)[c] >= 0.0);
                    turned = turned.max((n.spectrum(p)[c] - before.spectrum(p)[c]).abs());
                }
            }
        }
        assert!(turned > 1e-3, "nothing turned ({turned})");
    }

    /// Implicit frequency shifting at 10⁴ × the explicit step, by a current
    /// over a sloping bed (both terms of `c_σ`): every node's spectrum stays
    /// non-negative, and each direction's action `Σ Δσ_i N_i` changes by
    /// exactly what flows out through the ends of the grid in the backward
    /// Euler step (`Δt (c⁺_top N_top − c⁻_bottom N_0)` of the new state; the
    /// MUSCL correction moves action between bins only), to round-off.
    #[test]
    fn implicit_frequency_shifting_keeps_the_action_at_any_step() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 200.0, 0.0, 200.0, 2, 2);
        let ops = Arc::new(DGOperators2D::new(2));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bathymetry =
            Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| -(1.0 + 0.02 * y + 0.01 * x));
        let grid = SpectralGrid::new(0.06, 0.5, 12, 18);
        let mut model = WaveModel2D::new(Arc::new(mesh), ops, geom, &bathymetry, grid, 9.81)
            .with_implicit_frequency_shift(true);
        let np = model.n_points();
        let xy: Vec<[f64; 2]> = (0..model.mesh.n_elements)
            .flat_map(|k| {
                let (mesh, ops) = (&model.mesh, &model.ops);
                (0..ops.n_nodes).map(move |i| {
                    mesh.reference_to_physical(ElementIndex::new(k), ops.nodes_r[i], ops.nodes_s[i])
                })
            })
            .collect();
        let u: Vec<f64> = xy.iter().map(|p| 0.3 + 1e-3 * p[1]).collect();
        let v: Vec<f64> = xy.iter().map(|p| -0.2 + 2e-3 * p[0]).collect();
        model.set_currents(&u, &v);
        let e = model.grid.jonswap(0.5, 4.0, 3.3, 0.3, 2.0);
        let mut n = model.uniform_state(&e);
        let before = n.clone();
        // The explicit scheme's bound
        model.implicit_frequency_shift = false;
        let explicit = model.time_step_limits(1.0).frequency_shift.dt;
        model.implicit_frequency_shift = true;
        let dt = 1e4 * explicit;
        model.apply_implicit_frequency_shift(&mut n, dt);
        let (nf, nd) = (model.grid.n_freq(), model.grid.n_dir());
        let mut shifted = 0.0f64;
        for p in 0..np {
            for j in 0..nd {
                let theta = model.grid.theta[j];
                let action = |s: &WaveSolution| -> f64 {
                    (0..nf)
                        .map(|i| model.grid.d_sigma[i] * s.spectrum(p)[i * nd + j])
                        .sum()
                };
                let (bottom, top) = (
                    model.c_sigma(0, theta, p).min(0.0),
                    model.c_sigma(nf - 1, theta, p).max(0.0),
                );
                let outflow =
                    dt * (top * n.spectrum(p)[(nf - 1) * nd + j] - bottom * n.spectrum(p)[j]);
                let (a, b) = (action(&before), action(&n));
                assert!(
                    (a - b - outflow).abs() <= 1e-12 * a,
                    "node {p}, direction {j}: {a} → {b}, outflow {outflow}"
                );
                for i in 0..nf {
                    let c = i * nd + j;
                    assert!(n.spectrum(p)[c] >= 0.0);
                    shifted = shifted.max((n.spectrum(p)[c] - before.spectrum(p)[c]).abs());
                }
            }
        }
        assert!(shifted > 1e-3, "nothing shifted ({shifted})");
    }

    /// A new minimum depth keeps the water level: land stays as dry as it
    /// was (it took the floored depth for water, so land rose with the old
    /// floor, η = 0.1 m + B).
    #[test]
    fn a_new_depth_floor_keeps_the_water_level() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 200.0, 0.0, 100.0, 2, 1);
        let ops = Arc::new(DGOperators2D::new(1));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bed = |x: f64, _: f64| -5.0 + 0.04 * x;
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, bed);
        let grid = SpectralGrid::new(0.08, 0.3, 4, 8);
        let mut model = WaveModel2D::new(Arc::new(mesh), ops, geom, &bathymetry, grid, 9.81);
        let eta = vec![0.5; model.n_points()];
        model.set_water_level(&eta);
        let model = model.with_depth_min(0.01).with_absorbing_land(true);
        for (p, (&d, &b)) in model.depth().iter().zip(&bathymetry.data).enumerate() {
            assert_eq!(d, (0.5 - b).max(0.01), "node {p}");
            assert_eq!(model.absorbs_at(p), 0.5 - b <= 0.01, "node {p}");
        }
        assert!((0..model.n_points()).any(|p| model.absorbs_at(p)));
    }

    /// The node passes on their own (`apply_sources`, `apply_implicit_*`)
    /// empty dry land as `step` does, and leave every wet node as on floored
    /// land, bit for bit (each node's pass is its own).
    #[test]
    fn node_passes_absorb_on_land_and_leave_the_water_alone() {
        // P1 on a beach: the first two columns of elements dry, the third
        // partly, the rest wet
        let mesh = Arc::new(Mesh2D::uniform_rectangle(0.0, 400.0, 0.0, 100.0, 8, 2));
        let ops = Arc::new(DGOperators2D::new(1));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bed = |x: f64, y: f64| 2.0 - 0.02 * x + 1e-3 * y;
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, bed);
        let model = |absorbing: bool| {
            let grid = SpectralGrid::new(0.08, 0.5, 12, 12);
            let mut model = WaveModel2D::new(
                mesh.clone(),
                ops.clone(),
                geom.clone(),
                &bathymetry,
                grid,
                9.81,
            )
            .with_sources(SourceTerms::swan_defaults(9.81))
            .with_implicit_refraction(true)
            .with_implicit_frequency_shift(true)
            .with_absorbing_land(absorbing);
            model.set_wind(Wind {
                u10: 15.0,
                direction: 0.3,
            });
            let np = model.n_points();
            let u: Vec<f64> = (0..np).map(|p| 0.2 + 0.01 * (p % 7) as f64).collect();
            let v: Vec<f64> = (0..np).map(|p| -0.1 + 0.02 * (p % 5) as f64).collect();
            model.set_currents(&u, &v);
            model
        };
        let (absorbing, floored) = (model(true), model(false));
        let np = absorbing.n_points();
        let dry = (0..np).filter(|&p| absorbing.absorbs_at(p)).count();
        assert!(dry > 0 && dry < np / 2, "{dry} of {np} nodes dry");
        let e = absorbing.grid.jonswap(1.0, 6.0, 3.3, 0.3, 2.0);
        type Pass = fn(&WaveModel2D, &mut WaveSolution, f64);
        let passes: [(&str, Pass); 3] = [
            ("sources", WaveModel2D::apply_sources),
            ("refraction", WaveModel2D::apply_implicit_refraction),
            ("shift", WaveModel2D::apply_implicit_frequency_shift),
        ];
        for (name, pass) in passes {
            let (mut a, mut f) = (absorbing.uniform_state(&e), floored.uniform_state(&e));
            pass(&absorbing, &mut a, 30.0);
            pass(&floored, &mut f, 30.0);
            assert_ne!(
                a.data,
                absorbing.uniform_state(&e).data,
                "{name} did nothing"
            );
            for c in 0..absorbing.grid.n_components() {
                for p in 0..np {
                    let expected = if absorbing.absorbs_at(p) {
                        0.0
                    } else {
                        f.spectrum(p)[c]
                    };
                    assert_eq!(
                        a.spectrum(p)[c],
                        expected,
                        "{name}: node {p}, component {c}"
                    );
                }
            }
        }
    }

    /// A wind per node: from calm, one step's sea at each node is the one
    /// a uniform wind of that node's grows (the propagation of a calm sea is
    /// calm, so the sources alone act), bit for bit; and a uniform wind set
    /// afterwards replaces the per-node ones.
    #[test]
    fn each_node_grows_the_sea_of_its_own_wind() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 200.0, 0.0, 100.0, 2, 1);
        let ops = Arc::new(DGOperators2D::new(2));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |_, _| -50.0);
        let grid = SpectralGrid::new(0.08, 0.5, 12, 12);
        let mut model = WaveModel2D::new(Arc::new(mesh), ops, geom, &bathymetry, grid, 9.81)
            .with_sources(SourceTerms::swan_defaults(9.81));
        let (left, right) = ([10.4, 6.0], [-3.0, -5.2]);
        let winds: Vec<[f64; 2]> = (0..model.n_points())
            .map(|p| if p < model.ops.n_nodes { left } else { right })
            .collect();
        let uniform = |[u, v]: [f64; 2]| Wind {
            u10: u.hypot(v),
            direction: v.atan2(u),
        };
        let mut ws = WaveWorkspace::default();
        let mut grown = |model: &WaveModel2D| {
            let mut n = model.zero_state();
            model.step(&mut n, 60.0, &mut ws);
            n
        };
        model.set_winds(&winds);
        let per_node = grown(&model);
        model.set_wind(uniform(left));
        let by_left = grown(&model);
        model.set_wind(uniform(right));
        let by_right = grown(&model);
        let nn = model.ops.n_nodes;
        for c in 0..model.grid.n_components() {
            let (a, l, r): (Vec<f64>, Vec<f64>, Vec<f64>) = (
                per_node.component(c).collect(),
                by_left.component(c).collect(),
                by_right.component(c).collect(),
            );
            assert_eq!(a[..nn], l[..nn], "component {c}");
            assert_eq!(a[nn..], r[nn..], "component {c}");
        }
        let m0 = |n: &WaveSolution, p: usize| {
            let mut e = vec![0.0; model.grid.n_components()];
            model.energy_spectrum_into(n, p, &mut e);
            model.grid.parameters(&e).m0
        };
        assert!(
            m0(&per_node, 0) > 1.5 * m0(&per_node, nn),
            "the stronger wind grows more"
        );
        assert_eq!(model.wind_at(0).u10, model.wind_at(nn).u10);
    }

    /// The vector kernels (`simd`: the geographic term with directions as
    /// lanes, the implicit refraction and frequency shift with systems as
    /// lanes) give the scalar kernels' bits, over whole steps: P1 and P2,
    /// implicit and explicit spectral advection, van Leer and upwind, with a
    /// sheared current over a shoaling, banked bed (all rates nonzero), wind
    /// and sources, open and absorbing boundaries, and a direction count that
    /// leaves a short last vector.
    #[test]
    #[cfg(feature = "simd")]
    fn the_vector_kernels_give_the_scalar_bits() {
        use std::f64::consts::PI;
        use std::sync::atomic::Ordering;

        use crate::mesh::Bathymetry2D;
        use crate::waves::sources::{SourceTerms, Wind};

        let g = 9.81;
        let (lx, ly) = (12_000.0, 9_000.0);
        for order in [1, 2] {
            for (implicit, scheme) in [
                (true, SpectralAdvection::VanLeer),
                (true, SpectralAdvection::Upwind),
                (false, SpectralAdvection::VanLeer),
            ] {
                let mut mesh =
                    Mesh2D::uniform_rectangle_with_bc(0.0, lx, 0.0, ly, 6, 5, BoundaryTag::Open);
                for e in mesh.edges.iter_mut().filter(|e| e.right.is_none()) {
                    let (a, b) = e.vertices;
                    if mesh.vertices[a][1] > ly - 1.0 && mesh.vertices[b][1] > ly - 1.0 {
                        e.boundary_tag = Some(BoundaryTag::Wall);
                    }
                }
                let ops = DGOperators2D::new(order);
                let geom = GeometricFactors2D::compute(&mesh, &ops);
                let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
                    let s = 60.0 - 55.0 * x / lx;
                    -(s - 0.4 * s * (2.0 * PI * y / 4000.0).sin() * (2.0 * PI * x / 5000.0).cos())
                        .max(1.0)
                });
                let grid = SpectralGrid::new(0.05, 0.5, 10, 12);
                let sea = grid.jonswap(2.0, 8.0, 3.3, 0.4, 4.0);
                let mut model = WaveModel2D::new(
                    Arc::new(mesh),
                    Arc::new(ops),
                    Arc::new(geom),
                    &bathymetry,
                    grid,
                    g,
                )
                .with_sources(SourceTerms::swan_defaults(g))
                .with_boundary_spectrum(&sea)
                .with_wind(Wind {
                    u10: 12.0,
                    direction: 0.4,
                })
                .with_spectral_advection(scheme)
                .with_implicit_refraction(implicit)
                .with_implicit_frequency_shift(implicit);
                let (mesh, ops) = (model.mesh.clone(), model.ops.clone());
                let n = mesh.n_elements * ops.n_nodes;
                let (mut u, mut v) = (vec![0.0; n], vec![0.0; n]);
                for k in ElementIndex::iter(mesh.n_elements) {
                    for a in 0..ops.n_nodes {
                        let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[a], ops.nodes_s[a]);
                        let p = k.as_usize() * ops.n_nodes + a;
                        u[p] = 0.6 * (2.0 * PI * y / ly).sin();
                        v[p] = 0.3 * (2.0 * PI * x / lx).cos();
                    }
                }
                model.set_currents(&u, &v);
                let mut start = model.uniform_state(&sea);
                for (index, x) in start.data.iter_mut().enumerate() {
                    *x *= 1.0 + 0.3 * (index as f64 * 0.618).sin();
                }
                let dt = model.compute_dt(0.5);
                let run = |scalar: bool| {
                    SCALAR_ONLY.store(scalar, Ordering::Relaxed);
                    let mut state = start.clone();
                    let mut ws = WaveWorkspace::default();
                    for _ in 0..3 {
                        model.step(&mut state, dt, &mut ws);
                    }
                    SCALAR_ONLY.store(false, Ordering::Relaxed);
                    state.data.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                };
                assert!(
                    run(false) == run(true),
                    "P{order}, implicit {implicit}, {scheme:?}"
                );
            }
        }
    }
}
