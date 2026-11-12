//! Split-form DGSEM element kernel for the 2D SWE with bathymetry
//! (Wintermeyer, Winters, Gassner & Kopriva 2017).
//!
//! Selected with `SWEFormulation2D::EntropyStable` or `EntropyConservative`;
//! the two-point fluxes live in `flux::swe_2d_entropy`.
//!
//! # Semi-discrete form
//!
//! On a tensor-product GLL element, node `i = (a, j)` lies on r-line `j` and
//! s-line `a`. With `D` the 1D GLL differentiation matrix, the contravariant
//! vectors `Ja¹ = J∇r`, `Ja² = J∇s` at the nodes (`GeometricFactors2D`) and
//! `{{Ja}}_ic = ½(Ja_i + Ja_c)`, the element update is (Wintermeyer et al.
//! 2017, curvilinear form)
//!
//! ```text
//! J_i dq_i/dt = − Σ_c 2 D_ac F#(q_i, q_(c,j))·{{Ja¹}}_(i,(c,j)) − g h_i Σ_c D_ac B_(c,j) (0, {{Ja¹}}_(i,(c,j)))
//!               − Σ_c 2 D_jc F#(q_i, q_(a,c))·{{Ja²}}_(i,(a,c)) − g h_i Σ_c D_jc B_(a,c) (0, {{Ja²}}_(i,(a,c)))
//!               + Σ_f LIFT_f sJ_f (F(q⁻)·n − F*)
//!
//! F* = F#(q⁻, q⁺)·n + ½g h⁻ (B⁺ − B⁻)(0, n)    [ − ½λ H̄[[w]]  for EntropyStable ]
//! ```
//!
//! with `F(q⁻) = F#(q⁻, q⁻)`. On affine elements `Ja` is constant and this is
//! the Cartesian split form with `Ja/J = ∇r, ∇s` and the collocated bed slope
//! `g h_i ∇B_i`. At the faces `sJ n = ±Ja` of the face node, so the volume and
//! surface terms telescope as in the affine case.
//!
//! # Properties
//!
//! For any nodal bathymetry, continuous or discontinuous across faces:
//! - **Mass conservation**: the mass component of `F#` is symmetric and `D`
//!   is SBP (`M D + (M D)ᵀ = diag(−1, 0, …, 0, 1)`), so the volume term
//!   telescopes to the face flux and neighbours exchange exactly `∓F*`.
//! - **Well-balanced**: at lake at rest the pressure part of the volume term is
//!   `g h_i Σ_c D_ac h_c`, cancelled by `g h_i Σ_c D_ac B_c` up to round-off
//!   (`Σ_c D_ac η = 0`); at faces `½g h⁻h⁺ + ½g h⁻(B⁺ − B⁻) = ½g h⁻²`.
//!   Unlike the collocated formulation this holds for bathymetry of any degree.
//! - **Entropy**: `EntropyConservative` conserves the total energy
//!   `∫ ½h|u|² + ½gh² + ghB` on periodic domains (up to round-off);
//!   `EntropyStable` dissipates it at faces. The split form also removes the
//!   aliasing of the collocated nonlinear flux derivative (notes/review-2026-09-25.md §1.6).
//!
//! # Wetting and drying (`SWEFormulation2D::WetDry`)
//!
//! The entropy-stable interface dissipation is not positivity preserving (its
//! mass component is `−½λ[[η]]`, which drains a wet node next to a dry one),
//! and the flux-differencing volume term is not balanced in an element with
//! dry nodes, where η = B ≠ η₀. The wet/dry variant therefore (following
//! Wintermeyer et al. 2018 and the Trixi.jl shallow-water solvers):
//!
//! - uses HLL on Audusse et al. (2004) hydrostatically reconstructed states at
//!   every face, `F* = F_HLL(q*⁻, q*⁺)·n + ½g(h⁻² − h*⁻²)(0, n)` with
//!   `h* = max(0, η − max(B⁻, B⁺))`. At lake at rest `F* = F(q⁻)·n`, wet or
//!   dry, and its mass part is positivity preserving (h* ≤ h). Where both
//!   `h*` are dry, a wet side's flow into the dry bank meets a wall's ram
//!   pressure (Brufau et al. 2002), unless its velocity head reaches over
//!   the bank (see [`wall_ram_pressure`]): without it a node walled in by
//!   higher dry nodes kept its velocity forever;
//! - replaces the volume term of every element that has a node shallower than
//!   `h_dry` by a second-order finite-volume update on the GLL subcells
//!   (Hennemann et al. 2021; Rueda-Ramírez et al. 2021 for the reconstruction),
//!   per GLL line and in both directions:
//!
//!   ```text
//!   dq_a/dt = −(F̂_{a,a+1} − F̂_{a−1,a} − S_a) / w_a
//!   S_a     = −½g (h_a,L + h_a,R)(B_a,R − B_a,L) (0, m)
//!   ```
//!
//!   Subcell `a` has width `w_a` (the GLL weight) and contains node `a`.
//!   `h`, `η`, `u`, `v` are reconstructed linearly from the node values with a
//!   monotonized-central limiter (`η` not through a dry node below the
//!   water beside it: its `η` is its bed, not a water level); end subcells
//!   take their outer neighbour from the next element. `F̂` is the same hydrostatic HLL flux on the
//!   reconstructed face states `q_a,L`, `q_a,R`, and `S_a` is the Audusse et al.
//!   (2004) second-order bed term. At the element faces `F̂` is the physical
//!   flux `F(q)·m` of the node, which the surface term replaces by `F*`.
//!
//!   The subcell fluxes telescope, so the element still exchanges exactly
//!   `∓F*` with its neighbours (mass conservation, and the Zhang–Shu argument
//!   for the element means). Face depths stay between neighbouring node
//!   values (`h ≥ 0`), and lake at rest is kept to round-off, dry subcells
//!   included (see [`reconstruct_subcell`]). On smooth flow the subcells
//!   converge at second order (first order without the reconstruction, which
//!   made `WetDry` about 3× less accurate than `Standard` on Thacker's
//!   paraboloid at P2).
//!
//! Fully wet elements keep the flux-differencing volume term. The positivity
//! limiter and wet/dry correction (`SWEPhysics2D`) still run after every stage.

use crate::boundary::{BoundaryState, SWEBoundaryCondition2D};
use crate::flux::{
    HllSide, SWENodeState2D, entropy_stable_dissipation_2d, hll_flux_face_aligned,
    rotate_from_normal, wintermeyer_bed_interface_term_2d, wintermeyer_flux_2d,
};
use crate::mesh::Mesh2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::rhs::subcells::{subcell_slot, telescoped_interfaces};
use crate::solver::{SWESolution2D, SWEState2D};
use crate::source::HydrostaticReconstruction2D;
use crate::types::ElementIndex;

use super::swe_2d::{SWE2DRhsConfig, SWEFormulation2D};

#[cfg(feature = "simd")]
#[path = "swe_2d_split_form_batch.rs"]
mod batch;

/// SIMD lanes (`simd` feature): the element terms are evaluated this many
/// elements at a time, one per lane.
#[cfg(feature = "simd")]
pub(super) const VOLUME_BATCH: usize = 8;

/// Elements per [`SplitFormSWE2D::chunk_rhs`]: sorted into the
/// flux-differencing and the subcell kind before they are batched, so both
/// kinds fill their lanes.
pub(super) const ELEMENT_CHUNK: usize = 64;

/// The momentum a dry sill `δ = B_face − η` above a wet side of depth `h`
/// takes from its flow `uₙ` towards it, beyond the hydrostatic `½g h²`:
/// `w h uₙ(2uₙ + c)` for `uₙ > 0`, `c = √(g h)`, with
/// `w = max(0, 1 − (uₙ²/2g)/δ)`.
///
/// Where both hydrostatically reconstructed depths are dry (`h* ≤ h_min`) no
/// water passes, but the Audusse flux still returns only `½g h²`, whatever
/// the velocity: a node walled in by higher dry nodes keeps its velocity
/// forever (on the Frøya coastline, a 25 m single-node tidal pond held a
/// 3.7 m/s jet through a 5 h ebb). Following Brufau, Vázquez-Cendón &
/// García-Navarro (2002), such an interface is a solid wall: the HLL flux
/// against the mirror state `(h, −uₙ, u_t)`, whose speeds are `∓(|uₙ| + c)`,
/// has no mass or tangential part and the normal momentum flux
/// `½g h² + h uₙ² + (|uₙ| + c) h uₙ`, which is `½g h² + h uₙ(2uₙ + c)`
/// against the flow.
///
/// Two restrictions keep shorelines moving:
/// - **Only against the flow** (`uₙ > 0`). Flow away from the wall gets the
///   mirror's rarefaction, `½g h² + c h uₙ < ½g h²`, which would hold a
///   receding shoreline back.
/// - **Only a sill the flow cannot climb.** An advancing shoreline's front
///   node lies a little below its dry neighbour, within the velocity head
///   `uₙ²/2g` (Thacker's 0.7 m/s: 2.5 cm), and runs up; a pond lies metres
///   below its walls. `w` is 0 where the head reaches the sill and → 1 for a
///   sill far above it, and the flux stays continuous where `h*` crosses
///   `h_min` (`δ ≤ 0`).
///
/// At rest the term vanishes, so lake at rest is kept.
#[inline]
fn wall_ram_pressure(h: f64, un: f64, sill: f64, g: f64) -> f64 {
    let un = un.max(0.0);
    let head = un * un / (2.0 * g);
    if sill > head {
        (1.0 - head / sill) * h * un * (2.0 * un + (g * h).sqrt())
    } else {
        0.0
    }
}

/// Limited slope of one variable at node `ξ` of a subcell spanning
/// `[x_l, x_r]`, from the values at the neighbouring nodes `ξ_m < ξ < ξ_p`:
/// the central slope, capped so that the face values `q + σ(x − ξ)` stay
/// between the neighbouring node values (a monotonized-central limiter on the
/// non-uniform GLL subcells); zero at an extremum.
#[inline]
fn limited_slope(q: [f64; 3], xi: [f64; 3], (x_l, x_r): (f64, f64)) -> f64 {
    let (dm, dp) = (q[1] - q[0], q[2] - q[1]);
    if dm * dp <= 0.0 {
        return 0.0;
    }
    let central = (q[2] - q[0]) / (xi[2] - xi[0]);
    // An end subcell has its face on the node, where the value is the node
    // value; bound its slope on that side as the standard MC limiter does
    // (face halfway to the neighbour), so that a round-off difference cannot
    // let the central slope through
    let bound = |dq: f64, face: f64, node: f64| {
        let dx = if face > 0.0 { face } else { 0.5 * node };
        (dq / dx).abs()
    };
    let bound = bound(dm, xi[1] - x_l, xi[1] - xi[0]).min(bound(dp, x_r - xi[1], xi[2] - xi[1]));
    central.signum() * central.abs().min(bound)
}

/// Face states `(q_L, q_R)` of the subcell `[x_l, x_r]` around node `ξ`, from
/// a limited linear reconstruction of `h`, `η = h + B`, `u` and `v` between
/// the nodes `ξ_m < ξ < ξ_p` (second-order Audusse et al. 2004). The bed at
/// a face is `B = η − h`.
///
/// A dry node (shallower than `h_dry`) has no water level: its `η` is its
/// bed. As a bank above the wet surfaces of the stencil that closes the
/// interface either way, but below one the node must flood, and a monotone
/// stencil through its bed sets the limited face value of a wet node to
/// that bed (the stencil bank, strip, sill) or lifts the dry node's face bed
/// to the wet neighbour's surface (the stencil high water, dry, low water):
/// the interface closes, the node never floods, and the bed term of the
/// tilted subcell pushes the water against it. So `η` is not reconstructed
/// (zero slope) in a stencil with a dry node below one of its wet surfaces.
///
/// Every face value lies between the neighbouring node values, so `h ≥ 0` at
/// faces, and lake at rest (wet nodes at `η₀`, dry nodes with `B ≥ η₀`, so
/// every dry node is a bank) stays balanced, also across shorelines:
/// - at a wet node `η` is constant or has a minimum (a dry neighbour has
///   `η = B ≥ η₀`), so its slope is zero and `h + B = η₀` at both faces;
/// - at a dry node the depth has a minimum, so its face depths are zero, and
///   its face beds `B = η` lie between `η₀` and neighbouring beds `≥ η₀`, so
///   the hydrostatic reconstruction closes the interface to a wet side;
/// - each side of an interface then sees the flux `½g h_face² (0, m)`, and
///   these balance `S_a`: `½g(h_R² − h_L²) = −½g(h_L + h_R)(B_R − B_L)`.
#[inline]
fn reconstruct_subcell(
    nodes: [&SWENodeState2D; 3],
    xi: [f64; 3],
    subcell: (f64, f64),
    h_dry: f64,
) -> (SWENodeState2D, SWENodeState2D) {
    let [m, c, p] = nodes;
    let slope = |f: fn(&SWENodeState2D) -> f64| limited_slope([f(m), f(c), f(p)], xi, subcell);
    let wet_surface = nodes
        .iter()
        .filter(|n| n.h >= h_dry)
        .fold(f64::NEG_INFINITY, |top, n| top.max(n.eta()));
    let flooding = nodes.iter().any(|n| n.h < h_dry && n.eta() < wet_surface);
    let (s_h, s_eta, s_u, s_v) = (
        slope(|n| n.h),
        if flooding {
            0.0
        } else {
            slope(SWENodeState2D::eta)
        },
        slope(|n| n.u),
        slope(|n| n.v),
    );
    let at = |x: f64| {
        let d = x - xi[1];
        let h = (c.h + s_h * d).max(0.0);
        let (u, v) = (c.u + s_u * d, c.v + s_v * d);
        SWENodeState2D {
            h,
            hu: h * u,
            hv: h * v,
            u,
            v,
            b: c.eta() + s_eta * d - h,
        }
    };
    (at(subcell.0), at(subcell.1))
}

/// Interface flux of a split form.
#[derive(Clone, Copy, Debug, PartialEq)]
enum SurfaceFlux {
    /// Wintermeyer flux plus the bed interface term
    EntropyConservative,
    /// The same plus Lax–Friedrichs dissipation in entropy variables
    EntropyStable,
    /// HLL on hydrostatically reconstructed states (wet/dry)
    HydrostaticHll,
}

/// Per-thread scratch space for [`SplitFormSWE2D::element_rhs`].
pub(super) struct SplitFormWorkspace {
    /// Node states (with bed and velocity) of the current element
    nodes: Vec<SWENodeState2D>,
    /// RHS accumulator of the current element, times J (divided out at the end)
    rhs: Vec<SWEState2D>,
    /// Contravariant vector of the current line's direction (J∇r on r-lines,
    /// J∇s on s-lines) at each node of the line
    metric: Vec<(f64, f64)>,
    /// Subcell interface metrics of the current line (`n_1d + 1` values):
    /// `interfaces[a]` is the direction between subcells `a − 1` and `a`, the
    /// node metrics at the two line ends
    interfaces: Vec<(f64, f64)>,
    /// Reconstructed (left, right) face states of the subcells of one line
    faces: Vec<(SWENodeState2D, SWENodeState2D)>,
    /// Per face node (`face · n_1d + fi`): the first interior node of the
    /// neighbour element on the GLL line through it, and its distance from
    /// the face in this element's reference coordinate (`None` at boundaries)
    outer: Vec<Option<(SWENodeState2D, f64)>>,
    /// The RHS (`[h, hu, hv]`, volume and surface terms) of a chunk of
    /// elements, element `j` of the chunk from `j · n_nodes`
    /// ([`SplitFormSWE2D::chunk_rhs`])
    #[cfg_attr(not(feature = "simd"), allow(dead_code))]
    batch: [Vec<f64>; 3],
    /// Subcell interface mass fluxes of the chunk's subcell elements,
    /// element `j` from `j · subcell_interfaces(n_1d)`
    #[cfg_attr(not(feature = "simd"), allow(dead_code))]
    batch_mass: Vec<f64>,
    /// The mass component of `F*` at the chunk's face nodes, element `j`
    /// from `j · 4 n_1d` (if asked for)
    #[cfg_attr(not(feature = "simd"), allow(dead_code))]
    batch_face_mass: Vec<f64>,
    /// Bit `j`: element `j` of the chunk took the subcell volume term
    #[cfg_attr(not(feature = "simd"), allow(dead_code))]
    batch_subcells: u64,
}

impl SplitFormWorkspace {
    pub(super) fn new(n_nodes: usize) -> Self {
        // Unused trailing capacity keeps workspaces of different threads off
        // shared cache lines (see `padded` in swe_2d.rs).
        const SLACK: usize = 4;
        fn padded<T: Clone>(n: usize, value: T) -> Vec<T> {
            let mut v = Vec::with_capacity(n + SLACK);
            v.resize(n, value);
            v
        }
        let n_1d = (n_nodes as f64).sqrt().round() as usize;
        Self {
            nodes: padded(n_nodes, SWENodeState2D::default()),
            rhs: padded(n_nodes, SWEState2D::zero()),
            metric: padded(n_1d, (0.0, 0.0)),
            interfaces: padded(n_1d + 1, (0.0, 0.0)),
            faces: padded(n_1d, Default::default()),
            outer: padded(4 * n_1d, None),
            batch: std::array::from_fn(|_| padded(ELEMENT_CHUNK * n_nodes, 0.0)),
            batch_mass: padded(ELEMENT_CHUNK * 2 * n_1d * n_1d.saturating_sub(1), 0.0),
            batch_face_mass: padded(ELEMENT_CHUNK * 4 * n_1d, 0.0),
            batch_subcells: 0,
        }
    }
}

/// Split-form operator for one RHS evaluation.
pub(super) struct SplitFormSWE2D<'a, 'c, BC: SWEBoundaryCondition2D> {
    q: &'a SWESolution2D,
    mesh: &'a Mesh2D,
    ops: &'a DGOperators2D,
    geom: &'a GeometricFactors2D,
    config: &'a SWE2DRhsConfig<'c, BC>,
    time: f64,
    surface: SurfaceFlux,
    /// `WetDry` only: subcell finite volumes in elements with a node
    /// shallower than `.0` (`subcell_depth`), and the dry-node depth `.1`
    /// (`h_dry`) of their reconstruction
    subcells: Option<(f64, f64)>,
    reconstruction: HydrostaticReconstruction2D,
}

impl<'a, 'c, BC: SWEBoundaryCondition2D> SplitFormSWE2D<'a, 'c, BC> {
    /// Returns `None` for the standard (collocated) formulation.
    ///
    /// # Panics
    /// If `config.source_terms` also discretizes the bed slope (e.g. contains
    /// `BathymetrySource2D`): the split form already includes it.
    pub(super) fn new(
        q: &'a SWESolution2D,
        mesh: &'a Mesh2D,
        ops: &'a DGOperators2D,
        geom: &'a GeometricFactors2D,
        config: &'a SWE2DRhsConfig<'c, BC>,
        time: f64,
    ) -> Option<Self> {
        let (surface, subcells) = match config.formulation {
            SWEFormulation2D::Standard => return None,
            SWEFormulation2D::EntropyConservative => (SurfaceFlux::EntropyConservative, None),
            SWEFormulation2D::EntropyStable => (SurfaceFlux::EntropyStable, None),
            SWEFormulation2D::WetDry => (
                SurfaceFlux::HydrostaticHll,
                Some((config.subcell_depth.unwrap_or(config.h_dry), config.h_dry)),
            ),
        };
        assert!(
            !config
                .source_terms
                .is_some_and(|s| s.includes_bathymetry_slope()),
            "SWEFormulation2D::{:?} includes the bed slope −gh∇B in the DG operator; \
             remove BathymetrySource2D from source_terms",
            config.formulation
        );

        Some(Self {
            q,
            mesh,
            ops,
            geom,
            config,
            time,
            surface,
            subcells,
            reconstruction: HydrostaticReconstruction2D::new(
                config.equation.g,
                config.equation.h_min.meters(),
            ),
        })
    }

    /// Without the `simd` feature nothing is batched: every element is
    /// evaluated by itself in [`Self::element_rhs`].
    #[cfg(not(feature = "simd"))]
    pub(super) fn chunk_rhs(
        &self,
        _ks: &[usize],
        _ahead: &[u32],
        _faces: &[SWEState2D],
        _time_of: &dyn Fn(usize) -> f64,
        _face_mass: bool,
        _ws: &mut SplitFormWorkspace,
    ) -> u64 {
        0
    }

    /// Without the `simd` feature the interior-face fluxes come from the face
    /// pass ([`Self::edge_fluxes`]).
    #[cfg(not(feature = "simd"))]
    pub(super) fn fuses_faces(&self) -> bool {
        false
    }

    #[inline]
    fn bed(&self, k: ElementIndex, node: usize) -> f64 {
        self.config.bathymetry.map_or(0.0, |b| b.get(k, node))
    }

    /// Length of the face buffer [`Self::edge_fluxes`] fills: both sides of
    /// every edge.
    pub(super) fn face_buffer_len(&self) -> usize {
        2 * self.mesh.n_edges * self.ops.n_face_nodes
    }

    /// F* of both sides of interior edge `e`, into `out` (`2 · n_face_nodes`
    /// values): first the left element's, then the right's, each in its own
    /// face-node order and outward-normal direction. Boundary edges are left
    /// untouched (their faces are evaluated per element).
    ///
    /// One surface-flux evaluation serves both sides, so the numerical flux
    /// is computed once per face instead of once per element face, and the
    /// two sides exchange exactly opposite interface fluxes.
    pub(super) fn edge_fluxes(&self, e: usize, out: &mut [SWEState2D]) {
        let edge = &self.mesh.edges[e];
        let Some(right) = edge.right else {
            return;
        };
        let n_face_nodes = self.ops.n_face_nodes;
        let g = self.config.equation.g;
        let h_min = self.config.equation.h_min.meters();
        let (left_k, right_k) = (
            ElementIndex::new(edge.left.element),
            ElementIndex::new(right.element),
        );
        let node = |k: ElementIndex, i: usize| {
            SWENodeState2D::new(&self.q.get_state(k, i), self.bed(k, i), h_min)
        };
        let (left_out, right_out) = out.split_at_mut(n_face_nodes);
        // A parallelogram's faces are straight with a constant normal
        let left = self.geom.element_geometry(edge.left.element);
        let affine_normal = left.faces[edge.left.face].0;
        for (fi, &left_node) in self.ops.face_nodes[edge.left.face].iter().enumerate() {
            let normal = if left.affine {
                affine_normal
            } else {
                self.geom.normal(edge.left.element, edge.left.face, fi)
            };
            let rfi = n_face_nodes - 1 - fi;
            let right_node = self.ops.face_nodes[right.face][rfi];
            let (f_left, f_right) = self.surface_flux(
                &node(left_k, left_node),
                &node(right_k, right_node),
                normal,
                g,
            );
            left_out[fi] = f_left;
            right_out[rfi] = f_right;
        }
    }

    /// Interface flux `F*` between `q_a` and `q_b` across a face with unit
    /// normal `n` pointing from a to b: `F*·n` as seen from side a, and
    /// `F*·(−n)` as seen from side b (its own outward normal).
    ///
    /// The two-point flux and the dissipation are antisymmetric under
    /// swapping the sides and the normal, so they are evaluated once. Only
    /// the bed term (and, for HLL, the hydrostatic correction) is per side.
    #[inline]
    fn surface_flux(
        &self,
        q_a: &SWENodeState2D,
        q_b: &SWENodeState2D,
        n: (f64, f64),
        g: f64,
    ) -> (SWEState2D, SWEState2D) {
        match self.surface {
            SurfaceFlux::HydrostaticHll => {
                let (f_a, f_b) = self.hydrostatic_hll(q_a, q_b, n, g);
                (f_a, -f_b)
            }
            surface => {
                let mut shared = wintermeyer_flux_2d(q_a, q_b, n, g);
                if surface == SurfaceFlux::EntropyStable {
                    shared = shared + entropy_stable_dissipation_2d(q_a, q_b, n, g);
                }
                let minus_n = (-n.0, -n.1);
                (
                    shared + wintermeyer_bed_interface_term_2d(q_a.h, q_a.b, q_b.b, n, g),
                    -shared + wintermeyer_bed_interface_term_2d(q_b.h, q_b.b, q_a.b, minus_n, g),
                )
            }
        }
    }

    /// Volume, bed-slope and surface terms of element `k`, written to `out`
    /// (`[h, hu, hv]`, `n_nodes` values each), with interior-face fluxes from
    /// `faces` (filled by [`Self::edge_fluxes`]). Source terms are not
    /// included. `face_mass`, if given, receives the mass component of `F*` at
    /// every face node (outward normal, `face · n_face_nodes + fi`), and
    /// `subcell_mass`, if given, the mass flux through every subcell interface
    /// ([`crate::solver::rhs::subcells`]), or NaN in all of them if the element
    /// takes the flux-differencing volume term.
    ///
    /// `batched`: the slot of the workspace's batch holding this element's
    /// terms (and mass fluxes), if [`Self::chunk_rhs`] computed them: they
    /// are copied out, the same bit for bit as evaluated here.
    pub(super) fn element_rhs(
        &self,
        k: ElementIndex,
        ws: &mut SplitFormWorkspace,
        faces: &[SWEState2D],
        out: [&mut [f64]; 3],
        mut face_mass: Option<&mut [f64]>,
        mut subcell_mass: Option<&mut [f64]>,
        batched: Option<usize>,
    ) {
        let ops = self.ops;
        let n1 = ops.n_1d;
        let n_face_nodes = ops.n_face_nodes;
        let ki = k.as_usize();
        let g = self.config.equation.g;
        let h_min = self.config.equation.h_min.meters();

        if let Some(slot) = batched {
            let n = ops.n_nodes;
            for (out, batch) in out.into_iter().zip(&ws.batch) {
                out.copy_from_slice(&batch[slot * n..(slot + 1) * n]);
            }
            if let Some(mass) = face_mass {
                let len = mass.len();
                mass.copy_from_slice(&ws.batch_face_mass[slot * len..(slot + 1) * len]);
            }
            if let Some(mass) = subcell_mass {
                if (ws.batch_subcells >> slot) & 1 == 1 {
                    let len = mass.len();
                    mass.copy_from_slice(&ws.batch_mass[slot * len..(slot + 1) * len]);
                } else {
                    mass.fill(f64::NAN);
                }
            }
            return;
        }

        for i in 0..ops.n_nodes {
            ws.nodes[i] = SWENodeState2D::new(&self.q.get_state(k, i), self.bed(k, i), h_min);
        }
        ws.rhs.fill(SWEState2D::zero());

        // 1. Volume term (times J), line by line (node ordering i = j·n1 + a,
        //    r fastest): flux differencing with the collocated bed slope, or
        //    subcell finite volumes in elements with dry nodes. Each line uses
        //    the contravariant vector of its direction at its nodes.
        let subcells = self
            .subcells
            .is_some_and(|(depth, _)| ws.nodes.iter().any(|n| n.h < depth));
        match (subcells, subcell_mass.as_deref_mut()) {
            (true, _) => self.outer_nodes(k, ws),
            (false, Some(mass)) => mass.fill(f64::NAN),
            _ => {}
        }
        // A parallelogram has a constant metric: take it from the dense
        // per-element geometry and skip the pair averages (the Cartesian
        // split form)
        let element = self.geom.element_geometry(ki);
        let affine = element.affine;
        let (affine_r, affine_s) = element.metric.contravariant();
        for line in 0..n1 {
            let r_line = |a: usize| line * n1 + a;
            let s_line = |j: usize| j * n1 + line;
            // Line ends on faces 3/1 (r-lines) and 0/2 (s-lines); faces 2
            // and 3 list their nodes in reverse
            let (rev, fwd) = (n1 - 1 - line, line);

            for a in 0..n1 {
                ws.metric[a] = if affine {
                    affine_r
                } else {
                    self.geom.contravariant(ki, r_line(a)).0
                };
            }
            if subcells {
                let ends = [ws.outer[3 * n1 + rev], ws.outer[n1 + fwd]];
                let first = subcell_slot(n1, 0, line, 0);
                let mass = subcell_mass
                    .as_deref_mut()
                    .map(|m| &mut m[first..first + n1 - 1]);
                self.line_subcells(ws, r_line, ends, affine, g, mass);
            } else if affine {
                self.line_volume::<true>(ws, r_line, affine_r, g);
            } else {
                self.line_volume::<false>(ws, r_line, affine_r, g);
            }

            for j in 0..n1 {
                ws.metric[j] = if affine {
                    affine_s
                } else {
                    self.geom.contravariant(ki, s_line(j)).1
                };
            }
            if subcells {
                let ends = [ws.outer[fwd], ws.outer[2 * n1 + rev]];
                let first = subcell_slot(n1, 1, line, 0);
                let mass = subcell_mass
                    .as_deref_mut()
                    .map(|m| &mut m[first..first + n1 - 1]);
                self.line_subcells(ws, s_line, ends, affine, g, mass);
            } else if affine {
                self.line_volume::<true>(ws, s_line, affine_s, g);
            } else {
                self.line_volume::<false>(ws, s_line, affine_s, g);
            }
        }

        // 2. Surface terms (times J): LIFT sJ (F(q⁻)·n − F*). For GLL
        //    collocation LIFT only couples a face node to itself. F* of
        //    interior faces comes from the face pass (`edge_fluxes`), once per
        //    face for both sides.
        for face in 0..4 {
            let edge_index = self.mesh.element_edges[ki][face];
            let edge = &self.mesh.edges[edge_index];
            // This side's slots in the face buffer, if the face is interior
            let slots = edge.right.map(|_| {
                let side = usize::from(edge.left.element != ki || edge.left.face != face);
                &faces[(2 * edge_index + side) * n_face_nodes..][..n_face_nodes]
            });

            for (fi, &node) in ops.face_nodes[face].iter().enumerate() {
                let (normal, scale) = if affine {
                    element.faces[face]
                } else {
                    (
                        self.geom.normal(ki, face, fi),
                        self.geom.surface_jacobian(ki, face, fi),
                    )
                };
                let q_int = ws.nodes[node];
                let f_star = match slots {
                    Some(slots) => slots[fi],
                    None => self.boundary_flux(k, face, node, &q_int, normal, self.time),
                };
                if let Some(face_mass) = face_mass.as_deref_mut() {
                    face_mass[face * n_face_nodes + fi] = f_star.h;
                }
                let flux_diff = wintermeyer_flux_2d(&q_int, &q_int, normal, g) - f_star;

                ws.rhs[node] = ws.rhs[node] + (scale * ops.lift[face][(node, fi)]) * flux_diff;
            }
        }

        let [out_h, out_hu, out_hv] = out;
        for (i, r) in ws.rhs.iter().enumerate() {
            let j_inv = if affine {
                element.metric.det_j_inv
            } else {
                self.geom.jacobian_inv(ki, i)
            };
            out_h[i] = j_inv * r.h;
            out_hu[i] = j_inv * r.hu;
            out_hv[i] = j_inv * r.hv;
        }
    }

    /// Accumulate (times J) the flux-differencing volume term with the bed
    /// slope along one line of nodes `idx(0..n1)`,
    ///
    /// ```text
    /// −Σ_c 2 D_ac F#(q_a, q_c)·{{Ja}}_ac − g h_a Σ_c D_ac B_c {{Ja}}_ac (0, 1, 1)
    /// ```
    ///
    /// with `Ja` the contravariant vector of the line direction
    /// ([`SplitFormWorkspace::metric`]) and `{{Ja}}_ac = ½(Ja_a + Ja_c)`
    /// (Wintermeyer et al. 2017, curvilinear form). At lake at rest the two
    /// terms leave `½g h_a η₀ (D Ja)_a`, which cancels between the r- and
    /// s-lines by the discrete metric identities; on affine elements `Ja` is
    /// constant and this is the Cartesian split form.
    ///
    /// `F#` is symmetric in its states (and `{{Ja}}` in its nodes), so each
    /// pair is evaluated once and applied to both ends.
    ///
    /// `AFFINE`: the metric is constant along the line (a parallelogram,
    /// [`SplitFormWorkspace::metric`] filled with one value), so the averages
    /// are skipped.
    #[inline]
    fn line_volume<const AFFINE: bool>(
        &self,
        ws: &mut SplitFormWorkspace,
        idx: impl Fn(usize) -> usize,
        affine_metric: (f64, f64),
        g: f64,
    ) {
        let n1 = self.ops.n_1d;
        let d1 = &self.ops.dr_1d_row_major;
        for a in 0..n1 {
            let ia = idx(a);
            let q_a = ws.nodes[ia];
            let m_a = if AFFINE { affine_metric } else { ws.metric[a] };

            // Diagonal pair F#(q_a, q_a) = F(q_a); D_aa ≠ 0 only at line ends
            let mut acc = (2.0 * d1[a * n1 + a]) * wintermeyer_flux_2d(&q_a, &q_a, m_a, g);
            for c in (a + 1)..n1 {
                let ic = idx(c);
                let m = if AFFINE {
                    m_a
                } else {
                    let m_c = ws.metric[c];
                    (0.5 * (m_a.0 + m_c.0), 0.5 * (m_a.1 + m_c.1))
                };
                let f = wintermeyer_flux_2d(&q_a, &ws.nodes[ic], m, g);
                acc = acc + (2.0 * d1[a * n1 + c]) * f;
                ws.rhs[ic] = ws.rhs[ic] - (2.0 * d1[c * n1 + a]) * f;
            }

            if self.config.bathymetry.is_some() {
                // g h_a Σ_c D_ac {{Ja}}_ac B_c = ½g h_a [Ja_a (D B)_a + (D (Ja B))_a]
                let (mut db, mut db_x, mut db_y) = (0.0, 0.0, 0.0);
                for c in 0..n1 {
                    let d_b = d1[a * n1 + c] * ws.nodes[idx(c)].b;
                    db += d_b;
                    if !AFFINE {
                        db_x += d_b * ws.metric[c].0;
                        db_y += d_b * ws.metric[c].1;
                    }
                }
                let (db_x, db_y) = if AFFINE {
                    (db * m_a.0, db * m_a.1)
                } else {
                    (0.5 * (db * m_a.0 + db_x), 0.5 * (db * m_a.1 + db_y))
                };
                let force = g * q_a.h;
                acc = acc + SWEState2D::new(0.0, force * db_x, force * db_y);
            }

            ws.rhs[ia] = ws.rhs[ia] - acc;
        }
    }

    /// HLL flux on hydrostatically reconstructed states in direction `m`
    /// (unit normal or contravariant vector), as seen from side `a` and from
    /// side `b`: `|m| F_HLL(q*_a, q*_b, m̂) + ½g(h² − h*²)(0, m)`, each with
    /// its own side's depths. The mass parts are equal (conservative); at lake
    /// at rest each equals the physical flux `½g h² (0, m)` of its side.
    #[inline]
    fn hydrostatic_hll(
        &self,
        q_a: &SWENodeState2D,
        q_b: &SWENodeState2D,
        m: (f64, f64),
        g: f64,
    ) -> (SWEState2D, SWEState2D) {
        let h_min = self.reconstruction.h_min;
        let norm = (m.0 * m.0 + m.1 * m.1).sqrt();
        let (nx, ny) = (m.0 / norm, m.1 / norm);
        // Audusse et al. (2004): h* = max(0, η − max(B_a, B_b)) with the
        // velocity kept (zero on a side at or below h_min), so hu* = h* u.
        // The node states carry u and v: HLL needs no division to recover them.
        let b_face = q_a.b.max(q_b.b);
        let side = |q: &SWENodeState2D| {
            let h_star = (q.h + q.b - b_face).max(0.0);
            let (u, v) = if q.h > h_min { (q.u, q.v) } else { (0.0, 0.0) };
            let (un, ut) = (u * nx + v * ny, -u * ny + v * nx);
            let (hun, hut) = (h_star * un, h_star * ut);
            let (un, ut) = if h_star > h_min { (un, ut) } else { (0.0, 0.0) };
            HllSide {
                h: h_star,
                hun,
                hut,
                un,
                ut,
            }
        };
        let (a_star, b_star) = (side(q_a), side(q_b));
        let flux =
            norm * rotate_from_normal(&hll_flux_face_aligned(a_star, b_star, g, h_min), nx, ny);
        // HLL returns zero, without the pressure ½g h*², when both sides are
        // below h_min (thin reconstructed subcell faces at a shoreline): the
        // correction then restores all of ½g h²
        let hll_dry = a_star.h <= h_min && b_star.h <= h_min;
        // Pressure as in F(q)·m = F#(q, q)·m (no dry cutoff); where no water
        // passes, a wet side faces a wall (see `wall_ram_pressure`), which
        // lies along +m from side a and along −m from side b
        let correction = |q: &SWENodeState2D, h_star: f64, towards: f64| {
            let h = q.h;
            let mut dp = 0.5 * g * (h * h - if hll_dry { 0.0 } else { h_star * h_star });
            if hll_dry && h > h_min {
                let (un, sill) = (towards * (q.u * nx + q.v * ny), b_face - (h + q.b));
                dp += wall_ram_pressure(h, un, sill, g);
            }
            SWEState2D::new(0.0, dp * m.0, dp * m.1)
        };
        (
            flux + correction(q_a, a_star.h, 1.0),
            flux + correction(q_b, b_star.h, -1.0),
        )
    }

    /// Accumulate (times J) the subcell finite-volume update along one line of
    /// nodes `idx(0..n1)`:
    ///
    /// ```text
    /// J dq_a/dt = −(F̂_{a,a+1} − F̂_{a−1,a} − S_a) / w_a
    /// S_a       = −½g (h_a,L + h_a,R)(B_a,R − B_a,L) (0, m̄_a)
    ///             + ¼g [(h_a,R + h_a)(B_a − B_a,R) + (h_a,L + h_a)(B_a − B_a,L)]
    ///               (0, m_{a,a+1} − m_{a−1,a})
    /// ```
    ///
    /// with the hydrostatic HLL flux on the reconstructed subcell face states
    /// at the subcell interfaces, and the physical flux `F(q)·Ja` at the two
    /// ends, which the surface term then replaces by `F*`. See
    /// [`reconstruct_subcell`] for the face states.
    ///
    /// **Metrics.** Interface `(a, a+1)` has the direction
    /// `m_{a,a+1} = Σ_{k≤a} Σ_{l>a} 2 Q_kl {{Ja}}_kl` (`Q = W D`), the metric
    /// under which the flux-differencing volume term telescopes into subcell
    /// fluxes (Fisher et al. 2013; Hennemann et al. 2021 on curvilinear
    /// meshes); the line ends use `Ja` of the end nodes. Then
    /// `m_{a,a+1} − m_{a−1,a} = w_a (D Ja)_a`, so a uniform state is preserved
    /// by the discrete metric identities. On affine elements every `m` is `Ja`.
    ///
    /// **Bed term.** `m̄_a` is the mean of the subcell's two interface
    /// metrics. The second term vanishes on affine elements and on a flat bed
    /// (so momentum stays conserved there). At lake at rest it equals
    /// `¼g(h_L² + h_R² − 2h_a²)(m_{a,a+1} − m_{a−1,a})` (wet: `h + B = η₀` at
    /// the node and its faces; dry: all depths zero), which turns the metric
    /// part of the pressure difference, `¼g(h_L² + h_R²)(m_{a,a+1} −
    /// m_{a−1,a})`, into `½g h_a²(m_{a,a+1} − m_{a−1,a})`; that cancels between
    /// the r- and s-lines of the node by the metric identities. For smooth flow
    /// it is second-order small (a bed difference times a metric difference).
    ///
    /// `ends` are the outer neighbours of the two end nodes (see
    /// [`SplitFormWorkspace::outer`]); an end subcell without one (physical
    /// boundary) keeps its node state. `affine`: the metric is constant (every
    /// interface metric is `Ja`).
    ///
    /// `mass`, if given (`n₁ − 1` values), receives the mass flux `F̂_h`
    /// through each interface of the line, from node `a` to `a + 1`
    /// ([`crate::solver::rhs::subcells`]).
    #[inline]
    fn line_subcells(
        &self,
        ws: &mut SplitFormWorkspace,
        idx: impl Fn(usize) -> usize,
        ends: [Option<(SWENodeState2D, f64)>; 2],
        affine: bool,
        g: f64,
        mut mass: Option<&mut [f64]>,
    ) {
        let n1 = self.ops.n_1d;
        let w = &self.ops.weights_1d;
        let xi = &self.ops.nodes_1d;
        // Subcells run only in the wet/dry formulation, which sets h_dry
        let h_dry = self.subcells.map_or(0.0, |(_, h_dry)| h_dry);

        // Interface metrics (the telescoping sum of the flux-differencing term)
        telescoped_interfaces(self.ops, &ws.metric, affine, &mut ws.interfaces);

        let (first, last) = (idx(0), idx(n1 - 1));
        let (q_first, q_last) = (ws.nodes[first], ws.nodes[last]);
        let (m_first, m_last) = (ws.metric[0], ws.metric[n1 - 1]);
        ws.rhs[first] =
            ws.rhs[first] + (1.0 / w[0]) * wintermeyer_flux_2d(&q_first, &q_first, m_first, g);
        ws.rhs[last] =
            ws.rhs[last] - (1.0 / w[n1 - 1]) * wintermeyer_flux_2d(&q_last, &q_last, m_last, g);

        // Face states. Subcell a spans [x_a, x_a + w_a] with x_0 = −1; the
        // end nodes sit on the element faces, where the face value is the node
        // value, and take their outer slope neighbour from the next element.
        let mut x_left = -1.0;
        for a in 0..n1 {
            let q = ws.nodes[idx(a)];
            let x_right = x_left + w[a];
            let left = if a > 0 {
                Some((ws.nodes[idx(a - 1)], xi[a - 1]))
            } else {
                ends[0].map(|(s, d)| (s, -1.0 - d))
            };
            let right = if a + 1 < n1 {
                Some((ws.nodes[idx(a + 1)], xi[a + 1]))
            } else {
                ends[1].map(|(s, d)| (s, 1.0 + d))
            };
            ws.faces[a] = match (left, right) {
                (Some((l, x_l)), Some((r, x_r))) => {
                    reconstruct_subcell([&l, &q, &r], [x_l, xi[a], x_r], (x_left, x_right), h_dry)
                }
                _ => (q, q),
            };
            x_left = x_right;
        }

        for a in 0..n1 - 1 {
            let (ia, ib) = (idx(a), idx(a + 1));
            let m = ws.interfaces[a + 1];
            let (f_a, f_b) = self.hydrostatic_hll(&ws.faces[a].1, &ws.faces[a + 1].0, m, g);
            if let Some(mass) = mass.as_deref_mut() {
                mass[a] = f_a.h;
            }
            ws.rhs[ia] = ws.rhs[ia] - (1.0 / w[a]) * f_a;
            ws.rhs[ib] = ws.rhs[ib] + (1.0 / w[a + 1]) * f_b;
        }

        // Bed-slope term, zero in subcells without a slope (B_L = B_R), and
        // the metric balance term, zero on affine elements
        for (a, ((left, right), w_a)) in ws.faces.iter().zip(w).enumerate() {
            let ia = idx(a);
            let bed = -0.5 * g * (left.h + right.h) * (right.b - left.b) / w_a;
            let (force_x, force_y) = if affine {
                let m = ws.metric[0];
                (bed * m.0, bed * m.1)
            } else {
                let (m_l, m_r) = (ws.interfaces[a], ws.interfaces[a + 1]);
                let (h_a, b_a) = (ws.nodes[ia].h, ws.nodes[ia].b);
                let metric = 0.25
                    * g
                    * ((right.h + h_a) * (b_a - right.b) + (left.h + h_a) * (b_a - left.b))
                    / w_a;
                (
                    bed * 0.5 * (m_l.0 + m_r.0) + metric * (m_r.0 - m_l.0),
                    bed * 0.5 * (m_l.1 + m_r.1) + metric * (m_r.1 - m_l.1),
                )
            };
            ws.rhs[ia] = ws.rhs[ia] + SWEState2D::new(0.0, force_x, force_y);
        }
    }

    /// Fill [`SplitFormWorkspace::outer`] for element `k`.
    fn outer_nodes(&self, k: ElementIndex, ws: &mut SplitFormWorkspace) {
        let n1 = self.ops.n_1d;
        for face in 0..4 {
            for fi in 0..n1 {
                ws.outer[face * n1 + fi] = self.outer_node(k, face, fi);
            }
        }
    }

    /// The outer node of face node `fi` of `face` (see
    /// [`SplitFormWorkspace::outer`]): the neighbour's first interior node on
    /// the GLL line through it and its distance from the face in this
    /// element's reference coordinate, or `None` at a boundary.
    fn outer_node(&self, k: ElementIndex, face: usize, fi: usize) -> Option<(SWENodeState2D, f64)> {
        let ops = self.ops;
        let n1 = ops.n_1d;
        let h_min = self.config.equation.h_min.meters();
        // Element height normal to a face at a face node, in physical units / 2:
        // J / sJ there (constant on a parallelogram)
        let height = |e: usize, f: usize, fi: usize, node: usize| {
            let element = self.geom.element_geometry(e);
            if element.affine {
                element.metric.det_j / element.faces[f].1
            } else {
                self.geom.jacobian(e, node) / self.geom.surface_jacobian(e, f, fi)
            }
        };
        self.mesh.neighbor(k, face).map(|nb| {
            let nb_k = ElementIndex::new(nb.element);
            let nb_node = ops.face_nodes[nb.face][n1 - 1 - fi];
            // One node into the neighbour, normal to its face
            let inner = match nb.face {
                0 => nb_node + n1,
                1 => nb_node - 1,
                2 => nb_node - n1,
                _ => nb_node + 1,
            };
            let own_node = ops.face_nodes[face][fi];
            let distance = (ops.nodes_1d[1] - ops.nodes_1d[0])
                * height(nb.element, nb.face, n1 - 1 - fi, nb_node)
                / height(k.as_usize(), face, fi, own_node);
            let state = self.q.get_state(nb_k, inner);
            (
                SWENodeState2D::new(&state, self.bed(nb_k, inner), h_min),
                distance,
            )
        })
    }

    /// `F*` at the boundary face node `node` (state `q_int`) of `face` of
    /// element `k` at `time`, from the configured boundary condition. The
    /// boundary state takes the node's bed: no bed step at boundaries.
    fn boundary_flux(
        &self,
        k: ElementIndex,
        face: usize,
        node: usize,
        q_int: &SWENodeState2D,
        normal: (f64, f64),
        time: f64,
    ) -> SWEState2D {
        let g = self.config.equation.g;
        let h_min = self.config.equation.h_min.meters();
        let state = super::swe_2d::boundary_state(
            self.q,
            self.mesh,
            self.ops,
            self.config,
            time,
            k,
            face,
            node,
            normal,
        );
        match state {
            // State on the boundary: its physical flux F(q_b)·n
            BoundaryState::Exact(q) => {
                let q_b = SWENodeState2D::new(&q, q_int.b, h_min);
                wintermeyer_flux_2d(&q_b, &q_b, normal, g)
            }
            BoundaryState::Ghost(q) => {
                let q_ext = SWENodeState2D::new(&q, q_int.b, h_min);
                self.surface_flux(q_int, &q_ext, normal, g).0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::boundary::Reflective2D;
    use crate::equations::ShallowWater2D;
    use crate::flux::{SWENodeState2D, swe_entropy_variables_2d};
    use crate::mesh::{Bathymetry2D, Mesh2D};
    use crate::operators::{DGOperators2D, GeometricFactors2D};
    use crate::solver::rhs::swe_2d::{SWE2DRhsConfig, SWEFormulation2D, compute_rhs_swe_2d};
    use crate::solver::{SWESolution2D, SWEState2D};
    use crate::source::{BathymetrySource2D, CombinedSource2D, CoriolisSource2D};
    use crate::types::{Depth, ElementIndex};

    const G: f64 = 9.81;
    const L: f64 = 20_000.0;
    const SPLIT_FORMS: [SWEFormulation2D; 3] = [
        SWEFormulation2D::EntropyConservative,
        SWEFormulation2D::EntropyStable,
        SWEFormulation2D::WetDry,
    ];

    #[derive(Clone, Copy, Debug)]
    enum Bed {
        /// Smooth, nodal, degree ≥ p in every element; continuous across faces
        Smooth,
        /// Cell averages: constant per element, jumps at every face
        CellAverage,
        /// Smooth plus per-node noise: rough inside elements, jumps at faces
        Rough,
    }
    const BEDS: [Bed; 3] = [Bed::Smooth, Bed::CellAverage, Bed::Rough];

    fn smooth_bed(x: f64, y: f64) -> f64 {
        let tau = 2.0 * std::f64::consts::PI / L;
        -200.0 + 150.0 * (tau * x).sin() * (tau * y).cos()
    }

    fn setup(
        order: usize,
        periodic: bool,
        bed: Bed,
    ) -> (Mesh2D, DGOperators2D, GeometricFactors2D, Bathymetry2D) {
        let mesh = if periodic {
            Mesh2D::uniform_periodic(0.0, L, 0.0, L, 12, 12)
        } else {
            Mesh2D::uniform_rectangle(0.0, L, 0.0, 0.5 * L, 8, 5)
        };
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let mut bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, smooth_bed);
        match bed {
            Bed::Smooth => {}
            Bed::CellAverage => bathymetry.to_cell_average(),
            Bed::Rough => {
                let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
                for b in bathymetry.data.iter_mut() {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    *b += 30.0 * ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5);
                }
                bathymetry.compute_gradients(&ops, &geom);
            }
        }
        (mesh, ops, geom, bathymetry)
    }

    /// η = 0.3 m over `bathymetry`, velocity `speed`·(cos·cos, ½ sin·cos) —
    /// correlated with the bed slope so that no symmetry hides errors.
    fn state(
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        bathymetry: &Bathymetry2D,
        speed: f64,
    ) -> SWESolution2D {
        let tau = 2.0 * std::f64::consts::PI / L;
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let h = 0.3 - bathymetry.get(k, i);
                let u = speed * (tau * x).cos() * (tau * y).cos();
                let v = 0.5 * speed * (tau * x).sin() * (tau * y).cos();
                q.set_state(k, i, SWEState2D::from_primitives(h, u, v));
            }
        }
        q
    }

    /// Σ_k Σ_i ω_i J_ki f(k, i)
    fn integrate(
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        f: impl Fn(ElementIndex, usize) -> f64,
    ) -> f64 {
        let mut total = 0.0;
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                total += geom.mass[geom.node_index(k.as_usize(), i)] * f(k, i);
            }
        }
        total
    }

    /// Entropy rate dE/dt = ∫ w·dq/dt and the scale ∫ |w·dq/dt|.
    fn entropy_rate(
        q: &SWESolution2D,
        rhs: &SWESolution2D,
        bathymetry: &Bathymetry2D,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
    ) -> (f64, f64) {
        let w_dot_rhs = |k: ElementIndex, i: usize| {
            let node = SWENodeState2D::new(&q.get_state(k, i), bathymetry.get(k, i), 1e-6);
            let w = swe_entropy_variables_2d(&node, G);
            let r = rhs.get_state(k, i);
            w.h * r.h + w.hu * r.hu + w.hv * r.hv
        };
        (
            integrate(mesh, ops, geom, w_dot_rhs),
            integrate(mesh, ops, geom, |k, i| w_dot_rhs(k, i).abs()),
        )
    }

    fn max_momentum_rate(rhs: &SWESolution2D) -> f64 {
        rhs.data[1]
            .iter()
            .chain(rhs.data[2].iter())
            .fold(0.0_f64, |m, x| m.max(x.abs()))
    }

    #[test]
    fn test_lake_at_rest_any_nodal_bathymetry() {
        // notes/review-2026-09-25.md §1.1: the collocated scheme balances only
        // deg B ≤ p/2. The split form must balance smooth high-degree,
        // cell-averaged and rough nodal beds, with periodic and reflective-wall
        // boundaries.
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        for order in 1..=4 {
            for periodic in [true, false] {
                for bed in BEDS {
                    let (mesh, ops, geom, bathymetry) = setup(order, periodic, bed);
                    let q = state(&mesh, &ops, &bathymetry, 0.0);
                    for formulation in SPLIT_FORMS {
                        let config = SWE2DRhsConfig::new(&equation, &bc)
                            .with_coriolis(false)
                            .with_formulation(formulation)
                            .with_bathymetry(&bathymetry);
                        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

                        let max_h = rhs.data[0].iter().fold(0.0_f64, |m, x| m.max(x.abs()));
                        let max_mom = max_momentum_rate(&rhs);
                        // O(1e-12) m/s of spurious acceleration in ~300 m of water
                        assert!(
                            max_h < 1e-12 && max_mom < 1e-9,
                            "p={order}, periodic={periodic}, {bed:?}, {formulation:?}: \
                             max |dh/dt| = {max_h:.3e}, max |d(hu)/dt| = {max_mom:.3e}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_standard_formulation_not_balanced_for_high_degree_bathymetry() {
        // Negative control for the test above: the collocated scheme with
        // BathymetrySource2D is far from balanced on the same smooth bed.
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let bathy_source = BathymetrySource2D::new(G);
        let (mesh, ops, geom, bathymetry) = setup(2, true, Bed::Smooth);
        let q = state(&mesh, &ops, &bathymetry, 0.0);
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&bathy_source)
            .with_well_balanced(true);
        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
        let max_mom = max_momentum_rate(&rhs);
        assert!(
            max_mom > 1e-4,
            "expected a lake-at-rest residual, got {max_mom:.3e}"
        );
    }

    #[test]
    fn test_mass_conservation() {
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        for order in 1..=3 {
            for bed in BEDS {
                let (mesh, ops, geom, bathymetry) = setup(order, true, bed);
                let q = state(&mesh, &ops, &bathymetry, 1.0);
                let mass = integrate(&mesh, &ops, &geom, |k, i| q.get_var(k, i, 0));
                for formulation in SPLIT_FORMS {
                    let config = SWE2DRhsConfig::new(&equation, &bc)
                        .with_coriolis(false)
                        .with_formulation(formulation)
                        .with_bathymetry(&bathymetry);
                    let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
                    let mass_rate = integrate(&mesh, &ops, &geom, |k, i| rhs.get_var(k, i, 0));
                    assert!(
                        mass_rate.abs() / mass < 1e-14,
                        "p={order}, {bed:?}, {formulation:?}: d(mass)/dt / mass = {:.3e}",
                        mass_rate / mass
                    );
                }
            }
        }
    }

    #[test]
    fn test_entropy_conservation_and_stability() {
        // EntropyConservative: dE/dt = 0 to round-off (periodic or walls).
        // EntropyStable: dE/dt < 0. The dissipation acts on jumps of the entropy
        // variables (η, u, v), not of h, so the free surface is given element-wise
        // offsets to make the states jump across every face.
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        for order in 1..=3 {
            for periodic in [true, false] {
                for bed in BEDS {
                    let (mesh, ops, geom, bathymetry) = setup(order, periodic, bed);
                    let mut q = state(&mesh, &ops, &bathymetry, 1.0);
                    for k in ElementIndex::iter(mesh.n_elements) {
                        let offset = 0.05 * (k.as_usize() % 3) as f64;
                        for i in 0..ops.n_nodes {
                            let s = q.get_state(k, i);
                            q.set_state(k, i, SWEState2D::new(s.h + offset, s.hu, s.hv));
                        }
                    }

                    let config = SWE2DRhsConfig::new(&equation, &bc)
                        .with_coriolis(false)
                        .with_formulation(SWEFormulation2D::EntropyConservative)
                        .with_bathymetry(&bathymetry);
                    let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
                    let (rate, scale) = entropy_rate(&q, &rhs, &bathymetry, &mesh, &ops, &geom);
                    assert!(
                        rate.abs() < 1e-12 * scale,
                        "EC p={order}, periodic={periodic}, {bed:?}: \
                         dE/dt = {rate:.3e} (scale {scale:.3e})"
                    );

                    // The hydrostatic-HLL interfaces of WetDry dissipate too
                    for formulation in [SWEFormulation2D::EntropyStable, SWEFormulation2D::WetDry] {
                        let config = SWE2DRhsConfig::new(&equation, &bc)
                            .with_coriolis(false)
                            .with_formulation(formulation)
                            .with_bathymetry(&bathymetry);
                        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
                        let (rate, scale) = entropy_rate(&q, &rhs, &bathymetry, &mesh, &ops, &geom);
                        assert!(
                            rate < -1e-10 * scale,
                            "{formulation:?} p={order}, periodic={periodic}, {bed:?}: \
                             dE/dt = {rate:.3e} (scale {scale:.3e})"
                        );
                    }
                }
            }
        }
    }

    /// Periodic bed with dry islands and shoreline elements: B = −1 + 1.5
    /// cos(2πx/L) cos(2πy/L) (plus ±0.2 m per-node noise if `rough`), η = 0.3.
    fn shoreline_setup(
        order: usize,
        periodic: bool,
        rough: bool,
    ) -> (Mesh2D, DGOperators2D, GeometricFactors2D, Bathymetry2D) {
        let (mesh, ops, geom, mut bathymetry) = setup(order, periodic, Bed::Smooth);
        let tau = 2.0 * std::f64::consts::PI / L;
        let mut seed: u64 = 0x2545_f491_4f6c_dd1d;
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let mut b = -1.0 + 1.5 * (tau * x).cos() * (tau * y).cos();
                if rough {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    b += 0.4 * ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5);
                }
                bathymetry.data[k.as_usize() * ops.n_nodes + i] = b;
            }
        }
        bathymetry.compute_gradients(&ops, &geom);
        (mesh, ops, geom, bathymetry)
    }

    /// Lake at rest η = 0.3 over `bathymetry` (h = max(0, η − B)), plus the
    /// velocity field of [`state`] times `speed` at wet nodes.
    fn shoreline_state(
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        bathymetry: &Bathymetry2D,
        speed: f64,
    ) -> SWESolution2D {
        let mut q = state(mesh, ops, bathymetry, speed);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let s = q.get_state(k, i);
                let h = (0.3 - bathymetry.get(k, i)).max(0.0);
                let scale = if s.h > 0.0 { h / s.h } else { 0.0 };
                q.set_state(k, i, SWEState2D::new(h, s.hu * scale, s.hv * scale));
            }
        }
        q
    }

    /// The face pass evaluates each interior face once for both sides: the
    /// mass fluxes the two neighbours see are exactly opposite, for
    /// every split form, over shorelines and rough beds; momentum differs only
    /// by each side's own bed/hydrostatic term.
    #[test]
    fn test_face_pass_exchanges_opposite_mass_fluxes() {
        use super::SplitFormSWE2D;

        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        for formulation in SPLIT_FORMS {
            for (order, rough) in [(2, true), (3, false)] {
                let (mesh, ops, geom, bathymetry) = shoreline_setup(order, true, rough);
                let q = shoreline_state(&mesh, &ops, &bathymetry, 0.5);
                let config = SWE2DRhsConfig::new(&equation, &bc)
                    .with_formulation(formulation)
                    .with_bathymetry(&bathymetry);
                let split = SplitFormSWE2D::new(&q, &mesh, &ops, &geom, &config, 0.0).unwrap();
                let nf = ops.n_face_nodes;
                let mut slots = vec![SWEState2D::zero(); 2 * nf];
                let mut nonzero = 0;
                for e in 0..mesh.n_edges {
                    split.edge_fluxes(e, &mut slots);
                    for fi in 0..nf {
                        let (left, right) = (slots[fi], slots[nf + nf - 1 - fi]);
                        // Exact equality (== also equates the +0/−0 of dry faces)
                        assert!(
                            left.h == -right.h,
                            "{formulation:?}: {} vs {}",
                            left.h,
                            right.h
                        );
                        nonzero += usize::from(left.h != 0.0);
                    }
                }
                assert!(nonzero > 0, "{formulation:?}: no flow through faces");
            }
        }
    }

    #[test]
    fn test_wet_dry_lake_at_rest_with_shorelines() {
        // Elements cut by the shoreline have dry nodes where η = B ≠ 0.3, so the
        // flux-differencing volume term is not balanced there; WetDry switches
        // them to subcell finite volumes with hydrostatic reconstruction.
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        for order in 1..=4 {
            for periodic in [true, false] {
                for rough in [false, true] {
                    let (mesh, ops, geom, bathymetry) = shoreline_setup(order, periodic, rough);
                    let q = shoreline_state(&mesh, &ops, &bathymetry, 0.0);
                    assert!(q.h_data().contains(&0.0), "no dry nodes");

                    let config = SWE2DRhsConfig::new(&equation, &bc)
                        .with_coriolis(false)
                        .with_formulation(SWEFormulation2D::WetDry)
                        .with_bathymetry(&bathymetry);
                    let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
                    let max_h = rhs.data[0].iter().fold(0.0_f64, |m, x| m.max(x.abs()));
                    let max_mom = max_momentum_rate(&rhs);
                    assert!(
                        max_h < 1e-14 && max_mom < 1e-13,
                        "p={order}, periodic={periodic}, rough={rough}: \
                         max |dh/dt| = {max_h:.3e}, max |d(hu)/dt| = {max_mom:.3e}"
                    );

                    // Negative control: the wet-only split form is not balanced
                    if !rough {
                        let config = config.with_formulation(SWEFormulation2D::EntropyStable);
                        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
                        assert!(max_momentum_rate(&rhs) > 1e-6, "p={order}: balanced?");
                    }
                }
            }
        }
    }

    #[test]
    fn test_wet_dry_lake_at_rest_with_films_below_h_min() {
        // The HLL flux returns zero, pressure included, when both sides are
        // shallower than h_min; the hydrostatic correction must then supply
        // all of ½g h². Films of 5 mm under h_min = 1 cm make the dropped
        // pressure visible (~1e-7 m/s² before the fix).
        let equation = ShallowWater2D::with_h_min(G, Depth::new(1e-2));
        let bc = Reflective2D::new();
        for order in 1..=3 {
            let (mesh, ops, geom, mut bathymetry) = shoreline_setup(order, true, false);
            for b in bathymetry.data.iter_mut() {
                if (0.25..0.3).contains(b) {
                    *b = 0.3 - 5e-3;
                }
            }
            bathymetry.compute_gradients(&ops, &geom);
            let q = shoreline_state(&mesh, &ops, &bathymetry, 0.0);
            assert!(q.h_data().iter().any(|&h| (h - 5e-3).abs() < 1e-12));

            let config = SWE2DRhsConfig::new(&equation, &bc)
                .with_coriolis(false)
                .with_formulation(SWEFormulation2D::WetDry)
                .with_bathymetry(&bathymetry);
            let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
            let max_mom = max_momentum_rate(&rhs);
            assert!(max_mom < 1e-13, "p={order}: max |d(hu)/dt| = {max_mom:.3e}");
        }
    }

    #[test]
    fn test_wet_dry_mass_conservation_with_dry_regions() {
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        for order in 1..=3 {
            for rough in [false, true] {
                let (mesh, ops, geom, bathymetry) = shoreline_setup(order, true, rough);
                let q = shoreline_state(&mesh, &ops, &bathymetry, 1.0);
                let mass = integrate(&mesh, &ops, &geom, |k, i| q.get_var(k, i, 0));
                let config = SWE2DRhsConfig::new(&equation, &bc)
                    .with_coriolis(false)
                    .with_formulation(SWEFormulation2D::WetDry)
                    .with_bathymetry(&bathymetry);
                let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
                let mass_rate = integrate(&mesh, &ops, &geom, |k, i| rhs.get_var(k, i, 0));
                // Relative to the mass flux scale ∫|dh/dt|
                let scale = integrate(&mesh, &ops, &geom, |k, i| rhs.get_var(k, i, 0).abs());
                assert!(
                    mass_rate.abs() < 1e-14 * scale,
                    "p={order}, rough={rough}: d(mass)/dt = {mass_rate:.3e} (scale {scale:.3e}, mass {mass:.3e})"
                );
            }
        }
    }

    #[test]
    fn test_wet_dry_on_submesh_with_new_walls() {
        // A water-only submesh (Mesh2D::retain_elements) with an island cut out
        // of the middle: the new faces are walls, and the operator stays
        // conservative and well-balanced there
        use crate::mesh::BoundaryTag;
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let full = Mesh2D::uniform_rectangle(0.0, L, 0.0, L, 8, 8);
        let island = |k: ElementIndex| {
            let [x, y] = full.reference_to_physical(k, 0.0, 0.0);
            (0.3 * L..0.6 * L).contains(&x) && (0.4 * L..0.7 * L).contains(&y)
        };
        let (mesh, _) = full.retain_elements(|k| !island(k), BoundaryTag::Wall);
        assert!(mesh.n_elements < full.n_elements);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, smooth_bed);
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_formulation(SWEFormulation2D::WetDry)
            .with_bathymetry(&bathymetry);

        let rest = state(&mesh, &ops, &bathymetry, 0.0);
        let rhs = compute_rhs_swe_2d(&rest, &mesh, &ops, &geom, &config, 0.0);
        assert!(rhs.max_abs() < 1e-9, "lake at rest: {:.3e}", rhs.max_abs());

        let q = state(&mesh, &ops, &bathymetry, 1.0);
        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
        let mass_rate = integrate(&mesh, &ops, &geom, |k, i| rhs.get_var(k, i, 0));
        let scale = integrate(&mesh, &ops, &geom, |k, i| rhs.get_var(k, i, 0).abs());
        assert!(
            mass_rate.abs() < 1e-14 * scale,
            "d(mass)/dt = {mass_rate:.3e} (scale {scale:.3e})"
        );
    }

    /// A dry node below the water beside it must flood. Two P2 stencils
    /// closed that interface: a wet node between a bank and the dry node
    /// (the Frøya strait, a one-node strip between land at +4 m and a sill
    /// at +0.35 m, where the sill stayed dry under 0.65 m of water while a
    /// 5 m/s jet pushed at it), and a dry node between a higher and a lower
    /// wet node. Both took the dry node's η = B for a water level, so the
    /// limited η slope set the reconstructed face to the dry node's bed.
    #[test]
    fn test_wet_dry_floods_a_dry_node_below_the_water() {
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let ops = DGOperators2D::new(2);
        // (bed, surface) at the GLL nodes along x, uniform in y; the surface
        // of a dry node is ignored
        let cases: [(&str, &[(f64, f64)]); 2] = [
            (
                "bank | strip | sill | strip | bank",
                &[
                    (4.0, 0.0),
                    (-3.8, 0.8),
                    (0.35, 0.0),
                    (-3.8, 0.8),
                    (4.0, 0.0),
                ],
            ),
            (
                "high water | dry | low water",
                &[(-3.0, 0.8), (0.5, 0.0), (-3.0, -0.2)],
            ),
        ];
        for (name, nodes) in cases {
            let nx = (nodes.len() - 1) / (ops.n_1d - 1);
            let width = 100.0 * nx as f64;
            let mesh = Mesh2D::uniform_rectangle(0.0, width, 0.0, 100.0, nx, 1);
            let geom = GeometricFactors2D::compute(&mesh, &ops);
            let mut bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |_, _| 0.0);
            let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
            let position = |k: ElementIndex, i: usize| {
                let [x, _] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                (x / width * (nodes.len() - 1) as f64).round() as usize
            };
            for k in ElementIndex::iter(mesh.n_elements) {
                for i in 0..ops.n_nodes {
                    let (b, eta) = nodes[position(k, i)];
                    bathymetry.data[k.as_usize() * ops.n_nodes + i] = b;
                    q.set_state(k, i, SWEState2D::new((eta - b).max(0.0), 0.0, 0.0));
                }
            }
            bathymetry.compute_gradients(&ops, &geom);
            let config = SWE2DRhsConfig::new(&equation, &bc)
                .with_coriolis(false)
                .with_formulation(SWEFormulation2D::WetDry)
                .with_bathymetry(&bathymetry);
            let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
            for k in ElementIndex::iter(mesh.n_elements) {
                for i in 0..ops.n_nodes {
                    let (b, eta) = nodes[position(k, i)];
                    let below_water = nodes
                        .iter()
                        .skip(position(k, i).saturating_sub(1))
                        .take(if position(k, i) == 0 { 2 } else { 3 })
                        .any(|&(nb, n_eta)| n_eta > nb && n_eta > b);
                    if eta <= b && below_water {
                        let dh = rhs.get_var(k, i, 0);
                        assert!(dh > 1e-3, "{name}: dry node {i} of {k:?}: dh/dt = {dh:.3e}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_uniform_flow_flat_bottom_is_steady() {
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::from_primitives(2.0, 0.5, -0.2));
            }
        }
        for formulation in SPLIT_FORMS {
            let config = SWE2DRhsConfig::new(&equation, &bc)
                .with_coriolis(false)
                .with_formulation(formulation);
            let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
            assert!(
                rhs.max_abs() < 1e-12,
                "{formulation:?}: {:.3e}",
                rhs.max_abs()
            );
        }
    }

    #[test]
    #[should_panic(expected = "remove BathymetrySource2D")]
    fn test_rejects_double_counted_bed_slope() {
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let (mesh, ops, geom, bathymetry) = setup(1, true, Bed::Smooth);
        let q = state(&mesh, &ops, &bathymetry, 0.0);
        let bathy_source = BathymetrySource2D::new(G);
        let coriolis = CoriolisSource2D::f_plane(1.2e-4);
        let combined = CombinedSource2D::new(vec![&coriolis, &bathy_source]);
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_formulation(SWEFormulation2D::EntropyStable)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&combined);
        compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
    }

    #[test]
    #[cfg(all(feature = "parallel", feature = "simd"))]
    fn test_parallel_matches_serial() {
        use crate::solver::rhs::swe_2d::compute_rhs_swe_2d_parallel;

        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let coriolis = CoriolisSource2D::f_plane(1.2e-4);
        for periodic in [true, false] {
            let (mesh, ops, geom, bathymetry) = setup(3, periodic, Bed::Rough);
            let q = state(&mesh, &ops, &bathymetry, 1.0);
            for formulation in SPLIT_FORMS {
                let config = SWE2DRhsConfig::new(&equation, &bc)
                    .with_coriolis(false)
                    .with_formulation(formulation)
                    .with_bathymetry(&bathymetry)
                    .with_source_terms(&coriolis);
                let serial = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
                let parallel = compute_rhs_swe_2d_parallel(&q, &mesh, &ops, &geom, &config, 0.0);
                // Same kernel, same summation order: bitwise identical
                assert_eq!(
                    serial.data, parallel.data,
                    "periodic={periodic}, {formulation:?}"
                );
            }

            // Wet/dry subcells active in the shoreline elements
            let (mesh, ops, geom, bathymetry) = shoreline_setup(3, periodic, true);
            let q = shoreline_state(&mesh, &ops, &bathymetry, 1.0);
            let config = SWE2DRhsConfig::new(&equation, &bc)
                .with_coriolis(false)
                .with_formulation(SWEFormulation2D::WetDry)
                .with_bathymetry(&bathymetry)
                .with_source_terms(&coriolis);
            let serial = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
            let parallel = compute_rhs_swe_2d_parallel(&q, &mesh, &ops, &geom, &config, 0.0);
            assert_eq!(serial.data, parallel.data, "periodic={periodic}, WetDry");
        }
    }

    /// Every value of a solution as bits (NaN and the sign of zero included)
    #[cfg(feature = "simd")]
    fn bits(q: &SWESolution2D) -> Vec<u64> {
        q.data.iter().flatten().map(|x| x.to_bits()).collect()
    }

    /// The volume terms batched across elements and the face fluxes across
    /// edges (one per SIMD lane) give every element's RHS bit for bit:
    /// P1–P5 (P5 is not batched), on
    /// parallelograms, general quadrilaterals and a mesh of both (mixed
    /// batches), with and without bathymetry, with shoreline subcells in
    /// some lanes, and 35 elements (a short last batch). Also through the
    /// subset RHS of local time stepping, whose batches are arbitrary lists,
    /// and the face and subcell mass fluxes, serial and parallel.
    #[test]
    #[cfg(feature = "simd")]
    fn test_batched_kernels_are_bit_for_bit() {
        use super::batch::UNBATCHED;
        #[cfg(feature = "parallel")]
        use crate::solver::rhs::swe_2d::compute_rhs_swe_2d_parallel_mass_fluxes_into;
        use crate::solver::rhs::swe_2d::{
            compute_rhs_swe_2d_mass_fluxes_into, compute_rhs_swe_2d_subset_mass_fluxes_then,
            compute_rhs_swe_2d_subset_then, face_mass_len, subcell_mass_len,
        };

        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let unbatched = |f: &dyn Fn() -> SWESolution2D| {
            UNBATCHED.with(|u| u.set(true));
            let q = f();
            UNBATCHED.with(|u| u.set(false));
            q
        };
        let tau = 2.0 * std::f64::consts::PI / L;
        for order in 1..=5 {
            // 0: parallelograms, 1: all distorted, 2: distorted for x > L/2
            for distortion in 0..3 {
                let mut mesh = Mesh2D::uniform_rectangle(0.0, L, 0.0, 0.5 * L, 7, 5);
                if distortion > 0 {
                    let a = 0.03 * L;
                    for v in &mut mesh.vertices {
                        let [x, y] = *v;
                        if distortion == 1 || x > 0.5 * L {
                            let (sx, sy) = ((tau * x).sin(), (2.0 * tau * y).sin());
                            *v = [x + a * sx * sy, y - 0.7 * a * sx * sy];
                        }
                    }
                }
                let ops = DGOperators2D::new(order);
                let geom = GeometricFactors2D::compute(&mesh, &ops);
                if order > 1 {
                    assert_eq!(
                        geom.is_affine(),
                        distortion == 0,
                        "P{order}, distortion {distortion}"
                    );
                }
                let mut bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
                    -1.0 + 1.5 * (tau * x).cos() * (tau * y).cos()
                });
                let mut seed: u64 = 0x2545_f491_4f6c_dd1d;
                for b in bathymetry.data.iter_mut() {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    *b += 0.4 * ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5);
                }
                bathymetry.compute_gradients(&ops, &geom);
                let q = shoreline_state(&mesh, &ops, &bathymetry, 1.0);
                let deep = state(&mesh, &ops, &bathymetry, 1.0);

                for (formulation, q, with_bed) in [
                    (SWEFormulation2D::WetDry, &q, true),
                    (SWEFormulation2D::EntropyStable, &deep, true),
                    (SWEFormulation2D::EntropyConservative, &deep, false),
                ] {
                    let mut config = SWE2DRhsConfig::new(&equation, &bc)
                        .with_coriolis(false)
                        .with_formulation(formulation);
                    if with_bed {
                        config = config.with_bathymetry(&bathymetry);
                    }
                    let case = format!("P{order}, distortion {distortion}, {formulation:?}");
                    let batched = compute_rhs_swe_2d(q, &mesh, &ops, &geom, &config, 0.0);
                    let reference =
                        unbatched(&|| compute_rhs_swe_2d(q, &mesh, &ops, &geom, &config, 0.0));
                    assert!(bits(&batched) == bits(&reference), "{case}");

                    // The face and subcell mass fluxes (from the batch of the
                    // subcell elements), serial and parallel
                    let mass_fluxes = |parallel: bool| {
                        let mut rhs = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
                        let mut face = vec![0.0; face_mass_len(&mesh, &ops)];
                        let mut subcell = vec![0.0; subcell_mass_len(&mesh, &ops)];
                        if parallel {
                            #[cfg(feature = "parallel")]
                            compute_rhs_swe_2d_parallel_mass_fluxes_into(
                                q,
                                &mesh,
                                &ops,
                                &geom,
                                &config,
                                0.0,
                                &mut rhs,
                                &mut face,
                                &mut subcell,
                            );
                        } else {
                            compute_rhs_swe_2d_mass_fluxes_into(
                                q,
                                &mesh,
                                &ops,
                                &geom,
                                &config,
                                0.0,
                                &mut rhs,
                                &mut face,
                                &mut subcell,
                            );
                        }
                        let mut all = bits(&rhs);
                        all.extend(face.iter().chain(&subcell).map(|x| x.to_bits()));
                        all
                    };
                    UNBATCHED.with(|u| u.set(true));
                    let reference_fluxes = mass_fluxes(false);
                    UNBATCHED.with(|u| u.set(false));
                    assert!(
                        mass_fluxes(false) == reference_fluxes,
                        "{case}, mass fluxes"
                    );
                    if cfg!(feature = "parallel") {
                        assert!(mass_fluxes(true) == reference_fluxes, "{case}, parallel");
                    }

                    // Subset RHS over a scrambled list of most elements
                    let elements: Vec<u32> = (0..mesh.n_elements as u32)
                        .map(|k| (k * 11) % mesh.n_elements as u32)
                        .filter(|k| k % 5 != 2)
                        .collect();
                    let mut subset = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
                    compute_rhs_swe_2d_subset_then(
                        q,
                        &mesh,
                        &ops,
                        &geom,
                        &config,
                        &elements,
                        &|_| 0.0,
                        &mut subset,
                        None,
                        &|_, _, _| {},
                    );
                    let n = ops.n_nodes;
                    for &k in &elements {
                        let rows = k as usize * n..(k as usize + 1) * n;
                        for (a, b) in subset.data.iter().zip(&reference.data) {
                            assert!(
                                a[rows.clone()]
                                    .iter()
                                    .zip(&b[rows.clone()])
                                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                                "{case}, subset element {k}"
                            );
                        }
                    }

                    // And its elements' mass fluxes
                    let (n_face, n_sub) = (
                        face_mass_len(&mesh, &ops) / mesh.n_elements,
                        subcell_mass_len(&mesh, &ops) / mesh.n_elements,
                    );
                    let face = std::sync::Mutex::new(vec![f64::NAN; face_mass_len(&mesh, &ops)]);
                    let subcell =
                        std::sync::Mutex::new(vec![f64::NAN; subcell_mass_len(&mesh, &ops)]);
                    let mut subset = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
                    compute_rhs_swe_2d_subset_mass_fluxes_then(
                        q,
                        &mesh,
                        &ops,
                        &geom,
                        &config,
                        &elements,
                        &|_| 0.0,
                        &mut subset,
                        &|k, _, mass| {
                            face.lock().unwrap()[k * n_face..(k + 1) * n_face]
                                .copy_from_slice(mass.face);
                            subcell.lock().unwrap()[k * n_sub..(k + 1) * n_sub]
                                .copy_from_slice(mass.subcell);
                        },
                    );
                    let (face, subcell) =
                        (face.into_inner().unwrap(), subcell.into_inner().unwrap());
                    let n_rhs = 3 * mesh.n_elements * n;
                    let (ref_face, ref_subcell) =
                        reference_fluxes[n_rhs..].split_at(face_mass_len(&mesh, &ops));
                    for &k in &elements {
                        let k = k as usize;
                        let faces = k * n_face..(k + 1) * n_face;
                        let subs = k * n_sub..(k + 1) * n_sub;
                        assert!(
                            face[faces.clone()]
                                .iter()
                                .zip(&ref_face[faces])
                                .all(|(a, &b)| a.to_bits() == b),
                            "{case}, subset face mass fluxes of element {k}"
                        );
                        assert!(
                            subcell[subs.clone()]
                                .iter()
                                .zip(&ref_subcell[subs])
                                .all(|(a, &b)| a.to_bits() == b),
                            "{case}, subset subcell mass fluxes of element {k}"
                        );
                    }
                }
            }
        }
    }

    /// The batches against the per-element kernel where the gate above does
    /// not reach. Periodic pairs: faces that do not share their vertices, so
    /// that the edge's normal is not minus the right side's own and the
    /// neighbour's `sJ` not the element's (`FaceLink::shares_vertices`); on
    /// general quadrilaterals, with walls (the channel) and without. And a
    /// boundary condition that depends on time, in a subset pass that gives
    /// each element its own time, as local time stepping does.
    #[test]
    #[cfg(feature = "simd")]
    fn test_batches_are_bit_for_bit_at_periodic_faces_and_timed_boundaries() {
        use super::batch::UNBATCHED;
        use crate::boundary::{BCContext2D, SWEBoundaryCondition2D};
        use crate::solver::rhs::swe_2d::compute_rhs_swe_2d_subset_then;

        /// A ghost state that changes with time
        struct Pulsing;
        impl SWEBoundaryCondition2D for Pulsing {
            fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
                let q = ctx.interior_state;
                let a = 1.0 + 0.1 * (0.37 * ctx.time).sin();
                SWEState2D::new(a * q.h, -q.hu, 0.5 * a * q.hv)
            }
            fn name(&self) -> &'static str {
                "pulsing"
            }
        }

        let equation = ShallowWater2D::new(G);
        let bc = Pulsing;
        let unbatched = |f: &dyn Fn() -> SWESolution2D| {
            UNBATCHED.with(|u| u.set(true));
            let q = f();
            UNBATCHED.with(|u| u.set(false));
            q
        };
        let tau = 2.0 * std::f64::consts::PI / L;
        for order in 1..=4 {
            for (name, mut mesh) in [
                (
                    "periodic",
                    Mesh2D::uniform_periodic(0.0, L, 0.0, 0.5 * L, 7, 5),
                ),
                (
                    "channel",
                    Mesh2D::channel_periodic_x(0.0, L, 0.0, 0.5 * L, 7, 5),
                ),
            ] {
                // General quadrilaterals; the two sides of a periodic pair
                // are moved apart in shape as well
                let a = 0.03 * L;
                for v in &mut mesh.vertices {
                    let [x, y] = *v;
                    let (sx, sy) = ((0.7 * tau * x).sin(), (1.3 * tau * y).cos());
                    *v = [x + a * sx * sy, y - 0.7 * a * sx * sy];
                }
                assert!(
                    mesh.face_links
                        .iter()
                        .flatten()
                        .any(|link| !link.is_boundary() && !link.shares_vertices()),
                    "{name}: periodic pairs"
                );
                let ops = DGOperators2D::new(order);
                let geom = GeometricFactors2D::compute(&mesh, &ops);
                let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
                    -1.0 + 1.5 * (tau * x).cos() * (tau * y).cos()
                });
                let q = shoreline_state(&mesh, &ops, &bathymetry, 1.0);
                let deep = state(&mesh, &ops, &bathymetry, 1.0);
                for (formulation, q) in [
                    (SWEFormulation2D::WetDry, &q),
                    (SWEFormulation2D::EntropyStable, &deep),
                ] {
                    let config = SWE2DRhsConfig::new(&equation, &bc)
                        .with_coriolis(false)
                        .with_formulation(formulation)
                        .with_bathymetry(&bathymetry);
                    let case = format!("P{order}, {name}, {formulation:?}");
                    let rhs = || compute_rhs_swe_2d(q, &mesh, &ops, &geom, &config, 1.7);
                    assert!(bits(&rhs()) == bits(&unbatched(&rhs)), "{case}");

                    // Each element at its own time (35 elements: on this
                    // thread, so that the reference is unbatched)
                    let elements: Vec<u32> = (0..mesh.n_elements as u32)
                        .map(|k| (k * 11) % mesh.n_elements as u32)
                        .collect();
                    let subset = || {
                        let mut out = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
                        compute_rhs_swe_2d_subset_then(
                            q,
                            &mesh,
                            &ops,
                            &geom,
                            &config,
                            &elements,
                            &|k| 0.3 * k as f64,
                            &mut out,
                            None,
                            &|_, _, _| {},
                        );
                        out
                    };
                    assert!(
                        bits(&subset()) == bits(&unbatched(&subset)),
                        "{case}, own times"
                    );
                }
            }
        }
    }
}
