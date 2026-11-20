//! Baroclinic pressure gradient in σ-coordinates, from pairwise pressure
//! differences.
//!
//! # The σ-coordinate problem
//!
//! The horizontal pressure gradient force is `F = −(1/ρ₀) ∇p|_z`. On a
//! σ-surface the chain rule gives two large terms that must cancel,
//! `∇p|_z = ∇p|_σ + gρ ∇z|_σ`, and over steep bathymetry any inconsistency in
//! their discretisations appears as a spurious force as large as the physical
//! one (Haney 1991; Mellor et al. 1994). An early discretisation — the
//! element-local derivative of `p` along σ plus `gρ∇z`, with a first-order
//! half-layer pressure — made ~1e-4 m/s² in a stratified fjord at rest.
//!
//! # Pairwise pressure differences
//!
//! Every term of the DG derivative is a pressure difference of a pair of
//! columns. With `D` the element's nodal differentiation matrix
//! (`(Dx)_ij = r_x,i (D_r)_ij + s_x,i (D_s)_ij`, the metric at node `i`),
//!
//! ```text
//!     (∂p/∂x|_z)_i ≈ Σ_j Dx_ij Δp_ij,
//! ```
//!
//! with `Δp_ij` antisymmetric, in one of two forms ([`PressureGradientForm`]).
//! `p_c(z)` is the hydrostatic pressure of column `c` at depth `z`, and `z_i`,
//! `z_j` are the depths of nodes `i`, `j` on the σ-level.
//!
//! **Constant depth** (Stelling & van Kester 1994, in a DG form):
//!
//! ```text
//!     Δp_ij = ½ [ (p_j(z_i) − p_i(z_i)) + (p_j(z_j) − p_i(z_j)) ].
//! ```
//!
//! A smooth function of the node positions, so the sum is a high-order,
//! consistent approximation of `∂p/∂x|_z` (`Σ_j Dx_ij (x_j − x_i) = 1`).
//! Where a node lies below the other column's bed (a hydrostatically
//! inconsistent pair, the discrete Haney condition), only the difference at
//! the shallower node's depth, which both columns reach, is used; first order
//! there, but balanced. In a fluid at rest with `ρ = ρ(z)` every column
//! samples the same profile, so `Δp_ij` is the difference of two
//! interpolants of one profile: zero when `ρ` is linear in `z` (constant
//! `N²`), and the interpolation error otherwise — not amplified by the slope
//! as in the σ form.
//!
//! **σ-pairs** (the σ form `∇_σp + gρ∇_σz` in flux-differencing form):
//!
//! ```text
//!     Δp_ij = (p_j(z_j) − p_i(z_i)) + ½(ρ_i + ρ_j)(z_j − z_i),
//! ```
//!
//! whose second term is `½[ρ∇z + ∇(ρz) − z∇ρ]` at the nodes, the split form
//! of `ρ∇_σz`. Exact for linear `ρ(z)` as well (the trapezoid rule of the
//! pressure along the pair); for a curved profile its error is that rule's,
//! `ρ″Δz³/12` with `Δz` the depth difference of the pair, the σ form's
//! hydrostatic inconsistency (Haney 1991; Mellor et al. 1994): a pycnocline
//! sampled by levels far apart in neighbouring columns makes a large force.
//!
//! # Energy: why σ-pairs are the default
//!
//! In the continuum the work of the pressure gradient on the flow is the
//! conversion to potential energy by the density advection, which is what
//! keeps a stratified fluid at rest stable. Discretely that needs the PGF's
//! `ρ∇z` term to be the negative adjoint of the advection of the background
//! stratification along the sloping σ-levels. Under the GLL nodes'
//! summation by parts `MD + DᵀM = B`, the σ-pairs' term is exactly that for
//! the split-form advection of
//! [`crate::solver::rhs::advective_divergence_element`]: with `ρ̄ = a + bz`,
//! the advection makes `−b·½[∇·(Qz) + Q·∇z − z∇·Q]` of the background, and
//! summed against the nodes it pairs with `½[ρ′∇z + ∇(ρ′z) − z∇ρ′]` up to
//! face terms. In terms of the total potential energy `Σ gρz H_z` the
//! horizontal exchange is exact for any `ρ`, not only about a linear
//! background: both terms are sums of `(MD)_ij {{Q}}_ij {{ρ}}_ij (z_j − z_i)`
//! with opposite signs. (The vertical half, the Hermite pressure integral
//! against the vertical advection's face values, matches only for linear
//! profiles.) The constant-depth form has no such partner: its vertical
//! interpolation of the neighbour's pressure is not the transpose of
//! anything the advection does.
//!
//! Measured over a stratified seamount at rest (`examples/seamount_3d.rs`:
//! constant `N²`, 16 × 16 P2 elements of 4 km, 20 stretched levels, bed
//! 40–400 m, `r_x0` 0.32), the growth of the round-off velocity every 6 h:
//!
//! | PGF | horizontal advection | growth per 6 h |
//! |---|---|---|
//! | constant depth | conservative `∇·(Qφ)` | × 11 (× 18 on 2 km elements) |
//! | constant depth | split form | × 5 |
//! | σ-pairs | conservative | × 3.5 |
//! | σ-pairs | split form | none (7e-11 m/s after 2 days) |
//!
//! Neither horizontal viscosity (10 m²/s), vertical viscosity (1e-2 m²/s),
//! a penalty on the layer velocities' face jumps, nor a gentler slope
//! (`r_x0` 0.08: blew up after 3 days) stopped the growth of the first row.
//!
//! Exact energy exchange is not enough on its own. Rest is stable when a
//! conserved functional has its minimum there. The split form also conserves
//! the tracer variance `Σ H_z ρ²`, and for a linear `ρ(z)` energy plus
//! `g/(2|∂_z ρ|)` times the variance is such a functional. For a curved
//! profile there is none once a σ-level crosses the pycnocline between two
//! nodes of one element, as at the cliffs of a fjord coast. There the pair's
//! chord `Δρ/Δz` is far steeper than the stratification at either node, and
//! round-off grows (e-folding ≈ 15 min on the Frøya bed, 47 min in the x–z
//! cliff of `a_pycnocline_over_a_cliff_stays_at_rest`). The cure is a
//! reference stratification (next section).
//!
//! # About a reference: the anomaly, and the advection that is its adjoint
//!
//! With a reference stratification, a horizontally uniform `T̄(z)`, `S̄(z)`
//! with the density `ρ̄(z)` of its water
//! ([`crate::physics::Hydrostatic3D::with_reference_profile`]), the force is
//! the σ-pairs force of the anomaly `ρ′ = ρ − ρ̄(z)`, `ρ̄` taken at every
//! node's own height at the stage's `η` ([`reference_density_into`]), with
//! the anomaly's pressure integrated by the trapezoid rule,
//! `p_l − p_{l+1} = ½(ρ′_l + ρ′_{l+1})(z_{l+1} − z_l)`: linear in `ρ′`
//! (the monotone cubic's slopes are not). A fluid at rest in the reference
//! feels no force at all, over any bed (Mellor et al. 1998).
//!
//! **The force's work.** The layers move their volume along the edges of a
//! graph: the pairs of nodes `(i, j)` of an element on a level, each with the
//! volume flux `Φ_ij = m_i Dx_ij·Q_i − m_j Dx_ji·Q_j` from `i` to `j`
//! (`m = wJ` the nodal mass; on a parallelogram `2 m_i Dx_ij·{{Q}}_ij`, the
//! split form's), the element faces, and the σ-surfaces between the layers
//! of a column, each with `m Ω`. Summing the force against the layer
//! transports `Q`, by parts with the layers' continuity (the pressure
//! differences of the pairs and faces become those across the σ-surfaces):
//!
//! ```text
//!     −(ρ₀/g) Σ_a m_a Q_a·F_a = Σ_edges F_ab ½(ρ′_a + ρ′_b)(z_b − z_a)
//!                             = Σ_a m_a ρ′_a (H_z w)_a,
//!     (H_z w)_a = (1/m_a) Σ_b ½ F_ab (z_b − z_a),
//! ```
//!
//! over the edges the force acts along: the pairs of an element without a
//! thin column, and the σ-surfaces of the columns that are not thin (a face
//! joins nodes at one height, where the bed is continuous). `w` is the rate
//! at which the layer's water rises, `u·∇_σ z + Ω`, in the split form and
//! with `Ω` averaged to the layer centre by the trapezoid rule's weights:
//! the force's adjoint. The identity holds to round-off for transports
//! without a depth mean between walls (gate
//! `the_reference_advection_is_the_pressure_gradients_adjoint`: on
//! parallelograms and distorted elements, both metric forms, over a cliff,
//! with a shore element); a depth mean adds the free surface's own rate.
//!
//! **The tracers.** The conservative advection moves a tracer along an
//! edge by `−½ F_ab (C_b − C_a)/m_a`. In a fluid at rest in the reference
//! that jump is the reference's chord `C̄(z_b) − C̄(z_a)`, which across a
//! pair that straddles the pycnocline has nothing to do with the
//! stratification at either end. [`add_reference_advection`] replaces the
//! chord by the tangent, `γ_a (z_b − z_a)` with `γ_a` the reference's mean
//! gradient over the layer of `a`, on exactly the edges the force acts
//! along, so that the reference is advected as
//!
//! ```text
//!     H_z ∂C/∂t = −γ^C H_z w,    hence    H_z ∂ρ′/∂t = Γ H_z w,
//!     Γ = −(ρ_T γ^T + ρ_S γ^S) ≥ 0  (a statically stable reference),
//! ```
//!
//! with the same `w`. Then, for the scheme linearised about rest in the
//! reference,
//!
//! ```text
//!     d/dt [ ½ρ₀ Σ m H_z |u|² + ½ g Σ m H_z ρ′²/Γ ] = −g b + g b = 0,
//! ```
//!
//! `b` the sum above: the kinetic energy and the available potential energy
//! (Holliday & McIntyre 1981) with positive weights exchange exactly, the
//! operator is skew-adjoint in that norm (Bell, Peixoto & Thuburn 2017), and
//! no mode grows, whatever the bed, the mesh, or how the σ-levels cross the
//! pycnocline. The weights are those of the stratification at each node, so
//! nothing is stiff: the fastest mode is an internal wave's. Where `Γ = 0`
//! the anomaly does not change. Coriolis does no work; the viscosities and
//! the upwind faces only remove kinetic energy.
//!
//! **Conservation, and the mean displacement.** The tangent is not a flux
//! between two cells: along an edge the two ends' terms add up to
//! `F_ab τ_ab`, `τ_ab = (C̄_b − C̄_a) − ½(γ_a + γ_b)(z_b − z_a)`, the
//! trapezoid rule's error of the reference's gradient. (No single-valued
//! flux between cells paired by level index has this structure with bounded
//! weights: the conservative one's linearisation is the stiff pair
//! density's, `docs/stratified-rest-over-steep-beds.md`.) The element takes
//! that back: with `R = Σ_edges F_ab τ_ab` over its edges and
//! `N = Σ m H_z Γ` over its stably stratified layers, every layer gets
//! `−H_z Γ R/N`, a uniform displacement of the element's stratification at
//! the rate `R/N`. So each element keeps its inventories of `T` and `S` to
//! round-off, as the conservative scheme does (gates
//! `the_reference_advection_and_the_pressure_gradient_exchange_pseudo_energy_exactly`,
//! `a_curved_reference_keeps_constancy_and_the_tracers_inventories`).
//!
//! The force has the adjoint of that step. With the element's mean
//! displacement `κ = Σ m H_z ρ′/N` (the projection of the anomaly on a
//! uniform displacement of the reference, whose anomaly is `κΓ`, in the
//! norm above; [`ReferenceFields::set_displacements`]), every pressure
//! difference along an edge of the element leaves out the trapezoid rule's
//! error of that displacement's anomaly,
//!
//! ```text
//!     ½(ρ′_a + ρ′_b)(z_b − z_a) − κ [(ρ̄_b − ρ̄_a) + ½(Γ_a + Γ_b)(z_b − z_a)],
//! ```
//!
//! which integrates `κΓ` along the edge exactly, as `−κ(ρ̄_b − ρ̄_a)`. A
//! stratification displaced as a whole within an element then exerts no
//! force within it (but for the surface half-layer), where the trapezoid
//! rule alone gives the σ form's error of the displaced profile at every
//! straddle. With both, the identity above holds term by term again: the
//! rate of the potential energy gets `−g κ R`, and the force's work
//! `+g κ R`.
//!
//! Of a tracer's jump along an edge only the reference's share `θ_ab` is
//! treated so (`θ = clamp(ΔC/ΔC̄, 0, 1)`: none of a uniform tracer, all of
//! the reference itself), so constancy holds, and a stratification that has
//! drifted from the reference falls back on the conservative scheme. The
//! identity is exact for an equation of state linear in `T` and `S` (for
//! another, `Γ` is taken with its partials at the reference, and the
//! reference density's chords differ from the tracers' at the equation's
//! curvature).
//!
//! # Balanced reference
//!
//! The price of σ-pairs is the σ form's error at rest, for what is not in
//! the reference stratification. A stored reference state `ρ_s` (the
//! initial stratification, or a parent model's) takes it back
//! ([`BalancedReference`]):
//!
//! ```text
//!     F(ρ) = F_σ(ρ) + [F_z(ρ_s) − F_σ(ρ_s)],
//! ```
//!
//! with `F_σ`, `F_z` the σ-pair and constant-depth forces. The bracket is a
//! fixed field, so the dynamics (and their stability) are σ-pairs', while a
//! fluid resting in the reference state feels the constant-depth form's
//! force. (For a horizontally uniform `ρ_s(z)` take the reference
//! stratification above instead: it is exact at rest.)
//!
//! # Curvilinear elements
//!
//! On a general quadrilateral the metric varies over the element, and which
//! metric a pair is differenced with decides the energy argument above. The
//! work of the force on the layer transport `Q` is
//! `Σ_ij w_i D_ij Q_i·X_ij S_ij` for the pair's metric `X_ij` and an
//! antisymmetric `S_ij`, and under summation by parts (the boundary matrix is
//! diagonal and `S_ii = 0`) an advection whose pair transport is
//! `A_ij·Q_i + B_ij·Q_j` matches it when `X_ij = A_ij + B_ji`. The 3D
//! advection must reduce, for a constant field, to the layer continuity, the
//! 2D module's mass divergence ([`MetricForm`]), and must be a symmetric
//! two-point flux to conserve. So the force takes:
//!
//! - [`MetricForm::Conservative`] (`Standard` 2D module): each node's own
//!   metric, `(Dx)_ij = J_i⁻¹[(J∇r)_i (D_r)_ij + (J∇s)_i (D_s)_ij]·x̂`, the
//!   chain rule, the adjoint of the advection's `{{J∇r·Q}}_ij`;
//! - [`MetricForm::Averaged`] (the split forms): the pair's mean,
//!   `J_i⁻¹[{{J∇r}}_ij (D_r)_ij + {{J∇s}}_ij (D_s)_ij]·x̂`, the adjoint of
//!   `{{J∇r}}_ij·{{Q}}_ij`.
//!
//! On a parallelogram both are the affine derivative. Both are consistent
//! (the averaged one is `½[J∇r·D_r p + D_r(J∇r p)]/J` by the discrete metric
//! identities) and move momentum only across faces over level σ-surfaces, and
//! the balances at rest (`Δp_ij = 0`) do not depend on the element shape.
//! What differs is the aliasing of the metric times the pressure: on meshes
//! that are not asymptotically parallelograms (each element's bilinear term
//! a fixed fraction of its size, as on a coastline mesh) the averaged form
//! is one order lower pointwise (P2 1.1, P3 2.1 against the chain rule's 1.9
//! and 2.8, `tests/curvilinear_3d_test.rs`), while its element means are the
//! chain rule's and converge at about `N + 1`: the extra error is grid-scale,
//! with zero mean on every element. On the faces the normal, `sJ` and `J`
//! are those of each face node, as in the 3D transport.
//!
//! # Common to both forms
//!
//! Each column's density anomaly `ρ − ρ_ref` is a piecewise Hermite cubic in
//! `z` through the level values, with monotone harmonic-mean slopes
//! (Shchepetkin & McWilliams 2003, §5), continued linearly above the top and
//! below the bottom level. `p_c(z)` integrates it exactly from the surface.
//! A horizontally uniform `ρ(x)` over a flat bed is exact in both forms.
//!
//! **Faces.** The element-local sum ignores the neighbours. The jump of the
//! pressure across each face, in the same form, is lifted with a central
//! flux (`p* − p⁻ = ½Δp`), so a density front (or an `η` jump, for the full
//! PGF) sitting on a face exerts its force. In a fluid at rest the jump is
//! zero.
//!
//! **Wetting and drying.** A column thinner than `min_column_depth` exerts
//! and feels no pressure difference, and exchanges no baroclinic transport
//! ([`crate::solver::rhs::LayerTransport::compute`]). An element with such a
//! column moves its layers on the GLL subcells of the 2D wet/dry kernel with
//! their barotropic shares only, so it has no volume term here either: its
//! columns feel the pressure of their neighbours through its faces alone. (The
//! DG pairs of a shore element exchanged layer volume with dry nodes that
//! had no pressure-work partner, and a stratified shore at rest grew within
//! hours; baroclinic exchange through the subcells with its central-difference
//! force grew too, with an e-folding of 38 min; TODO P1.3.)
//!
//! **Reference density.** The integrated density is `ρ − rho_ref`:
//! `rho_ref = 0` gives the full PGF, whose `ρ`-uniform part is `−g∇η`
//! (DG-coupled through the face lift); `rho_ref = ρ₀` gives the
//! baroclinic-only PGF of mode splitting, where `−g∇η` comes from the 2D
//! module.

use crate::mesh::Mesh2D;
use crate::mesh::data::Bathymetry2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::TracerReferenceProfile;
use crate::solver::core::blocks::{Blocks, Pooled, for_each_block};
use crate::solver::rhs::transport_3d::{LayerTransport, MetricForm};
use crate::solver::state::Solution3D;
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

/// How [`compute_pressure_gradient`] differences the pressures of two columns
/// (see the module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PressureGradientForm {
    /// `(p_j − p_i) + ½(ρ_i + ρ_j)(z_j − z_i)` along the σ-level: consistent
    /// in energy with the split-form advection, so a stratified fluid over a
    /// slope stays stable (for curved profiles only while no σ-level crosses
    /// the pycnocline within an element, see the module docs); the σ form's
    /// error at rest for curved profiles (take it back with a
    /// [`BalancedReference`]).
    #[default]
    SigmaPairs,
    /// Differences at a common depth (Stelling & van Kester 1994): at rest
    /// only the vertical interpolation error, but not consistent in energy:
    /// over a slope a stratified fluid at rest is unstable (round-off grew
    /// tenfold every 6 h over a seamount). For diagnostics and comparison.
    ConstantDepth,
}

/// The force of a stored reference state in the constant-depth form minus
/// that in the σ-pairs form, `F_z(ρ_s) − F_σ(ρ_s)` (m/s²,
/// `[element][node][level]`), added to the σ-pairs force so that a fluid
/// resting in the reference state is balanced as by the constant-depth form
/// while the dynamics keep σ-pairs' energy consistency (see the module
/// docs).
///
/// Built from a state at its free surface `η`: a tide that moves the levels
/// by `η(1 + σ)` moves the reference's sampling with them, a relative error
/// of `η/D` in the correction.
#[derive(Clone, Debug)]
pub struct BalancedReference {
    /// x and y components of the correction.
    pub correction: [Vec<f64>; 2],
}

impl BalancedReference {
    /// The correction of `state` (its `rho` must be current), with the
    /// arguments of [`compute_pressure_gradient`]: with a `reference`, of
    /// the state's anomaly about it in both forms.
    pub fn new(
        state: &Solution3D,
        mesh: &Mesh2D,
        bathymetry: &Bathymetry2D,
        sigma: &SigmaGrid,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        g: f64,
        rho_0: f64,
        rho_ref: f64,
        min_column_depth: f64,
        metric: MetricForm,
        reference: Option<&ReferenceFields>,
    ) -> Self {
        let n = state.n_elements * ops.n_nodes * sigma.n_levels();
        let force = |form| {
            let (mut fx, mut fy) = (vec![0.0; n], vec![0.0; n]);
            compute_pressure_gradient(
                state,
                mesh,
                bathymetry,
                sigma,
                ops,
                geom,
                g,
                rho_0,
                rho_ref,
                min_column_depth,
                form,
                metric,
                reference,
                &mut fx,
                &mut fy,
                Blocks::All(state.n_elements),
            );
            [fx, fy]
        };
        let [sx, sy] = force(PressureGradientForm::SigmaPairs);
        let [mut zx, mut zy] = force(PressureGradientForm::ConstantDepth);
        for (z, s) in zx.iter_mut().zip(&sx).chain(zy.iter_mut().zip(&sy)) {
            *z -= s;
        }
        Self {
            correction: [zx, zy],
        }
    }

    /// Add the correction to the forces `fx`, `fy`.
    pub fn add_to(&self, fx: &mut [f64], fy: &mut [f64], elements: Blocks) {
        let [cx, cy] = &self.correction;
        let len = cx.len() / elements.n().max(1);
        for k in elements.iter() {
            let block = k * len..(k + 1) * len;
            for (f, c) in fx[block.clone()].iter_mut().zip(&cx[block.clone()]) {
                *f += c;
            }
            for (f, c) in fy[block.clone()].iter_mut().zip(&cy[block]) {
                *f += c;
            }
        }
    }
}

/// Level spacing (m) below which a column is treated as collapsed: its
/// density slope is taken as zero.
const MIN_LEVEL_SPACING: f64 = 1e-9;

/// The most σ-levels a reference stratification's kernels take (their
/// per-level buffers are on the stack).
const MAX_LEVELS: usize = 128;

/// One water column's density anomaly as a piecewise Hermite cubic in `z`,
/// and its hydrostatic pressure (per `g`) at the level centres.
#[derive(Clone, Copy)]
struct ColumnView<'a> {
    /// Level-centre depths, bottom first (ascending).
    z: &'a [f64],
    /// Density anomaly at the levels.
    rho: &'a [f64],
    /// `∂ρ/∂z` at the levels (monotone harmonic mean).
    slope: &'a [f64],
    /// `∫_z^η (ρ − ρ_ref) dz'` at the levels.
    pressure: &'a [f64],
    /// The same as [`Self::pressure_at`] evaluates it at the levels: equal
    /// to `pressure` but for rounding. The pressure differences take this,
    /// so that their own and cross-column terms round alike (round-off at
    /// rest stays ≤ 1.5e-14 m/s², not 3e-14).
    at_level: &'a [f64],
    /// Bed elevation.
    bed: f64,
    /// At least the minimum column depth deep.
    wet: bool,
    /// The reference stratification's density and gradient `Γ` at the
    /// levels (zeros without one, where `rho` is not an anomaly).
    reference: &'a [f64],
    stratification: &'a [f64],
}

impl ColumnView<'_> {
    /// `∫_z^η (ρ − ρ_ref) dz'` at depth `z`, and whether `z` is in the water
    /// column (above the bed). `cursor` is a level at or below the interval
    /// that holds `z` (`z[cursor] ≤ z`, or 0); it is moved up to that
    /// interval, so that a caller evaluating at rising depths finds each
    /// interval by stepping, not searching.
    #[inline]
    fn pressure_at(&self, z: f64, cursor: &mut usize) -> (f64, bool) {
        let n = self.z.len();
        let valid = z >= self.bed;
        let (top, bottom) = (n - 1, 0);
        let p = if z >= self.z[top] {
            let dz = z - self.z[top];
            self.pressure[top] - (self.rho[top] * dz + 0.5 * self.slope[top] * dz * dz)
        } else if z <= self.z[bottom] {
            let dz = self.z[bottom] - z;
            self.pressure[bottom] + self.rho[bottom] * dz - 0.5 * self.slope[bottom] * dz * dz
        } else {
            // z[m] ≤ z < z[m + 1]; z < z[top] ends the walk
            while self.z[*cursor + 1] <= z {
                *cursor += 1;
            }
            let m = *cursor;
            let h = self.z[m + 1] - self.z[m];
            let t = (z - self.z[m]) / h;
            self.pressure[m + 1]
                + h * (hermite_integral(1.0, self, m, h) - hermite_integral(t, self, m, h))
        };
        (p, valid)
    }
}

/// `(1/h) ∫_{z_m}^{z_m + t h}` of the Hermite cubic on level interval `m`.
#[inline]
fn hermite_integral(t: f64, column: &ColumnView, m: usize, h: f64) -> f64 {
    hermite_integral_of(t, column.rho, column.slope, m, h)
}

/// [`hermite_integral`] of the profile `rho` with slopes `slope`.
#[inline]
fn hermite_integral_of(t: f64, rho: &[f64], slope: &[f64], m: usize, h: f64) -> f64 {
    let (t2, t3, t4) = (t * t, t * t * t, t * t * t * t);
    let h00 = 0.5 * t4 - t3 + t;
    let h10 = 0.25 * t4 - 2.0 / 3.0 * t3 + 0.5 * t2;
    let h01 = -0.5 * t4 + t3;
    let h11 = 0.25 * t4 - t3 / 3.0;
    rho[m] * h00 + h * slope[m] * h10 + rho[m + 1] * h01 + h * slope[m + 1] * h11
}

/// The pressure difference `Δp_l` (per `g`) of columns `a` and `b` at every
/// level `l` into `out`, in `form`: for
/// [`PressureGradientForm::ConstantDepth`]
/// `½[(p_b(z_a) − p_a(z_a)) + (p_b(z_b) − p_a(z_b))]`, with `z_a`, `z_b` the
/// two columns' own depths of that level, using only the depths both columns
/// reach; for [`PressureGradientForm::SigmaPairs`]
/// `(p_b(z_b) − p_a(z_a)) + ½(ρ_a + ρ_b)(z_b − z_a)`. Zero if either column
/// is thin (a dry bank exerts no pressure on the water beside it).
///
/// A column's pressure at its own level is its table value, and the depths
/// at which each column is evaluated in the other rise with the level, so
/// one upward walk per column finds every interval.
///
/// `displacement` is the mean displacement `κ` of the two columns' element
/// about its reference stratification (zero without one, and for the two
/// sides of a face): the σ-pairs difference then leaves out the trapezoid
/// rule's error of that displacement's anomaly along the pair,
/// `κ [(ρ̄_b − ρ̄_a) + ½(Γ_a + Γ_b)(z_b − z_a)]` (see "About a reference").
fn pressure_differences(
    a: &ColumnView,
    b: &ColumnView,
    form: PressureGradientForm,
    displacement: f64,
    out: &mut [f64],
) {
    if !(a.wet && b.wet) {
        out.fill(0.0);
        return;
    }
    if form == PressureGradientForm::SigmaPairs {
        for (l, dp) in out.iter_mut().enumerate() {
            *dp = (b.at_level[l] - a.at_level[l]) + 0.5 * (a.rho[l] + b.rho[l]) * (b.z[l] - a.z[l]);
        }
        if displacement != 0.0 {
            for (l, dp) in out.iter_mut().enumerate() {
                *dp -= displacement
                    * ((b.reference[l] - a.reference[l])
                        + 0.5 * (a.stratification[l] + b.stratification[l]) * (b.z[l] - a.z[l]));
            }
        }
        return;
    }
    let (mut cursor_a, mut cursor_b) = (0, 0);
    for (l, dp) in out.iter_mut().enumerate() {
        let (pb_at_a, b_reaches_a) = b.pressure_at(a.z[l], &mut cursor_b);
        let (pa_at_b, a_reaches_b) = a.pressure_at(b.z[l], &mut cursor_a);
        let (pa, pb) = (a.at_level[l], b.at_level[l]);
        *dp = match (b_reaches_a, a_reaches_b) {
            (true, true) => 0.5 * ((pb_at_a - pa) + (pb - pa_at_b)),
            (true, false) => pb_at_a - pa,
            (false, true) => pb - pa_at_b,
            // Impossible for columns with z ≥ bed at their own nodes (one of
            // the two depths is the shallower one, which both reach); keep
            // the symmetric form for degenerate input.
            (false, false) => 0.5 * ((pb_at_a - pa) + (pb - pa_at_b)),
        };
    }
}

/// Per-column profiles of the nodes of one element or face, stored flat as
/// `[node][level]`.
struct Columns {
    n_levels: usize,
    z: Vec<f64>,
    rho: Vec<f64>,
    slope: Vec<f64>,
    pressure: Vec<f64>,
    at_level: Vec<f64>,
    bed: Vec<f64>,
    wet: Vec<bool>,
    min_column_depth: f64,
    /// The reference's density and gradient at the levels (zeros without)
    reference: Vec<f64>,
    stratification: Vec<f64>,
}

impl Columns {
    fn new(n_columns: usize, n_levels: usize, min_column_depth: f64) -> Self {
        let n = n_columns * n_levels;
        Self {
            min_column_depth,
            wet: vec![true; n_columns],
            n_levels,
            z: vec![0.0; n],
            rho: vec![0.0; n],
            slope: vec![0.0; n],
            pressure: vec![0.0; n],
            at_level: vec![0.0; n],
            bed: vec![0.0; n_columns],
            reference: vec![0.0; n],
            stratification: vec![0.0; n],
        }
    }

    fn view(&self, c: usize) -> ColumnView<'_> {
        let range = c * self.n_levels..(c + 1) * self.n_levels;
        ColumnView {
            z: &self.z[range.clone()],
            rho: &self.rho[range.clone()],
            slope: &self.slope[range.clone()],
            pressure: &self.pressure[range.clone()],
            at_level: &self.at_level[range.clone()],
            bed: self.bed[c],
            wet: self.wet[c],
            reference: &self.reference[range.clone()],
            stratification: &self.stratification[range],
        }
    }

    /// Fill column `c` from `ρ − rho_ref` of the state's column at (`el`,
    /// `node`), or with a `reference` from the density anomaly about it.
    fn fill(
        &mut self,
        c: usize,
        state: &Solution3D,
        bathymetry: &Bathymetry2D,
        sigma: &SigmaGrid,
        el: ElementIndex,
        node: usize,
        rho_ref: f64,
        reference: Option<&ReferenceFields>,
    ) {
        let nl = self.n_levels;
        let range = c * nl..(c + 1) * nl;
        let eta = state.eta.get(el.as_usize(), node);
        let bed = bathymetry.get(el, node);
        let depth = eta - bed;
        self.bed[c] = bed;
        self.wet[c] = depth >= self.min_column_depth;
        let z = &mut self.z[range.clone()];
        let rho = &mut self.rho[range.clone()];
        let slope = &mut self.slope[range.clone()];
        let pressure = &mut self.pressure[range.clone()];
        let at_level = &mut self.at_level[range.clone()];
        for (zl, &s) in z.iter_mut().zip(sigma.sigma_rho()) {
            *zl = eta + s * depth;
        }
        let column = state.rho_column(el, node);
        let Some(fields) = reference else {
            for (r, &c) in rho.iter_mut().zip(column) {
                *r = c - rho_ref;
            }
            monotone_slopes(z, rho, slope);
            // ∫_z^η from the surface down: the top half-layer with the
            // linear continuation, then exact integrals of the cubic
            // between levels
            let dz = eta - z[nl - 1];
            pressure[nl - 1] = rho[nl - 1] * dz + 0.5 * slope[nl - 1] * dz * dz;
            for l in (0..nl - 1).rev() {
                let h = z[l + 1] - z[l];
                pressure[l] = pressure[l + 1]
                    + h * (0.5 * (rho[l] + rho[l + 1]) + h * (slope[l] - slope[l + 1]) / 12.0);
            }
            // `pressure_at(z[l])`: the end levels exactly, the others from
            // the interval above them at t = 0
            for (l, p) in at_level.iter_mut().enumerate() {
                *p = if l == 0 || l == nl - 1 {
                    pressure[l]
                } else {
                    let h = z[l + 1] - z[l];
                    pressure[l + 1]
                        + h * (hermite_integral_of(1.0, rho, slope, l, h)
                            - hermite_integral_of(0.0, rho, slope, l, h))
                };
            }
            return;
        };

        // The anomaly about the reference at the column's own levels, its
        // pressure by the trapezoid rule (no slopes: linear in the anomaly,
        // so that the force has an adjoint), less the rule's error of the
        // element's mean displacement κ (see "About a reference")
        let start = (el.as_usize() * state.n_nodes + node) * nl;
        let density = &fields.density[start..start + nl];
        let gamma = &fields.stratification[start..start + nl];
        self.reference[range.clone()].copy_from_slice(density);
        self.stratification[range].copy_from_slice(gamma);
        let displacement = fields.displacement[el.as_usize()];
        for ((r, &c), &b) in rho.iter_mut().zip(column).zip(density) {
            *r = c - b;
        }
        slope.fill(0.0);
        pressure[nl - 1] = rho[nl - 1] * (eta - z[nl - 1]);
        for l in (0..nl - 1).rev() {
            let h = z[l + 1] - z[l];
            let trapezoid_error =
                (density[l + 1] - density[l]) + 0.5 * (gamma[l] + gamma[l + 1]) * h;
            pressure[l] =
                pressure[l + 1] + h * 0.5 * (rho[l] + rho[l + 1]) - displacement * trapezoid_error;
        }
        at_level.copy_from_slice(pressure);
    }
}

/// Monotone slopes `∂ρ/∂z` at the levels `z` of the profile `rho`: the
/// harmonic mean of the neighbouring secants, zero at extrema; one-sided at
/// the ends. Exact for linear `ρ(z)`.
fn monotone_slopes(z: &[f64], rho: &[f64], slope: &mut [f64]) {
    let nl = z.len();
    let secant = |l: usize| {
        let h = z[l + 1] - z[l];
        if h > MIN_LEVEL_SPACING {
            (rho[l + 1] - rho[l]) / h
        } else {
            0.0
        }
    };
    if nl == 1 {
        slope[0] = 0.0;
    } else {
        slope[0] = secant(0);
        slope[nl - 1] = secant(nl - 2);
        for (l, s) in slope.iter_mut().enumerate().take(nl - 1).skip(1) {
            let (a, b) = (secant(l - 1), secant(l));
            *s = if a * b > 0.0 {
                2.0 * a * b / (a + b)
            } else {
                0.0
            };
        }
    }
}

/// Compute the σ-coordinate pressure gradient force at constant depth (see
/// the module docs).
///
/// Output is stored in `grad_px` and `grad_py` with layout `[element][node][level]`.
/// The result is the force per unit mass: -1/ρ₀ ∇p.
///
/// The density used in the hydrostatic pressure and its gradient is `ρ − rho_ref`.
/// This lets the caller select which part of the PGF is computed:
///
/// - `rho_ref = 0` → **full** PGF (barotropic + baroclinic). The barotropic part
///   is exactly the free-surface term `−g∇η`.
/// - `rho_ref = ρ₀` → **baroclinic-only** PGF. The ρ₀ contribution integrates to
///   exactly `−g∇η`, so subtracting ρ₀ removes the barotropic term. Under mode
///   splitting the barotropic `−g∇η` is supplied by the 2D sub-model's `½gh²`
///   flux, so the 3D internal mode must use the baroclinic-only PGF to avoid
///   double-counting the surface pressure gradient (see `ModeSplitIntegrator`).
///
/// # Arguments
/// * `state` - 3D solution state (contains η and ρ)
/// * `mesh` - Mesh (face neighbours, for the lifted pressure jumps)
/// * `bathymetry` - Bed elevation B
/// * `sigma` - Vertical grid configuration
/// * `ops` - 2D DG operators (for gradients)
/// * `geom` - Geometric factors (per node: general quadrilaterals)
/// * `g` - Gravitational acceleration (m/s²)
/// * `rho_0` - Reference density ρ₀ used in the `-1/ρ₀` normalization (kg/m³)
/// * `rho_ref` - Density subtracted from ρ before integrating (0 = full PGF, ρ₀ = baroclinic-only)
/// * `min_column_depth` - Columns shallower than this (m) exert and feel no
///   pressure difference (3D wetting and drying)
/// * `form` - How two columns' pressures are differenced (see the module docs)
/// * `metric` - The 3D advection's volume form, whose adjoint the volume term
///   is on general quadrilaterals (see "Curvilinear elements")
/// * `grad_px` - Output x-component of PGF (m/s²)
/// * `grad_py` - Output y-component of PGF (m/s²)
pub fn compute_pressure_gradient(
    state: &Solution3D,
    mesh: &Mesh2D,
    bathymetry: &Bathymetry2D,
    sigma: &SigmaGrid,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    g: f64,
    rho_0: f64,
    rho_ref: f64,
    min_column_depth: f64,
    form: PressureGradientForm,
    metric: MetricForm,
    reference: Option<&ReferenceFields>,
    grad_px: &mut [f64],
    grad_py: &mut [f64],
    elements: Blocks,
) {
    let (nn, nfn) = (ops.n_nodes, ops.n_face_nodes);
    let nl = sigma.n_levels();
    let scale = -g / rho_0;

    let n = state.n_elements * nn * nl;
    for_each_block(
        elements,
        [&mut grad_px[..n], &mut grad_py[..n]],
        || {
            let mut scratch = Pooled::take(
                |s: &PressureScratch| s.fits(nn, nfn, nl),
                || PressureScratch::new(nn, nfn, nl),
            );
            scratch.own.min_column_depth = min_column_depth;
            scratch.across.min_column_depth = min_column_depth;
            scratch
        },
        |scratch, k, [out_x, out_y]| {
            let PressureScratch {
                own,
                across,
                px,
                py,
                dp,
            } = &mut **scratch;
            pressure_gradient_element(
                k, state, mesh, bathymetry, sigma, ops, geom, rho_ref, reference, form, metric,
                own, across, px, py, dp,
            );
            for ((fx, fy), (&dx, &dy)) in out_x.iter_mut().zip(out_y).zip(px.iter().zip(&*py)) {
                *fx = scale * dx;
                *fy = scale * dy;
            }
        },
    );
}

/// Buffers of one element of [`compute_pressure_gradient`].
struct PressureScratch {
    own: Columns,
    across: Columns,
    /// ∂p/∂x, ∂p/∂y per g, [node][level]
    px: Vec<f64>,
    py: Vec<f64>,
    /// Δp of one pair of columns, [level]
    dp: Vec<f64>,
}

impl PressureScratch {
    fn new(nn: usize, nfn: usize, nl: usize) -> Self {
        Self {
            own: Columns::new(nn, nl, 0.0),
            across: Columns::new(nfn, nl, 0.0),
            px: vec![0.0; nn * nl],
            py: vec![0.0; nn * nl],
            dp: vec![0.0; nl],
        }
    }

    fn fits(&self, nn: usize, nfn: usize, nl: usize) -> bool {
        self.own.n_levels == nl && self.own.bed.len() == nn && self.across.bed.len() == nfn
    }
}

/// `∂p/∂x`, `∂p/∂y` per `g` of element `k` into `px`, `py` (`[node][level]`),
/// with the columns of its nodes in `own` and of a face's neighbours in
/// `across`, and one pair's pressure differences in `dp` (see
/// [`compute_pressure_gradient`]).
fn pressure_gradient_element(
    k: usize,
    state: &Solution3D,
    mesh: &Mesh2D,
    bathymetry: &Bathymetry2D,
    sigma: &SigmaGrid,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    rho_ref: f64,
    reference: Option<&ReferenceFields>,
    form: PressureGradientForm,
    metric: MetricForm,
    own: &mut Columns,
    across: &mut Columns,
    px: &mut [f64],
    py: &mut [f64],
    dp: &mut [f64],
) {
    let nn = ops.n_nodes;
    let el = ElementIndex::new(k);
    for i in 0..nn {
        own.fill(i, state, bathymetry, sigma, el, i, rho_ref, reference);
    }
    px.fill(0.0);
    py.fill(0.0);

    // An element with a thin column moves its layers on the GLL subcells
    // with their barotropic shares only (`LayerTransport`): no baroclinic
    // exchange within it, so no force within it either
    if own.wet[..nn].iter().all(|&w| w) {
        // The element's mean displacement about its reference, if any
        let displacement = reference.map_or(0.0, |fields| fields.displacement[k]);
        dg_volume_term(k, ops, geom, form, metric, displacement, own, px, py, dp);
    }

    // Face terms: LIFT (n·(p* − p⁻)), p* − p⁻ = ½ Δp at a common depth
    let (nfn, nl) = (ops.n_face_nodes, own.n_levels);
    for f in 0..4 {
        let Some(nb) = mesh.neighbor(el, f) else {
            continue;
        };
        let nb_el = ElementIndex::new(nb.element);
        for fi in 0..nfn {
            let nb_node = ops.face_nodes[nb.face][nfn - 1 - fi];
            across.fill(
                fi, state, bathymetry, sigma, nb_el, nb_node, rho_ref, reference,
            );
        }
        for (fi, &node) in ops.face_nodes[f].iter().enumerate() {
            // The two sides of a face are at one height: no displacement
            // term (their pressures hold their own elements')
            pressure_differences(&own.view(node), &across.view(fi), form, 0.0, dp);
            let normal = geom.normal(k, f, fi);
            let lift_scale = geom.lift_scale(k, f, fi, node);
            for (l, &dp) in dp.iter().enumerate() {
                let jump = 0.5 * dp;
                if jump == 0.0 {
                    continue;
                }
                let (jx, jy) = (lift_scale * normal.0 * jump, lift_scale * normal.1 * jump);
                for i in 0..nn {
                    let lift = ops.lift[f][(i, fi)];
                    px[i * nl + l] += lift * jx;
                    py[i * nl + l] += lift * jy;
                }
            }
        }
    }
}

/// The derivative coefficients of the pair of nodes `(i, j)` of element `k`:
/// `(Dx, Dy)_ij` (the force at `i`) and `(Dx, Dy)_ji` (at `j`), with the
/// contravariant vectors the pair is differenced with at each node, each
/// node's own or both their mean (`averaged`, see "Curvilinear elements").
/// The pressure gradient's volume term and its adjoint
/// ([`add_reference_advection`]) take the same.
#[inline]
fn pair_derivatives(
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    k: usize,
    i: usize,
    j: usize,
    averaged: bool,
) -> [(f64, f64); 2] {
    let (ar_i, as_i) = geom.contravariant(k, i);
    let j_inv_i = geom.jacobian_inv(k, i);
    let (ar_j, as_j) = geom.contravariant(k, j);
    let j_inv_j = geom.jacobian_inv(k, j);
    let ((ar_ij, as_ij), (ar_ji, as_ji)) = if averaged {
        let mean = |a: (f64, f64), b: (f64, f64)| (0.5 * (a.0 + b.0), 0.5 * (a.1 + b.1));
        let m = (mean(ar_i, ar_j), mean(as_i, as_j));
        (m, m)
    } else {
        ((ar_i, as_i), (ar_j, as_j))
    };
    let (dr_ij, ds_ij) = (ops.dr[(i, j)], ops.ds[(i, j)]);
    let (dr_ji, ds_ji) = (ops.dr[(j, i)], ops.ds[(j, i)]);
    [
        (
            j_inv_i * (ar_ij.0 * dr_ij + as_ij.0 * ds_ij),
            j_inv_i * (ar_ij.1 * dr_ij + as_ij.1 * ds_ij),
        ),
        (
            j_inv_j * (ar_ji.0 * dr_ji + as_ji.0 * ds_ji),
            j_inv_j * (ar_ji.1 * dr_ji + as_ji.1 * ds_ji),
        ),
    ]
}

/// The DG volume term of `pressure_gradient_element`: `Σ_j Dx_ij Δp_ij`,
/// with `Δp_ji = −Δp_ij`, and the metric of the pair in the advection's form
/// (see "Curvilinear elements"); `displacement` is the element's (see
/// [`pressure_differences`]).
#[allow(clippy::too_many_arguments)]
fn dg_volume_term(
    k: usize,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    form: PressureGradientForm,
    metric: MetricForm,
    displacement: f64,
    own: &Columns,
    px: &mut [f64],
    py: &mut [f64],
    dp: &mut [f64],
) {
    let (nn, nl) = (ops.n_nodes, own.n_levels);
    let averaged = metric.averaged_on(geom, k);
    for i in 0..nn {
        let a = own.view(i);
        for j in i + 1..nn {
            let b = own.view(j);
            let [(dx_ij, dy_ij), (dx_ji, dy_ji)] = pair_derivatives(ops, geom, k, i, j, averaged);
            pressure_differences(&a, &b, form, displacement, dp);
            for (l, &dp) in dp.iter().enumerate() {
                px[i * nl + l] += dx_ij * dp;
                py[i * nl + l] += dy_ij * dp;
                px[j * nl + l] -= dx_ji * dp;
                py[j * nl + l] -= dy_ji * dp;
            }
        }
    }
}

/// The reference stratification at a stage's levels (see "About a
/// reference"): what the pressure gradient ([`compute_pressure_gradient`])
/// and the tracers' advection ([`add_reference_advection`], and the vertical
/// reconstruction about it, [`crate::solver::rhs::AdvectedTracer::reference`])
/// read of it. All `[element][node][level]` but the displacements.
#[derive(Clone, Debug, Default)]
pub struct ReferenceFields {
    /// `T̄` and `S̄` at the layer centres.
    pub tracers: [Vec<f64>; 2],
    /// Their gradients over each layer,
    /// `γ = (C̄(z_{l+½}) − C̄(z_{l−½})) / H_z` between the σ-surfaces that
    /// bound it (zero in a column without water).
    pub gradients: [Vec<f64>; 2],
    /// `ρ̄`, the density of `(T̄, S̄)`, at the layer centres.
    pub density: Vec<f64>,
    /// The stratification `Γ = −(ρ_T γ^T + ρ_S γ^S)` of each layer, with the
    /// equation of state's partials at the layer centre's `(T̄, S̄)`
    /// (positive where the reference is statically stable).
    pub stratification: Vec<f64>,
    /// The mean displacement `κ` (m) of every element's stratification
    /// about the reference, `[element]` ([`Self::set_displacements`]).
    pub displacement: Vec<f64>,
    /// The free surface the fields of each column were last filled at
    /// (`[element][node]`, NaN before the first): [`Self::fill`] leaves an
    /// element whose columns have not moved as it is.
    filled_at: Vec<f64>,
}

impl ReferenceFields {
    /// `profile` at the levels of the columns of `elements` at `state`'s
    /// `η`, with the density `density(T, S)` and its partials
    /// `partials(T, S) = (∂ρ/∂T, ∂ρ/∂S)` (the state's equation of state).
    /// A layer's gradient is the reference's mean over the layer, so that
    /// the gradients of a column add up to the reference's change over it
    /// whatever the levels resolve.
    ///
    /// The fields are functions of the columns' `η` alone, so an element
    /// whose `η` is that of its last fill is left as it is (a stage's
    /// pressure gradient and its transport take the same levels, and a fluid
    /// at rest never moves them): the same profile, equation of state, bed
    /// and levels must be given every time (a new [`Self::default`] for
    /// others).
    #[allow(clippy::too_many_arguments)]
    pub fn fill(
        &mut self,
        profile: &TracerReferenceProfile,
        density: impl Fn(f64, f64) -> f64 + Sync,
        partials: impl Fn(f64, f64) -> (f64, f64) + Sync,
        state: &Solution3D,
        bathymetry: &Bathymetry2D,
        sigma: &SigmaGrid,
        elements: Blocks,
    ) {
        let (nn, nl) = (state.n_nodes, sigma.n_levels());
        let n = state.n_elements * nn * nl;
        let Self {
            tracers: [temp, salt],
            gradients: [temp_gradient, salt_gradient],
            density: rho,
            stratification,
            displacement,
            filled_at,
        } = self;
        for field in [
            &mut *temp,
            &mut *salt,
            &mut *temp_gradient,
            &mut *salt_gradient,
            &mut *rho,
            &mut *stratification,
        ] {
            field.resize(n, 0.0);
        }
        displacement.resize(state.n_elements, 0.0);
        filled_at.resize(state.n_elements * nn, f64::NAN);
        let (sigma_rho, sigma_w) = (sigma.sigma_rho(), sigma.sigma_w());
        // 1/Δσ of every layer: a layer's thickness is Δσ times the depth
        assert!(nl <= MAX_LEVELS, "at most {MAX_LEVELS} levels, not {nl}");
        let mut inverse_d_sigma = [0.0; MAX_LEVELS];
        for (inverse, w) in inverse_d_sigma.iter_mut().zip(sigma_w.windows(2)) {
            *inverse = 1.0 / (w[1] - w[0]);
        }
        for_each_block(
            elements,
            [
                &mut temp[..],
                &mut salt[..],
                &mut temp_gradient[..],
                &mut salt_gradient[..],
                &mut rho[..],
                &mut stratification[..],
                &mut filled_at[..],
            ],
            || (),
            |_,
             k,
             [
                temp,
                salt,
                temp_gradient,
                salt_gradient,
                rho,
                stratification,
                filled_at,
            ]| {
                let eta_k = &state.eta.data[k * nn..(k + 1) * nn];
                if filled_at.iter().zip(eta_k).all(|(a, b)| a == b) {
                    return;
                }
                filled_at.copy_from_slice(eta_k);
                for (i, &eta) in eta_k.iter().enumerate() {
                    let depth = eta - bathymetry.data[k * nn + i];
                    // No gradient in a column without water
                    let inverse_depth = if depth > MIN_LEVEL_SPACING {
                        1.0 / depth
                    } else {
                        0.0
                    };
                    let mut below = profile.at(eta + sigma_w[0] * depth);
                    for l in 0..nl {
                        let c = i * nl + l;
                        let (t, s) = profile.at(eta + sigma_rho[l] * depth);
                        (temp[c], salt[c]) = (t, s);
                        rho[c] = density(t, s);
                        let above = profile.at(eta + sigma_w[l + 1] * depth);
                        let inverse = inverse_d_sigma[l] * inverse_depth;
                        let (gt, gs) =
                            ((above.0 - below.0) * inverse, (above.1 - below.1) * inverse);
                        (temp_gradient[c], salt_gradient[c]) = (gt, gs);
                        let (rho_t, rho_s) = partials(t, s);
                        stratification[c] = -(rho_t * gt + rho_s * gs);
                        below = above;
                    }
                }
            },
        );
    }

    /// Set the mean displacement of the elements of `elements` from
    /// `state`'s density (after [`Self::fill`] on them): the projection of
    /// the anomaly `ρ′ = ρ − ρ̄` on a uniform displacement of the reference
    /// (whose anomaly is `κ Γ`) in the available potential energy's norm,
    ///
    /// ```text
    ///     κ = Σ m H_z ρ′ / Σ m H_z Γ
    /// ```
    ///
    /// over the element's layers that are stably stratified (`Γ > 0`) in
    /// columns at least `min_column_depth` deep; zero where there are none.
    pub fn set_displacements(
        &mut self,
        state: &Solution3D,
        geom: &GeometricFactors2D,
        bathymetry: &Bathymetry2D,
        sigma: &SigmaGrid,
        min_column_depth: f64,
        elements: Blocks,
    ) {
        let (nn, nl) = (state.n_nodes, sigma.n_levels());
        let (density, stratification) = (&self.density, &self.stratification);
        let d_sigma = sigma.d_sigma();
        for_each_block(
            elements,
            [&mut self.displacement[..state.n_elements]],
            || (),
            |_, k, [displacement]| {
                let (mut anomaly, mut weight) = (0.0, 0.0);
                for i in 0..nn {
                    let idx = k * nn + i;
                    let depth = state.eta.data[idx] - bathymetry.data[idx];
                    if depth < min_column_depth {
                        continue;
                    }
                    let mass = geom.node_mass(k, i) * depth;
                    for l in 0..nl {
                        let gamma = stratification[idx * nl + l];
                        if gamma > 0.0 {
                            let volume = mass * d_sigma[l];
                            anomaly += volume * (state.rho[idx * nl + l] - density[idx * nl + l]);
                            weight += volume * gamma;
                        }
                    }
                }
                displacement[0] = if weight > 0.0 { anomaly / weight } else { 0.0 };
            },
        );
    }

    /// A bound (1/s) on the frequency of the reference's internal waves
    /// along the σ-levels of element `k` (after [`Self::fill`] on it): of
    /// the oscillation in which the flow along a pair of nodes on a sloping
    /// level displaces the stratification, [`add_reference_advection`], and
    /// the anomaly's pressure gradient pushes back. `buoyancy` is `g/ρ₀`;
    /// `scratch` holds two numbers per node. Zero in an element with a
    /// column thinner than `min_column_depth`, where the pairs do not act.
    ///
    /// In the pseudo-energy's coordinates, `r = ρ′ √(g m H_z/Γ)` and
    /// `v = u √(ρ₀ m H_z)`, the pairs' part of the operator linearised about
    /// rest is `[[0, −Bᵀ], [B, 0]]` on each level, with
    ///
    /// ```text
    ///     B[i; i]  =  ½ N_i Σ_j (z_j − z_i) D_ij              (½ N ∇_σ z)
    ///     B[i; j]  = −½ N_i (z_j − z_i) √(m_j D_j / m_i D_i) D_ji
    /// ```
    ///
    /// (`N² = (g/ρ₀) Γ`, `D_ij` the pair's derivative coefficients, `D_i`
    /// the depth): frequencies `N |∇_σ z|`, the buoyancy frequency times
    /// the slope of the level. Its eigenvalues are `±i` times `B`'s singular
    /// values, at most `√(‖B‖₁ ‖B‖∞)`: the row and column sums here, with
    /// `z_j − z_i = (1 + σ)(η_j − η_i) − σ (B_j − B_i)` bounded over the
    /// levels by each column's largest `N` and largest `N |σ|`. The
    /// element's mean displacement is an orthogonal projection in these
    /// coordinates and does not raise the bound.
    #[allow(clippy::too_many_arguments)]
    pub fn sloping_level_rate(
        &self,
        state: &Solution3D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        bathymetry: &Bathymetry2D,
        sigma: &SigmaGrid,
        buoyancy: f64,
        min_column_depth: f64,
        metric: MetricForm,
        k: usize,
        scratch: &mut [f64],
    ) -> f64 {
        let (nn, nl) = (ops.n_nodes, sigma.n_levels());
        let eta = |i: usize| state.eta.data[k * nn + i];
        let bed = |i: usize| bathymetry.data[k * nn + i];
        if (0..nn).any(|i| eta(i) - bed(i) < min_column_depth) {
            return 0.0;
        }
        // Each column's largest N and largest N |σ|
        let (largest, deepest) = scratch[..2 * nn].split_at_mut(nn);
        for i in 0..nn {
            let column = &self.stratification[(k * nn + i) * nl..][..nl];
            let (mut n2, mut n2_sigma2) = (0.0_f64, 0.0_f64);
            for (&gamma, &s) in column.iter().zip(sigma.sigma_rho()) {
                let gamma = gamma.max(0.0);
                n2 = n2.max(gamma);
                n2_sigma2 = n2_sigma2.max(gamma * s * s);
            }
            largest[i] = (buoyancy * n2).sqrt();
            deepest[i] = (buoyancy * n2_sigma2).sqrt();
        }
        if largest.iter().all(|&n| n == 0.0) {
            return 0.0;
        }
        let averaged = metric.averaged_on(geom, k);
        let weight = |i: usize| geom.node_mass(k, i) * (eta(i) - bed(i));
        let (mut rows, mut columns) = (0.0_f64, 0.0_f64);
        for i in 0..nn {
            // ∇η and ∇B at the node, and the sums over the other nodes of
            // its row and of its velocity's two columns
            let (mut grad_eta, mut grad_bed) = ((0.0, 0.0), (0.0, 0.0));
            let (mut row, mut column) = (0.0, (0.0, 0.0));
            for j in (0..nn).filter(|&j| j != i) {
                let [(dx_ij, dy_ij), (dx_ji, dy_ji)] =
                    pair_derivatives(ops, geom, k, i, j, averaged);
                let (d_eta, d_bed) = (eta(j) - eta(i), bed(j) - bed(i));
                grad_eta = (grad_eta.0 + dx_ij * d_eta, grad_eta.1 + dy_ij * d_eta);
                grad_bed = (grad_bed.0 + dx_ij * d_bed, grad_bed.1 + dy_ij * d_bed);
                let ratio = (weight(j) / weight(i)).sqrt();
                row += (largest[i] * d_eta.abs() + deepest[i] * d_bed.abs())
                    * ratio
                    * (dx_ji.abs() + dy_ji.abs());
                let from_j = (largest[j] * d_eta.abs() + deepest[j] * d_bed.abs()) / ratio;
                column = (
                    column.0 + from_j * dx_ij.abs(),
                    column.1 + from_j * dy_ij.abs(),
                );
            }
            let own_x = largest[i] * grad_eta.0.abs() + deepest[i] * grad_bed.0.abs();
            let own_y = largest[i] * grad_eta.1.abs() + deepest[i] * grad_bed.1.abs();
            rows = rows.max(0.5 * (own_x + own_y + row));
            columns = columns.max(0.5 * (own_x + column.0).max(own_y + column.1));
        }
        (rows * columns).sqrt()
    }
}

/// Add to the tracers' inventory tendencies `rhs` (`[T, S]`, `∂(H_z C)/∂t`,
/// layout of [`Solution3D`]) what turns the conservative advection of the
/// reference stratification into its advection by the vertical velocity
/// that is the pressure gradient's adjoint (see "About a reference").
///
/// The edges are those the pressure gradient acts along: the pairs of nodes
/// of an element on a level (where no column of the element is thin and its
/// layers move by the pairs, not on the subcells), and the σ-surfaces between
/// the layers of a column that is not thin. Each carries a volume flux
/// `f_ab` out of node `a` (per unit horizontal area of `a`): the split
/// form's `Φ_ij/m_i`, or `±Ω`. The conservative advection changes `C_a` by
/// `−½ f_ab (C_b − C_a)` along it. Of that jump the reference's share is
/// `θ_ab (C̄_b − C̄_a)` (`θ` is 1 in a fluid stratified like the reference),
/// and this adds
///
/// ```text
///     ½ f_ab θ_ab [(C̄_b − C̄_a) − γ_a (z_b − z_a)]
/// ```
///
/// so that the reference's share is advected as `−γ_a w_a`, with `γ_a` the
/// reference's gradient over the layer ([`ReferenceFields::gradients`]) and
/// `H_z,a w_a = ½ Σ_b f_ab (z_b − z_a)`, the adjoint of the pressure
/// gradient's sum over the same edges. The other edges (a subcell element's
/// interfaces, which carry the layers' shares of the barotropic flux, the
/// element faces, a thin column) are left to the conservative advection.
///
/// The tracer these terms would make in an element (the trapezoid rule's
/// error of the reference's gradient along its edges) is taken back as a
/// uniform displacement of the element's stratification, `−H_z Γ R/N` at
/// every layer with `R` the element's sum of the terms and
/// `N = Σ m H_z Γ` over its stably stratified layers: every element keeps
/// its inventory of `T` and of `S` to round-off (unless it has no such
/// layer). The pressure gradient's displacement term
/// ([`ReferenceFields::set_displacements`]) is this step's adjoint.
///
/// `tracers` are the advected concentrations, `reference` the reference at
/// the levels of `state`'s `η`, `transport` the stage's layer transports.
/// A uniform tracer gets nothing (`θ = 0`); a reference linear in `z` adds
/// nothing (its chords are its tangents).
#[allow(clippy::too_many_arguments)]
pub fn add_reference_advection(
    rhs: [&mut [f64]; 2],
    tracers: [&[f64]; 2],
    reference: &ReferenceFields,
    state: &Solution3D,
    transport: &LayerTransport,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    sigma: &SigmaGrid,
    bathymetry: &Bathymetry2D,
    min_column_depth: f64,
    elements: Blocks,
) {
    let (nn, nl) = (ops.n_nodes, sigma.n_levels());
    assert!(nl <= MAX_LEVELS, "at most {MAX_LEVELS} levels, not {nl}");
    let n = state.n_elements * nn * nl;
    let (sigma_rho, d_sigma) = (sigma.sigma_rho(), sigma.d_sigma());
    let metric = transport.metric;
    let [rhs_t, rhs_s] = rhs;
    for_each_block(
        elements,
        [&mut rhs_t[..n], &mut rhs_s[..n]],
        || (),
        |_, k, [rhs_t, rhs_s]| {
            let mut out = [rhs_t, rhs_s];
            let block = k * nn * nl..(k + 1) * nn * nl;
            let fields = [0, 1].map(|t| EdgeFields {
                reference: &reference.tracers[t][block.clone()],
                gradient: &reference.gradients[t][block.clone()],
                tracer: &tracers[t][block.clone()],
            });
            let (hu, hv) = (&transport.hu[block.clone()], &transport.hv[block.clone()]);
            let eta = |i: usize| state.eta.data[k * nn + i];
            let depth = |i: usize| state.eta.data[k * nn + i] - bathymetry.data[k * nn + i];
            let wet = |i: usize| depth(i) >= min_column_depth;
            // What the edges' terms make of each tracer in the element, per
            // level (m² · m C/s)
            let mut made = [[0.0; MAX_LEVELS]; 2];
            // An edge's half flux (½ F, out of its first end) and height
            // difference on every level
            let (mut half, mut dz) = ([0.0; MAX_LEVELS], [0.0; MAX_LEVELS]);

            // The pairs of the split form, with the pressure gradient's own
            // coefficients, where it has its volume term: in an element
            // without a thin column that moves its layers by the pairs
            if !transport.subcell_elements[k] && (0..nn).all(wet) {
                let averaged = metric.averaged_on(geom, k);
                // Only nodes on one GLL line are a pair
                let apart = |a: usize, b: usize| ops.dr[(a, b)] == 0.0 && ops.ds[(a, b)] == 0.0;
                for i in 0..nn {
                    let m_i = geom.node_mass(k, i);
                    for j in (i + 1..nn).filter(|&j| !(apart(i, j) && apart(j, i))) {
                        let [(dx_ij, dy_ij), (dx_ji, dy_ji)] =
                            pair_derivatives(ops, geom, k, i, j, averaged);
                        let m_j = geom.node_mass(k, j);
                        let (ci, cj) = (i * nl..(i + 1) * nl, j * nl..(j + 1) * nl);
                        // The pair's volume flux from i to j on every level,
                        // m_i Dx_ij·Q_i − m_j Dx_ji·Q_j, and z_j − z_i,
                        // (η_j − η_i) + σ_l (D_j − D_i)
                        let (ax, ay) = (m_i * dx_ij, m_i * dy_ij);
                        let (bx, by) = (m_j * dx_ji, m_j * dy_ji);
                        let (d_eta, d_depth) = (eta(j) - eta(i), depth(j) - depth(i));
                        let (hu_i, hv_i) = (&hu[ci.clone()], &hv[ci.clone()]);
                        let (hu_j, hv_j) = (&hu[cj.clone()], &hv[cj.clone()]);
                        for l in 0..nl {
                            half[l] = 0.5
                                * ((ax * hu_i[l] + ay * hv_i[l]) - (bx * hu_j[l] + by * hv_j[l]));
                            dz[l] = d_eta + sigma_rho[l] * d_depth;
                        }
                        for (t, field) in fields.iter().enumerate() {
                            // i < j: the two ends' rows are disjoint
                            let (low, high) = out[t].split_at_mut(cj.start);
                            edges_over_levels(
                                &half[..nl],
                                &dz[..nl],
                                field.rows(ci.clone()),
                                field.rows(cj.clone()),
                                (1.0 / m_i, 1.0 / m_j),
                                &mut low[ci.clone()],
                                &mut high[..nl],
                                &mut made[t][..nl],
                            );
                        }
                    }
                }
            }
            // The σ-surfaces of every column that is not thin: Ω up through
            // surface l, between the layers l − 1 and l (per unit area: the
            // node's mass in `made`)
            let (mut lower, mut upper) = ([0.0; MAX_LEVELS], [0.0; MAX_LEVELS]);
            let mut made_column = [0.0; MAX_LEVELS];
            let surfaces = nl - 1;
            for i in (0..nn).filter(|&i| wet(i)) {
                let (depth, mass) = (depth(i), geom.node_mass(k, i));
                let omega = &transport.omega[(k * nn + i) * (nl + 1)..][..nl + 1];
                for l in 1..nl {
                    half[l - 1] = 0.5 * omega[l];
                    dz[l - 1] = (sigma_rho[l] - sigma_rho[l - 1]) * depth;
                }
                let column = i * nl..(i + 1) * nl;
                for (t, field) in fields.iter().enumerate() {
                    lower[..surfaces].fill(0.0);
                    upper[..surfaces].fill(0.0);
                    made_column[..surfaces].fill(0.0);
                    edges_over_levels(
                        &half[..surfaces],
                        &dz[..surfaces],
                        field.rows(column.start..column.end - 1),
                        field.rows(column.start + 1..column.end),
                        (1.0, 1.0),
                        &mut lower[..surfaces],
                        &mut upper[..surfaces],
                        &mut made_column[..surfaces],
                    );
                    let out = &mut out[t][column.clone()];
                    for l in 0..surfaces {
                        out[l] += lower[l];
                        out[l + 1] += upper[l];
                        made[t][l] += mass * made_column[l];
                    }
                }
            }
            // Taken back as a uniform displacement of the element's stable
            // stratification: the element keeps its inventories
            let made = made.map(|levels| levels[..nl].iter().sum::<f64>());
            if made == [0.0; 2] {
                return;
            }
            let stratification = &reference.stratification[block];
            let mut weight = 0.0;
            for i in (0..nn).filter(|&i| wet(i)) {
                let mass = geom.node_mass(k, i) * depth(i);
                for (&gamma, &ds) in stratification[i * nl..(i + 1) * nl].iter().zip(d_sigma) {
                    weight += mass * ds * gamma.max(0.0);
                }
            }
            if weight <= 0.0 {
                return;
            }
            for i in (0..nn).filter(|&i| wet(i)) {
                let scale = depth(i) / weight;
                let column = i * nl..(i + 1) * nl;
                for (t, out) in out.iter_mut().enumerate() {
                    let taken = scale * made[t];
                    for ((o, &gamma), &ds) in out[column.clone()]
                        .iter_mut()
                        .zip(&stratification[column.clone()])
                        .zip(d_sigma)
                    {
                        *o -= taken * ds * gamma.max(0.0);
                    }
                }
            }
        },
    );
}

/// One tracer's rows of an element for [`add_reference_advection`]
/// (`[node][level]`): the reference, its gradient over each layer, and the
/// tracer itself.
#[derive(Clone, Copy)]
struct EdgeFields<'a> {
    reference: &'a [f64],
    gradient: &'a [f64],
    tracer: &'a [f64],
}

impl<'a> EdgeFields<'a> {
    /// The fields over `range` of the element's rows: one end of an edge
    /// per level.
    fn rows(&self, range: std::ops::Range<usize>) -> EdgeFields<'a> {
        EdgeFields {
            reference: &self.reference[range.clone()],
            gradient: &self.gradient[range.clone()],
            tracer: &self.tracer[range],
        }
    }
}

/// The terms of [`add_reference_advection`] of one edge per level, between
/// the ends `a` and `b`: with the half flux `half` (½ F, out of `a`) and the
/// height difference `dz = z_b − z_a` of every level's edge,
/// `½ F θ [(C̄_b − C̄_a) − γ dz]` at each end with its own `γ`, added to
/// `out_a`, `out_b` over the ends' masses (`inverse_mass`), and their sum to
/// `made`. `θ` is the share of the reference's jump along the edge that the
/// tracer's own jump holds, in `[0, 1]`: 1 in a fluid stratified like the
/// reference, `b` at a fraction `b` of its strength, 0 where the tracer is
/// uniform or stratified the other way. Every slice has one value per level.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn edges_over_levels(
    half: &[f64],
    dz: &[f64],
    a: EdgeFields,
    b: EdgeFields,
    inverse_mass: (f64, f64),
    out_a: &mut [f64],
    out_b: &mut [f64],
    made: &mut [f64],
) {
    let n = half.len();
    let (dz, out_a, out_b, made) = (&dz[..n], &mut out_a[..n], &mut out_b[..n], &mut made[..n]);
    let (ref_a, ref_b) = (&a.reference[..n], &b.reference[..n]);
    let (grad_a, grad_b) = (&a.gradient[..n], &b.gradient[..n]);
    let (c_a, c_b) = (&a.tracer[..n], &b.tracer[..n]);
    // Not inlined: the slices are then known not to overlap, and the loop
    // (a division per level) is taken several levels at once
    for l in 0..n {
        let chord = ref_b[l] - ref_a[l];
        // No reference jump, no share (0/0 is NaN, which `clamp` keeps)
        let ratio = ((c_b[l] - c_a[l]) / chord).clamp(0.0, 1.0);
        let share = if chord == 0.0 { 0.0 } else { ratio };
        let weight = half[l] * share;
        let at_a = weight * (chord - grad_a[l] * dz[l]);
        let at_b = weight * (chord - grad_b[l] * dz[l]);
        out_a[l] += at_a * inverse_mass.0;
        out_b[l] += at_b * inverse_mass.1;
        made[l] += at_a + at_b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vertical::{SongHaidvogelStretching, UniformStretching};

    const G: f64 = 9.81;
    const RHO0: f64 = 1025.0;

    /// A channel `[0, length] × [0, width]` with the bed `bed(x, y)`, the
    /// density `rho(x, z)` at every node and level, and `η = eta(x)`.
    struct Column3D {
        mesh: Mesh2D,
        ops: DGOperators2D,
        geom: GeometricFactors2D,
        sigma: SigmaGrid,
        bathymetry: Bathymetry2D,
        state: Solution3D,
    }

    impl Column3D {
        fn new(
            length: f64,
            nx: usize,
            order: usize,
            sigma: SigmaGrid,
            bed: impl Fn(f64, f64) -> f64,
            eta: impl Fn(f64) -> f64,
            rho: impl Fn(f64, f64) -> f64,
        ) -> Self {
            let mesh = Mesh2D::uniform_rectangle(0.0, length, 0.0, length / nx as f64, nx, 1);
            let ops = DGOperators2D::new(order);
            let geom = GeometricFactors2D::compute(&mesh, &ops);
            let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, bed);
            let nl = sigma.n_levels();
            let mut state = Solution3D::new(mesh.n_elements, ops.n_nodes, nl);
            for k in 0..mesh.n_elements {
                let el = ElementIndex::new(k);
                for i in 0..ops.n_nodes {
                    let [x, _] = mesh.reference_to_physical(el, ops.nodes_r[i], ops.nodes_s[i]);
                    let idx = k * ops.n_nodes + i;
                    let e = eta(x);
                    state.eta.data[idx] = e;
                    let depth = e - bathymetry.data[idx];
                    for (l, &s) in sigma.sigma_rho().iter().enumerate() {
                        state.rho[idx * nl + l] = rho(x, e + s * depth);
                    }
                }
            }
            Self {
                mesh,
                ops,
                geom,
                sigma,
                bathymetry,
                state,
            }
        }

        /// The PGF (x, y) at every node and level.
        fn pgf(&self, rho_ref: f64, form: PressureGradientForm) -> (Vec<f64>, Vec<f64>) {
            let n = self.state.rho.len();
            let (mut fx, mut fy) = (vec![f64::NAN; n], vec![f64::NAN; n]);
            compute_pressure_gradient(
                &self.state,
                &self.mesh,
                &self.bathymetry,
                &self.sigma,
                &self.ops,
                &self.geom,
                G,
                RHO0,
                rho_ref,
                0.0,
                form,
                MetricForm::Conservative,
                None,
                &mut fx,
                &mut fy,
                Blocks::All(self.state.n_elements),
            );
            (fx, fy)
        }

        /// The σ-pairs force of the density anomaly about `profile(z)`.
        fn anomaly_pgf(&self, profile: impl Fn(f64) -> f64) -> (Vec<f64>, Vec<f64>) {
            let n = self.state.rho.len();
            let nl = self.sigma.n_levels();
            // The reference's density alone (no displacements)
            let mut reference = ReferenceFields {
                density: vec![0.0; n],
                stratification: vec![0.0; n],
                displacement: vec![0.0; self.state.n_elements],
                ..ReferenceFields::default()
            };
            for (idx, column) in reference.density.chunks_exact_mut(nl).enumerate() {
                let eta = self.state.eta.data[idx];
                let depth = eta - self.bathymetry.data[idx];
                for (rho, &s) in column.iter_mut().zip(self.sigma.sigma_rho()) {
                    *rho = profile(eta + s * depth);
                }
            }
            let (mut fx, mut fy) = (vec![f64::NAN; n], vec![f64::NAN; n]);
            compute_pressure_gradient(
                &self.state,
                &self.mesh,
                &self.bathymetry,
                &self.sigma,
                &self.ops,
                &self.geom,
                G,
                RHO0,
                RHO0,
                0.0,
                PressureGradientForm::SigmaPairs,
                MetricForm::Conservative,
                Some(&reference),
                &mut fx,
                &mut fy,
                Blocks::All(self.state.n_elements),
            );
            (fx, fy)
        }

        /// Largest |F| over every node and level.
        fn max_force(&self, rho_ref: f64, form: PressureGradientForm) -> f64 {
            let (fx, fy) = self.pgf(rho_ref, form);
            largest(&fx, &fy)
        }

        /// The σ-pairs force with the state itself as the balanced reference.
        fn balanced_force(&self) -> f64 {
            let reference = BalancedReference::new(
                &self.state,
                &self.mesh,
                &self.bathymetry,
                &self.sigma,
                &self.ops,
                &self.geom,
                G,
                RHO0,
                RHO0,
                0.0,
                MetricForm::Conservative,
                None,
            );
            let (mut fx, mut fy) = self.pgf(RHO0, PressureGradientForm::SigmaPairs);
            reference.add_to(&mut fx, &mut fy, Blocks::All(self.state.n_elements));
            largest(&fx, &fy)
        }
    }

    /// σ-pairs alone on the pycnocline fjord: measured ≤ 6.8e-4 m/s².
    const SIGMA_PAIRS_BOUND: f64 = 1e-3;

    const FORMS: [PressureGradientForm; 2] = [
        PressureGradientForm::SigmaPairs,
        PressureGradientForm::ConstantDepth,
    ];

    /// Largest |F| over every node and level.
    fn largest(fx: &[f64], fy: &[f64]) -> f64 {
        // NaN-propagating (f64::max would drop a NaN)
        fx.iter()
            .zip(fy)
            .map(|(x, y)| x.hypot(*y))
            .fold(0.0, |m, f| if f.is_nan() || f > m { f } else { m })
    }

    /// The review's fjord (notes/review-2026-09-25.md §3.3): the bed drops from 30 to 400 m
    /// over ≈ 1.5 km.
    fn fjord_bed(x: f64, _y: f64) -> f64 {
        -(30.0 + 185.0 * (1.0 + ((x - 1500.0) / 375.0).tanh()))
    }

    /// 5 kg/m³ lighter in the upper 20 m.
    fn pycnocline(z: f64) -> f64 {
        RHO0 - 2.5 + 2.5 * (-(z + 10.0) / 3.0).tanh()
    }

    /// TODO P4.3 gate: constant N² at rest over the steep fjord slope, at
    /// every order and for stretched levels, is balanced to round-off in
    /// both forms (the old element-local σ form: 2.6e-6 m/s² for
    /// N² = 1e-4 s⁻² on a 30→400 m slope at every resolution, notes/review-2026-09-25.md
    /// §3.3(a); σ-pairs integrate a linear profile exactly).
    #[test]
    fn constant_n2_at_rest_is_balanced_over_a_steep_slope() {
        // N² = 1e-4 s⁻²: ∂ρ/∂z = −ρ₀N²/g
        let n2 = 1e-4;
        let linear = |_: f64, z: f64| RHO0 - RHO0 * n2 / G * z;
        for order in 1..=4 {
            for sigma in [
                SigmaGrid::new(30, UniformStretching),
                SigmaGrid::new(20, SongHaidvogelStretching::new(5.0, 0.4, 10.0)),
            ] {
                let case = Column3D::new(3000.0, 6, order, sigma, fjord_bed, |_| 0.0, linear);
                for rho_ref in [RHO0, 0.0] {
                    for form in FORMS {
                        // Measured ≤ 1.5e-14 (the old σ form: 1.3e-5 – 2.5e-3)
                        let force = case.max_force(rho_ref, form);
                        assert!(
                            force < 1e-13,
                            "P{order}, rho_ref {rho_ref}, {form:?}: spurious |F| = {force:.3e} m/s²"
                        );
                    }
                }
            }
        }
    }

    /// TODO P4.3 gate: the review's stratified fjord at rest (a 5 kg/m³
    /// pycnocline in the upper 20 m over a 30→400 m slope), on 30
    /// surface-stretched levels (θs = 5, θb = 0.4), against ≈ 5e-5 m/s² of
    /// real estuarine forcing. Spurious |F| (m/s²):
    ///
    /// | | P1, 500 m | P3, 250 m | P3, 100 m |
    /// |---|---|---|---|
    /// | old element-local σ form | 1.6e-3 | 1.1e-4 | 2.2e-5 |
    /// | σ-pairs | 6.8e-4 | 1.5e-4 | 1.7e-5 |
    /// | constant depth | 2.0e-7 | 1.5e-6 | 3.3e-6 |
    /// | σ-pairs, balanced reference | 2.0e-7 | 1.5e-6 | 3.3e-6 |
    ///
    /// σ-pairs alone carry the σ form's error (their default use is with
    /// a [`BalancedReference`], which gives a fluid resting in it the
    /// constant-depth form's force).
    ///
    /// On 30 *uniform* levels both schemes stay at 2e-4 – 1.6e-3 (new 2.0e-4
    /// – 4.1e-4): 13 m apart at 400 m, the deep columns do not sample the 3 m
    /// pycnocline, so no column-to-column comparison can match them. That is
    /// a vertical-resolution requirement (resolve the pycnocline in every
    /// column, as NorKyst's stretched levels do), not a PGF one.
    #[test]
    fn stratified_fjord_at_rest_has_a_small_spurious_force() {
        for (order, nx) in [(1, 6), (3, 12), (3, 30)] {
            let case = Column3D::new(
                3000.0,
                nx,
                order,
                SigmaGrid::new(30, SongHaidvogelStretching::new(5.0, 0.4, 10.0)),
                fjord_bed,
                |_| 0.0,
                |_, z| pycnocline(z),
            );
            let force = case.max_force(RHO0, PressureGradientForm::ConstantDepth);
            let balanced = case.balanced_force();
            let sigma_pairs = case.max_force(RHO0, PressureGradientForm::SigmaPairs);
            println!(
                "P{order}, {} m: constant depth {force:.3e}, balanced {balanced:.3e}, \
                 σ-pairs {sigma_pairs:.3e} m/s²",
                3000 / nx
            );
            for (name, force) in [("constant depth", force), ("balanced σ-pairs", balanced)] {
                assert!(
                    force < 5e-6,
                    "P{order}, {} m, {name}: spurious |F| = {force:.3e} m/s²",
                    3000 / nx
                );
            }
            assert!(
                sigma_pairs < SIGMA_PAIRS_BOUND,
                "P{order}, {} m, σ-pairs: spurious |F| = {sigma_pairs:.3e} m/s²",
                3000 / nx
            );
        }
    }

    /// The anomaly about a reference profile on the review's stratified
    /// fjord: at rest in the profile the force vanishes exactly (the σ-pairs
    /// form of the full density leaves ≈ 1e-3 m/s² here, and a balanced
    /// reference state the constant-depth form's interpolation error,
    /// ≈ 2e-7–2e-6 m/s²), and a departure from the profile that is uniform in
    /// each column (a horizontal density gradient) feels exactly the σ-pairs
    /// force of the departure alone: the anomaly's pressure is its trapezoid
    /// rule, which is the monotone cubic's of a column-uniform density.
    #[test]
    fn a_reference_profile_balances_the_fjord_at_rest_exactly() {
        let a = 1e-4;
        for (order, nx) in [(1, 6), (3, 12)] {
            for anomaly in [0.0, a] {
                let case = |rho: &dyn Fn(f64, f64) -> f64| {
                    Column3D::new(
                        3000.0,
                        nx,
                        order,
                        SigmaGrid::new(30, SongHaidvogelStretching::new(5.0, 0.4, 10.0)),
                        fjord_bed,
                        |_| 0.0,
                        rho,
                    )
                };
                let stratified = case(&|x, z| pycnocline(z) + anomaly * (x - 1500.0));
                let (fx, fy) = stratified.anomaly_pgf(pycnocline);
                let departure = case(&|x, _| RHO0 + anomaly * (x - 1500.0));
                let (dx, dy) = departure.pgf(RHO0, PressureGradientForm::SigmaPairs);
                let error = largest(
                    &fx.iter().zip(&dx).map(|(f, d)| f - d).collect::<Vec<_>>(),
                    &fy.iter().zip(&dy).map(|(f, d)| f - d).collect::<Vec<_>>(),
                );
                // Measured 0 at rest; with the departure the round-off of
                // subtracting the profile (1e-13 kg/m³ of 1e3)
                assert!(
                    error < 1e-14,
                    "P{order}, anomaly {anomaly}: off the departure's force by {error:.3e} m/s²"
                );
                if anomaly == 0.0 {
                    assert_eq!(largest(&fx, &fy), 0.0, "P{order}: a force at rest");
                }
            }
        }
    }

    /// A basin of 3 × 2 P2 elements (distorted by `distort` of an element)
    /// over a cliff from 5 to 125 m with land in one corner, 6 stretched
    /// levels, at rest but for a velocity `amplitude × random` with no depth
    /// mean in every column that is not thin (0.1 m): the mesh, the
    /// operators, the bed, the levels and the state (its `rho` not set).
    fn cliff_basin(
        distort: f64,
        amplitude: f64,
    ) -> (
        Mesh2D,
        DGOperators2D,
        GeometricFactors2D,
        Bathymetry2D,
        SigmaGrid,
        Solution3D,
    ) {
        use std::f64::consts::PI;
        let (lx, ly) = (900.0, 600.0);
        let mut mesh = Mesh2D::uniform_rectangle(0.0, lx, 0.0, ly, 3, 2);
        for v in &mut mesh.vertices {
            let [x, y] = *v;
            let bump = (PI * x / lx).sin() * (PI * y / ly).sin();
            *v = [x + distort * 300.0 * bump, y - distort * 300.0 * bump];
        }
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
            let cliff = -65.0 - 60.0 * ((x - 450.0) / 120.0).tanh();
            let land =
                0.25 * (1.0 + ((x - 750.0) / 60.0).tanh()) * (1.0 + ((y - 450.0) / 60.0).tanh());
            cliff + (2.0 - cliff) * land
        });
        let sigma = SigmaGrid::new(6, SongHaidvogelStretching::new(5.0, 0.4, 10.0));
        let (nn, nl) = (ops.n_nodes, sigma.n_levels());
        let mut state = Solution3D::new(mesh.n_elements, nn, nl);
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for idx in 0..mesh.n_elements * nn {
            let bed = bathymetry.data[idx];
            let eta = bed.max(0.0);
            state.eta.data[idx] = eta;
            if eta - bed < 0.1 {
                continue;
            }
            for field in [&mut state.u, &mut state.v] {
                let column = &mut field[idx * nl..(idx + 1) * nl];
                column.iter_mut().for_each(|u| *u = amplitude * random());
                let mean: f64 = column.iter().zip(sigma.d_sigma()).map(|(u, d)| u * d).sum();
                column.iter_mut().for_each(|u| *u -= mean);
            }
        }
        (mesh, ops, geom, bathymetry, sigma, state)
    }

    /// The layer transports of `state` alone (no barotropic pass), in
    /// `metric`'s form, with walls around.
    fn own_transports(
        state: &Solution3D,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        sigma: &SigmaGrid,
        bathymetry: &Bathymetry2D,
        metric: MetricForm,
    ) -> LayerTransport {
        use crate::solver::rhs::{Boundaries3D, Exterior3D};
        let mut transport =
            LayerTransport::new(state.n_elements, ops, sigma.n_levels()).with_metric(metric);
        transport.compute(
            state,
            None,
            mesh,
            ops,
            geom,
            sigma,
            bathymetry,
            &Boundaries3D::default(),
            &Exterior3D::default(),
            0.1,
            Blocks::All(state.n_elements),
        );
        transport
    }

    /// A thermocline over a stratification that is stable everywhere, as a
    /// reference profile (S uniform), for the linear equation of state
    /// [`thermocline_density`].
    fn thermocline() -> TracerReferenceProfile {
        TracerReferenceProfile::from_smooth_fn(-130.0, 5.0, 2701, |z| {
            (10.0 + 4.0 * ((z + 15.0) / 4.0).tanh() + 0.01 * z, 34.0)
        })
    }

    /// Thermal expansion of [`thermocline_density`] (1/°C).
    const ALPHA: f64 = 2e-4;

    /// `ρ = ρ₀ (1 − α (T − 10))`.
    fn thermocline_density(t: f64, _s: f64) -> f64 {
        RHO0 * (1.0 - ALPHA * (t - 10.0))
    }

    /// `(∂ρ/∂T, ∂ρ/∂S)` of [`thermocline_density`].
    fn thermocline_partials(_t: f64, _s: f64) -> (f64, f64) {
        (-RHO0 * ALPHA, 0.0)
    }

    /// TODO P1.3 gate, the identity the stability at rest stands on: about
    /// rest in a reference profile, the advection of the reference and the
    /// anomaly's pressure gradient exchange pseudo-energy exactly. For any
    /// layer transports `Q` without a depth mean and any density anomaly
    /// `ρ′`,
    ///
    /// ```text
    ///     g Σ m (ρ′/Γ) H_z ∂ρ/∂t = −ρ₀ Σ m Q·F(ρ′),
    /// ```
    ///
    /// the rate of the available potential energy `½ g ρ′²/Γ` under the
    /// tracers' whole advection of the fluid at rest in the reference (the
    /// conservative kernel, about the reference, with
    /// [`add_reference_advection`]) against the work of the force on the
    /// flow: on parallelograms and distorted elements, in both metric forms,
    /// over a cliff from 5 to 125 m under a thermocline, through element
    /// faces, and in a shore element with dry nodes (where the force has only
    /// its face terms). The element's inventory of the advected reference is
    /// kept, too.
    #[test]
    fn the_reference_advection_and_the_pressure_gradient_exchange_pseudo_energy_exactly() {
        use crate::solver::rhs::{
            AdvectedTracer, Boundaries3D, ExtrapolationTracerBC3D, VerticalAdvection,
            apply_tracer_transport_3d,
        };
        for (distort, metric) in [
            (0.0, MetricForm::Conservative),
            (0.12, MetricForm::Conservative),
            (0.12, MetricForm::Averaged),
        ] {
            let (mesh, ops, geom, bathymetry, sigma, mut state) = cliff_basin(distort, 1.0);
            let (nn, nl) = (ops.n_nodes, sigma.n_levels());
            let n = mesh.n_elements * nn * nl;
            let all = Blocks::All(mesh.n_elements);
            let (profile, density, partials) =
                (thermocline(), thermocline_density, thermocline_partials);
            let mut fields = ReferenceFields::default();
            fields.fill(
                &profile,
                density,
                partials,
                &state,
                &bathymetry,
                &sigma,
                all,
            );
            // Any anomaly about the reference
            let mut seed = 0x2545_f491_4f6c_dd1d_u64;
            let anomaly: Vec<f64> = (0..n)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    0.3 * ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
                })
                .collect();
            for ((rho, &base), &a) in state.rho.iter_mut().zip(&fields.density).zip(&anomaly) {
                *rho = base + a;
            }
            fields.set_displacements(&state, &geom, &bathymetry, &sigma, 0.1, all);
            let transport = own_transports(&state, &mesh, &ops, &geom, &sigma, &bathymetry, metric);

            // The force of the anomaly, and its work on the transports
            let (mut fx, mut fy) = (vec![0.0; n], vec![0.0; n]);
            compute_pressure_gradient(
                &state,
                &mesh,
                &bathymetry,
                &sigma,
                &ops,
                &geom,
                G,
                RHO0,
                RHO0,
                0.1,
                PressureGradientForm::SigmaPairs,
                metric,
                Some(&fields),
                &mut fx,
                &mut fy,
                all,
            );

            // The tracers at rest in the reference: their whole advection,
            // and the layers' own (a uniform field's), as inventory rates
            let [reference_t, reference_s] = &fields.tracers;
            let ones = vec![1.0; n];
            let boundaries = Boundaries3D::default();
            let (mut rate_t, mut rate_s) = (vec![0.0; n], vec![0.0; n]);
            apply_tracer_transport_3d(
                [
                    AdvectedTracer {
                        reference: Some(reference_t),
                        ..AdvectedTracer::new(&mut rate_t, reference_t, &ExtrapolationTracerBC3D)
                    },
                    AdvectedTracer {
                        reference: Some(reference_s),
                        ..AdvectedTracer::new(&mut rate_s, reference_s, &ExtrapolationTracerBC3D)
                    },
                ],
                &transport,
                &mesh,
                &ops,
                &geom,
                &boundaries,
                VerticalAdvection::LimitedAkima,
                all,
            );
            let (mut rate_h, mut unused) = (vec![0.0; n], vec![0.0; n]);
            apply_tracer_transport_3d(
                [
                    AdvectedTracer::new(&mut rate_h, &ones, &ExtrapolationTracerBC3D),
                    AdvectedTracer::new(&mut unused, &ones, &ExtrapolationTracerBC3D),
                ],
                &transport,
                &mesh,
                &ops,
                &geom,
                &boundaries,
                VerticalAdvection::LimitedAkima,
                all,
            );
            let conservative: f64 = (0..n)
                .map(|p| geom.node_mass(p / nl / nn, p / nl % nn) * rate_t[p])
                .sum();
            add_reference_advection(
                [&mut rate_t, &mut rate_s],
                [reference_t, reference_s],
                &fields,
                &state,
                &transport,
                &ops,
                &geom,
                &sigma,
                &bathymetry,
                0.1,
                all,
            );

            let (rho_t, _) = partials(0.0, 0.0);
            let (mut potential, mut work, mut scale) = (0.0, 0.0, 0.0);
            let (mut inventory, mut inventory_scale) = (0.0, 0.0);
            for p in 0..n {
                let (idx, mass) = (p / nl, geom.node_mass(p / nl / nn, p / nl % nn));
                inventory += mass * rate_t[p];
                inventory_scale += mass * rate_t[p].abs();
                work += mass * RHO0 * (transport.hu[p] * fx[p] + transport.hv[p] * fy[p]);
                // H ∂ρ/∂t at the node: the temperature's rate less the
                // layer's own, times ∂ρ/∂T
                let gamma = fields.stratification[p];
                if state.eta.data[idx] - bathymetry.data[idx] < 0.1 {
                    continue;
                }
                assert!(gamma > 0.0, "test regime: Γ = {gamma:.2e} at a wet node");
                let lifted = rho_t * (rate_t[p] - reference_t[p] * rate_h[p]);
                potential += mass * G * anomaly[p] / gamma * lifted;
                scale += mass * (G * anomaly[p] / gamma * lifted).abs();
            }
            println!(
                "distortion {distort}, {metric:?}: the potential energy's rate {potential:.6e}, \
                 the force's work {work:.6e} (scale {scale:.2e}); the elements' inventory \
                 rate {inventory:.2e} (conservative {conservative:.2e}, scale \
                 {inventory_scale:.2e})"
            );
            assert!(scale > 1.0, "test regime: scale {scale:.2e}");
            assert!(
                (potential + work).abs() < 1e-11 * scale,
                "distortion {distort}, {metric:?}: the potential energy's rate \
                 {potential:.12e} against the force's work {work:.12e} (scale {scale:.2e})"
            );
            // Walls around: the advected reference's inventory does not change
            assert!(
                inventory.abs() < 1e-12 * inventory_scale,
                "distortion {distort}, {metric:?}: the inventory changes at {inventory:.3e} \
                 (scale {inventory_scale:.2e})"
            );
        }
    }

    /// The step's bound on the reference's waves along sloping levels
    /// ([`ReferenceFields::sloping_level_rate`]) against the kernels. In the
    /// pseudo-energy's coordinates, the pairs' advection of the reference
    /// (the split form's chords with [`add_reference_advection`], no flux
    /// through the σ-surfaces) is a matrix `B` from the layers' velocities
    /// to the anomaly's rate, assembled here column by column in every
    /// element of the cliff basin; its largest singular value (power
    /// iteration on `BᵀB`) is the largest frequency of the oscillation the
    /// pairs and the anomaly's pressure gradient make. The bound is at
    /// least that in every element without a thin column (measured 1.3 to
    /// 1.9 times it; on distorted elements too, in both metric forms), and
    /// none where the pairs do not act.
    #[test]
    fn the_sloping_level_rate_bounds_the_pairs_frequency() {
        for (distort, metric) in [
            (0.0, MetricForm::Conservative),
            (0.12, MetricForm::Conservative),
            (0.12, MetricForm::Averaged),
        ] {
            let (mesh, ops, geom, bathymetry, sigma, state) = cliff_basin(distort, 0.0);
            let (nn, nl) = (ops.n_nodes, sigma.n_levels());
            let (ne, per_element) = (mesh.n_elements, nn * nl);
            let n = ne * per_element;
            let mut fields = ReferenceFields::default();
            fields.fill(
                &thermocline(),
                thermocline_density,
                thermocline_partials,
                &state,
                &bathymetry,
                &sigma,
                Blocks::All(ne),
            );
            let depth =
                |k: usize, i: usize| state.eta.data[k * nn + i] - bathymetry.data[k * nn + i];
            let wet = |k: usize| (0..nn).all(|i| depth(k, i) >= 0.1);
            // m H_z of a layer
            let volume = |k: usize, i: usize, l: usize| {
                geom.node_mass(k, i) * depth(k, i) * sigma.d_sigma()[l]
            };
            let [reference_t, reference_s] = &fields.tracers;
            let mut transport = LayerTransport::new(ne, &ops, nl).with_metric(metric);
            // B of every element, [row = node, level][column = node, level, x|y]
            let columns = 2 * per_element;
            let mut b = vec![0.0; ne * per_element * columns];
            let (mut rate_t, mut rate_s) = (vec![0.0; n], vec![0.0; n]);
            for column in 0..columns {
                let (at, y) = (column / 2, column % 2 == 1);
                // A unit of √(ρ₀ m H_z) u at the column's node and level of
                // every element, as its layer transport H_z u
                transport.hu.fill(0.0);
                transport.hv.fill(0.0);
                for k in (0..ne).filter(|&k| wet(k)) {
                    let (i, l) = (at / nl, at % nl);
                    let h = depth(k, i) * sigma.d_sigma()[l];
                    let q = h / (RHO0 * volume(k, i, l)).sqrt();
                    let field = if y {
                        &mut transport.hv
                    } else {
                        &mut transport.hu
                    };
                    field[k * per_element + at] = q;
                }
                rate_t.fill(0.0);
                rate_s.fill(0.0);
                add_reference_advection(
                    [&mut rate_t, &mut rate_s],
                    [reference_t, reference_s],
                    &fields,
                    &state,
                    &transport,
                    &ops,
                    &geom,
                    &sigma,
                    &bathymetry,
                    0.1,
                    Blocks::All(ne),
                );
                for k in (0..ne).filter(|&k| wet(k)) {
                    let averaged = metric.averaged_on(&geom, k);
                    let block = k * per_element;
                    // The split form's own advection of the reference along
                    // the pairs, −(1/m_i) Σ_j ½ Φ_ij (T̄_j − T̄_i)
                    for i in 0..nn {
                        for j in i + 1..nn {
                            let [(dx_ij, dy_ij), (dx_ji, dy_ji)] =
                                pair_derivatives(&ops, &geom, k, i, j, averaged);
                            let (m_i, m_j) = (geom.node_mass(k, i), geom.node_mass(k, j));
                            for l in 0..nl {
                                let (a, c) = (block + i * nl + l, block + j * nl + l);
                                let half = 0.5
                                    * (m_i * (dx_ij * transport.hu[a] + dy_ij * transport.hv[a])
                                        - m_j
                                            * (dx_ji * transport.hu[c] + dy_ji * transport.hv[c]));
                                let chord = reference_t[c] - reference_t[a];
                                rate_t[a] -= half * chord / m_i;
                                rate_t[c] -= half * chord / m_j;
                            }
                        }
                    }
                    // … as the rate of ρ′ √(g m H_z / Γ)
                    for row in 0..per_element {
                        let (i, l) = (row / nl, row % nl);
                        let gamma = fields.stratification[block + row];
                        assert!(gamma > 0.0, "the thermocline is stable");
                        let h = depth(k, i) * sigma.d_sigma()[l];
                        let rate = -RHO0 * ALPHA * rate_t[block + row] / h;
                        b[(block + row) * columns + column] =
                            rate * (G * volume(k, i, l) / gamma).sqrt();
                    }
                }
            }
            let mut scratch = vec![0.0; 2 * nn];
            let mut steep = 0;
            for k in 0..ne {
                let bound = fields.sloping_level_rate(
                    &state,
                    &ops,
                    &geom,
                    &bathymetry,
                    &sigma,
                    G / RHO0,
                    0.1,
                    metric,
                    k,
                    &mut scratch,
                );
                let b = &b[k * per_element * columns..(k + 1) * per_element * columns];
                if !wet(k) {
                    assert!(b.iter().all(|&x| x == 0.0), "pairs act in element {k}");
                    assert_eq!(bound, 0.0, "a bound without pairs in element {k}");
                    continue;
                }
                // The largest singular value, from below
                let mut x: Vec<f64> = (0..columns)
                    .map(|c| 1.0 + 0.37 * (c as f64).sin())
                    .collect();
                let mut y = vec![0.0; per_element];
                let mut frequency = 0.0_f64;
                for _ in 0..500 {
                    for (row, y) in y.iter_mut().enumerate() {
                        *y = b[row * columns..(row + 1) * columns]
                            .iter()
                            .zip(&x)
                            .map(|(b, x)| b * x)
                            .sum();
                    }
                    let norm_x = x.iter().map(|x| x * x).sum::<f64>().sqrt();
                    let norm_y = y.iter().map(|y| y * y).sum::<f64>().sqrt();
                    frequency = norm_y / norm_x;
                    for (c, x) in x.iter_mut().enumerate() {
                        *x = (0..per_element)
                            .map(|row| b[row * columns + c] * y[row])
                            .sum();
                    }
                    let norm = x.iter().map(|x| x * x).sum::<f64>().sqrt();
                    x.iter_mut().for_each(|x| *x /= norm);
                }
                println!(
                    "distortion {distort}, {metric:?}, element {k}: frequency {frequency:.4e} \
                     1/s, bound {bound:.4e} ({:.2})",
                    bound / frequency
                );
                assert!(
                    frequency <= bound,
                    "element {k}: frequency {frequency:.4e} above the bound {bound:.4e}"
                );
                assert!(
                    bound < 2.5 * frequency,
                    "element {k}: the bound {bound:.4e} is slack on {frequency:.4e}"
                );
                steep += usize::from(frequency > 5e-3);
            }
            assert!(steep >= 2, "the cliff's elements carry the frequency");
        }
    }

    /// Constancy, a linear reference, and every element's inventory: a
    /// uniform tracer gets nothing from [`add_reference_advection`] whatever
    /// the reference and the flow (it holds no share of the reference's
    /// stratification); a tracer in a reference linear in `z` gets nothing
    /// either (the reference's chords are its tangents), to round-off; and
    /// what a tracer in a curved reference gets adds up to nothing in every
    /// element.
    #[test]
    fn the_reference_advection_keeps_constancy_a_linear_reference_and_the_elements_inventories() {
        let (mesh, ops, geom, bathymetry, sigma, state) = cliff_basin(0.12, 1.0);
        let (nn, nl) = (ops.n_nodes, sigma.n_levels());
        let n = mesh.n_elements * nn * nl;
        let transport = own_transports(
            &state,
            &mesh,
            &ops,
            &geom,
            &sigma,
            &bathymetry,
            MetricForm::Averaged,
        );
        let all = Blocks::All(mesh.n_elements);
        let (density, partials) = (thermocline_density, thermocline_partials);
        let apply = |profile: &TracerReferenceProfile, tracers: [&[f64]; 2]| {
            let mut fields = ReferenceFields::default();
            fields.fill(profile, density, partials, &state, &bathymetry, &sigma, all);
            let (mut t, mut s) = (vec![0.0; n], vec![0.0; n]);
            add_reference_advection(
                [&mut t, &mut s],
                tracers,
                &fields,
                &state,
                &transport,
                &ops,
                &geom,
                &sigma,
                &bathymetry,
                0.1,
                all,
            );
            (fields, t, s)
        };
        let largest = |x: &[f64]| x.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        // A thermocline (and a linear salinity) as the reference, uniform
        // tracers
        let curved = TracerReferenceProfile::from_smooth_fn(-130.0, 5.0, 2701, |z| {
            (
                10.0 + 4.0 * ((z + 15.0) / 4.0).tanh() + 0.01 * z,
                34.0 - 0.01 * z,
            )
        });
        let (uniform_t, uniform_s) = (vec![11.5; n], vec![33.0; n]);
        let (fields, t, s) = apply(&curved, [&uniform_t, &uniform_s]);
        assert_eq!(largest(&t), 0.0, "uniform T");
        assert_eq!(largest(&s), 0.0, "uniform S");
        // The tracers in the curved reference: a term of the advection's
        // size, which adds up to nothing in every element
        let (_, t, _) = apply(&curved, [&fields.tracers[0], &fields.tracers[1]]);
        assert!(largest(&t) > 1e-3, "test regime: {:.2e}", largest(&t));
        for (k, element) in t.chunks_exact(nn * nl).enumerate() {
            let (mut sum, mut scale) = (0.0, 0.0);
            for (i, column) in element.chunks_exact(nl).enumerate() {
                let mass = geom.node_mass(k, i);
                sum += mass * column.iter().sum::<f64>();
                scale += mass * column.iter().map(|c| c.abs()).sum::<f64>();
            }
            assert!(
                sum.abs() <= 1e-13 * scale,
                "element {k} gains {sum:.3e} (scale {scale:.2e})"
            );
        }
        // The tracers in a linear reference: nothing
        let linear = TracerReferenceProfile::new(
            vec![-130.0, 5.0],
            vec![4.0, 4.0 + 0.03 * 135.0],
            vec![36.0, 36.0 - 0.01 * 135.0],
        );
        let (fields, _, _) = apply(&linear, [&uniform_t, &uniform_s]);
        let (_, t, s) = apply(&linear, [&fields.tracers[0], &fields.tracers[1]]);
        // Against Ω and the pairs' fluxes of ≈ 1 m/s times a degree
        assert!(largest(&t) < 1e-12, "linear T: {:.3e}", largest(&t));
        assert!(largest(&s) < 1e-12, "linear S: {:.3e}", largest(&s));
    }

    /// A density that varies only horizontally, over a flat bed: the force
    /// is `−(g/ρ₀)(η − z) ∂ρ/∂x`, reproduced exactly for a polynomial `ρ(x)`
    /// of the element's degree.
    #[test]
    fn horizontal_density_gradient_over_a_flat_bed_is_exact() {
        let (a, depth) = (2e-4, 50.0);
        let case = Column3D::new(
            4000.0,
            4,
            2,
            SigmaGrid::new(10, UniformStretching),
            |_, _| -depth,
            |_| 0.0,
            |x, _| RHO0 + a * x,
        );
        for form in FORMS {
            let (fx, fy) = case.pgf(RHO0, form);
            let nl = case.sigma.n_levels();
            for (idx, (&x_force, &y_force)) in fx.iter().zip(&fy).enumerate() {
                let z = case.sigma.sigma_rho()[idx % nl] * depth;
                let expected = -G / RHO0 * (0.0 - z) * a;
                assert!(
                    (x_force - expected).abs() < 1e-9 * expected.abs().max(1e-6),
                    "{form:?}: {x_force:.6e} vs {expected:.6e}"
                );
                assert!(y_force.abs() < 1e-15, "{form:?}: {y_force:.3e}");
            }
        }
    }

    /// Constant density over a sloping bed with a flat surface: no force, for
    /// the full PGF too (the old σ form needed exact `∇B` for this).
    #[test]
    fn constant_density_over_a_sloping_bed_has_no_force() {
        let case = Column3D::new(
            1000.0,
            2,
            1,
            SigmaGrid::new(5, UniformStretching),
            |x, _| -100.0 - 0.1 * x,
            |_| 0.0,
            |_, _| 1000.0,
        );
        for rho_ref in [0.0, 1000.0] {
            for form in FORMS {
                let force = case.max_force(rho_ref, form);
                assert!(
                    force < 1e-12,
                    "rho_ref {rho_ref}, {form:?}: |F| = {force:.3e}"
                );
            }
        }
    }

    /// Regression (TODO P0.6): with constant density ρ = ρ₀ and a tilted free
    /// surface over a flat bottom, the FULL PGF must be the barotropic term
    /// −g∇η, while the BAROCLINIC-ONLY PGF (rho_ref = ρ₀) must vanish. If the
    /// baroclinic path still carried −g∇η, the mode-split G-term would double
    /// the surface pressure gradient already supplied by the 2D ½gh² flux.
    #[test]
    fn baroclinic_only_pgf_excludes_surface_slope() {
        let slope = 1.0e-4;
        let case = Column3D::new(
            1000.0,
            4,
            1,
            SigmaGrid::new(4, UniformStretching),
            |_, _| -100.0,
            |x| slope * x,
            |_, _| RHO0,
        );
        for form in FORMS {
            let (full_x, full_y) = case.pgf(0.0, form);
            let expected = -G * slope;
            let x_err = full_x
                .iter()
                .map(|v| (v - expected).abs())
                .fold(0.0, f64::max);
            let y_err = full_y.iter().map(|v| v.abs()).fold(0.0, f64::max);
            assert!(
                x_err < 1e-9,
                "full PGF x error {x_err:.3e} (−g∇η = {expected:.3e})"
            );
            assert!(y_err < 1e-9, "full PGF y {y_err:.3e}");
            let baroclinic = case.max_force(RHO0, form);
            assert!(
                baroclinic < 1e-12,
                "{form:?}: baroclinic-only PGF {baroclinic:.3e}"
            );
        }
    }

    /// A density front on an element face: the element-local sums see two
    /// uniform elements and no force; the lifted pressure jump gives the
    /// front its force, and the force on the two sides balances (it only
    /// moves momentum across the face).
    #[test]
    fn density_front_on_a_face_exerts_a_force() {
        let depth = 20.0;
        let case = Column3D::new(
            2000.0,
            2,
            2,
            SigmaGrid::new(4, UniformStretching),
            |_, _| -depth,
            |_| 0.0,
            |_, _| RHO0,
        );
        let mut case = case;
        let nl = case.sigma.n_levels();
        let nn = case.ops.n_nodes;
        // Heavier water in the right element, up to the shared face
        case.state.rho[nn * nl..].fill(RHO0 + 1.0);
        for form in FORMS {
            let (fx, _) = case.pgf(RHO0, form);
            // Depth-integrated, area-weighted force on each element
            let force = |k: usize| -> f64 {
                let per_node: Vec<f64> = (0..nn)
                    .map(|i| (0..nl).map(|l| fx[(k * nn + i) * nl + l]).sum::<f64>())
                    .collect();
                case.geom.integrate_element(k, &per_node)
            };
            let (left, right) = (force(0), force(1));
            // Heavier water on the right pushes towards the left, on both sides
            assert!(
                left < 0.0 && right < 0.0,
                "left {left:.3e}, right {right:.3e}"
            );
            assert!(
                (left - right).abs() < 1e-12 * left.abs(),
                "{form:?}: the face force is shared: left {left:.6e}, right {right:.6e}"
            );
        }
    }

    /// Smooth baroclinic field over a gentle slope: `ρ' = a sin(kx) e^{z/H}`,
    /// `∂p/∂x|_z = g a k H cos(kx) (1 − e^{z/H})`, at P2 under joint
    /// refinement of the mesh and the levels. The nodal P2 derivative is
    /// second order; measured ratios 1.69, 1.80, 1.75 for the constant-depth
    /// form (1.74 over a flat bed, so not the one-sided pairs at the bottom
    /// of the slope), 1.73, 1.93, 1.98 for σ-pairs (errors 2.3e-5 → 4.7e-7).
    #[test]
    fn smooth_baroclinic_force_converges() {
        let (a, h_scale, length) = (0.5, 40.0, 8000.0);
        let wavenumber = 2.0 * std::f64::consts::PI / length;
        for form in FORMS {
            let errors: Vec<f64> = [(4, 10), (8, 20), (16, 40), (32, 80)]
                .iter()
                .map(|&(nx, levels)| {
                    let case = Column3D::new(
                        length,
                        nx,
                        2,
                        SigmaGrid::new(levels, UniformStretching),
                        |x, _| -100.0 - 0.005 * x,
                        |_| 0.0,
                        |x, z| RHO0 + a * (wavenumber * x).sin() * (z / h_scale).exp(),
                    );
                    let (fx, _) = case.pgf(RHO0, form);
                    let nl = levels;
                    let mut err = 0.0_f64;
                    for k in 0..case.mesh.n_elements {
                        let el = ElementIndex::new(k);
                        for i in 0..case.ops.n_nodes {
                            let [x, _] = case.mesh.reference_to_physical(
                                el,
                                case.ops.nodes_r[i],
                                case.ops.nodes_s[i],
                            );
                            let idx = k * case.ops.n_nodes + i;
                            let depth = -case.bathymetry.data[idx];
                            for (l, &s) in case.sigma.sigma_rho().iter().enumerate() {
                                let z = s * depth;
                                let exact = -G / RHO0
                                    * a
                                    * wavenumber
                                    * h_scale
                                    * (wavenumber * x).cos()
                                    * (1.0 - (z / h_scale).exp());
                                err = err.max((fx[idx * nl + l] - exact).abs());
                            }
                        }
                    }
                    err
                })
                .collect();
            println!("{form:?}: errors {errors:?}");
            for pair in errors.windows(2) {
                let order = (pair[0] / pair[1]).log2();
                assert!(
                    order > 1.6,
                    "{form:?}: order {order:.2} (errors {errors:?})"
                );
            }
        }
    }
}
