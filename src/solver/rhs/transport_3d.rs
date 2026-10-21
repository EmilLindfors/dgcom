//! Layer volume transport of the 3D stages, consistent with the barotropic mode.
//!
//! # Formulation
//!
//! In σ-coordinates the volume of layer `l` obeys
//!
//! ```text
//!     ∂H_z,l/∂t + ∇·Q_l + Ω_{l+1/2} − Ω_{l−1/2} = 0,    H_z,l = D·Δσ_l,
//! ```
//!
//! with `Q_l` the horizontal transport of the layer (m²/s) and `Ω` the volume
//! flux through the σ-surfaces (m/s). A tracer `C` is advected with the same
//! fluxes, `∂(H_z C)/∂t = −∇·(Q C) − δ(Ω C)`. Constant `C` stays constant, and
//! the inventory `Σ H_z C` is conserved, only if the fluxes that move the
//! tracer are exactly the fluxes that move the water (Shchepetkin &
//! McWilliams 2005, §3; ROMS `set_depth`/`omega`).
//!
//! Under mode splitting the free surface is moved by the barotropic pass:
//! `η̄ − ηⁿ = −Δt·∇·DU_avg2` ([`crate::time::BarotropicTransport`]). The layer
//! transports of every 3D stage are therefore built as (ROMS `set_massflux`,
//! corrector step):
//!
//! 1. `Q_l = H_z,l u_l` at the nodes, and the central average
//!    `{{Q_l}}·n` at the element faces (at open boundaries the interior's
//!    `Q_l·n`, at walls none; [`crate::solver::rhs::boundary_3d`]);
//! 2. corrected by the layer's share of the difference to the barotropic
//!    transport, `Q_l += Δσ_l (DU_avg2 − Σ_m Q_m)` at the nodes and on the
//!    faces, so that `Σ_l Q_l = DU_avg2` exactly. The dissipative part of the
//!    2D face flux is shared out the same way;
//! 3. `Ω` integrated up from the bed, with the layer's share of the barotropic
//!    `∂η/∂t` and its volume source `s_l` from rivers ([`crate::source::river`]):
//!    `Ω_{l+1/2} = Ω_{l−1/2} − ∇·Q_l − Δσ_l ∂η/∂t + s_l`.
//!
//! The DG divergence ([`transport_divergence_element`]) is linear in the nodal
//! transport and the face flux, and its volume term takes the 2D module's
//! metric form ([`MetricForm`]: on general quadrilaterals the split forms'
//! averaged metric is a different operator from the collocated form's), so
//! `Σ_l ∇·Q_l = ∇·DU_avg2 = Σ_l s_l − ∂η/∂t`
//! and `Ω` at the surface vanishes to round-off. That holds wherever the barotropic pass
//! keeps the nodal identity and its 2D module has no mass source of its own.
//!
//! **Volume sources.** Whatever `Ω` leaves at the surface is water that no
//! transport carried: the 2D module's own mass sources (a nesting band or
//! sponge relaxing the level, `SourceTerm2D::changes_mass`), and in the
//! elements described below the nodal residual of the pass. The layers take
//! it as a volume source in proportion to `Δσ_l`, which closes `Ω`
//! ([`LayerTransport::volume_source`]), and the tracers get that water at
//! the concentration it joins ([`LayerTransport::add_volume_sources`]), so it
//! changes no concentration. Without that term a relaxed level thickened the
//! layers with no tracer in the water: at Mausund a uniform 10 °C fell to
//! 8.5 °C within an hour (TODO P1.3).
//!
//! **Wetting and drying.** `WetDry` elements with a shallow node move their
//! mass on GLL subcells instead (finite volumes with Audusse et al.'s
//! hydrostatic reconstruction, which closes the interface to a dry bank;
//! [`crate::solver::rhs::subcells`]). The pass reports and averages the mass
//! flux through every subcell interface, and in the elements it moved on
//! the subcells throughout, the layers move through the same interfaces,
//! each with its share of the barotropic flux ([`LayerTransport::compute`],
//! [`subcell_divergence_element`]). So the nodal identity holds there too,
//! a dry bank passes no layer volume, and such a shore element exchanges no
//! baroclinic transport within itself (as ROMS masks its wet–dry cells). A thin column exchanges no
//! baroclinic transport across faces either, as the pressure gradient exerts
//! no force between thin columns. Where the pass keeps only element balances
//! (an element that switched to the subcells during the step, or one the
//! positivity limiter changed, see `BarotropicTransport`), the surface
//! residual is such a volume source too. Besides the element's true source
//! it integrates to zero over the element (the element balance), so every
//! layer's continuity still holds for the element as a whole; the mode
//! splitter carries the tracers of those elements as element means per
//! level ([`from_inventory`]), and their volume sources bring the element's
//! mean concentration: constant and conservative there too.
//!
//! [`apply_tracer_transport_3d`] then advects a tracer with these fluxes in
//! inventory form: in split form within the elements
//! ([`advective_divergence_element`], which the 3D model's energy balance
//! over sloping σ-levels needs; central on the subcell interfaces of a
//! subcell element), upwind in `C` on the face fluxes, and on
//! `Ω` by one of
//! the [`VerticalAdvection`] schemes (by default fourth-order Akima under a
//! TVD limiter, [`VerticalAdvection::LimitedAkima`]).
//!
//! # Momentum
//!
//! The horizontal velocity is advected with the same fluxes, also in
//! inventory form ([`apply_momentum_transport_3d`]; ROMS `rhs3d`/`step3d_uv`):
//!
//! ```text
//!     ∂(H_z u)_l/∂t = −∇·(Q_l u_l) − (Ω_{l+1/2} u_{l+1/2} − Ω_{l−1/2} u_{l−1/2}) + …
//! ```
//!
//! in split form within the elements (by default, [`MomentumAdvectionForm`]),
//! upwind in `u` on the face fluxes and (by default) centred on `Ω`. A velocity that is
//! uniform in space changes its inventory exactly as the layer thickness
//! changes, and the layer momentum `∫ H_z,l u_l` is changed by the advection
//! only through open boundaries. In the split form with centred `Ω` fluxes
//! the advection changes the kinetic energy only through open boundaries and
//! by the upwind faces' dissipation: it cannot feed a grid-scale
//! instability, as the conservative form's aliasing did (P2 lock exchange
//! without horizontal viscosity: NaN within 2 h in conservative form; fronts
//! at Fr 0.46 in split form).

use crate::mesh::Mesh2D;
use crate::mesh::data::Bathymetry2D;
use crate::mesh::data::BoundaryTag;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::SWEFormulation2D;
use crate::solver::core::blocks::{Pooled, for_each_block, max_over_blocks};
use crate::solver::rhs::advection_3d::{TracerBCContext3D, TracerBoundaryCondition3D};
use crate::solver::rhs::boundary_3d::{Boundaries3D, Exterior3D, ExteriorField, FaceExterior};
use crate::solver::rhs::subcells::{
    line_metric, line_node, subcell_interfaces, subcell_slot, telescoped_interfaces,
};
use crate::solver::state::Solution3D;
use crate::source::RiverInflow;
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

/// How the curvilinear metric enters the volume term of a horizontal flux
/// divergence. The 3D kernels take the 2D module's form
/// ([`Self::of`]), so that the layer continuity is the barotropic mass
/// equation's operator and `Σ_l ∇·Q_l = ∇·DU_avg2` holds node by node (see
/// the module docs). On a parallelogram the metric is constant and the two
/// are the same operator; on a general quadrilateral they differ by the
/// aliasing of the metric times the flux.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MetricForm {
    /// The collocated strong form `J⁻¹[D_r(J∇r·q) + D_s(J∇s·q)]`
    /// (`SWEFormulation2D::Standard`).
    #[default]
    Conservative,
    /// Two-point fluxes with the averaged metric,
    /// `J⁻¹ Σ_j [D_r,ij {{J∇r}}_ij + D_s,ij {{J∇s}}_ij]·(q_i + q_j)`: the
    /// mass flux of the split forms (`EntropyConservative`, `EntropyStable`,
    /// `WetDry`; Wintermeyer et al. 2017).
    Averaged,
}

impl MetricForm {
    /// The form of the 2D module's mass equation under `formulation`.
    pub fn of(formulation: SWEFormulation2D) -> Self {
        match formulation {
            SWEFormulation2D::Standard => Self::Conservative,
            SWEFormulation2D::EntropyConservative
            | SWEFormulation2D::EntropyStable
            | SWEFormulation2D::WetDry => Self::Averaged,
        }
    }

    /// Whether element `k` takes the averaged two-point volume term (on a
    /// parallelogram it is the conservative one, which is cheaper).
    #[inline]
    pub(crate) fn averaged_on(self, geom: &GeometricFactors2D, k: usize) -> bool {
        self == Self::Averaged && !geom.element_is_affine(k)
    }
}

/// The DG divergence of a transport `q = (hu, hv)` on element `k`, in the
/// strong form of the 2D shallow-water kernel with the volume term in
/// `metric`'s form (conservative shown):
///
/// ```text
///     J⁻¹[Dr·(J∇r·q) + Ds·(J∇s·q)] − J⁻¹ Σ_f LIFT_f sJ_f (q·n − F*)
/// ```
///
/// `hu`, `hv` are the element's nodal values; `face` holds the numerical flux
/// `F*` out of every face node, `face · n_face_nodes + fi`. The result is
/// linear in `(hu, hv, face)`, which the layer transports rely on.
pub fn transport_divergence_element(
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    metric: MetricForm,
    k: usize,
    hu: &[f64],
    hv: &[f64],
    face: &[f64],
    div: &mut [f64],
) {
    let nn = ops.n_nodes;
    if metric.averaged_on(geom, k) {
        // ½ Σ_j D_ij (Ja_i + Ja_j)·(q_i + q_j)
        for (i, d) in div.iter_mut().enumerate() {
            let (ar_i, as_i) = geom.contravariant(k, i);
            let (mut dr, mut ds) = (0.0, 0.0);
            for j in 0..nn {
                let (ar_j, as_j) = geom.contravariant(k, j);
                let (qx, qy) = (hu[i] + hu[j], hv[i] + hv[j]);
                dr += ops.dr[(i, j)] * ((ar_i.0 + ar_j.0) * qx + (ar_i.1 + ar_j.1) * qy);
                ds += ops.ds[(i, j)] * ((as_i.0 + as_j.0) * qx + (as_i.1 + as_j.1) * qy);
            }
            *d = 0.5 * geom.jacobian_inv(k, i) * (dr + ds);
        }
    } else {
        for (i, d) in div.iter_mut().enumerate() {
            let (mut dr, mut ds) = (0.0, 0.0);
            for j in 0..nn {
                let ((ar_x, ar_y), (as_x, as_y)) = geom.contravariant(k, j);
                let (fr, fs) = (ar_x * hu[j] + ar_y * hv[j], as_x * hu[j] + as_y * hv[j]);
                dr += ops.dr[(i, j)] * fr;
                ds += ops.ds[(i, j)] * fs;
            }
            *d = geom.jacobian_inv(k, i) * (dr + ds);
        }
    }
    subtract_face_lift(ops, geom, k, hu, hv, face, div);
}

/// The face part of the strong-form divergence, `− J⁻¹ Σ_f LIFT_f sJ_f (q·n −
/// F*)`, subtracted from `div`.
fn subtract_face_lift(
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    k: usize,
    hu: &[f64],
    hv: &[f64],
    face: &[f64],
    div: &mut [f64],
) {
    let nfn = ops.n_face_nodes;
    for f in 0..4 {
        let f_star = &face[f * nfn..(f + 1) * nfn];
        for (fi, &node) in ops.face_nodes[f].iter().enumerate() {
            let (nx, ny) = geom.normal(k, f, fi);
            let scale = geom.lift_scale(k, f, fi, node);
            let jump = nx * hu[node] + ny * hv[node] - f_star[fi];
            for (i, d) in div.iter_mut().enumerate() {
                *d -= scale * ops.lift[f][(i, fi)] * jump;
            }
        }
    }
}

/// The divergence of a transport on the GLL subcells of element `k`, the
/// form of the 2D wet/dry kernel's subcell finite volumes
/// ([`crate::solver::rhs::subcells`]):
///
/// ```text
///     J_i⁻¹ Σ_dir (q_{a+½} − q_{a−½}) / w_a  − J⁻¹ Σ_f LIFT_f sJ_f (q·n − F*),
/// ```
///
/// with `q_{a+½}` the flux through each interior interface (`interfaces`, one
/// per subcell slot, already along the interface's contravariant direction)
/// and, at the two ends of a line, the node's own `m·(hu, hv)` with its
/// contravariant vector `m`, which the face term replaces by `F*` (`face`).
/// Linear in `(hu, hv, interfaces, face)`. With the 2D kernel's interface
/// mass fluxes and `F*_h` it is that kernel's mass tendency node by node.
pub fn subcell_divergence_element(
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    k: usize,
    hu: &[f64],
    hv: &[f64],
    interfaces: &[f64],
    face: &[f64],
    div: &mut [f64],
) {
    let n1 = ops.n_1d;
    let w = &ops.weights_1d;
    div.fill(0.0);
    let mut metric = [(0.0, 0.0); MAX_N_1D];
    for dir in 0..2 {
        for line in 0..n1 {
            line_metric(ops, geom, k, dir, line, &mut metric[..n1]);
            let end_flux = |a: usize| {
                let node = line_node(n1, dir, line, a);
                metric[a].0 * hu[node] + metric[a].1 * hv[node]
            };
            for a in 0..n1 {
                let right = if a + 1 < n1 {
                    interfaces[subcell_slot(n1, dir, line, a)]
                } else {
                    end_flux(a)
                };
                let left = if a > 0 {
                    interfaces[subcell_slot(n1, dir, line, a - 1)]
                } else {
                    end_flux(a)
                };
                div[line_node(n1, dir, line, a)] += (right - left) / w[a];
            }
        }
    }
    for (i, d) in div.iter_mut().enumerate() {
        *d *= geom.jacobian_inv(k, i);
    }
    subtract_face_lift(ops, geom, k, hu, hv, face, div);
}

/// Largest number of GLL nodes per direction of the subcell kernels' stack
/// buffers (polynomial order 15).
const MAX_N_1D: usize = 16;

/// The DG divergence of the advective flux `q φ` of a field `φ` carried by
/// the transport `q = (hu, hv)` on element `k`, with the volume term in
/// split (flux-differencing) form:
///
/// ```text
///     J⁻¹ Σ_j [Dr_ij ({{J∇r·q}}_ij) + Ds_ij ({{J∇s·q}}_ij)] 2{{φ}}_ij
///         − J⁻¹ Σ_f LIFT_f sJ_f (q·n φ − F*),
/// ```
///
/// `{{a}}_ij = ½(a_i + a_j)`. It is `½[∇·(qφ) + q·∇φ + φ∇·q]` at the nodes
/// (Kennedy & Gruber 2008; Gassner 2013 for the GLL summation-by-parts
/// property that makes it conservative), where
/// [`transport_divergence_element`] of `qφ` is `∇·(qφ)` alone, and the two
/// differ by aliasing only. With `φ` constant both are `φ∇·q`, so the
/// continuity of the layers (which keeps the conservative form) still
/// keeps a constant field constant.
///
/// Why the split form: the 3D model's energy balance needs the advection of
/// the background stratification to be the negative adjoint of the
/// baroclinic pressure gradient's `ρ∇z` term (see
/// [`crate::solver::rhs::baroclinic`]), and under GLL summation by parts
/// that holds for this form, not for the conservative one. Over a
/// stratified seamount at rest the conservative form let round-off grow
/// tenfold every 6 h (`examples/seamount_3d.rs`). For the momentum it is
/// what keeps the kinetic energy (see [`MomentumAdvectionForm::Split`]).
///
/// That is the [`MetricForm::Conservative`] pairing, `{{J∇r·q}}_ij`; with
/// [`MetricForm::Averaged`] the contravariant transport of a pair is
/// `{{J∇r}}_ij·{{q}}_ij` instead (the 2D split forms' mass flux). Either way
/// a constant `φ` gives `φ` times [`transport_divergence_element`] of `q` in
/// the same form, the layer continuity.
///
/// `hu`, `hv` and `phi` are the element's nodal values, `face` the
/// numerical flux `F*` of `qφ` out of every face node, `fr`, `fs` scratch
/// of `n_nodes` values.
pub fn advective_divergence_element(
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    metric: MetricForm,
    k: usize,
    hu: &[f64],
    hv: &[f64],
    phi: &[f64],
    face: &[f64],
    fr: &mut [f64],
    fs: &mut [f64],
    div: &mut [f64],
) {
    let (nn, nfn) = (ops.n_nodes, ops.n_face_nodes);
    if metric.averaged_on(geom, k) {
        // 2{{Ja}}·{{q}} = ½(Ja_i + Ja_j)·(q_i + q_j)
        for (i, d) in div.iter_mut().enumerate() {
            let (ar_i, as_i) = geom.contravariant(k, i);
            let mut sum = 0.0;
            for j in 0..nn {
                let (ar_j, as_j) = geom.contravariant(k, j);
                let (qx, qy) = (hu[i] + hu[j], hv[i] + hv[j]);
                let flux = 0.5
                    * (ops.dr[(i, j)] * ((ar_i.0 + ar_j.0) * qx + (ar_i.1 + ar_j.1) * qy)
                        + ops.ds[(i, j)] * ((as_i.0 + as_j.0) * qx + (as_i.1 + as_j.1) * qy));
                sum += flux * (phi[i] + phi[j]);
            }
            *d = 0.5 * geom.jacobian_inv(k, i) * sum;
        }
    } else {
        for j in 0..nn {
            let ((ar_x, ar_y), (as_x, as_y)) = geom.contravariant(k, j);
            fr[j] = ar_x * hu[j] + ar_y * hv[j];
            fs[j] = as_x * hu[j] + as_y * hv[j];
        }
        for (i, d) in div.iter_mut().enumerate() {
            let mut sum = 0.0;
            for j in 0..nn {
                let flux = ops.dr[(i, j)] * (fr[i] + fr[j]) + ops.ds[(i, j)] * (fs[i] + fs[j]);
                sum += flux * (phi[i] + phi[j]);
            }
            *d = 0.5 * geom.jacobian_inv(k, i) * sum;
        }
    }
    for f in 0..4 {
        let f_star = &face[f * nfn..(f + 1) * nfn];
        for (fi, &node) in ops.face_nodes[f].iter().enumerate() {
            let (nx, ny) = geom.normal(k, f, fi);
            let scale = geom.lift_scale(k, f, fi, node);
            let jump = (nx * hu[node] + ny * hv[node]) * phi[node] - f_star[fi];
            for (i, d) in div.iter_mut().enumerate() {
                *d -= scale * ops.lift[f][(i, fi)] * jump;
            }
        }
    }
}

/// Buffers of the all-levels kernels ([`transport_divergence_levels`],
/// [`advective_divergence_levels`]) for one element.
pub struct LevelScratch {
    /// The contravariant transports `J∇r·q`, `J∇s·q`, `[node][level]`.
    fr: Vec<f64>,
    fs: Vec<f64>,
    /// One value per level.
    acc: Vec<f64>,
}

impl LevelScratch {
    /// Buffers for elements of `n_nodes` nodes and `n_levels` levels.
    pub fn new(n_nodes: usize, n_levels: usize) -> Self {
        Self {
            fr: vec![0.0; n_nodes * n_levels],
            fs: vec![0.0; n_nodes * n_levels],
            acc: vec![0.0; n_levels],
        }
    }

    fn fits(&self, n_nodes: usize, n_levels: usize) -> bool {
        self.fr.len() == n_nodes * n_levels && self.acc.len() == n_levels
    }
}

/// [`transport_divergence_element`] of every level of element `k` at once:
/// `hu`, `hv` and `div` are the element's `[node][level]` blocks (the layout
/// of [`Solution3D`]) and `face` its `[face node][level]` block (that of
/// [`LayerTransport::face`]).
///
/// Level by level the operations and their order are those of
/// [`transport_divergence_element`], so the result is the same to the bit;
/// the levels are the innermost loop, contiguous, which vectorises, and the
/// metric of each node pair is read once for all levels instead of once per
/// level (the 3D kernels' per-level loops gathered every value at a stride
/// of the level count).
#[allow(clippy::too_many_arguments)]
pub fn transport_divergence_levels(
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    metric: MetricForm,
    k: usize,
    hu: &[f64],
    hv: &[f64],
    face: &[f64],
    div: &mut [f64],
    scratch: &mut LevelScratch,
) {
    let nn = ops.n_nodes;
    let nl = hu.len() / nn;
    debug_assert!(scratch.fits(nn, nl));
    let LevelScratch { fr, fs, acc } = scratch;
    let acc = &mut acc[..nl];
    if metric.averaged_on(geom, k) {
        for i in 0..nn {
            let (ar_i, as_i) = geom.contravariant(k, i);
            let (hu_i, hv_i) = (level_row(hu, i, nl), level_row(hv, i, nl));
            let d = &mut div[i * nl..(i + 1) * nl];
            d.fill(0.0);
            acc.fill(0.0);
            for j in 0..nn {
                let (ar_j, as_j) = geom.contravariant(k, j);
                let (rx, ry) = (ar_i.0 + ar_j.0, ar_i.1 + ar_j.1);
                let (sx, sy) = (as_i.0 + as_j.0, as_i.1 + as_j.1);
                let (dr, ds) = (ops.dr[(i, j)], ops.ds[(i, j)]);
                let (hu_j, hv_j) = (level_row(hu, j, nl), level_row(hv, j, nl));
                let (hu_i, hv_i, d, acc) = (&hu_i[..nl], &hv_i[..nl], &mut d[..nl], &mut acc[..nl]);
                for l in 0..nl {
                    let (qx, qy) = (hu_i[l] + hu_j[l], hv_i[l] + hv_j[l]);
                    d[l] += dr * (rx * qx + ry * qy);
                    acc[l] += ds * (sx * qx + sy * qy);
                }
            }
            let scale = 0.5 * geom.jacobian_inv(k, i);
            for (d, &a) in d.iter_mut().zip(acc.iter()) {
                *d = scale * (*d + a);
            }
        }
    } else {
        contravariant_levels(geom, k, nl, hu, hv, fr, fs);
        for i in 0..nn {
            let d = &mut div[i * nl..(i + 1) * nl];
            d.fill(0.0);
            acc.fill(0.0);
            for j in 0..nn {
                let (dr, ds) = (ops.dr[(i, j)], ops.ds[(i, j)]);
                let (fr_j, fs_j) = (level_row(fr, j, nl), level_row(fs, j, nl));
                let (d, acc) = (&mut d[..nl], &mut acc[..nl]);
                for l in 0..nl {
                    d[l] += dr * fr_j[l];
                    acc[l] += ds * fs_j[l];
                }
            }
            let scale = geom.jacobian_inv(k, i);
            for (d, &a) in d.iter_mut().zip(acc.iter()) {
                *d = scale * (*d + a);
            }
        }
    }
    subtract_face_lift_levels(ops, geom, k, hu, hv, None, face, div, acc);
}

/// [`advective_divergence_element`] of every level of element `k` at once,
/// for `N` fields carried by the same transport (u and v, T and S, k and ψ),
/// with the blocks of [`transport_divergence_levels`] (each `phi[t]` as `hu`,
/// each `face[t]` as its `face`, each `div[t]` as its `div`): for each field
/// the same to the bit, level by level. The fields share the metric of each
/// node pair and its transport flux.
#[allow(clippy::too_many_arguments)]
pub fn advective_divergence_levels<const N: usize>(
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    metric: MetricForm,
    k: usize,
    hu: &[f64],
    hv: &[f64],
    phi: [&[f64]; N],
    face: [&[f64]; N],
    mut div: [&mut [f64]; N],
    scratch: &mut LevelScratch,
) {
    let nn = ops.n_nodes;
    let nl = hu.len() / nn;
    debug_assert!(scratch.fits(nn, nl));
    let LevelScratch { fr, fs, acc } = scratch;
    if metric.averaged_on(geom, k) {
        // 2{{Ja}}·{{q}} = ½(Ja_i + Ja_j)·(q_i + q_j)
        for i in 0..nn {
            let (ar_i, as_i) = geom.contravariant(k, i);
            let (hu_i, hv_i) = (level_row(hu, i, nl), level_row(hv, i, nl));
            let phi_i = phi.map(|phi| level_row(phi, i, nl));
            let mut d = div.each_mut().map(|div| &mut div[i * nl..][..nl]);
            d.iter_mut().for_each(|d| d.fill(0.0));
            for j in 0..nn {
                let (ar_j, as_j) = geom.contravariant(k, j);
                let (rx, ry) = (ar_i.0 + ar_j.0, ar_i.1 + ar_j.1);
                let (sx, sy) = (as_i.0 + as_j.0, as_i.1 + as_j.1);
                let (dr, ds) = (ops.dr[(i, j)], ops.ds[(i, j)]);
                let (hu_j, hv_j) = (level_row(hu, j, nl), level_row(hv, j, nl));
                let phi_j = phi.map(|phi| level_row(phi, j, nl));
                for l in 0..nl {
                    let (qx, qy) = (hu_i[l] + hu_j[l], hv_i[l] + hv_j[l]);
                    let flux = 0.5 * (dr * (rx * qx + ry * qy) + ds * (sx * qx + sy * qy));
                    for t in 0..N {
                        d[t][l] += flux * (phi_i[t][l] + phi_j[t][l]);
                    }
                }
            }
            let scale = 0.5 * geom.jacobian_inv(k, i);
            d.iter_mut()
                .flat_map(|d| d.iter_mut())
                .for_each(|d| *d *= scale);
        }
    } else {
        contravariant_levels(geom, k, nl, hu, hv, fr, fs);
        for i in 0..nn {
            let (fr_i, fs_i) = (level_row(fr, i, nl), level_row(fs, i, nl));
            let phi_i = phi.map(|phi| level_row(phi, i, nl));
            let mut d = div.each_mut().map(|div| &mut div[i * nl..][..nl]);
            d.iter_mut().for_each(|d| d.fill(0.0));
            for j in 0..nn {
                let (dr, ds) = (ops.dr[(i, j)], ops.ds[(i, j)]);
                let (fr_j, fs_j) = (level_row(fr, j, nl), level_row(fs, j, nl));
                let phi_j = phi.map(|phi| level_row(phi, j, nl));
                for l in 0..nl {
                    let flux = dr * (fr_i[l] + fr_j[l]) + ds * (fs_i[l] + fs_j[l]);
                    for t in 0..N {
                        d[t][l] += flux * (phi_i[t][l] + phi_j[t][l]);
                    }
                }
            }
            let scale = 0.5 * geom.jacobian_inv(k, i);
            d.iter_mut()
                .flat_map(|d| d.iter_mut())
                .for_each(|d| *d *= scale);
        }
    }
    for t in 0..N {
        subtract_face_lift_levels(
            ops,
            geom,
            k,
            hu,
            hv,
            Some(phi[t]),
            face[t],
            div[t],
            &mut acc[..nl],
        );
    }
}

/// Node `i`'s levels of an element's `[node][level]` block.
#[inline]
fn level_row(values: &[f64], i: usize, nl: usize) -> &[f64] {
    &values[i * nl..][..nl]
}

/// The contravariant transports `J∇r·q`, `J∇s·q` of element `k`'s
/// `[node][level]` blocks `hu`, `hv` into `fr`, `fs`.
fn contravariant_levels(
    geom: &GeometricFactors2D,
    k: usize,
    nl: usize,
    hu: &[f64],
    hv: &[f64],
    fr: &mut [f64],
    fs: &mut [f64],
) {
    for (j, (((fr, fs), hu), hv)) in fr
        .chunks_exact_mut(nl)
        .zip(fs.chunks_exact_mut(nl))
        .zip(hu.chunks_exact(nl))
        .zip(hv.chunks_exact(nl))
        .enumerate()
    {
        let ((ar_x, ar_y), (as_x, as_y)) = geom.contravariant(k, j);
        for l in 0..nl {
            fr[l] = ar_x * hu[l] + ar_y * hv[l];
            fs[l] = as_x * hu[l] + as_y * hv[l];
        }
    }
}

/// [`subtract_face_lift`] of every level (of the advective flux `q φ` if
/// `phi`), with the blocks of [`transport_divergence_levels`]; `jump` holds
/// one value per level.
#[allow(clippy::too_many_arguments)]
fn subtract_face_lift_levels(
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    k: usize,
    hu: &[f64],
    hv: &[f64],
    phi: Option<&[f64]>,
    face: &[f64],
    div: &mut [f64],
    jump: &mut [f64],
) {
    let nfn = ops.n_face_nodes;
    let nl = jump.len();
    for f in 0..4 {
        for (fi, &node) in ops.face_nodes[f].iter().enumerate() {
            let (nx, ny) = geom.normal(k, f, fi);
            let scale = geom.lift_scale(k, f, fi, node);
            let f_star = &face[(f * nfn + fi) * nl..(f * nfn + fi + 1) * nl];
            let (hu, hv) = (
                &hu[node * nl..(node + 1) * nl],
                &hv[node * nl..(node + 1) * nl],
            );
            match phi {
                None => {
                    for l in 0..nl {
                        jump[l] = nx * hu[l] + ny * hv[l] - f_star[l];
                    }
                }
                Some(phi) => {
                    let phi = &phi[node * nl..(node + 1) * nl];
                    for l in 0..nl {
                        jump[l] = (nx * hu[l] + ny * hv[l]) * phi[l] - f_star[l];
                    }
                }
            }
            for (i, d) in div.chunks_exact_mut(nl).enumerate() {
                let lift = scale * ops.lift[f][(i, fi)];
                for (d, &jump) in d.iter_mut().zip(jump.iter()) {
                    *d -= lift * jump;
                }
            }
        }
    }
}

/// The barotropic side of one baroclinic step, as the layer transports need
/// it: `DU_avg2` and the rate of the free surface over the step.
#[derive(Clone, Copy)]
pub struct BarotropicFlux<'a> {
    /// Length of the baroclinic step (s).
    pub dt: f64,
    /// The longest forward-Euler step in the 3D stages' scheme (s): `dt`
    /// over its SSP coefficient (`dt` for SSP-RK3, `dt/2` for SSP-RK(4,3)).
    /// A stage's explicit Courant number is taken for it.
    pub euler_dt: f64,
    /// Nodal barotropic transport `DU_avg2` (m²/s), `[element][node]`.
    pub hu: &'a [f64],
    pub hv: &'a [f64],
    /// Barotropic mass flux out of every element face node (m²/s),
    /// `(k · 4 + face) · n_face_nodes + fi`.
    pub face: &'a [f64],
    /// Barotropic mass flux through every subcell interface,
    /// `k · n_sub + slot` ([`crate::solver::rhs::subcells`]); NaN in the
    /// elements where the pass did not move the mass on the subcells
    /// throughout ([`crate::time::BarotropicTransport::subcell`]).
    pub subcell: &'a [f64],
    /// `∂η/∂t = (η̄ − ηⁿ)/Δt` over the step (m/s), `[element][node]`.
    pub eta_rate: &'a [f64],
    /// The rivers of the step, if any: volume sources of the layers, so
    /// that `∂η/∂t = −∇·DU_avg2 + Σ Q̄/A_k` (see [`crate::source::river`]).
    pub rivers: Option<RiverInflow<'a>>,
    /// The elements whose fields the mode splitter carries as element means
    /// per level (`[element]`; empty for none): there the volume source of
    /// [`LayerTransport::volume_source`] brings the element's mean
    /// concentration per level ([`LayerTransport::add_volume_sources`]).
    pub element_means: &'a [bool],
}

/// Layer transports and `Ω` of one 3D stage (see the module docs).
pub struct LayerTransport {
    n_levels: usize,
    /// Layer transport `Q_l = H_z u` at the nodes (m²/s), corrected to
    /// `DU_avg2`; layout of [`Solution3D`], `[element][node][level]`.
    pub hu: Vec<f64>,
    pub hv: Vec<f64>,
    /// Layer volume flux out of every element face node (m²/s),
    /// `((k · 4 + face) · n_face_nodes + fi) · n_levels + level`.
    pub face: Vec<f64>,
    /// Layer volume flux through every subcell interface of the subcell
    /// elements, along the interface's contravariant direction (as the 2D
    /// kernel's), `((k · n_sub + slot) · n_levels + level`
    /// ([`crate::solver::rhs::subcells`]; zero in other elements).
    pub subcell: Vec<f64>,
    /// Elements whose layers move through their subcell interfaces: those
    /// with a column thinner than the minimum column depth, where the 2D
    /// wet/dry kernel moves the mass on the GLL subcells (see
    /// [`Self::compute`]).
    pub subcell_elements: Vec<bool>,
    /// `Ω` at the w-points (m/s), `n_levels + 1` per column, bed first.
    pub omega: Vec<f64>,
    /// The volume source of every column that no transport carries (m/s,
    /// `[element][node]`): the surface residual of `Ω`, negated, which the
    /// layers take in proportion to `Δσ_l` (see [`Self::compute`]). Where
    /// the barotropic pass keeps the nodal identity it is round-off, except
    /// where its 2D module has a mass source of its own (the nesting band's
    /// relaxation of `η`, `crate::boundary::NestingRelaxation2D`). The
    /// tracers get its water through [`Self::add_volume_sources`].
    pub volume_source: Vec<f64>,
    /// The longest forward-Euler step of the stages over the water depth,
    /// `Δt_E/D` at the nodes (s/m, `[element][node]`; zero without a
    /// barotropic pass, in a column with no water): a layer's explicit
    /// outflow Courant number is `Δt_E/D · |Ω|/Δσ_l`. Set by
    /// [`Self::compute`]; the vertical advection bounds the water an end layer
    /// sheds by it ([`VerticalAdvection::LimitedAkima`]).
    pub dt_over_depth: Vec<f64>,
    /// σ-thickness `Δσ_l` of the layers (uniform until the first
    /// [`Self::compute`]).
    pub d_sigma: Vec<f64>,
    /// Largest `|Ω|` at the surface before the residual was spread over the
    /// column (m/s): round-off where the barotropic pass keeps the nodal
    /// identity `η̄ − ηⁿ = −Δt∇·DU_avg2`.
    pub surface_residual: f64,
    /// The volume form of every horizontal divergence of these transports:
    /// the 2D module's ([`MetricForm::of`]; see [`Self::with_metric`]).
    pub metric: MetricForm,
}

impl LayerTransport {
    /// Buffers for `n_elements` elements of `ops` and `n_levels` layers.
    pub fn new(n_elements: usize, ops: &DGOperators2D, n_levels: usize) -> Self {
        let (nn, nfn) = (ops.n_nodes, ops.n_face_nodes);
        Self {
            n_levels,
            hu: vec![0.0; n_elements * nn * n_levels],
            hv: vec![0.0; n_elements * nn * n_levels],
            face: vec![0.0; n_elements * 4 * nfn * n_levels],
            subcell: vec![0.0; n_elements * subcell_interfaces(ops.n_1d) * n_levels],
            subcell_elements: vec![false; n_elements],
            omega: vec![0.0; n_elements * nn * (n_levels + 1)],
            volume_source: vec![0.0; n_elements * nn],
            dt_over_depth: vec![0.0; n_elements * nn],
            d_sigma: vec![1.0 / n_levels as f64; n_levels],
            surface_residual: 0.0,
            metric: MetricForm::default(),
        }
    }

    /// The transports with `metric`'s volume form (the default is
    /// [`MetricForm::Conservative`], the same on parallelograms).
    pub fn with_metric(mut self, metric: MetricForm) -> Self {
        self.metric = metric;
        self
    }

    /// Number of layers.
    pub fn n_levels(&self) -> usize {
        self.n_levels
    }

    /// Layer transports and `Ω` of `state` (its `η`, `u`, `v`), corrected to
    /// the barotropic transport `barotropic`, with the physical boundaries
    /// `boundaries`.
    ///
    /// With `barotropic = None` they are the state's own transports: `H_z u`
    /// at the nodes and the central average on the faces, uncorrected, with
    /// `Ω` closed by the free-surface rate their divergence implies,
    /// `∂η/∂t = −Σ_l ∇·Q_l`. (The slow forcing of the mode splitter uses them
    /// before the barotropic pass of a step exists.)
    ///
    /// # Wetting and drying
    ///
    /// A column thinner than `min_column_depth` exchanges no baroclinic
    /// transport: across a face with such a column on either side the layers
    /// carry only their share `Δσ_l F*` of the barotropic flux, as the
    /// baroclinic pressure gradient exerts no force between such columns.
    /// An element with a thin column is a subcell element
    /// ([`Self::subcell_elements`]): its layers move through the GLL subcell
    /// interfaces of the 2D wet/dry kernel ([`crate::solver::rhs::subcells`])
    /// instead of the DG volume term, each with its share of the barotropic
    /// flux, `q_l = Δσ_l F̄`, with `F̄` the pass's flux through the interface
    /// (where it moved the element's mass on the subcells throughout; else
    /// the central `m·{{DU_avg2}}`, none beside a thin column). So the layers
    /// sum to the 2D update node by node and `Ω` closes at the surface (the
    /// 2D kernel blocks a dry bank, its hydrostatic reconstruction giving
    /// `F̄ = 0`), and the element exchanges no baroclinic transport within
    /// itself, matching the pressure gradient, which exerts no force within
    /// it (its faces exchange both, with any wet neighbour). Baroclinic
    /// exchange through the subcells, with the matching central-difference
    /// force, kept energy exact but was unstable under a pycnocline: a shore
    /// element of the Frøya coastline mesh grew with an e-folding of 38 min
    /// (TODO P1.3).
    pub fn compute(
        &mut self,
        state: &Solution3D,
        barotropic: Option<BarotropicFlux>,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        sigma: &SigmaGrid,
        bathymetry: &Bathymetry2D,
        boundaries: &Boundaries3D,
        exterior: &Exterior3D,
        min_column_depth: f64,
    ) {
        let (nn, nfn, nl) = (ops.n_nodes, ops.n_face_nodes, self.n_levels);
        assert_eq!(state.n_levels, nl, "layer count of the state");
        let n_elements = state.n_elements;
        let n1 = ops.n_1d;
        let n_sub = subcell_interfaces(n1);
        let d_sigma = sigma.d_sigma();
        self.d_sigma.copy_from_slice(d_sigma);
        let Self {
            hu: layer_hu,
            hv: layer_hv,
            face: layer_face,
            subcell: layer_subcell,
            subcell_elements,
            omega: layer_omega,
            volume_source,
            dt_over_depth,
            metric,
            ..
        } = self;
        let metric = *metric;
        let thin = |flat: usize| state.eta.data[flat] - bathymetry.data[flat] < min_column_depth;
        // The subcell elements: those the pass moved on the subcells
        // throughout (so the layers sum to its update node by node), or
        // without a pass those with a thin column
        for (k, sub) in subcell_elements.iter_mut().enumerate() {
            *sub = match barotropic {
                Some(b) => b.subcell[k * n_sub..(k + 1) * n_sub]
                    .iter()
                    .all(|f| f.is_finite()),
                None => (k * nn..(k + 1) * nn).any(thin),
            };
        }
        let subcell_elements: &[bool] = subcell_elements;

        // 1. Nodal layer transports, corrected to DU_avg2
        for_each_block(
            n_elements,
            [&mut layer_hu[..], &mut layer_hv[..], &mut dt_over_depth[..]],
            || (),
            |_, k, [hu_k, hv_k, dt_k]| {
                let bed = bathymetry.element(ElementIndex::new(k));
                for (i, &b) in bed.iter().enumerate() {
                    let idx = k * nn + i;
                    let depth = state.eta.data[idx] - b;
                    dt_k[i] = match barotropic {
                        Some(flux) if depth > 0.0 => flux.euler_dt / depth,
                        _ => 0.0,
                    };
                    let column = idx * nl..(idx + 1) * nl;
                    let local = i * nl..(i + 1) * nl;
                    let (mut sum_u, mut sum_v) = (0.0, 0.0);
                    for (((hu, hv), (&u, &v)), &ds) in hu_k[local.clone()]
                        .iter_mut()
                        .zip(&mut hv_k[local.clone()])
                        .zip(state.u[column.clone()].iter().zip(&state.v[column]))
                        .zip(d_sigma)
                    {
                        *hu = depth * ds * u;
                        *hv = depth * ds * v;
                        sum_u += *hu;
                        sum_v += *hv;
                    }
                    let Some(barotropic) = barotropic else {
                        continue;
                    };
                    let (corr_u, corr_v) = (barotropic.hu[idx] - sum_u, barotropic.hv[idx] - sum_v);
                    for ((hu, hv), &ds) in hu_k[local.clone()]
                        .iter_mut()
                        .zip(&mut hv_k[local])
                        .zip(d_sigma)
                    {
                        *hu += ds * corr_u;
                        *hv += ds * corr_v;
                    }
                }
            },
        );
        let (layer_hu, layer_hv): (&[f64], &[f64]) = (layer_hu, layer_hv);

        // 2. Face fluxes: central average of the nodal transports, corrected
        // to the barotropic face flux
        for_each_block(
            n_elements,
            [&mut layer_face[..]],
            || (),
            |_, k, [face_k]| {
                let el = ElementIndex::new(k);
                for f in 0..4 {
                    let face_exterior = boundaries.exterior(mesh, el, f);
                    for (fi, &node) in ops.face_nodes[f].iter().enumerate() {
                        let (nx, ny) = geom.normal(k, f, fi);
                        let flat = k * nn + node;
                        let interior = flat * nl;
                        // The profile of the layer fluxes: central between
                        // elements, the interior's at open boundaries (the
                        // shear leaves with the flow) or central with a
                        // nesting parent's, none at walls
                        let across = match face_exterior {
                            FaceExterior::Element(nb) => Across::Node(
                                (nb.element * nn + ops.face_nodes[nb.face][nfn - 1 - fi]) * nl,
                            ),
                            FaceExterior::Open(tag) => match exterior.velocity {
                                Some([u, v]) if u.at(tag, flat, 0).is_some() => {
                                    let depth = state.eta.data[flat] - bathymetry.data[flat];
                                    Across::Parent(u, v, tag, depth)
                                }
                                _ => Across::Node(interior),
                            },
                            FaceExterior::Wall => Across::Nothing,
                        };
                        let local = f * nfn + fi;
                        let fluxes = &mut face_k[local * nl..(local + 1) * nl];
                        // A thin column on either side: no baroclinic
                        // exchange (the barotropic share only)
                        let blocked =
                            thin(flat) || matches!(across, Across::Node(e) if thin(e / nl));
                        let mut sum = 0.0;
                        for (l, flux) in fluxes.iter_mut().enumerate() {
                            let q_in = nx * layer_hu[interior + l] + ny * layer_hv[interior + l];
                            *flux = match across {
                                _ if blocked => 0.0,
                                Across::Node(e) => {
                                    0.5 * (q_in + nx * layer_hu[e + l] + ny * layer_hv[e + l])
                                }
                                Across::Parent(u, v, tag, depth) => {
                                    let u = u.at(tag, flat, l).expect("checked");
                                    let v = v.at(tag, flat, l).expect("checked");
                                    0.5 * (q_in + depth * d_sigma[l] * (nx * u + ny * v))
                                }
                                Across::Nothing => 0.0,
                            };
                            sum += *flux;
                        }
                        let Some(barotropic) = barotropic else {
                            continue;
                        };
                        let corr = barotropic.face[(k * 4 + f) * nfn + fi] - sum;
                        for (flux, &ds) in fluxes.iter_mut().zip(d_sigma) {
                            *flux += ds * corr;
                        }
                    }
                }
            },
        );
        let layer_face: &[f64] = layer_face;

        // 2b. Subcell interface fluxes of the subcell elements: each layer's
        // share of the barotropic flux, none between a column and a thin one
        for_each_block(
            n_elements,
            [&mut layer_subcell[..]],
            || (),
            |_, k, [sub_k]| {
                if !subcell_elements[k] {
                    sub_k.fill(0.0);
                    return;
                }
                let pass = barotropic.and_then(|b| {
                    let fluxes = &b.subcell[k * n_sub..(k + 1) * n_sub];
                    fluxes.iter().all(|f| f.is_finite()).then_some(fluxes)
                });
                let mut metric = [(0.0, 0.0); MAX_N_1D];
                let mut interfaces = [(0.0, 0.0); MAX_N_1D + 1];
                for dir in 0..2 {
                    for line in 0..n1 {
                        let affine = line_metric(ops, geom, k, dir, line, &mut metric[..n1]);
                        telescoped_interfaces(ops, &metric[..n1], affine, &mut interfaces);
                        for a in 0..n1 - 1 {
                            let slot = subcell_slot(n1, dir, line, a);
                            let m = interfaces[a + 1];
                            let (ia, ib) = (
                                k * nn + line_node(n1, dir, line, a),
                                k * nn + line_node(n1, dir, line, a + 1),
                            );
                            // The column transport along the interface
                            let column = |flat: usize| {
                                (0..nl)
                                    .map(|l| {
                                        m.0 * layer_hu[flat * nl + l]
                                            + m.1 * layer_hv[flat * nl + l]
                                    })
                                    .sum::<f64>()
                            };
                            let bar = match pass {
                                Some(fluxes) => fluxes[slot],
                                None if thin(ia) || thin(ib) => 0.0,
                                None => 0.5 * (column(ia) + column(ib)),
                            };
                            for (flux, &ds) in
                                sub_k[slot * nl..(slot + 1) * nl].iter_mut().zip(d_sigma)
                            {
                                *flux = ds * bar;
                            }
                        }
                    }
                }
            },
        );
        let layer_subcell: &[f64] = layer_subcell;

        // 3. Ω from the bed up
        let sigma_w = sigma.sigma_w();
        let residual = max_over_blocks(
            n_elements,
            [&mut layer_omega[..], &mut volume_source[..]],
            || {
                Pooled::take(
                    |s: &OmegaScratch| s.fits(nn, nfn, nl, n_sub),
                    || OmegaScratch::new(nn, nfn, nl, n_sub),
                )
            },
            |scratch, k, [omega_k, source_k]| {
                let OmegaScratch {
                    hu,
                    hv,
                    face,
                    sub,
                    layer_div,
                    div,
                    source,
                    levels,
                } = &mut **scratch;
                // The rivers' volume sources of the layers, per node (their
                // profiles follow each node's depth)
                let rivers = barotropic
                    .and_then(|b| b.rivers)
                    .filter(|r| !r.rivers.in_element(k).is_empty());
                source.fill(0.0);
                // Every layer's divergence, `[node][level]`
                if subcell_elements[k] {
                    for l in 0..nl {
                        for i in 0..nn {
                            hu[i] = layer_hu[(k * nn + i) * nl + l];
                            hv[i] = layer_hv[(k * nn + i) * nl + l];
                        }
                        for (slot, flux) in face.iter_mut().enumerate() {
                            *flux = layer_face[((k * 4 * nfn) + slot) * nl + l];
                        }
                        for (slot, flux) in sub.iter_mut().enumerate() {
                            *flux = layer_subcell[(k * n_sub + slot) * nl + l];
                        }
                        subcell_divergence_element(ops, geom, k, hu, hv, sub, face, layer_div);
                        for (i, &d) in layer_div.iter().enumerate() {
                            div[i * nl + l] = d;
                        }
                    }
                } else {
                    let block = k * nn * nl..(k + 1) * nn * nl;
                    transport_divergence_levels(
                        ops,
                        geom,
                        metric,
                        k,
                        &layer_hu[block.clone()],
                        &layer_hv[block],
                        &layer_face[k * 4 * nfn * nl..(k + 1) * 4 * nfn * nl],
                        div,
                        levels,
                    );
                }
                let mut largest = 0.0_f64;
                for i in 0..nn {
                    let idx = k * nn + i;
                    let div = &div[i * nl..(i + 1) * nl];
                    let rate = match barotropic {
                        Some(barotropic) => barotropic.eta_rate[idx],
                        None => -div.iter().sum::<f64>(),
                    };
                    let omega = &mut omega_k[i * (nl + 1)..(i + 1) * (nl + 1)];
                    if let Some(rivers) = rivers {
                        source.fill(0.0);
                        rivers.add_layer_rates(k, i, source);
                    }
                    omega[0] = 0.0;
                    for l in 0..nl {
                        omega[l + 1] = omega[l] - div[l] - d_sigma[l] * rate + source[l];
                    }
                    let residual = omega[nl];
                    largest = largest.max(residual.abs());
                    // The layers take the residual as a volume source in
                    // proportion to Δσ_l, which closes Ω at the surface
                    for (w, &s) in omega.iter_mut().zip(sigma_w) {
                        *w -= (s + 1.0) * residual;
                    }
                    source_k[i] = -residual;
                }
                largest
            },
        );
        self.surface_residual = residual.max(0.0);
    }

    /// The transports of the control volumes around the w-points of `layers`
    /// (the w-cells), for advecting a field that lives at the w-points (the
    /// turbulence of [`crate::physics::GlsMixing`]; ROMS `gls_corstep`).
    /// `self` must have one level more than `layers`.
    ///
    /// The w-cell of an interior w-point spans from the centre of the layer
    /// below to the centre of the layer above; those of the bed and the
    /// surface are the lower half of the bed layer and the upper half of the
    /// top layer ([`w_cell_thicknesses`]). Each carries half of the
    /// horizontal transport of the two layers it overlaps, nodally and on
    /// the faces, and passes the mean of the layer's two `Ω` through a layer
    /// centre (none through the bed and the surface). Averaging the
    /// continuity of the two layers gives the w-cell's, so a field advected
    /// with these fluxes keeps the constancy and conservation of
    /// [`apply_tracer_transport_3d`].
    pub fn stagger_from(&mut self, layers: &LayerTransport) {
        let nl = layers.n_levels;
        assert_eq!(self.n_levels, nl + 1, "w-cells of {nl} layers");
        let nw = nl + 1;
        self.metric = layers.metric;
        self.subcell_elements
            .copy_from_slice(&layers.subcell_elements);
        let n_elements = self.subcell_elements.len();
        // Every field is `[element][…][level]`: element `k`'s block of the
        // w-cells' is made from its block of the layers'
        fn block(values: &[f64], k: usize, n_elements: usize) -> &[f64] {
            let len = values.len() / n_elements;
            &values[k * len..(k + 1) * len]
        }
        for_each_block(
            n_elements,
            [
                &mut self.hu[..],
                &mut self.hv[..],
                &mut self.face[..],
                &mut self.subcell[..],
                &mut self.omega[..],
            ],
            || (),
            |_, k, [hu, hv, face, subcell, omega]| {
                for (w_cells, layer_values) in [
                    (hu, &layers.hu),
                    (hv, &layers.hv),
                    (face, &layers.face),
                    (subcell, &layers.subcell),
                ] {
                    for (w, l) in w_cells
                        .chunks_exact_mut(nw)
                        .zip(block(layer_values, k, n_elements).chunks_exact(nl))
                    {
                        w[0] = 0.5 * l[0];
                        for j in 1..nl {
                            w[j] = 0.5 * (l[j - 1] + l[j]);
                        }
                        w[nl] = 0.5 * l[nl - 1];
                    }
                }
                for (w, omega) in omega
                    .chunks_exact_mut(nw + 1)
                    .zip(block(&layers.omega, k, n_elements).chunks_exact(nl + 1))
                {
                    w[0] = 0.0;
                    for l in 0..nl {
                        w[l + 1] = 0.5 * (omega[l] + omega[l + 1]);
                    }
                    w[nw] = 0.0;
                }
            },
        );
        w_cell_thicknesses(&layers.d_sigma, &mut self.d_sigma);
        // Each w-cell averages its two layers' continuity, sources included
        self.volume_source.copy_from_slice(&layers.volume_source);
        self.surface_residual = layers.surface_residual;
    }

    /// Add the inventory tendency of [`Self::volume_source`]'s water to
    /// `rhs` (`∂(H C)/∂t` of a field `field` over cells of σ-thickness
    /// `cells`: the layers' `Δσ_l`, or the w-cells' of
    /// [`w_cell_thicknesses`], whose sources these transports' are, see
    /// [`Self::stagger_from`]): `Δσ_c s C`, the water at the concentration
    /// it joins, so the source changes no concentration (constancy) and
    /// adds exactly what its volume holds (conservation).
    ///
    /// `C` is the node's own value, or in the elements of `element_means`
    /// the element's mean per cell, `Σ w_i J_i H_ic C_ic / Σ w_i J_i H_ic`:
    /// there the barotropic pass kept only the element balance, and the
    /// nodal sources hold its nodal residual, which integrates to zero over
    /// the element, besides the element's true source; at the mean
    /// concentration the residual moves no tracer, as in
    /// [`from_inventory`].
    #[allow(clippy::too_many_arguments)]
    pub fn add_volume_sources(
        &self,
        rhs: &mut [f64],
        field: &[f64],
        cells: &[f64],
        eta: &[f64],
        bathymetry: &Bathymetry2D,
        geom: &GeometricFactors2D,
        element_means: &[bool],
    ) {
        let (nn, nc) = (bathymetry.n_nodes, cells.len());
        let n_elements = bathymetry.n_elements;
        for_each_block(
            n_elements,
            [&mut rhs[..n_elements * nn * nc]],
            || (),
            |_, k, [rhs_k]| {
                let source = &self.volume_source[k * nn..(k + 1) * nn];
                let field_k = &field[k * nn * nc..(k + 1) * nn * nc];
                if !element_means.get(k).copied().unwrap_or(false) {
                    for (i, &s) in source.iter().enumerate() {
                        for (c, &dc) in cells.iter().enumerate() {
                            rhs_k[i * nc + c] += dc * s * field_k[i * nc + c];
                        }
                    }
                    return;
                }
                let bed = bathymetry.element(ElementIndex::new(k));
                let depth = |i: usize| eta[k * nn + i] - bed[i];
                for (c, &dc) in cells.iter().enumerate() {
                    let (mut inventory, mut volume) = (0.0, 0.0);
                    for i in 0..nn {
                        let weight = geom.node_mass(k, i) * layer_volume_of(depth(i), dc);
                        inventory += weight * field_k[i * nc + c];
                        volume += weight;
                    }
                    for (i, &s) in source.iter().enumerate() {
                        // An element without water has no mean: its own values
                        let value = if volume > 0.0 {
                            inventory / volume
                        } else {
                            field_k[i * nc + c]
                        };
                        rhs_k[i * nc + c] += dc * s * value;
                    }
                }
            },
        );
    }
}

/// σ-thicknesses of the w-cells of layers of σ-thickness `d_sigma` into
/// `out` (one more): half the bed layer, the mean of each pair of adjacent
/// layers, half the top layer (see [`LayerTransport::stagger_from`]). They
/// sum to one, and a w-cell's thickness is `D` times its value.
pub fn w_cell_thicknesses(d_sigma: &[f64], out: &mut [f64]) {
    let nl = d_sigma.len();
    assert_eq!(out.len(), nl + 1, "w-cells of {nl} layers");
    out[0] = 0.5 * d_sigma[0];
    for j in 1..nl {
        out[j] = 0.5 * (d_sigma[j - 1] + d_sigma[j]);
    }
    out[nl] = 0.5 * d_sigma[nl - 1];
}

/// One element's layers of [`LayerTransport::compute`]: a layer's nodal
/// transport and face fluxes, the divergence of every layer, and the
/// layers' volume sources.
struct OmegaScratch {
    hu: Vec<f64>,
    hv: Vec<f64>,
    face: Vec<f64>,
    /// A layer's subcell interface fluxes
    sub: Vec<f64>,
    /// A layer's divergence (subcell elements)
    layer_div: Vec<f64>,
    /// Every layer's divergence, `[node][level]`
    div: Vec<f64>,
    source: Vec<f64>,
    levels: LevelScratch,
}

impl OmegaScratch {
    fn new(nn: usize, nfn: usize, nl: usize, n_sub: usize) -> Self {
        Self {
            hu: vec![0.0; nn],
            hv: vec![0.0; nn],
            face: vec![0.0; 4 * nfn],
            sub: vec![0.0; n_sub],
            layer_div: vec![0.0; nn],
            div: vec![0.0; nn * nl],
            source: vec![0.0; nl],
            levels: LevelScratch::new(nn, nl),
        }
    }

    fn fits(&self, nn: usize, nfn: usize, nl: usize, n_sub: usize) -> bool {
        self.hu.len() == nn
            && self.face.len() == 4 * nfn
            && self.sub.len() == n_sub
            && self.div.len() == nn * nl
            && self.source.len() == nl
    }
}

/// What lies across a face for the layer volume fluxes.
#[derive(Clone, Copy)]
enum Across<'a> {
    /// A node's layer transports, at this offset.
    Node(usize),
    /// A nesting parent's velocity, over the interior's depth.
    Parent(
        ExteriorField<'a>,
        ExteriorField<'a>,
        crate::mesh::data::BoundaryTag,
        f64,
    ),
    /// A wall.
    Nothing,
}

/// Layer thickness `H_z = D·Δσ` (m), floored at the same small positive value
/// as the 3D advection kernels, so that dividing an inventory by it is safe.
#[inline]
pub(crate) fn layer_thickness_of(depth: f64, d_sigma: f64) -> f64 {
    (depth * d_sigma).max(crate::solver::rhs::advection_3d::MIN_LAYER_THICKNESS)
}

/// The volume of a cell of σ-thickness `d_sigma` per unit area, `D·Δσ` (m),
/// zero where the column has no water. Inventories hold exactly this
/// volume's content: [`layer_thickness_of`]'s floor guards divisions only,
/// and as an inventory's volume it gave a node wetting from dry the
/// concentration of the floor's phantom water (1e-12 m; 2.3e-7 of a 4 μm
/// film's tracers at a beach's wetting front, TODO P1.3).
#[inline]
pub(crate) fn layer_volume_of(depth: f64, d_sigma: f64) -> f64 {
    (depth * d_sigma).max(0.0)
}

/// Depth (m) below which a node's (or, as a mean, an element's) tracers are
/// left as they were: it holds no water to define a concentration.
const DRY_DEPTH: f64 = 1e-6;

/// Multiply a field by the thickness of its cells under `η`: concentration
/// `C` → inventory `H_z C`, with `H_z = D·Δσ` for the σ-thicknesses
/// `d_sigma` of the cells in a column (the layers' [`SigmaGrid::d_sigma`],
/// or the w-cells' of [`w_cell_thicknesses`] for a field at the w-points),
/// zero without water ([`layer_volume_of`]).
pub fn to_inventory(tracer: &mut [f64], eta: &[f64], d_sigma: &[f64], bathymetry: &Bathymetry2D) {
    let (nn, nl) = (bathymetry.n_nodes, d_sigma.len());
    let n_elements = bathymetry.n_elements;
    for_each_block(
        n_elements,
        [&mut tracer[..n_elements * nn * nl]],
        || (),
        |_, k, [block]| {
            let nodes = k * nn..(k + 1) * nn;
            for ((column, &e), &b) in block
                .chunks_exact_mut(nl)
                .zip(&eta[nodes.clone()])
                .zip(&bathymetry.data[nodes])
            {
                for (c, &ds) in column.iter_mut().zip(d_sigma) {
                    *c *= layer_volume_of(e - b, ds);
                }
            }
        },
    );
}

/// [`to_inventory`] of `values` into `q` at the nodes with (almost) no water
/// only, shallower than `DRY_DEPTH`, where [`from_inventory`] keeps the last
/// value: their inventories become that value's, the rest are left alone.
pub(crate) fn dry_inventories(
    q: &mut [f64],
    values: &[f64],
    eta: &[f64],
    d_sigma: &[f64],
    bathymetry: &Bathymetry2D,
) {
    let nl = d_sigma.len();
    for (idx, (&e, &b)) in eta.iter().zip(&bathymetry.data).enumerate() {
        let depth = e - b;
        if depth < DRY_DEPTH {
            let column = idx * nl..(idx + 1) * nl;
            for ((q, &c), &ds) in q[column.clone()]
                .iter_mut()
                .zip(&values[column])
                .zip(d_sigma)
            {
                *q = c * layer_volume_of(depth, ds);
            }
        }
    }
}

/// Inventory `q = H_z C` → concentration, written to `out`, for cells of
/// σ-thickness `d_sigma` in a column (as [`to_inventory`]).
///
/// - Elements not marked in `element_means`: `C = q / H_z` at every node.
/// - Marked elements: per level, the element's inventory over its volume,
///   `M = Σ w_i J_i q_il / Σ w_i J_i H_z,il`, plus each node's departure from
///   the same mean of the values `out` holds (the last ones):
///   `C_il = out_il + (M − Σ w_i J_i H_z,il out_il / Σ w_i J_i H_z,il)`. The
///   mode splitter marks the elements where the barotropic pass kept only
///   the element balance, not the nodal identity `η̄ − ηⁿ = −Δt∇·DU_avg2`
///   (`WetDry` subcell and positivity-limited elements, see
///   `BarotropicTransport`): there the nodal quotient is not constant for a
///   constant tracer, but the element means are (and are conservative, see
///   the module docs). The departures are zero for a constant tracer and
///   have zero weighted mean, so the conversion keeps both properties; and
///   they keep the element's vertical structure, where the mean alone
///   flattened the stratification along the σ-levels: across a shoreline
///   element spanning 0.6 to 105 m of water a σ-level crosses 100 m of
///   depth, and the flattened density drove 0.1–1 m/s at rest over the
///   Frøya bed (2026-10-03). The structure within a marked element is held
///   while it is marked; only its means move with the fluxes.
/// - A node with (almost) no water, shallower than `DRY_DEPTH`, and a level
///   of a marked element with a mean depth below it, keep the value `out`
///   already holds (the last concentration) instead of dividing by a
///   vanishing layer.
///
/// Converting back with [`to_inventory`] keeps each element's
/// inventory per level exactly.
pub fn from_inventory(
    q: &[f64],
    eta: &[f64],
    d_sigma: &[f64],
    bathymetry: &Bathymetry2D,
    geom: &GeometricFactors2D,
    element_means: &[bool],
    out: &mut [f64],
) {
    let nl = d_sigma.len();
    let nn = bathymetry.n_nodes;
    let n_elements = bathymetry.n_elements;
    for_each_block(
        n_elements,
        [&mut out[..n_elements * nn * nl]],
        || (),
        |_, k, [out_k]| {
            let bed = bathymetry.element(ElementIndex::new(k));
            let eta_k = &eta[k * nn..(k + 1) * nn];
            let q_k = &q[k * nn * nl..(k + 1) * nn * nl];
            if !element_means[k] {
                for (i, (&e, &b)) in eta_k.iter().zip(bed).enumerate() {
                    if e - b < DRY_DEPTH {
                        continue;
                    }
                    for (l, &ds) in d_sigma.iter().enumerate() {
                        out_k[i * nl + l] = q_k[i * nl + l] / layer_thickness_of(e - b, ds);
                    }
                }
                return;
            }
            let area: f64 = (0..nn).map(|i| geom.node_mass(k, i)).sum();
            for (l, &ds) in d_sigma.iter().enumerate() {
                // The new mean, and the last values' mean with the same
                // weights: each node keeps its departure from it
                let (mut inventory, mut last, mut volume) = (0.0, 0.0, 0.0);
                for (i, (&e, &b)) in eta_k.iter().zip(bed).enumerate() {
                    let weight = geom.node_mass(k, i) * layer_volume_of(e - b, ds);
                    inventory += geom.node_mass(k, i) * q_k[i * nl + l];
                    last += weight * out_k[i * nl + l];
                    volume += weight;
                }
                if volume > area * ds * DRY_DEPTH {
                    // Not at a dry node: it has no weight in the mean, so
                    // the shifts summed there step by step (a dry shore node
                    // of the Frøya day drifted to 33 °C, and wetting brought
                    // the drift back into the water, 2026-10-16)
                    let shift = (inventory - last) / volume;
                    for (i, (&e, &b)) in eta_k.iter().zip(bed).enumerate() {
                        if e - b >= DRY_DEPTH {
                            out_k[i * nl + l] += shift;
                        }
                    }
                }
            }
        },
    );
}

/// The form of the horizontal momentum advection `∇·(Q_l u_l)` within the
/// elements ([`apply_momentum_transport_3d`]). Both are conservative and keep
/// a uniform velocity uniform; they differ by aliasing only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MomentumAdvectionForm {
    /// Split (flux-differencing) form, `½[∇·(Qu) + Q·∇u + u∇·Q]`
    /// ([`advective_divergence_element`]): kinetic-energy preserving within
    /// the elements (Gassner, Winters & Kopriva 2016), so the advection only
    /// removes kinetic energy, at the upwind faces. The default: in the
    /// conservative form the aliasing feeds the grid-scale shear instability
    /// of a sharp interface, and the P2 lock exchange blew up without
    /// horizontal viscosity (as did the P2 fjord estuary with ν = 5 m²/s);
    /// in this form both run with none.
    #[default]
    Split,
    /// Strong conservative form, the DG divergence of `Q_l u_l` at the nodes
    /// ([`transport_divergence_element`]). For comparison.
    Conservative,
}

/// A tracer for [`apply_tracer_transport_3d`]: its concentration and
/// where its inventory tendency goes, its exterior values, and the
/// reference stratification its vertical reconstruction is taken about, if
/// any (`[element][node][level]`, e.g. its values at rest).
pub struct AdvectedTracer<'a> {
    /// Overwritten with the inventory tendency (layout of [`Solution3D`]).
    pub rhs: &'a mut [f64],
    /// The concentration (layout of [`Solution3D`]).
    pub tracer: &'a [f64],
    /// The exterior value on inflow through a physical boundary.
    pub bc: &'a dyn TracerBoundaryCondition3D,
    /// A nesting parent's values outside its open faces, which take the
    /// place of `bc`'s where they have one.
    pub exterior: Option<ExteriorField<'a>>,
    /// The reference stratification of the vertical reconstruction.
    pub reference: Option<&'a [f64]>,
    /// The tracer's physical range `(min, max)`, which the water leaving an
    /// end layer stays within ([`UNBOUNDED`] for none).
    pub range: (f64, f64),
}

/// The [`AdvectedTracer::range`] of a tracer with no bounds.
pub const UNBOUNDED: (f64, f64) = (f64::NEG_INFINITY, f64::INFINITY);

impl<'a> AdvectedTracer<'a> {
    /// `tracer` into `rhs` with `bc`'s exterior values, no parent, no
    /// reference and no range.
    pub fn new(
        rhs: &'a mut [f64],
        tracer: &'a [f64],
        bc: &'a dyn TracerBoundaryCondition3D,
    ) -> Self {
        Self {
            rhs,
            tracer,
            bc,
            exterior: None,
            reference: None,
            range: UNBOUNDED,
        }
    }
}

/// Overwrite each tracer's `rhs` with the inventory tendency `∂(H_z C)/∂t`
/// of its concentration `C`, advected by the layer transports `transport`:
///
/// ```text
///     ∂(H_z C)_l/∂t = −∇·(Q_l C_l) − (Ω_{l+1/2} C_{l+1/2} − Ω_{l−1/2} C_{l−1/2})
/// ```
///
/// Horizontally the nodal flux is `Q_l C_l` and the face flux `F_l C↑`, with
/// `C↑` the concentration upwind of the layer's face flux `F_l`: the element's
/// own on outflow, the neighbour's on inflow, and the tracer's `bc` exterior
/// value on inflow through a physical boundary (a nesting parent's value, from
/// its `exterior`, where it has one). Vertically `C_{l±1/2}` is reconstructed
/// by `vertical` ([`VerticalAdvection`]).
///
/// For constant `C` the tendency is `C·(−∇·Q_l − δΩ_l) = C·Δσ_l ∂η/∂t`, the
/// change of the layer's thickness: constancy. Both fluxes are single-valued
/// on each face, so the inventory is conserved.
///
/// With a `reference`, at every σ-surface the surface value is `vertical`'s,
/// less the share `ψ` of the reference's own departure from the centred
/// value:
///
/// ```text
///     C_s = V(C)_s − ψ_s [V(C_ref)_s − ½(C_ref,l−1 + C_ref,l)],
///     ψ_s = clamp((C_l − C_l−1)/(C_ref,l − C_ref,l−1), 0, 1),
/// ```
///
/// with both reconstructions taken for the same `Ω`, the correction moving
/// `C_s` towards the column's centred value `½(C_l−1 + C_l)` and never past
/// it (so a bounded `V` stays bounded where its limiter clipped `C` but not
/// `C_ref`). In a column at its
/// reference (`ψ = 1`) this is the centred value, so about a fluid resting
/// in the reference the flux of the background is the arithmetic mean to
/// first order, whatever `V` does to the departures. A dissipative or
/// higher-order `V` otherwise feeds a grid-scale instability of the
/// stratified rest state: it mixes the background at a rate set by the
/// perturbation's own `|Ω|`, and that drives the perturbation (TODO P1.3,
/// `docs/stratified-rest-over-steep-beds.md`: e-folding 26–75 min on the
/// Frøya coastline mesh for upwind to the Hermite mean; centred holds).
///
/// - A uniform tracer stays uniform (`ψ = 0`): constancy, whatever the
///   reference.
/// - A column stratified like its reference at a fraction of its strength
///   (`C = a + b C_ref`, `0 < b ≤ 1`) has `ψ = b`, and its surface values are
///   exactly centred (every [`VerticalAdvection`] commutes with
///   `C → a + bC` for `b > 0`).
/// - Jumps beyond the reference's, or against it, keep `V`'s full limiting
///   (fronts).
/// - The flux is single-valued on every surface: conservative.
///
/// Up to two tracers go through each element together, sharing
/// its transports; each tracer's tendency is what it would be alone, to the
/// bit.
pub fn apply_tracer_transport_3d<const N: usize>(
    tracers: [AdvectedTracer; N],
    transport: &LayerTransport,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    boundaries: &Boundaries3D,
    vertical: VerticalAdvection,
) {
    let (nn, nl) = (ops.n_nodes, transport.n_levels);
    let context = LayerContext {
        transport,
        mesh,
        ops,
        geom,
        boundaries,
    };
    let fields = tracers
        .each_ref()
        .map(|t| (t.tracer, t.reference, t.bc, t.exterior, t.range));
    let n = mesh.n_elements * nn * nl;
    for_each_block(
        mesh.n_elements,
        tracers.map(|t| &mut t.rhs[..n]),
        || TransportScratch::take(ops, nl),
        |scratch, k, mut rhs_k| {
            let inflow = |t: usize, f, fi, node, tag, interior, flux, l| {
                let (_, _, bc, exterior, _) = fields[t];
                exterior
                    .and_then(|e: ExteriorField| e.at(tag, node, l))
                    .unwrap_or_else(|| {
                        bc.exterior_value(&TracerBCContext3D {
                            element: k,
                            face: f,
                            level: l,
                            face_node: fi,
                            boundary_tag: Some(tag),
                            interior_value: interior,
                            normal_velocity: flux,
                        })
                    })
            };
            if context.levels_at_once(k, true) {
                let div = context.flux_divergence_levels(
                    k,
                    fields.map(|(tracer, ..)| tracer),
                    inflow,
                    scratch,
                );
                for (rhs_k, div) in rhs_k.iter_mut().zip(div.chunks_exact(nn * nl)) {
                    for (r, &d) in rhs_k.iter_mut().zip(div) {
                        *r = -d;
                    }
                }
            } else {
                for (t, rhs_k) in rhs_k.iter_mut().enumerate() {
                    for l in 0..nl {
                        let inflow = |f, fi, node, tag, interior, flux| {
                            inflow(t, f, fi, node, tag, interior, flux, l)
                        };
                        let div = context.flux_divergence(k, l, fields[t].0, true, inflow, scratch);
                        for (i, &d) in div.iter().enumerate() {
                            rhs_k[i * nl + l] = -d;
                        }
                    }
                }
            }

            for (rhs_k, &(tracer, reference, _, _, range)) in rhs_k.iter_mut().zip(&fields) {
                subtract_vertical_flux(
                    rhs_k, tracer, reference, range, transport, k, nn, vertical, scratch,
                );
            }
        },
    );
}

/// Subtract `δ(Ω φ)`, with `φ` at the σ-surfaces reconstructed by `scheme`,
/// from `rhs_k` (element `k`'s block, `[node][level]`) in every column of
/// element `k`.
fn subtract_vertical_flux(
    rhs_k: &mut [f64],
    field: &[f64],
    reference: Option<&[f64]>,
    range: (f64, f64),
    transport: &LayerTransport,
    k: usize,
    nn: usize,
    scheme: VerticalAdvection,
    scratch: &mut TransportScratch,
) {
    let nl = transport.n_levels;
    let TransportScratch {
        vertical: flux,
        slope,
        reference_surface,
        ..
    } = scratch;
    for i in 0..nn {
        let idx = k * nn + i;
        let omega = &transport.omega[idx * (nl + 1)..(idx + 1) * (nl + 1)];
        let column = &field[idx * nl..(idx + 1) * nl];
        let shed = EndLayerShed {
            range,
            dt_over_depth: transport.dt_over_depth[idx],
        };
        scheme.surface_values(column, omega, &transport.d_sigma, shed, slope, flux);
        if let Some(reference) = reference {
            let reference = &reference[idx * nl..(idx + 1) * nl];
            let unbounded = EndLayerShed {
                range: UNBOUNDED,
                ..shed
            };
            scheme.surface_values(
                reference,
                omega,
                &transport.d_sigma,
                unbounded,
                slope,
                reference_surface,
            );
            for l in 1..nl {
                let jump = reference[l] - reference[l - 1];
                if jump == 0.0 {
                    continue;
                }
                let share = ((column[l] - column[l - 1]) / jump).clamp(0.0, 1.0);
                let correction =
                    -share * (reference_surface[l] - 0.5 * (reference[l - 1] + reference[l]));
                // Only ever towards the column's centred value, not past it:
                // a limited value stays bounded
                let to_centred = 0.5 * (column[l - 1] + column[l]) - flux[l];
                flux[l] += correction.clamp(to_centred.min(0.0), to_centred.max(0.0));
            }
        }
        for l in 0..=nl {
            flux[l] *= omega[l];
        }
        for (l, r) in rhs_k[i * nl..(i + 1) * nl].iter_mut().enumerate() {
            *r -= flux[l + 1] - flux[l];
        }
    }
}

/// Reconstruction of a tracer at the σ-surfaces `l ± 1/2` for its vertical
/// advection `−δ(Ω C)` ([`apply_tracer_transport_3d`]). All are
/// conservative, keep a constant tracer constant and take the layers'
/// thicknesses into account (`H_z ∝ Δσ` within a column), so a linear
/// profile is reconstructed exactly on stretched levels too, except by
/// `Upwind`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VerticalAdvection {
    /// Second-order centred, `C_{l−1/2} = ½(C_{l−1} + C_l)`: non-dissipative
    /// and linear, the momentum's default (ROMS `UV_C2VADVECTION`). It
    /// oscillates at fronts and is exact for a linear profile only on
    /// uniform levels. Stable with the SSP-RK3 stages up to a vertical
    /// Courant number `|Ω|Δt/H_z` ≈ 1.7 (√3).
    Centred,
    /// First-order upwind on `Ω`: monotone, but it diffuses with
    /// `κ ≈ |w|Δz/2`. A mode-1 internal seiche on 10 uniform levels runs
    /// 1.1 % slow and loses 1.2 % of its amplitude per period (Akima: 0.37 %
    /// and 0.14 %).
    Upwind,
    /// Fourth-order Akima (ROMS `TS_A4VADVECTION`; Shchepetkin & McWilliams
    /// 2005, §3): the surface value of the cubic that has the two layers'
    /// means and, at their centres, the harmonic-mean slopes
    /// `d = 2δ₋δ₊/(δ₋ + δ₊)` of the neighbouring gradients (zero at an
    /// extremum, one-sided at the bed and the surface). On layers `a` below
    /// and `b` above,
    ///
    /// ```text
    ///     C_{l−1/2} = (b C_{l−1} + a C_l)/(a + b) + ab (d_{l−1} − d_l) / (3(a + b)),
    /// ```
    ///
    /// ROMS's `½(C_{l−1} + C_l) − (d_l − d_{l−1})/6` on uniform levels (ROMS
    /// applies that in index space on stretched levels too). Centred: no
    /// numerical diffusion, but not monotone; it over- and undershoots
    /// slightly at extrema and fronts, where the zero slope leaves
    /// second-order centred.
    ///
    /// With the SSP-RK3 stages it is stable up to a vertical Courant number
    /// `|Ω|Δt/H_z` ≈ 1.2 (the fourth-order centred operator's eigenvalues
    /// reach 1.37 `|Ω|/H_z` on the imaginary axis, RK3's limit there is √3);
    /// upwind up to 1.
    Akima,
    /// Third-order upwind-biased with a TVD limiter: the surface value of the
    /// parabola with the means of the upwind layer, the one beyond it and the
    /// downwind one (`C_{l−1/2} = (5C_{l−1} + 2C_l − C_{l−2})/6` on uniform
    /// levels, upwind `C_{l−1}`), its increment over the upwind layer's mean
    /// limited to at most the upwind and downwind differences and zero at an
    /// extremum (Koren 1993). This is the spatial part of HSIMT (Wu & Zhu
    /// 2010, ROMS `TS_HSIMT`), whose Courant-number terms the SSP-RK3 stages
    /// replace. Out of the bed or the surface layer, where there is no layer
    /// beyond, the surface value is linear through the two end layers'
    /// means: first order there would freeze the end layer's mean where the
    /// horizontal flow vanishes (at a wall), and a mode-1 internal seiche
    /// with a thick bed layer drifted by a third of its amplitude per period.
    /// Its increment is bounded by the end layer's own gradient, extrapolated
    /// from the gradients between the three end layers: kept for linear and
    /// smoothly curved profiles, first order where the end layer sits on a
    /// plateau over a front below it. Unbounded, the water leaving such a
    /// layer was cooler than its mean (a warm one), and the warm water
    /// flowing in along it carried its mean past the column's range: a light
    /// gravity current's head in a P2 lock exchange 2.4e-2 °C above it.
    ///
    /// Monotone in the interior: no new extrema up to a vertical Courant
    /// number `|Ω|Δt/H_z` of ½ (SSP-RK3 inherits the forward-Euler bound).
    /// The end layers' means can still leave the column's range where a
    /// profile is linear to the end, as the water at the bed or the surface
    /// beyond the end layer's mean is carried into the layer above or below
    /// (that is the profile, not an overshoot), but not the tracer's own
    /// range ([`AdvectedTracer::range`]): the end layer's linear profile is kept
    /// within it, so that a fresh surface layer over a halocline does not shed
    /// more salt than it has (and, beyond an outflow Courant number of ½,
    /// only `(1 − c)/c` of the distance to the bound, so that its mean stays
    /// in the range: [`in_range`]). The limiter adds some
    /// diffusion at extrema and fronts.
    Tvd,
    /// [`Self::Akima`]'s surface values under [`Self::Tvd`]'s limiter: the
    /// increment over the upwind layer's mean at most the upwind and
    /// downwind differences and zero at an extremum (out of an end layer,
    /// the end layer's own extrapolated gradient takes the upwind
    /// difference's place, as in [`Self::Tvd`]), i.e. bounded by the layers
    /// below and above. Where a profile is smooth and monotone Akima's value lies
    /// within those bounds and is kept. At the foot of a front, where the
    /// upwind layer's harmonic-mean slope is small and the next one steep,
    /// Akima's centred value lies far towards the downwind layer's, more
    /// than the upwind gradient allows, and the scheme oscillates there; it
    /// is clipped to the bound.
    ///
    /// Akima's undershoots at a sharp interface are in the layers' element
    /// means, which no conservative slope limiter of the nodal values can
    /// change: a P2 lock exchange on 125 m (ν = 10 m²/s) left 7.27 °C in a
    /// layer of a 7.5–12.5 °C range. The bounds have to be on the flux.
    /// Monotone in the interior up to a vertical Courant number of ½, and the
    /// end layers kept within the tracer's range, as [`Self::Tvd`]. The tracers'
    /// default.
    #[default]
    LimitedAkima,
    /// The mean between the two layer centres of the Hermite cubic with the
    /// harmonic-mean slopes of the baroclinic pressure integral,
    /// `½(C_{l−1} + C_l) + h(d_{l−1} − d_l)/12` with `h` the centres'
    /// spacing: the energy partner of that pressure integral
    /// (experimental, TODO P1.3).
    HermiteMean,
}

impl VerticalAdvection {
    /// `C` at the `nl + 1` σ-surfaces of `column` (bed first) into
    /// `surface`, for the volume fluxes `omega` (which decide the upwind
    /// side) through layers of σ-thickness `d_sigma`. The bed and surface
    /// pass no flux; their values are zero. `slope` holds `nl` values of
    /// scratch. `shed` is the tracer's physical range and the stage's
    /// Courant scale, which [`Self::Tvd`] and [`Self::LimitedAkima`] keep the
    /// water leaving an end layer within (see [`EndLayerShed`]).
    fn surface_values(
        self,
        column: &[f64],
        omega: &[f64],
        d_sigma: &[f64],
        shed: EndLayerShed,
        slope: &mut [f64],
        surface: &mut [f64],
    ) {
        let nl = column.len();
        surface[0] = 0.0;
        surface[nl] = 0.0;
        match self {
            Self::Centred => {
                for l in 1..nl {
                    surface[l] = 0.5 * (column[l - 1] + column[l]);
                }
            }
            Self::Upwind => {
                for l in 1..nl {
                    surface[l] = if omega[l] >= 0.0 {
                        column[l - 1]
                    } else {
                        column[l]
                    };
                }
            }
            Self::Akima if nl > 1 => akima_surface_values(column, d_sigma, slope, surface),
            Self::HermiteMean if nl > 1 => {
                // Akima's slopes are the pressure integral's harmonic means
                akima_surface_values(column, d_sigma, slope, surface);
                let weight = curvature_weight();
                for l in 1..nl {
                    let h = 0.5 * (d_sigma[l - 1] + d_sigma[l]);
                    surface[l] = 0.5 * (column[l - 1] + column[l])
                        + weight * h * (slope[l - 1] - slope[l]) / 6.0;
                }
            }
            Self::LimitedAkima if nl > 1 => {
                akima_surface_values(column, d_sigma, slope, surface);
                for l in 1..nl {
                    let (up, far, down) = upwind_stencil(l, nl, omega[l]);
                    let Some(far) = far else {
                        // Out of an end layer: bounded by a ghost layer, and
                        // by the range
                        let increment =
                            end_layer_limited(surface[l] - column[up], column, d_sigma, up, down);
                        surface[l] =
                            column[up] + shed.limit(increment, column[up], omega[l], d_sigma[up]);
                        continue;
                    };
                    surface[l] = column[up]
                        + tvd_limited(
                            surface[l] - column[up],
                            column[up] - column[far],
                            column[down] - column[up],
                        );
                }
            }
            Self::Tvd => {
                for l in 1..nl {
                    let (up, far, down) = upwind_stencil(l, nl, omega[l]);
                    let Some(far) = far else {
                        // Out of an end layer: linear through the two layers'
                        // means (first order would freeze the end layer's
                        // mean against a wall, where u = 0), bounded by a
                        // ghost layer
                        let (a, b) = (d_sigma[up], d_sigma[down]);
                        let increment = a / (a + b) * (column[down] - column[up]);
                        let increment = end_layer_limited(increment, column, d_sigma, up, down);
                        surface[l] =
                            column[up] + shed.limit(increment, column[up], omega[l], d_sigma[up]);
                        continue;
                    };
                    let (a, b, c) = (d_sigma[up], d_sigma[down], d_sigma[far]);
                    let (behind, ahead) = (column[up] - column[far], column[down] - column[up]);
                    let increment = a * b / ((a + c) * (a + b + c)) * behind
                        + a * (a + c) / ((a + b) * (a + b + c)) * ahead;
                    surface[l] = column[up] + tvd_limited(increment, behind, ahead);
                }
            }
            Self::Akima | Self::LimitedAkima | Self::HermiteMean => {}
        }
    }
}

/// Experiment (TODO P1.3): the weight of Akima's curvature correction in
/// [`VerticalAdvection::HermiteMean`] (½), settable from tests.
#[cfg(test)]
pub(crate) static CURVATURE_WEIGHT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0x3FE0_0000_0000_0000);

#[cfg(test)]
fn curvature_weight() -> f64 {
    f64::from_bits(CURVATURE_WEIGHT.load(std::sync::atomic::Ordering::Relaxed))
}

#[cfg(not(test))]
fn curvature_weight() -> f64 {
    0.5
}

/// The layers around σ-surface `l` of a column of `nl` for the volume flux
/// `omega` through it: the upwind layer, the one beyond it (if any) and the
/// downwind layer.
fn upwind_stencil(l: usize, nl: usize, omega: f64) -> (usize, Option<usize>, usize) {
    if omega >= 0.0 {
        (l - 1, l.checked_sub(2), l)
    } else {
        (l, Some(l + 1).filter(|&f| f < nl), l - 1)
    }
}

/// Koren's (1993) TVD bound on the `increment` of a surface value over its
/// upwind layer's mean: at most the differences `behind` (upwind layer
/// minus the one beyond) and `ahead` (downwind minus upwind), and zero at an
/// extremum or against the flow's gradient.
fn tvd_limited(increment: f64, behind: f64, ahead: f64) -> f64 {
    if behind * ahead > 0.0 && increment * ahead > 0.0 {
        increment.signum() * increment.abs().min(behind.abs()).min(ahead.abs())
    } else {
        0.0
    }
}

/// What keeps the water an end layer sheds within the tracer's physical
/// range: the range itself and the stage's Courant scale at the column.
#[derive(Clone, Copy, Debug)]
struct EndLayerShed {
    /// The tracer's physical range `(min, max)` ([`UNBOUNDED`] for none).
    range: (f64, f64),
    /// `Δt_E/D` of the column ([`LayerTransport::dt_over_depth`]), zero for
    /// none: the Courant number is not known and taken as small.
    dt_over_depth: f64,
}

impl EndLayerShed {
    /// No range, so nothing to keep.
    #[cfg(test)]
    const NONE: Self = Self {
        range: UNBOUNDED,
        dt_over_depth: 0.0,
    };

    /// The `increment` of the surface value out of the end layer (mean
    /// `mean`, σ-thickness `d_sigma`) through the σ-surface of volume flux
    /// `omega` (out of the layer), kept within the range at the layer's own
    /// outflow Courant number `Δt_E |Ω|/(D Δσ)` (see [`in_range`]).
    fn limit(self, increment: f64, mean: f64, omega: f64, d_sigma: f64) -> f64 {
        let courant = self.dt_over_depth * omega.abs() / d_sigma;
        in_range(increment, mean, self.range, courant)
    }
}

/// The `increment` of the surface value out of an end layer over its mean
/// `mean`, kept so that the end layer stays within `range` after a
/// forward-Euler step in which its outflow is the Courant number `courant`
/// (`Δt|Ω|/H_z`) of its thickness.
///
/// The surface value `mean + increment` at the layer's inner surface and its
/// reflection `mean − increment` at the bed or the free surface are the two
/// ends of the layer's linear profile. The layer loses (gains) the water at
/// the inner surface's value, so it keeps the part of the profile nearer its
/// end: its mean changes by `−c δ/(1 − c)` of the increment `δ` at a Courant
/// number `c` of the water it sheds while it thins. For `c ≤ ½` (the
/// interior's monotone bound too) that is within `δ`, so keeping the profile
/// within the range leaves the layer's mean in it. Beyond ½ the mean stays in
/// the range only if `δ` is at most `(1 − c)/c` of the distance from the mean
/// to the bound, which is the same factor that Koren's limiter loses at that
/// Courant number to stay monotone (`φ ≤ 2(1 − c)/c`); at `c ≥ 1` the layer
/// keeps only its own mean (first order). The default
/// [`crate::physics::ImplicitVerticalAdvection`] holds the explicit outflow
/// of a layer to 0.8, a factor ¼, and without it the step bound allows 1.
///
/// Without the range the end layer's own gradient ([`end_layer_limited`])
/// lets the profile reach beyond the layers' means by half a layer. That is
/// the profile, not an overshoot, until it reaches a bound of the tracer: the
/// surface layer of fresh river water (`S` = 0.03 over 0.9 over 1.8) over a
/// halocline has a linear profile of −0.4 at the surface, and its mean fell
/// to −3.6 within an hour where the bed rises under it (the landward flank of
/// a 60 m sill in 400 m of water, `examples/sill_fjord_3d.rs`). The bound
/// matters only within that distance of the range; a mean beyond the range
/// keeps only an increment towards it. (The layer's horizontal outflow is not
/// counted in `c`: its Courant number is a fraction of the vertical one in
/// thin layers, `TODO.md` P4.5.)
fn in_range(increment: f64, mean: f64, (min, max): (f64, f64), courant: f64) -> f64 {
    // Not scaled below ½, where the profile in the range is the bound:
    // bit for bit as before
    let share = if courant > 0.5 {
        (1.0 - courant).max(0.0) / courant
    } else {
        1.0
    };
    // An infinite distance (no bound) stays infinite at share 0
    let reach = |distance: f64| {
        if share < 1.0 && distance.is_finite() {
            share * distance
        } else {
            distance
        }
    };
    // `max` and `min`, not `clamp`: the bounds can be NaN beside a NaN mean
    increment
        .max(reach((mean - max).min(0.0)))
        .min(reach((mean - min).max(0.0)))
}

/// [`tvd_limited`] for the `increment` of the surface value out of the end
/// layer `up` (the bed or the surface layer) towards its neighbour `down`,
/// where there is no layer beyond to bound it. The difference behind is the
/// end layer's own gradient times its thickness, the gradient at its centre
/// extrapolated linearly from those between the three end layers' centres.
/// A linear profile keeps its linear surface value, as does a curved one
/// unless it flattens sharply towards the end (on uniform levels, the
/// difference next to the end layer less than half the one beyond); where
/// the end layer sits on a plateau over a front below it, its water leaves
/// at its own mean. With two layers there is no curvature to see, and the
/// increment is only kept between the two layers' means.
fn end_layer_limited(
    increment: f64,
    column: &[f64],
    d_sigma: &[f64],
    up: usize,
    down: usize,
) -> f64 {
    let ahead = column[down] - column[up];
    let next = (2 * down).checked_sub(up).filter(|&n| n < column.len());
    let Some(next) = next else {
        return increment.max(ahead.min(0.0)).min(ahead.max(0.0));
    };
    let (a, b, c) = (d_sigma[up], d_sigma[down], d_sigma[next]);
    // Gradients into the column between the centres, midway between them
    let (h1, h2) = (0.5 * (a + b), 0.5 * (b + c));
    let (g1, g2) = (ahead / h1, (column[next] - column[down]) / h2);
    let gradient = g1 - (g2 - g1) * h1 / (h1 + h2);
    tvd_limited(increment, gradient * a, ahead)
}

/// [`VerticalAdvection::Akima`]'s values at the interior σ-surfaces of
/// `column` (`nl > 1` layers of σ-thickness `d_sigma`) into `surface`, with
/// the harmonic-mean slopes in `slope`.
fn akima_surface_values(column: &[f64], d_sigma: &[f64], slope: &mut [f64], surface: &mut [f64]) {
    let nl = column.len();
    // The gradient across surface l (between the centres of layers l − 1
    // and l), repeated at the bed and the surface
    let gradient = |l: usize| {
        let l = l.clamp(1, nl - 1);
        2.0 * (column[l] - column[l - 1]) / (d_sigma[l - 1] + d_sigma[l])
    };
    for (l, d) in slope.iter_mut().enumerate() {
        let (below, above) = (gradient(l), gradient(l + 1));
        let product = below * above;
        *d = if product > 0.0 {
            2.0 * product / (below + above)
        } else {
            0.0
        };
    }
    for l in 1..nl {
        let (a, b) = (d_sigma[l - 1], d_sigma[l]);
        surface[l] =
            (b * column[l - 1] + a * column[l] + a * b * (slope[l - 1] - slope[l]) / 3.0) / (a + b);
    }
}

/// Add the inventory tendency `∂(H_z u)/∂t` of the momentum advection by the
/// layer transports `transport` to `rhs_u` and `rhs_v`:
///
/// ```text
///     ∂(H_z u)_l/∂t += −∇·(Q_l u_l) − (Ω_{l+1/2} u_{l+1/2} − Ω_{l−1/2} u_{l−1/2})
/// ```
///
/// for the advected velocity `(u, v)`; `Q` and `Ω` come from `transport`.
/// Within the elements `∇·(Q_l u_l)` is in the `form` of
/// [`MomentumAdvectionForm`] (`Hydrostatic3D` uses the split form by
/// default). Horizontally the face flux is `F_l u↑`, with `u↑` upwind of the layer's
/// face flux as for the tracers. Flowing in through an open boundary it is
/// a nesting parent's velocity (`exterior`) where there is one, otherwise
/// the interior's (zero gradient, ROMS's "gradient" condition for the 3D
/// velocity; see [`crate::solver::rhs::boundary_3d`]). Walls carry no volume, so no
/// momentum. Vertically `u` at the σ-surfaces is reconstructed by `vertical`
/// (`Hydrostatic3D` uses [`VerticalAdvection::Centred`] by default).
///
/// A velocity uniform in space gets `u·Δσ_l ∂η/∂t`: divided by the new layer
/// thickness it stays uniform. The layer momentum `∫ H_z,l u_l` changes
/// only through open faces. In the split form with
/// [`VerticalAdvection::Centred`] the kinetic energy `½ Σ_l H_z,l |u_l|²`
/// changes only through open faces and by the upwind faces' dissipation,
/// `−½ Σ_faces |F_l| |u_l − u_l'|²`.
pub fn apply_momentum_transport_3d(
    rhs_u: &mut [f64],
    rhs_v: &mut [f64],
    u: &[f64],
    v: &[f64],
    transport: &LayerTransport,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    boundaries: &Boundaries3D,
    exterior: Option<[ExteriorField; 2]>,
    form: MomentumAdvectionForm,
    vertical: VerticalAdvection,
) {
    let (nn, nl) = (ops.n_nodes, transport.n_levels);
    let split = form == MomentumAdvectionForm::Split;
    let context = LayerContext {
        transport,
        mesh,
        ops,
        geom,
        boundaries,
    };
    let [exterior_u, exterior_v] = exterior.map_or([None, None], |[u, v]| [Some(u), Some(v)]);
    let n = mesh.n_elements * nn * nl;
    for_each_block(
        mesh.n_elements,
        [&mut rhs_u[..n], &mut rhs_v[..n]],
        || TransportScratch::take(ops, nl),
        |scratch, k, [rhs_u, rhs_v]| {
            let exterior = [exterior_u, exterior_v];
            let inflow = |t: usize, _, _, node, tag, interior, _, l| {
                exterior[t]
                    .and_then(|e: ExteriorField| e.at(tag, node, l))
                    .unwrap_or(interior)
            };
            let mut rhs_k = [rhs_u, rhs_v];
            if context.levels_at_once(k, split) {
                let div = context.flux_divergence_levels(k, [u, v], inflow, scratch);
                for (rhs_k, div) in rhs_k.iter_mut().zip(div.chunks_exact(nn * nl)) {
                    for (r, &d) in rhs_k.iter_mut().zip(div) {
                        *r -= d;
                    }
                }
            } else {
                for (t, (rhs_k, field)) in rhs_k.iter_mut().zip([u, v]).enumerate() {
                    for l in 0..nl {
                        let inflow = |f, fi, node, tag, interior, flux| {
                            inflow(t, f, fi, node, tag, interior, flux, l)
                        };
                        let div = context.flux_divergence(k, l, field, split, inflow, scratch);
                        for (i, &d) in div.iter().enumerate() {
                            rhs_k[i * nl + l] -= d;
                        }
                    }
                }
            }
            for (rhs_k, field) in rhs_k.iter_mut().zip([u, v]) {
                subtract_vertical_flux(
                    rhs_k, field, None, UNBOUNDED, transport, k, nn, vertical, scratch,
                );
            }
        },
    );
}

/// What the horizontal flux divergence of a layer needs.
struct LayerContext<'a> {
    transport: &'a LayerTransport,
    mesh: &'a Mesh2D,
    ops: &'a DGOperators2D,
    geom: &'a GeometricFactors2D,
    boundaries: &'a Boundaries3D,
}

impl LayerContext<'_> {
    /// `∇·(Q_l φ)` on element `k`, layer `l`, of the field `φ` (layout of
    /// [`Solution3D`]): in the element in split form
    /// ([`advective_divergence_element`]) if `split`, else conservative
    /// ([`transport_divergence_element`] of `Q_l φ`); face flux `F_l φ↑`
    /// with `φ↑` upwind of `F_l`. On inflow through an open face `φ↑` is
    /// `inflow(face, face_node, node, tag, interior value, F_l)`, with `node`
    /// the face node's `[element][node]` index. Written to (and returned
    /// from) `scratch.div`.
    fn flux_divergence<'s>(
        &self,
        k: usize,
        l: usize,
        field: &[f64],
        split: bool,
        inflow: impl Fn(usize, usize, usize, BoundaryTag, f64, f64) -> f64,
        scratch: &'s mut TransportScratch,
    ) -> &'s [f64] {
        let (ops, transport) = (self.ops, self.transport);
        let (nn, nfn, nl) = (ops.n_nodes, ops.n_face_nodes, transport.n_levels);
        let TransportScratch {
            hu,
            hv,
            phi,
            fr,
            fs,
            face,
            sub,
            div,
            ..
        } = scratch;
        for i in 0..nn {
            let idx = (k * nn + i) * nl + l;
            hu[i] = transport.hu[idx];
            hv[i] = transport.hv[idx];
            phi[i] = field[idx];
        }
        let el = ElementIndex::new(k);
        for f in 0..4 {
            let exterior = self.boundaries.exterior(self.mesh, el, f);
            for (fi, &node) in ops.face_nodes[f].iter().enumerate() {
                let slot = (k * 4 + f) * nfn + fi;
                let flux = transport.face[slot * nl + l];
                let interior = field[(k * nn + node) * nl + l];
                let upwind = match exterior {
                    _ if flux >= 0.0 => interior,
                    FaceExterior::Element(nb) => {
                        let nb_node = ops.face_nodes[nb.face][nfn - 1 - fi];
                        field[(nb.element * nn + nb_node) * nl + l]
                    }
                    FaceExterior::Open(tag) => inflow(f, fi, k * nn + node, tag, interior, flux),
                    // A wall passes no volume: only a round-off flux
                    FaceExterior::Wall => interior,
                };
                face[f * nfn + fi] = flux * upwind;
            }
        }
        if transport.subcell_elements[k] {
            // Through the subcell interfaces the 2D wet/dry kernel moves the
            // water through, with the central value: no diffusion along
            // σ-levels that cut steeply through the stratification (upwind
            // moved a steep shore's stratification by 8e-3 °C)
            let n1 = ops.n_1d;
            let n_sub = subcell_interfaces(n1);
            for dir in 0..2 {
                for line in 0..n1 {
                    for a in 0..n1 - 1 {
                        let slot = subcell_slot(n1, dir, line, a);
                        let flux = transport.subcell[(k * n_sub + slot) * nl + l];
                        let (ia, ib) =
                            (line_node(n1, dir, line, a), line_node(n1, dir, line, a + 1));
                        sub[slot] = flux * 0.5 * (phi[ia] + phi[ib]);
                    }
                }
            }
            for ((hu, hv), &phi) in hu.iter_mut().zip(hv.iter_mut()).zip(phi.iter()) {
                *hu *= phi;
                *hv *= phi;
            }
            subcell_divergence_element(ops, self.geom, k, hu, hv, sub, face, div);
        } else if split {
            let metric = transport.metric;
            advective_divergence_element(ops, self.geom, metric, k, hu, hv, phi, face, fr, fs, div);
        } else {
            for ((hu, hv), &phi) in hu.iter_mut().zip(hv.iter_mut()).zip(phi.iter()) {
                *hu *= phi;
                *hv *= phi;
            }
            transport_divergence_element(ops, self.geom, transport.metric, k, hu, hv, face, div);
        }
        div
    }

    /// Whether [`Self::flux_divergence_levels`] takes element `k` (in split
    /// form, unless its layers move through the subcell interfaces).
    fn levels_at_once(&self, k: usize, split: bool) -> bool {
        split && !self.transport.subcell_elements[k]
    }

    /// [`Self::flux_divergence`] in split form of every layer of element `k`
    /// at once, for `N` (at most [`MAX_FIELDS`]) fields
    /// ([`advective_divergence_levels`], the same to the bit), into (and
    /// returned from) `scratch.div_levels`: field `t`'s `[node][level]` block
    /// at `t · n_nodes · n_levels`. `inflow` takes the field's index first and
    /// the level last. Only where [`Self::levels_at_once`].
    fn flux_divergence_levels<'s, const N: usize>(
        &self,
        k: usize,
        fields: [&[f64]; N],
        inflow: impl Fn(usize, usize, usize, usize, BoundaryTag, f64, f64, usize) -> f64,
        scratch: &'s mut TransportScratch,
    ) -> &'s [f64] {
        const { assert!(N <= MAX_FIELDS, "the scratch holds MAX_FIELDS fields") };
        let (ops, transport) = (self.ops, self.transport);
        let (nn, nfn, nl) = (ops.n_nodes, ops.n_face_nodes, transport.n_levels);
        let TransportScratch {
            face_levels,
            div_levels,
            levels,
            ..
        } = scratch;
        let (face_len, div_len) = (4 * nfn * nl, nn * nl);
        let block = k * nn * nl..(k + 1) * nn * nl;
        let el = ElementIndex::new(k);
        for f in 0..4 {
            let exterior = self.boundaries.exterior(self.mesh, el, f);
            for (fi, &node) in ops.face_nodes[f].iter().enumerate() {
                let slot = (k * 4 + f) * nfn + fi;
                let fluxes = &transport.face[slot * nl..(slot + 1) * nl];
                for (t, field) in fields.iter().enumerate() {
                    let interior = &field[(k * nn + node) * nl..(k * nn + node + 1) * nl];
                    let start = t * face_len + (f * nfn + fi) * nl;
                    let out = &mut face_levels[start..start + nl];
                    match exterior {
                        FaceExterior::Element(nb) => {
                            let nb_node = ops.face_nodes[nb.face][nfn - 1 - fi];
                            let start = (nb.element * nn + nb_node) * nl;
                            let outside = &field[start..start + nl];
                            for l in 0..nl {
                                let flux = fluxes[l];
                                out[l] = flux * if flux >= 0.0 { interior[l] } else { outside[l] };
                            }
                        }
                        FaceExterior::Open(tag) => {
                            for l in 0..nl {
                                let (flux, interior) = (fluxes[l], interior[l]);
                                out[l] = flux
                                    * if flux >= 0.0 {
                                        interior
                                    } else {
                                        inflow(t, f, fi, k * nn + node, tag, interior, flux, l)
                                    };
                            }
                        }
                        // A wall passes no volume: only a round-off flux
                        FaceExterior::Wall => {
                            for l in 0..nl {
                                out[l] = fluxes[l] * interior[l];
                            }
                        }
                    }
                }
            }
        }
        let mut faces = face_levels.chunks_exact(face_len);
        let mut divs = div_levels.chunks_exact_mut(div_len);
        advective_divergence_levels(
            ops,
            self.geom,
            transport.metric,
            k,
            &transport.hu[block.clone()],
            &transport.hv[block.clone()],
            fields.map(|field| &field[block.clone()]),
            [(); N].map(|()| faces.next().expect("a face block per field")),
            [(); N].map(|()| divs.next().expect("a divergence block per field")),
            levels,
        );
        &div_levels[..N * div_len]
    }
}

/// The most fields one call of the all-levels kernels takes (a pair: u
/// and v, T and S, k and ψ).
const MAX_FIELDS: usize = 2;

/// Buffers of the transport kernels ([`apply_tracer_transport_3d`],
/// [`apply_momentum_transport_3d`]), sized for one element.
struct TransportScratch {
    hu: Vec<f64>,
    hv: Vec<f64>,
    phi: Vec<f64>,
    fr: Vec<f64>,
    fs: Vec<f64>,
    face: Vec<f64>,
    /// A layer's advective fluxes through the subcell interfaces
    sub: Vec<f64>,
    div: Vec<f64>,
    vertical: Vec<f64>,
    slope: Vec<f64>,
    /// The surface values of a column's reference (`[level]`, bed first)
    reference_surface: Vec<f64>,
    /// Every level's face fluxes, `[face node][level]`, and divergence,
    /// `[node][level]`, of up to [`MAX_FIELDS`] fields
    /// ([`LayerContext::flux_divergence_levels`])
    face_levels: Vec<f64>,
    div_levels: Vec<f64>,
    levels: LevelScratch,
}

impl TransportScratch {
    /// Buffers for elements of `ops` and `n_levels` layers, from this
    /// thread's cache.
    fn take(ops: &DGOperators2D, n_levels: usize) -> Pooled<Self> {
        Pooled::take(
            |s: &Self| {
                s.hu.len() == ops.n_nodes
                    && s.face.len() == 4 * ops.n_face_nodes
                    && s.slope.len() == n_levels
            },
            || Self::new(ops, n_levels),
        )
    }

    /// Buffers for elements of `ops` and `n_levels` layers.
    fn new(ops: &DGOperators2D, n_levels: usize) -> Self {
        Self {
            hu: vec![0.0; ops.n_nodes],
            hv: vec![0.0; ops.n_nodes],
            phi: vec![0.0; ops.n_nodes],
            fr: vec![0.0; ops.n_nodes],
            fs: vec![0.0; ops.n_nodes],
            face: vec![0.0; 4 * ops.n_face_nodes],
            sub: vec![0.0; subcell_interfaces(ops.n_1d)],
            div: vec![0.0; ops.n_nodes],
            vertical: vec![0.0; n_levels + 1],
            slope: vec![0.0; n_levels],
            reference_surface: vec![0.0; n_levels + 1],
            face_levels: vec![0.0; MAX_FIELDS * 4 * ops.n_face_nodes * n_levels],
            div_levels: vec![0.0; MAX_FIELDS * ops.n_nodes * n_levels],
            levels: LevelScratch::new(ops.n_nodes, n_levels),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::data::BoundaryTag;
    use crate::solver::rhs::advection_3d::{
        ExtrapolationTracerBC3D, FixedTracerBC3D, UpwindTracerBC3D,
    };
    use std::f64::consts::PI;

    const LX: f64 = 2000.0;
    const LY: f64 = 1000.0;

    /// A closed, periodic or open basin over a sloping, uneven bed with a
    /// sheared, non-uniform flow, a barotropic transport that is not the depth
    /// integral of the 3D velocities, and the free-surface rate that transport
    /// implies.
    struct Case {
        mesh: Mesh2D,
        ops: DGOperators2D,
        geom: GeometricFactors2D,
        sigma: SigmaGrid,
        bathymetry: Bathymetry2D,
        state: Solution3D,
        du_hu: Vec<f64>,
        du_hv: Vec<f64>,
        du_face: Vec<f64>,
        /// No subcell fluxes from a pass (NaN): thin elements take the
        /// central DU_avg2
        du_subcell: Vec<f64>,
        eta_rate: Vec<f64>,
        boundaries: Boundaries3D,
    }

    fn wave(x: f64, y: f64) -> f64 {
        (2.0 * PI * x / LX).sin() * (PI * y / LY).cos()
    }

    impl Case {
        fn new(mesh: Mesh2D, closed: bool) -> Self {
            let ops = DGOperators2D::new(2);
            let geom = GeometricFactors2D::compute(&mesh, &ops);
            let sigma = SigmaGrid::uniform(4);
            let (nn, nfn, nl) = (ops.n_nodes, ops.n_face_nodes, sigma.n_levels());
            let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
                -20.0 - 0.01 * x + 2.0 * (PI * y / LY).cos()
            });
            let mut state = Solution3D::new(mesh.n_elements, nn, nl);
            let mut du_hu = vec![0.0; mesh.n_elements * nn];
            let mut du_hv = vec![0.0; mesh.n_elements * nn];
            for k in 0..mesh.n_elements {
                let el = ElementIndex::new(k);
                for i in 0..nn {
                    let [x, y] = mesh.reference_to_physical(el, ops.nodes_r[i], ops.nodes_s[i]);
                    let idx = k * nn + i;
                    state.eta.data[idx] = 0.3 * wave(x, y);
                    for l in 0..nl {
                        let s = sigma.sigma_rho()[l];
                        state.u[idx * nl + l] = 0.4 * wave(x, y) + 0.2 * (s + 0.5);
                        state.v[idx * nl + l] = -0.2 + 0.1 * wave(y, x) - 0.1 * s;
                        state.temp[idx * nl + l] = 8.0 + 1e-3 * x - 2e-3 * y + l as f64;
                    }
                    // A transport with no normal component at the walls
                    let taper = if closed {
                        (PI * x / LX).sin() * (PI * y / LY).sin()
                    } else {
                        1.0
                    };
                    du_hu[idx] = 3.0 * taper * (1.0 + 0.5 * wave(x, y));
                    du_hv[idx] = -1.5 * taper * wave(y, x);
                }
            }
            // The barotropic face flux as a 2D kernel gives it: central average
            // of the nodal transport plus a jump term, single-valued on each
            // face, zero through walls; at open faces the interior's normal
            // transport plus an offset (an open-boundary condition's flux
            // differs from the interior's)
            let boundaries = Boundaries3D::default();
            let mut du_face = vec![0.0; mesh.n_elements * 4 * nfn];
            for k in 0..mesh.n_elements {
                let el = ElementIndex::new(k);
                for f in 0..4 {
                    let exterior = boundaries.exterior(&mesh, el, f);
                    for (fi, &node) in ops.face_nodes[f].iter().enumerate() {
                        let (nx, ny) = geom.normal(k, f, fi);
                        let a = k * nn + node;
                        let qa = nx * du_hu[a] + ny * du_hv[a];
                        du_face[(k * 4 + f) * nfn + fi] = match exterior {
                            FaceExterior::Element(nb) => {
                                let b = nb.element * nn + ops.face_nodes[nb.face][nfn - 1 - fi];
                                let qb = nx * du_hu[b] + ny * du_hv[b];
                                0.5 * (qa + qb) - 0.3 * (state.eta.data[b] - state.eta.data[a])
                            }
                            FaceExterior::Open(_) => qa + 0.4 * state.eta.data[a],
                            FaceExterior::Wall => 0.0,
                        };
                    }
                }
            }
            // ∂η/∂t = −∇·DU_avg2, the nodal identity of the barotropic pass
            let mut eta_rate = vec![0.0; du_hu.len()];
            for k in 0..mesh.n_elements {
                transport_divergence_element(
                    &ops,
                    &geom,
                    MetricForm::Conservative,
                    k,
                    &du_hu[k * nn..(k + 1) * nn],
                    &du_hv[k * nn..(k + 1) * nn],
                    &du_face[k * 4 * nfn..(k + 1) * 4 * nfn],
                    &mut eta_rate[k * nn..(k + 1) * nn],
                );
            }
            eta_rate.iter_mut().for_each(|r| *r = -*r);
            let du_subcell = vec![f64::NAN; mesh.n_elements * subcell_interfaces(ops.n_1d)];
            Self {
                mesh,
                ops,
                geom,
                sigma,
                bathymetry,
                state,
                du_hu,
                du_hv,
                du_subcell,
                du_face,
                eta_rate,
                boundaries,
            }
        }

        fn closed() -> Self {
            Self::new(Mesh2D::uniform_rectangle(0.0, LX, 0.0, LY, 4, 3), true)
        }

        /// Open on all four sides, with flow through them.
        fn open() -> Self {
            let mesh = Mesh2D::uniform_rectangle_with_bc(0.0, LX, 0.0, LY, 4, 3, BoundaryTag::Open);
            Self::new(mesh, false)
        }

        fn periodic() -> Self {
            Self::new(Mesh2D::uniform_periodic(0.0, LX, 0.0, LY, 4, 3), false)
        }

        fn transport(&self) -> LayerTransport {
            self.transport_with(&Exterior3D::default())
        }

        fn transport_with(&self, exterior: &Exterior3D) -> LayerTransport {
            let mut transport =
                LayerTransport::new(self.mesh.n_elements, &self.ops, self.sigma.n_levels());
            let barotropic = BarotropicFlux {
                dt: 1.0,
                euler_dt: 1.0,
                hu: &self.du_hu,
                hv: &self.du_hv,
                face: &self.du_face,
                subcell: &self.du_subcell,
                eta_rate: &self.eta_rate,
                rivers: None,
                element_means: &[],
            };
            transport.compute(
                &self.state,
                Some(barotropic),
                &self.mesh,
                &self.ops,
                &self.geom,
                &self.sigma,
                &self.bathymetry,
                &self.boundaries,
                exterior,
                0.0,
            );
            transport
        }

        fn tracer_rhs(
            &self,
            transport: &LayerTransport,
            tracer: &[f64],
            bc: &dyn TracerBoundaryCondition3D,
        ) -> Vec<f64> {
            let mut rhs = vec![f64::NAN; tracer.len()];
            apply_tracer_transport_3d(
                [AdvectedTracer {
                    rhs: &mut rhs,
                    tracer,
                    bc,
                    exterior: None,
                    reference: None,
                    range: UNBOUNDED,
                }],
                transport,
                &self.mesh,
                &self.ops,
                &self.geom,
                &self.boundaries,
                VerticalAdvection::default(),
            );
            rhs
        }

        /// `∫ Σ_l φ_l dA` (GLL quadrature) of a field in the 3D layout.
        fn integral(&self, field: &[f64]) -> f64 {
            let (nn, nl) = (self.ops.n_nodes, self.sigma.n_levels());
            (0..self.mesh.n_elements)
                .map(|k| {
                    let sums: Vec<f64> = (0..nn)
                        .map(|i| field[(k * nn + i) * nl..][..nl].iter().sum())
                        .collect();
                    self.geom.integrate_element(k, &sums)
                })
                .sum()
        }
    }

    fn max_abs(values: impl IntoIterator<Item = f64>) -> f64 {
        values.into_iter().fold(0.0, |m, x| m.max(x.abs()))
    }

    /// The layer transports add up to the barotropic transport, at the nodes
    /// and on every face, and walls carry nothing.
    #[test]
    fn layer_transports_add_up_to_the_barotropic_transport() {
        for case in [Case::closed(), Case::open()] {
            layer_transports_add_up(&case);
        }
    }

    fn layer_transports_add_up(case: &Case) {
        let transport = case.transport();
        let (nn, nfn, nl) = (
            case.ops.n_nodes,
            case.ops.n_face_nodes,
            case.sigma.n_levels(),
        );
        let scale = max_abs(case.du_hu.iter().chain(&case.du_hv).copied());
        for idx in 0..case.mesh.n_elements * nn {
            let column = idx * nl..(idx + 1) * nl;
            let sum_u: f64 = transport.hu[column.clone()].iter().sum();
            let sum_v: f64 = transport.hv[column].iter().sum();
            assert!((sum_u - case.du_hu[idx]).abs() < 1e-14 * scale);
            assert!((sum_v - case.du_hv[idx]).abs() < 1e-14 * scale);
        }
        for (slot, &du) in case.du_face.iter().enumerate() {
            let sum: f64 = transport.face[slot * nl..(slot + 1) * nl].iter().sum();
            assert!(
                (sum - du).abs() < 1e-14 * scale,
                "face slot {slot}: Σ F_l = {sum}, DU_avg2 = {du}"
            );
        }
        for k in 0..case.mesh.n_elements {
            for f in 0..4 {
                let exterior = case
                    .boundaries
                    .exterior(&case.mesh, ElementIndex::new(k), f);
                if exterior == FaceExterior::Wall {
                    let slots = (k * 4 + f) * nfn * nl..(k * 4 + f + 1) * nfn * nl;
                    assert_eq!(max_abs(transport.face[slots].iter().copied()), 0.0);
                }
            }
        }
    }

    /// The Courant scale the end layers' bound reads is the stage's longest
    /// Euler step over the water depth at each node, and zero (no Courant
    /// number known) without a barotropic pass.
    #[test]
    fn the_courant_scale_is_the_euler_step_over_the_depth() {
        let case = Case::closed();
        let transport = case.transport();
        assert!(!transport.dt_over_depth.is_empty());
        for (idx, &scale) in transport.dt_over_depth.iter().enumerate() {
            let depth = case.state.eta.data[idx] - case.bathymetry.data[idx];
            assert!(depth > 0.0);
            assert!(
                (scale - 1.0 / depth).abs() < 1e-15 * scale,
                "node {idx}: {scale} against Δt_E/D = {}",
                1.0 / depth
            );
        }
        let mut alone = LayerTransport::new(case.mesh.n_elements, &case.ops, case.sigma.n_levels());
        alone.compute(
            &case.state,
            None,
            &case.mesh,
            &case.ops,
            &case.geom,
            &case.sigma,
            &case.bathymetry,
            &case.boundaries,
            &Exterior3D::default(),
            0.0,
        );
        assert!(alone.dt_over_depth.iter().all(|&s| s == 0.0));
    }

    /// At an open face each layer carries the 2D flux in the interior's
    /// vertical profile: `F_l = Q_l·n + Δσ_l (F_2D − Σ_m Q_m·n)`, so the
    /// shear leaves (and enters) with the flow. Before, every layer took the
    /// uniform share `Δσ_l F_2D`, as at a wall.
    #[test]
    fn open_faces_carry_the_interior_layer_profile() {
        let case = Case::open();
        let transport = case.transport();
        let (nn, nfn, nl) = (
            case.ops.n_nodes,
            case.ops.n_face_nodes,
            case.sigma.n_levels(),
        );
        let d_sigma = case.sigma.d_sigma();
        let scale = max_abs(transport.face.iter().copied());
        let (mut max_err, mut max_shear) = (0.0_f64, 0.0_f64);
        for k in 0..case.mesh.n_elements {
            for f in 0..4 {
                let el = ElementIndex::new(k);
                if !matches!(
                    case.boundaries.exterior(&case.mesh, el, f),
                    FaceExterior::Open(_)
                ) {
                    continue;
                }
                for (fi, &node) in case.ops.face_nodes[f].iter().enumerate() {
                    let (nx, ny) = case.geom.normal(k, f, fi);
                    let column = (k * nn + node) * nl;
                    let q: Vec<f64> = (0..nl)
                        .map(|l| nx * transport.hu[column + l] + ny * transport.hv[column + l])
                        .collect();
                    let q_sum: f64 = q.iter().sum();
                    let slot = (k * 4 + f) * nfn + fi;
                    for l in 0..nl {
                        let expected = q[l] + d_sigma[l] * (case.du_face[slot] - q_sum);
                        max_err = max_err.max((transport.face[slot * nl + l] - expected).abs());
                        max_shear = max_shear.max((q[l] - d_sigma[l] * q_sum).abs());
                    }
                }
            }
        }
        assert!(
            max_shear > 0.05 * scale,
            "test regime: no shear at the open faces"
        );
        assert!(
            max_err < 1e-14 * scale,
            "open-face layer fluxes off the interior profile by {max_err:.3e}"
        );
    }

    /// With `∂η/∂t = −∇·DU_avg2` (the barotropic pass's nodal identity), Ω
    /// integrated from the bed closes at the surface to round-off: nothing is
    /// left for the linear correction.
    #[test]
    fn omega_vanishes_at_the_surface_without_correction() {
        for case in [Case::closed(), Case::periodic(), Case::open()] {
            let transport = case.transport();
            let scale = max_abs(transport.omega.iter().copied());
            assert!(scale > 1e-4, "test flow should drive a non-trivial Ω");
            assert!(
                transport.surface_residual < 1e-12 * scale,
                "surface residual {:.2e} (Ω scale {scale:.2e})",
                transport.surface_residual
            );
        }
    }

    /// Constancy: a uniform tracer changes its inventory exactly as its layer
    /// thickness changes, `∂(H_z C)/∂t = C Δσ_l ∂η/∂t`, so `C` stays put. The
    /// boundary value must not matter (walls carry no volume).
    #[test]
    fn uniform_tracer_follows_the_layer_thickness() {
        let c = 34.7;
        let fixed = FixedTracerBC3D::new(5.0);
        let same = FixedTracerBC3D::new(c);
        let cases: [(Case, [&dyn TracerBoundaryCondition3D; 2]); 2] = [
            (Case::closed(), [&ExtrapolationTracerBC3D, &fixed]),
            // Open: the inflow must be the same water
            (Case::open(), [&ExtrapolationTracerBC3D, &same]),
        ];
        for (case, bcs) in cases {
            uniform_tracer_follows(&case, c, bcs);
        }
    }

    fn uniform_tracer_follows(case: &Case, c: f64, bcs: [&dyn TracerBoundaryCondition3D; 2]) {
        let transport = case.transport();
        let tracer = vec![c; case.state.temp.len()];
        let (nn, nl) = (case.ops.n_nodes, case.sigma.n_levels());
        let scale = c * max_abs(case.eta_rate.iter().copied());
        for bc in bcs {
            let rhs = case.tracer_rhs(&transport, &tracer, bc);
            for idx in 0..case.mesh.n_elements * nn {
                for l in 0..nl {
                    let expected = c * case.sigma.d_sigma()[l] * case.eta_rate[idx];
                    let got = rhs[idx * nl + l];
                    assert!(
                        (got - expected).abs() < 1e-12 * scale,
                        "node {idx}, layer {l}: {got:.6e} vs C·Δσ·∂η/∂t = {expected:.6e}"
                    );
                }
            }
        }
    }

    /// Conservation: the inventory tendency integrates to zero in a closed
    /// basin, whatever the tracer's boundary value (the P0.22 regression: heat
    /// and salt must not cross a coastline that water cannot), and on a
    /// periodic mesh.
    #[test]
    fn tracer_inventory_is_conserved() {
        let bcs: [&dyn TracerBoundaryCondition3D; 3] = [
            &ExtrapolationTracerBC3D,
            &FixedTracerBC3D::new(5.0),
            &UpwindTracerBC3D::new(5.0),
        ];
        for case in [Case::closed(), Case::periodic()] {
            let transport = case.transport();
            let hu_scale = max_abs(transport.hu.iter().copied());
            let scale = case.integral(&vec![hu_scale / 1000.0; case.state.temp.len()])
                * max_abs(case.state.temp.iter().copied());
            for bc in bcs {
                let rhs = case.tracer_rhs(&transport, &case.state.temp, bc);
                let tendency = case.integral(&rhs);
                assert!(
                    tendency.abs() < 1e-12 * scale,
                    "inventory tendency {tendency:.3e} (advective scale {scale:.3e})"
                );
            }
        }
    }

    /// Open faces: the inventory changes by exactly the tracer carried through
    /// them, `d/dt ∫ Σ_l H_z C = −∮ Σ_l F_l C_up` (the interior's value on
    /// outflow, the boundary condition's on inflow): nothing is created or
    /// lost at the boundary.
    #[test]
    fn open_boundaries_change_the_inventory_by_the_boundary_flux() {
        let case = Case::open();
        let transport = case.transport();
        let (nn, nfn, nl) = (
            case.ops.n_nodes,
            case.ops.n_face_nodes,
            case.sigma.n_levels(),
        );
        let tracer = &case.state.temp;
        let bcs: [&dyn TracerBoundaryCondition3D; 2] =
            [&ExtrapolationTracerBC3D, &UpwindTracerBC3D::new(5.0)];
        for bc in bcs {
            let tendency = case.integral(&case.tracer_rhs(&transport, tracer, bc));
            let (mut outflow, mut scale) = (0.0, 0.0);
            for k in 0..case.mesh.n_elements {
                let el = ElementIndex::new(k);
                for f in 0..4 {
                    let FaceExterior::Open(tag) = case.boundaries.exterior(&case.mesh, el, f)
                    else {
                        continue;
                    };
                    for (fi, &node) in case.ops.face_nodes[f].iter().enumerate() {
                        let slot = (k * 4 + f) * nfn + fi;
                        let weight = case.ops.weights_1d[fi] * case.geom.surface_jacobian(k, f, fi);
                        for l in 0..nl {
                            let flux = transport.face[slot * nl + l];
                            let interior = tracer[(k * nn + node) * nl + l];
                            let c = if flux >= 0.0 {
                                interior
                            } else {
                                bc.exterior_value(&TracerBCContext3D {
                                    element: k,
                                    face: f,
                                    level: l,
                                    face_node: fi,
                                    boundary_tag: Some(tag),
                                    interior_value: interior,
                                    normal_velocity: flux,
                                })
                            };
                            outflow += weight * flux * c;
                            scale += (weight * flux * c).abs();
                        }
                    }
                }
            }
            assert!(
                (tendency + outflow).abs() < 1e-12 * scale,
                "inventory tendency {tendency:.6e} vs boundary outflow {outflow:.6e}"
            );
        }
    }

    /// Vertically the tracer is upwinded on Ω: upward flux takes the lower
    /// layer's value, downward the upper layer's. Inventory form: the flux
    /// difference is not divided by `H_z`.
    #[test]
    fn vertical_tracer_flux_is_upwind() {
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 1, 1);
        let ops = DGOperators2D::new(1);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let nl = 3;
        let mut transport = LayerTransport::new(1, &ops, nl);
        let tracer: Vec<f64> = (0..ops.n_nodes).flat_map(|_| [1.0, 2.0, 4.0]).collect();
        for (w, expected) in [(1.0, [-1.0, -1.0, 2.0]), (-1.0, [2.0, 2.0, -4.0])] {
            for column in transport.omega.chunks_exact_mut(nl + 1) {
                column.copy_from_slice(&[0.0, w, w, 0.0]);
            }
            let mut rhs = vec![0.0; tracer.len()];
            apply_tracer_transport_3d(
                [AdvectedTracer {
                    rhs: &mut rhs,
                    tracer: &tracer,
                    bc: &ExtrapolationTracerBC3D,
                    exterior: None,
                    reference: None,
                    range: UNBOUNDED,
                }],
                &transport,
                &mesh,
                &ops,
                &geom,
                &Boundaries3D::default(),
                VerticalAdvection::Upwind,
            );
            for column in rhs.chunks_exact(nl) {
                for (got, want) in column.iter().zip(expected) {
                    assert!((got - want).abs() < 1e-14, "Ω = {w}: {column:?}");
                }
            }
        }
    }

    const ALL_SCHEMES: [VerticalAdvection; 6] = [
        VerticalAdvection::Centred,
        VerticalAdvection::Upwind,
        VerticalAdvection::Tvd,
        VerticalAdvection::Akima,
        VerticalAdvection::LimitedAkima,
        VerticalAdvection::HermiteMean,
    ];

    /// The vertical tracer tendency of `tracer` (one column per node of a
    /// single P1 element, 4 stretched layers, no horizontal transport) for
    /// `Ω` = `omega` at the σ-surfaces, about `reference` if any.
    fn vertical_tendency(
        scheme: VerticalAdvection,
        tracer: &[f64],
        reference: Option<&[f64]>,
        omega: [f64; 5],
    ) -> Vec<f64> {
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 1, 1);
        let ops = DGOperators2D::new(1);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let mut transport = LayerTransport::new(1, &ops, 4);
        transport.d_sigma.copy_from_slice(&[0.4, 0.3, 0.2, 0.1]);
        for column in transport.omega.as_chunks_mut::<5>().0 {
            column.copy_from_slice(&omega);
        }
        let mut rhs = vec![0.0; tracer.len()];
        apply_tracer_transport_3d(
            [AdvectedTracer {
                rhs: &mut rhs,
                tracer,
                bc: &ExtrapolationTracerBC3D,
                exterior: None,
                reference,
                range: UNBOUNDED,
            }],
            &transport,
            &mesh,
            &ops,
            &geom,
            &Boundaries3D::default(),
            scheme,
        );
        rhs
    }

    /// One 4-layer column, bed first, repeated at the 4 nodes of a P1 element.
    fn columns(column: [f64; 4]) -> Vec<f64> {
        (0..4).flat_map(|_| column).collect()
    }

    /// Two up and one down: both upwind sides at the interior surfaces.
    const OMEGAS: [[f64; 5]; 2] = [[0.0, 0.3, -0.15, 0.3, 0.0], [0.0, -0.3, 0.15, -0.3, 0.0]];

    /// A curved stratification (warm above a pycnocline), where the schemes
    /// differ from centred.
    const STRATIFIED: [f64; 4] = [6.9, 7.0, 9.8, 10.0];

    /// About a reference equal to the tracer, every scheme's vertical flux is
    /// centred's (TODO P1.3: the background's surface values must be the
    /// arithmetic mean), as is a column stratified like the reference at a
    /// fraction of its strength.
    #[test]
    fn vertical_flux_about_its_own_stratification_is_centred() {
        let reference = columns(STRATIFIED);
        for omega in OMEGAS {
            for scaled in [1.0, 0.6, 0.1] {
                let tracer: Vec<f64> = reference.iter().map(|r| 3.0 + scaled * r).collect();
                let centred = vertical_tendency(VerticalAdvection::Centred, &tracer, None, omega);
                for scheme in ALL_SCHEMES {
                    let about = vertical_tendency(scheme, &tracer, Some(&reference), omega);
                    let error = about
                        .iter()
                        .zip(&centred)
                        .map(|(a, c)| (a - c).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        error < 1e-13,
                        "{scheme:?} at {scaled}: off centred by {error:.2e}"
                    );
                    if scheme != VerticalAdvection::Centred && scaled == 1.0 {
                        let plain = vertical_tendency(scheme, &tracer, None, omega);
                        assert!(
                            plain
                                .iter()
                                .zip(&centred)
                                .any(|(a, c)| (a - c).abs() > 1e-3),
                            "{scheme:?} should differ from centred on a curved profile"
                        );
                    }
                }
            }
        }
    }

    /// A uniform tracer stays uniform about any reference (constancy: its
    /// tendency is `C δΩ`, the layer thickness's change), and the vertical
    /// fluxes conserve the column's inventory about any reference.
    #[test]
    fn vertical_flux_about_a_reference_keeps_constancy_and_inventory() {
        let reference = columns(STRATIFIED);
        let uniform = columns([4.0; 4]);
        let mixed = columns([7.2, 6.8, 9.9, 9.0]);
        for omega in OMEGAS {
            for scheme in ALL_SCHEMES {
                let about = vertical_tendency(scheme, &uniform, Some(&reference), omega);
                for column in about.as_chunks::<4>().0 {
                    for (l, &r) in column.iter().enumerate() {
                        let want = -4.0 * (omega[l + 1] - omega[l]);
                        assert!(
                            (r - want).abs() < 1e-14,
                            "{scheme:?} level {l}: {r} against {want}"
                        );
                    }
                }
                let about = vertical_tendency(scheme, &mixed, Some(&reference), omega);
                for column in about.as_chunks::<4>().0 {
                    let sum: f64 = column.iter().sum();
                    assert!(sum.abs() < 1e-14, "{scheme:?}: the column gained {sum:.2e}");
                }
            }
        }
    }

    /// A front far sharper than the reference keeps the scheme's limiting:
    /// with the bounded schemes its surface value stays between the two
    /// layers'.
    #[test]
    fn vertical_flux_about_a_reference_keeps_a_front_bounded() {
        let reference = columns([7.0, 7.1, 7.2, 7.3]);
        let front = columns([7.0, 7.0, 12.0, 12.0]);
        let omega = [0.0, 0.3, 0.3, 0.3, 0.0];
        for scheme in [
            VerticalAdvection::Upwind,
            VerticalAdvection::Tvd,
            VerticalAdvection::LimitedAkima,
        ] {
            let plain = vertical_tendency(scheme, &front, None, omega);
            let about = vertical_tendency(scheme, &front, Some(&reference), omega);
            // Below the front, layer 1 passes up water at a surface value
            // between 7 and 12 and gets 7: its tendency lies in
            // [−0.3·(12 − 7), 0] for a bounded surface value
            for (rhs, name) in [(&plain, "plain"), (&about, "about")] {
                for column in rhs.as_chunks::<4>().0 {
                    assert!(
                        (-1.5 - 1e-14..=1e-14).contains(&column[1]),
                        "{scheme:?} {name}: {column:?}"
                    );
                }
            }
        }
    }

    /// Forward-Euler steps of a column's layer inventories `H_z C` and
    /// thicknesses under the vertical advection of `scheme` alone, for `Ω` at
    /// the σ-surfaces (`omega`, bed first, up positive; no horizontal
    /// transport), `dt` apart: the concentrations after each step
    /// (`H_z' = −δΩ`, `(H_z C)' = −δ(Ω C_s)`), and the column's inventory.
    /// The column's depth is that of its layers, which no step changes
    /// (`Ω` is zero at both ends), so `shed`'s `dt_over_depth` is `dt/Σ hz`
    /// (or zero, for a bound that does not know the Courant number).
    fn euler_steps(
        scheme: VerticalAdvection,
        shed: EndLayerShed,
        mut column: Vec<f64>,
        mut hz: Vec<f64>,
        omega: &[f64],
        dt: f64,
        steps: usize,
    ) -> (Vec<Vec<f64>>, f64) {
        let nl = column.len();
        let d_sigma = vec![1.0 / nl as f64; nl];
        let (mut slope, mut surface) = (vec![0.0; nl], vec![0.0; nl + 1]);
        let inventory =
            |column: &[f64], hz: &[f64]| column.iter().zip(hz).map(|(c, h)| c * h).sum();
        let before: f64 = inventory(&column, &hz);
        let mut history = Vec::new();
        for _ in 0..steps {
            scheme.surface_values(&column, omega, &d_sigma, shed, &mut slope, &mut surface);
            for l in 0..nl {
                let q = column[l] * hz[l]
                    - dt * (omega[l + 1] * surface[l + 1] - omega[l] * surface[l]);
                hz[l] -= dt * (omega[l + 1] - omega[l]);
                column[l] = q / hz[l];
            }
            history.push(column.clone());
        }
        let after: f64 = inventory(&column, &hz);
        assert!(
            (after - before).abs() < 1e-12 * before.abs().max(1.0),
            "{scheme:?}: the column's inventory went from {before} to {after}"
        );
        (history, after)
    }

    /// A surface layer that sheds water downward as the water column thins
    /// under it (the landward flank of a sill, where the bed rises under the
    /// brackish layer of a river), over a halocline: the surface layer is
    /// fresh (0.03) and the layers below salt up linearly. A linear profile
    /// through the three means is negative at the surface, so the unbounded
    /// second-order value for the water that leaves takes more salt out of the
    /// layer than it has, and its mean falls through zero; within the range the
    /// layer's salinity only falls by the share of the layer that leaves.
    #[test]
    fn a_surface_layer_shedding_water_over_a_halocline_stays_in_range() {
        let shed = EndLayerShed {
            range: (0.0, 42.0),
            dt_over_depth: 10.0 / 60.0,
        };
        // Bed first; all the layers thick against the shedding
        let column = vec![33.0, 33.0, 25.0, 1.8, 0.9, 0.03];
        let hz = vec![10.0; 6];
        // Downward through the surface layer's lower surface only: that layer
        // thins by a tenth of its thickness per step, the one under it grows
        let mut omega = vec![0.0; 7];
        omega[5] = -0.1;
        for scheme in [VerticalAdvection::LimitedAkima, VerticalAdvection::Tvd] {
            let (history, _) = euler_steps(
                scheme,
                EndLayerShed::NONE,
                column.clone(),
                hz.clone(),
                &omega,
                10.0,
                4,
            );
            let unbounded = history.iter().map(|c| c[5]).fold(f64::INFINITY, f64::min);
            assert!(
                unbounded < -0.01,
                "{scheme:?}: without the range the surface layer ended at {unbounded}, not \
                 below zero: the test does not test"
            );
            let (history, _) =
                euler_steps(scheme, shed, column.clone(), hz.clone(), &omega, 10.0, 4);
            let bounded = history.iter().map(|c| c[5]).fold(f64::INFINITY, f64::min);
            assert!(
                bounded >= 0.0,
                "{scheme:?}: the surface layer fell to {bounded}"
            );
            // and no other layer leaves the range either
            for column in &history {
                assert!(
                    column.iter().all(|&c| (0.0..=33.0 + 1e-12).contains(&c)),
                    "{scheme:?}: {column:?}"
                );
            }
        }
    }

    /// The same at the bed: a bed layer at the range's top, shedding water
    /// upward, stays below it.
    #[test]
    fn a_bed_layer_shedding_water_stays_in_range() {
        let shed = EndLayerShed {
            range: (-2.0, 40.0),
            dt_over_depth: 10.0 / 60.0,
        };
        // A bed layer at the top of the range under a steep thermocline
        let column = vec![39.99, 39.1, 38.2, 20.0, 10.0, 5.0];
        let hz = vec![10.0; 6];
        let mut omega = vec![0.0; 7];
        omega[1] = 0.1;
        for scheme in [VerticalAdvection::LimitedAkima, VerticalAdvection::Tvd] {
            let (history, _) = euler_steps(
                scheme,
                EndLayerShed::NONE,
                column.clone(),
                hz.clone(),
                &omega,
                10.0,
                4,
            );
            let unbounded = history
                .iter()
                .map(|c| c[0])
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(
                unbounded > 40.01,
                "{scheme:?}: {unbounded}: the test does not test"
            );
            let (history, _) =
                euler_steps(scheme, shed, column.clone(), hz.clone(), &omega, 10.0, 4);
            let bounded = history
                .iter()
                .map(|c| c[0])
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(
                bounded <= 40.0,
                "{scheme:?}: the bed layer rose to {bounded}"
            );
        }
    }

    /// The same layers at an outflow Courant number beyond ½ (the explicit
    /// part of the default implicit split reaches 0.8, the step bound
    /// without it 1): the layer's mean falls by `c δ/(1 − c)` of the
    /// increment `δ`, which a profile within the range does not bound
    /// beyond ½. One forward-Euler step, as in a stage.
    #[test]
    fn a_layer_shedding_water_at_a_high_courant_number_stays_in_range() {
        let (dt, depth) = (10.0, 60.0);
        let surface_column = vec![33.0, 33.0, 25.0, 1.8, 0.9, 0.03];
        let bed_column = vec![39.99, 39.1, 38.2, 20.0, 10.0, 5.0];
        let hz = vec![10.0; 6];
        for courant in [0.55, 0.6, 0.8, 0.95] {
            for scheme in [VerticalAdvection::LimitedAkima, VerticalAdvection::Tvd] {
                // `Ω dt/H_z = c` with H_z = 10 m
                let mut down = vec![0.0; 7];
                down[5] = -courant;
                let mut up = vec![0.0; 7];
                up[1] = courant;
                for (column, omega, range, end) in [
                    (&surface_column, &down, (0.0, 42.0), 5),
                    (&bed_column, &up, (-2.0, 40.0), 0),
                ] {
                    let step = |dt_over_depth| {
                        let shed = EndLayerShed {
                            range,
                            dt_over_depth,
                        };
                        let (history, _) =
                            euler_steps(scheme, shed, column.clone(), hz.clone(), omega, dt, 1);
                        history[0].clone()
                    };
                    // Without the Courant number the end layer ends out of the range
                    let blind = step(0.0);
                    assert!(
                        blind[end] < range.0 || blind[end] > range.1,
                        "{scheme:?}, c = {courant}: the blind bound ended at {} within {range:?}: \
                         the test does not test",
                        blind[end]
                    );
                    let aware = step(dt / depth);
                    assert!(
                        aware
                            .iter()
                            .all(|&c| c >= range.0 - 1e-12 && c <= range.1 + 1e-12),
                        "{scheme:?}, c = {courant}: {aware:?} leaves {range:?}"
                    );
                }
            }
        }
    }

    /// Away from the range's ends the bound changes nothing, to the bit, and
    /// a mean outside the range leaves first order.
    #[test]
    fn the_range_leaves_a_column_in_its_interior_alone() {
        let column = [10.0, 10.5, 11.2, 11.9, 12.1, 12.4];
        let d_sigma = [1.0 / 6.0; 6];
        for scheme in [VerticalAdvection::LimitedAkima, VerticalAdvection::Tvd] {
            for w in [1.0, -1.0] {
                let mut omega = [w; 7];
                (omega[0], omega[6]) = (0.0, 0.0);
                let at = |range| {
                    let (mut slope, mut surface) = ([0.0; 6], [0.0; 7]);
                    let shed = EndLayerShed {
                        range,
                        dt_over_depth: 0.0,
                    };
                    scheme.surface_values(
                        &column,
                        &omega,
                        &d_sigma,
                        shed,
                        &mut slope,
                        &mut surface,
                    );
                    surface
                };
                assert_eq!(at((-2.0, 40.0)), at(UNBOUNDED), "{scheme:?}, Ω = {w}");
            }
        }
        // A mean beyond the range keeps only an increment towards it: 41 above
        // a range ending at 40 has a profile ending at 41 + 0.3 or 41 - 0.3
        assert_eq!(in_range(-0.3, 41.0, (-2.0, 40.0), 0.0), 0.0);
        assert_eq!(in_range(0.3, 41.0, (-2.0, 40.0), 0.0), 0.3);
        assert_eq!(in_range(0.3, -3.0, (-2.0, 40.0), 0.0), 0.0);
        assert_eq!(in_range(-0.3, -3.0, (-2.0, 40.0), 0.0), -0.3);
        // The profile's end value `mean − increment` stays within the range
        assert_eq!(in_range(5.0, 1.0, (0.0, 42.0), 0.0), 1.0);
        assert_eq!(in_range(-5.0, 41.0, (0.0, 42.0), 0.0), -1.0);
        assert_eq!(in_range(0.5, 1.0, (0.0, 42.0), 0.0), 0.5);
        assert_eq!(in_range(0.5, 1.0, UNBOUNDED, 0.0), 0.5);
    }

    /// Up to a Courant number of ½ the range is the profile's, bit for bit; beyond
    /// it the increment is `(1 − c)/c` of the distance from the mean to the
    /// bound, none at 1 and above; without a bound nothing changes.
    #[test]
    fn the_range_shrinks_with_the_courant_number_beyond_one_half() {
        let range = (0.0, 42.0);
        for c in [0.0, 0.25, 0.5] {
            assert_eq!(in_range(5.0, 1.0, range, c), 1.0, "c = {c}");
            assert_eq!(in_range(-5.0, 41.0, range, c), -1.0, "c = {c}");
        }
        assert!((in_range(5.0, 1.0, range, 0.8) - 0.25).abs() < 1e-15);
        assert!((in_range(-5.0, 41.0, range, 0.6) + 0.4 / 0.6).abs() < 1e-15);
        for c in [1.0, 1.5, 100.0] {
            assert_eq!(in_range(5.0, 1.0, range, c), 0.0, "c = {c}");
            assert_eq!(in_range(-5.0, 41.0, range, c), 0.0, "c = {c}");
        }
        // An increment already inside the share is kept
        assert_eq!(in_range(0.1, 1.0, range, 0.8), 0.1);
        // No bound, no Courant number to keep
        for c in [0.0, 0.8, 3.0] {
            assert_eq!(in_range(5.0, 1.0, UNBOUNDED, c), 5.0, "c = {c}");
            assert_eq!(in_range(-5.0, 1.0, UNBOUNDED, c), -5.0, "c = {c}");
        }
        // One bound: the other side is free
        assert_eq!(in_range(-5.0, 1.0, (0.0, f64::INFINITY), 0.8), -5.0);
        assert!((in_range(5.0, 1.0, (0.0, f64::INFINITY), 0.8) - 0.25).abs() < 1e-15);
    }

    /// Surface values of `scheme` on `column` with layers `d_sigma` and a
    /// uniform `Ω = w` through the interior surfaces.
    fn surfaces(scheme: VerticalAdvection, column: &[f64], d_sigma: &[f64], w: f64) -> Vec<f64> {
        let nl = column.len();
        let mut omega = vec![w; nl + 1];
        (omega[0], omega[nl]) = (0.0, 0.0);
        let (mut slope, mut surface) = (vec![0.0; nl], vec![f64::NAN; nl + 1]);
        let none = EndLayerShed::NONE;
        scheme.surface_values(column, &omega, d_sigma, none, &mut slope, &mut surface);
        surface
    }

    /// Layers stretched by up to 2.5× between neighbours, and a linear
    /// profile `C = 3 + 2z` sampled as layer means (their centre values).
    fn stretched_linear() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let d_sigma = vec![0.05, 0.125, 0.2, 0.25, 0.375];
        let mut faces = vec![-1.0];
        for ds in &d_sigma {
            faces.push(faces.last().unwrap() + ds);
        }
        let column = (0..5).map(|l| 3.0 + (faces[l] + faces[l + 1])).collect();
        (column, d_sigma, faces)
    }

    /// The Akima surface values: exact for a linear profile (on stretched
    /// levels too), the plain average next to an extremum (zero slope
    /// there), and independent of the sign of Ω (centred).
    #[test]
    fn vertical_akima_values_are_centred_and_exact_for_linear_profiles() {
        let akima = VerticalAdvection::Akima;
        let linear = [1.0, 3.0, 5.0, 7.0, 9.0];
        for w in [1.0, -1.0] {
            let surface = surfaces(akima, &linear, &[1.0; 5], w);
            for l in 1..linear.len() {
                assert!((surface[l] - (2.0 * l as f64)).abs() < 1e-14, "{surface:?}");
            }
        }
        let (column, d_sigma, faces) = stretched_linear();
        for w in [1.0, -1.0] {
            let surface = surfaces(akima, &column, &d_sigma, w);
            for l in 1..5 {
                let want = 3.0 + 2.0 * faces[l];
                assert!((surface[l] - want).abs() < 1e-13, "{surface:?}");
            }
        }
        // Harmonic-mean slopes d = [1, 4/3, 2] (one-sided at the ends):
        // C_{1/2} = ½(1 + 2 − (4/3 − 1)/3), C_{3/2} = ½(2 + 4 − (2 − 4/3)/3),
        // ROMS's index-space formula on uniform levels
        let surface = surfaces(akima, &[1.0, 2.0, 4.0], &[1.0; 3], 1.0);
        assert!((surface[1] - 13.0 / 9.0).abs() < 1e-14, "{surface:?}");
        assert!((surface[2] - 26.0 / 9.0).abs() < 1e-14, "{surface:?}");
        // The same on uniform levels of any thickness
        let thin = surfaces(akima, &[1.0, 2.0, 4.0], &[0.1; 3], 1.0);
        assert!((thin[1] - 13.0 / 9.0).abs() < 1e-13, "{thin:?}");
        // Level 1 is a maximum: its slope is zero
        let peaked = [1.0, 3.0, 2.0, 1.0];
        let surface = surfaces(akima, &peaked, &[1.0; 4], -1.0);
        let d = [2.0, 0.0, -1.0, -1.0];
        for l in 1..4 {
            let want = 0.5 * (peaked[l - 1] + peaked[l] - (d[l] - d[l - 1]) / 3.0);
            assert!((surface[l] - want).abs() < 1e-14, "{surface:?}");
        }
        // One level: nothing to reconstruct
        assert_eq!(surfaces(akima, &[4.0], &[1.0], 1.0), vec![0.0, 0.0]);
    }

    /// The TVD surface values: third-order upwind `(5C_up + 2C_down −
    /// C_far)/6` where the limiter allows it, linear out of an end layer,
    /// exact for a linear profile (on stretched levels too), first order at
    /// an extremum, and never outside the two neighbouring layers' values.
    #[test]
    fn vertical_tvd_values_are_third_order_upwind_and_bounded() {
        let tvd = VerticalAdvection::Tvd;
        // Smooth and monotone: 1, 2, 4, 7 (unlimited increments)
        let column = [1.0, 2.0, 4.0, 7.0];
        let up = surfaces(tvd, &column, &[1.0; 4], 1.0);
        assert_eq!(up[1], 1.5, "linear out of the bed layer");
        assert!((up[2] - (5.0 * 2.0 + 2.0 * 4.0 - 1.0) / 6.0).abs() < 1e-14);
        assert!((up[3] - (5.0 * 4.0 + 2.0 * 7.0 - 2.0) / 6.0).abs() < 1e-14);
        let down = surfaces(tvd, &column, &[1.0; 4], -1.0);
        assert!((down[1] - (5.0 * 2.0 + 2.0 * 1.0 - 4.0) / 6.0).abs() < 1e-14);
        assert!((down[2] - (5.0 * 4.0 + 2.0 * 2.0 - 7.0) / 6.0).abs() < 1e-14);
        assert_eq!(down[3], 5.5, "linear out of the surface layer");

        let (column, d_sigma, faces) = stretched_linear();
        for w in [1.0, -1.0] {
            let surface = surfaces(tvd, &column, &d_sigma, w);
            for l in 1..5 {
                let want = 3.0 + 2.0 * faces[l];
                assert!((surface[l] - want).abs() < 1e-13, "Ω = {w}: {surface:?}");
            }
        }

        // A step and a spike: no value outside its neighbours
        for column in [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ] {
            for w in [1.0, -1.0] {
                let surface = surfaces(tvd, &column, &[1.0; 6], w);
                for l in 1..6 {
                    let (lo, hi) = (column[l - 1].min(column[l]), column[l - 1].max(column[l]));
                    assert!(
                        (lo..=hi).contains(&surface[l]),
                        "Ω = {w}: {surface:?} for {column:?}"
                    );
                }
            }
        }
        // Upwind of the spike's peak (an extremum): first order
        let surface = surfaces(tvd, &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], &[1.0; 6], 1.0);
        assert_eq!(surface[3], 1.0);
    }

    /// The limited Akima surface values: Akima's where they lie within the
    /// TVD bounds (a smooth monotone profile, a linear one on stretched
    /// levels), never outside the two neighbouring layers' values, and the
    /// upwind layer's value next to an extremum. At the foot of a front
    /// Akima's value is clipped to the upwind gradient.
    #[test]
    fn vertical_limited_akima_values_are_akimas_within_the_tvd_bounds() {
        let (akima, limited) = (VerticalAdvection::Akima, VerticalAdvection::LimitedAkima);
        for w in [1.0, -1.0] {
            let column = [1.0, 2.0, 4.0, 7.0, 11.0];
            assert_eq!(
                surfaces(limited, &column, &[1.0; 5], w),
                surfaces(akima, &column, &[1.0; 5], w),
                "Ω = {w}"
            );
            let (column, d_sigma, faces) = stretched_linear();
            let surface = surfaces(limited, &column, &d_sigma, w);
            for l in 1..5 {
                let want = 3.0 + 2.0 * faces[l];
                assert!((surface[l] - want).abs() < 1e-13, "Ω = {w}: {surface:?}");
            }
        }

        // The foot of a front: harmonic-mean slopes 0, 0.18, 0 at layers
        // 1–3, so Akima's value at surface 3 is ½(0.1 + 1 + 0.18/3) = 0.58,
        // 0.48 above the upwind layer's 0.1 when the gradient behind it is
        // 0.1: clipped to 0.2
        let front = [0.0, 0.0, 0.1, 1.0, 1.0, 1.0];
        let unlimited = surfaces(akima, &front, &[1.0; 6], 1.0);
        assert!((unlimited[3] - 0.58).abs() < 1e-14, "{unlimited:?}");
        let clipped = surfaces(limited, &front, &[1.0; 6], 1.0);
        assert!((clipped[3] - 0.2).abs() < 1e-14, "{clipped:?}");
        for column in [front, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] {
            for w in [1.0, -1.0] {
                let surface = surfaces(limited, &column, &[1.0; 6], w);
                for l in 1..6 {
                    let (lo, hi) = (column[l - 1].min(column[l]), column[l - 1].max(column[l]));
                    assert!(
                        (lo..=hi).contains(&surface[l]),
                        "Ω = {w}: {surface:?} for {column:?}"
                    );
                }
            }
        }
        // Upwind of the spike's peak (an extremum): first order
        let surface = surfaces(limited, &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], &[1.0; 6], 1.0);
        assert_eq!(surface[3], 1.0);
    }

    /// Out of an end layer the surface value is bounded by the end layer's
    /// own gradient (`end_layer_limited`). A surface layer at a plateau over
    /// a front below it, the water sinking (a light gravity current's head):
    /// its water leaves at its own mean. A value between the two layers'
    /// means took cooler water out of the warmest layer, which was warmed
    /// past the column's range by the warm water flowing in along it (P2
    /// lock exchange on 125 m without viscosity: 6.0e-5 °C with limited
    /// Akima, 2.4e-2 °C with TVD). The same at the bed, rising; a linear
    /// profile (on stretched levels too) and a smooth curved one keep
    /// their values.
    #[test]
    fn vertical_values_out_of_an_end_layer_at_a_plateau_are_its_mean() {
        let plateau = [7.5, 8.0, 10.0, 12.4, 12.5];
        let bed_plateau: Vec<f64> = plateau.iter().rev().copied().collect();
        for scheme in [VerticalAdvection::LimitedAkima, VerticalAdvection::Tvd] {
            let sinking = surfaces(scheme, &plateau, &[1.0; 5], -1.0);
            assert_eq!(sinking[4], 12.5, "{scheme:?}: {sinking:?}");
            let rising = surfaces(scheme, &bed_plateau, &[1.0; 5], 1.0);
            assert_eq!(rising[1], 12.5, "{scheme:?}: {rising:?}");
            // Flattening less sharply (the difference next to the end layer
            // half the one beyond): kept, TVD's linear 13.0, Akima's 13.11
            let curved = surfaces(scheme, &[0.0, 4.0, 8.0, 12.0, 14.0], &[1.0; 5], -1.0);
            assert!((curved[4] - 13.0).abs() < 0.5, "{scheme:?}: {curved:?}");
            let (column, d_sigma, faces) = stretched_linear();
            for w in [1.0, -1.0] {
                let surface = surfaces(scheme, &column, &d_sigma, w);
                for l in [1, 4] {
                    let want = 3.0 + 2.0 * faces[l];
                    assert!((surface[l] - want).abs() < 1e-13, "{scheme:?}: {surface:?}");
                }
            }
        }
    }

    /// `column` advected up a column of `d_sigma` layers by `Ω = w` for
    /// `steps` SSP-RK3 steps at the Courant number `courant` (on the thinnest
    /// layer), `scheme` at the interior surfaces, the same water flowing in
    /// at the bed and out at the surface.
    fn advect_column(
        scheme: VerticalAdvection,
        mut column: Vec<f64>,
        d_sigma: &[f64],
        courant: f64,
        steps: usize,
    ) -> Vec<f64> {
        let nl = column.len();
        let w = 1.0;
        let dt = courant * d_sigma.iter().copied().fold(f64::MAX, f64::min) / w;
        let omega = vec![w; nl + 1];
        let (mut slope, mut surface) = (vec![0.0; nl], vec![0.0; nl + 1]);
        let mut rate = |c: &[f64]| -> Vec<f64> {
            let none = EndLayerShed::NONE;
            scheme.surface_values(c, &omega, d_sigma, none, &mut slope, &mut surface);
            (surface[0], surface[nl]) = (c[0], c[nl - 1]);
            (0..nl)
                .map(|l| -w * (surface[l + 1] - surface[l]) / d_sigma[l])
                .collect()
        };
        for _ in 0..steps {
            let q0 = column.clone();
            let k: Vec<f64> = rate(&column);
            let q1: Vec<f64> = q0.iter().zip(&k).map(|(q, k)| q + dt * k).collect();
            let k: Vec<f64> = rate(&q1);
            let q2: Vec<f64> = (0..nl)
                .map(|l| 0.75 * q0[l] + 0.25 * (q1[l] + dt * k[l]))
                .collect();
            let k: Vec<f64> = rate(&q2);
            column = (0..nl)
                .map(|l| q0[l] / 3.0 + 2.0 / 3.0 * (q2[l] + dt * k[l]))
                .collect();
        }
        column
    }

    /// A front advected 16 layers up a column: TVD and limited Akima stay
    /// within [0, 1] and spread it over 6 layers, upwind over 18; Akima
    /// overshoots by 16 % (and spreads it over 19). A halocline-like tanh
    /// profile: the error is 800× smaller than upwind's with Akima and with
    /// limited Akima (whose limiter does not act on it: the same error to
    /// the last digit), 60× with TVD (whose limiter acts in the profile's
    /// tails), and 450× and 40× on stretched layers.
    #[test]
    fn vertical_schemes_on_a_front_and_a_smooth_profile() {
        use VerticalAdvection::{Akima, LimitedAkima, Tvd, Upwind};
        let nl = 100;
        let uniform = vec![1.0 / nl as f64; nl];
        let front: Vec<f64> = (0..nl).map(|l| if l < 30 { 1.0 } else { 0.0 }).collect();
        // 40 layers at 0.4 per step: the front moves 16 layers
        let run = |scheme| advect_column(scheme, front.clone(), &uniform, 0.4, 40);
        let (upwind, akima, tvd) = (run(Upwind), run(Akima), run(Tvd));
        let limited = run(LimitedAkima);
        let range = |c: &[f64]| {
            c.iter()
                .fold((f64::MAX, f64::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)))
        };
        let smeared = |c: &[f64]| c.iter().filter(|&&x| x > 0.01 && x < 0.99).count();
        for (name, c) in [("TVD", &tvd), ("limited Akima", &limited)] {
            let (lo, hi) = range(c);
            assert!(
                lo >= -1e-12 && hi <= 1.0 + 1e-12,
                "{name} left [0, 1]: [{lo}, {hi}]"
            );
            assert!(
                smeared(c) * 2 < smeared(&upwind),
                "{name}: front over {} layers, upwind {}",
                smeared(c),
                smeared(&upwind)
            );
        }
        let (lo, hi) = range(&upwind);
        assert!(
            lo >= -1e-12 && hi <= 1.0 + 1e-12,
            "upwind left [0, 1]: [{lo}, {hi}]"
        );
        let (lo, hi) = range(&akima);
        assert!(
            hi > 1.1,
            "test regime: Akima should overshoot: [{lo}, {hi}]"
        );

        // A smooth profile on uniform and stretched layers (2.5 % thicker
        // from each layer to the next), advected 16 thin layers
        let stretched: Vec<f64> = {
            let raw: Vec<f64> = (0..nl).map(|l| 1.05_f64.powf(l as f64 / 2.0)).collect();
            let total: f64 = raw.iter().sum();
            raw.iter().map(|r| r / total).collect()
        };
        for d_sigma in [&uniform, &stretched] {
            let mut faces = vec![0.0];
            for ds in d_sigma {
                faces.push(faces.last().unwrap() + ds);
            }
            // Layer means of a halocline-like ½(1 + tanh((σ − 0.4)/0.08)), and
            // of it shifted by the travel w·t
            let mean = |shift: f64| -> Vec<f64> {
                let primitive = |x: f64| {
                    let y = (x - shift - 0.4) / 0.08;
                    0.5 * (x + 0.08 * y.cosh().ln())
                };
                (0..nl)
                    .map(|l| (primitive(faces[l + 1]) - primitive(faces[l])) / d_sigma[l])
                    .collect()
            };
            let steps = 40;
            let thinnest = d_sigma.iter().copied().fold(f64::MAX, f64::min);
            let travel = 0.4 * thinnest * steps as f64;
            let exact = mean(travel);
            let error = |scheme| {
                let c = advect_column(scheme, mean(0.0), d_sigma, 0.4, steps);
                // Away from the inflow at the bed and the outflow at the
                // surface (both first order here)
                max_abs((nl / 4..nl - 4).map(|l| c[l] - exact[l]))
            };
            let (upwind, akima, tvd) = (error(Upwind), error(Akima), error(Tvd));
            let limited = error(LimitedAkima);
            assert!(
                akima * 200.0 < upwind && tvd * 20.0 < upwind && limited < 1.01 * akima,
                "smooth profile: upwind {upwind:.3e}, Akima {akima:.3e}, TVD {tvd:.3e},                  limited Akima {limited:.3e}"
            );
        }
    }

    impl Case {
        /// Momentum inventory tendencies `(∂(H_z u)/∂t, ∂(H_z v)/∂t)` of the
        /// advection of `(u, v)` in `form`.
        fn momentum_rhs(
            &self,
            transport: &LayerTransport,
            u: &[f64],
            v: &[f64],
            form: MomentumAdvectionForm,
        ) -> (Vec<f64>, Vec<f64>) {
            let mut rhs_u = vec![0.0; u.len()];
            let mut rhs_v = vec![0.0; v.len()];
            apply_momentum_transport_3d(
                &mut rhs_u,
                &mut rhs_v,
                u,
                v,
                transport,
                &self.mesh,
                &self.ops,
                &self.geom,
                &self.boundaries,
                None,
                form,
                VerticalAdvection::Centred,
            );
            (rhs_u, rhs_v)
        }
    }

    /// Momentum constancy: a velocity uniform in space changes its inventory
    /// exactly as the layer thickness changes, `∂(H_z u)/∂t = u Δσ_l ∂η/∂t`,
    /// so it stays uniform, through walls, open faces and periodic ones, in
    /// either form. (The
    /// old velocity-form advection, `∇·(u u)` with its own Rusanov flux and
    /// `Ω` only in the vertical term, was not built on the layer transports.)
    #[test]
    fn uniform_velocity_follows_the_layer_thickness() {
        let (u0, v0) = (0.7, -0.3);
        for case in [Case::closed(), Case::open(), Case::periodic()] {
            let transport = case.transport();
            let n = case.state.u.len();
            for form in [
                MomentumAdvectionForm::Split,
                MomentumAdvectionForm::Conservative,
            ] {
                let (rhs_u, rhs_v) =
                    case.momentum_rhs(&transport, &vec![u0; n], &vec![v0; n], form);
                let (nn, nl) = (case.ops.n_nodes, case.sigma.n_levels());
                let scale = max_abs(case.eta_rate.iter().copied());
                for idx in 0..case.mesh.n_elements * nn {
                    for l in 0..nl {
                        let thickness_rate = case.sigma.d_sigma()[l] * case.eta_rate[idx];
                        for (c, rhs) in [(u0, &rhs_u), (v0, &rhs_v)] {
                            let got = rhs[idx * nl + l];
                            assert!(
                                (got - c * thickness_rate).abs() < 1e-12 * scale,
                                "{form:?}, node {idx}, layer {l}: {got:.6e} vs u·Δσ·∂η/∂t = {:.6e}",
                                c * thickness_rate
                            );
                        }
                    }
                }
            }
        }
    }

    /// Conservation: the momentum advection moves momentum between layers and
    /// elements but, summed over the layers, creates none in a closed basin
    /// (walls carry no volume, so no momentum) or on a periodic mesh, in
    /// either form.
    #[test]
    fn momentum_advection_conserves_the_column_momentum() {
        for case in [Case::closed(), Case::periodic()] {
            let transport = case.transport();
            for form in [
                MomentumAdvectionForm::Split,
                MomentumAdvectionForm::Conservative,
            ] {
                let (rhs_u, rhs_v) =
                    case.momentum_rhs(&transport, &case.state.u, &case.state.v, form);
                let hu_scale = max_abs(transport.hu.iter().copied());
                let u_scale = max_abs(case.state.u.iter().chain(&case.state.v).copied());
                let scale = case.integral(&vec![hu_scale * u_scale / 1000.0; rhs_u.len()]);
                assert!(
                    max_abs(rhs_u.iter().copied()) > 1e-3 * hu_scale * u_scale / 1000.0,
                    "test regime: no advection"
                );
                for rhs in [rhs_u, rhs_v] {
                    let tendency = case.integral(&rhs);
                    assert!(
                        tendency.abs() < 1e-12 * scale,
                        "{form:?}: momentum tendency {tendency:.3e} (advective scale {scale:.3e})"
                    );
                }
            }
        }
    }

    /// TODO P4.5 gate: the split form keeps the kinetic energy. In a closed
    /// basin or on a periodic mesh, with a sheared, non-uniform flow and
    /// barotropically corrected layer transports, the momentum advection in
    /// split form with centred `Ω` fluxes changes the kinetic energy
    /// `½ ∫ Σ_l H_z,l |u_l|²` only by the upwind faces' dissipation,
    ///
    /// ```text
    ///     ∫ Σ_l [u·A_u + v·A_v − ½|u|² ∂H_z,l/∂t] = −½ ∮ Σ_l |F_l| |u_l − u_l′|²
    /// ```
    ///
    /// (GLL summation by parts: the split volume term and the continuity's
    /// strong form leave `F(φ − φ′)[½(φ + φ′) − φ↑]` at every face, the
    /// centred σ-surface values nothing). The conservative form's volume
    /// term leaves an aliasing remainder of either sign: the energy source
    /// that fed the P2 lock exchange's grid-scale instability.
    #[test]
    fn split_momentum_advection_only_dissipates_kinetic_energy() {
        for case in [Case::closed(), Case::periodic()] {
            let transport = case.transport();
            let (nn, nfn, nl) = (
                case.ops.n_nodes,
                case.ops.n_face_nodes,
                case.sigma.n_levels(),
            );
            // Rough at the node scale, discontinuous across the faces
            let offset = |idx: usize| 0.05 * (1.37 * idx as f64).sin();
            let u: Vec<f64> = (case.state.u.iter().enumerate())
                .map(|(idx, &u)| u + offset(idx))
                .collect();
            let v: Vec<f64> = (case.state.v.iter().enumerate())
                .map(|(idx, &v)| v - 0.5 * offset(idx))
                .collect();
            let (u, v) = (&u[..], &v[..]);
            // ½ ∮ Σ_l |F_l| |Δu|², each face once (from both sides, halved)
            let mut dissipation = 0.0;
            for k in 0..case.mesh.n_elements {
                for f in 0..4 {
                    let el = ElementIndex::new(k);
                    let FaceExterior::Element(nb) = case.boundaries.exterior(&case.mesh, el, f)
                    else {
                        continue;
                    };
                    for (fi, &node) in case.ops.face_nodes[f].iter().enumerate() {
                        let slot = (k * 4 + f) * nfn + fi;
                        let weight = case.ops.weights_1d[fi] * case.geom.surface_jacobian(k, f, fi);
                        let nb_node = case.ops.face_nodes[nb.face][nfn - 1 - fi];
                        let (a, b) = ((k * nn + node) * nl, (nb.element * nn + nb_node) * nl);
                        for l in 0..nl {
                            let jump =
                                (u[a + l] - u[b + l]).powi(2) + (v[a + l] - v[b + l]).powi(2);
                            dissipation +=
                                0.25 * weight * transport.face[slot * nl + l].abs() * jump;
                        }
                    }
                }
            }
            let energy_rate = |form| {
                let (rhs_u, rhs_v) = case.momentum_rhs(&transport, u, v, form);
                let mut rate = vec![0.0; u.len()];
                for (idx, r) in rate.iter_mut().enumerate() {
                    let thickness_rate = case.sigma.d_sigma()[idx % nl] * case.eta_rate[idx / nl];
                    *r = u[idx] * rhs_u[idx] + v[idx] * rhs_v[idx]
                        - 0.5 * (u[idx].powi(2) + v[idx].powi(2)) * thickness_rate;
                }
                case.integral(&rate)
            };
            let split = energy_rate(MomentumAdvectionForm::Split);
            let conservative = energy_rate(MomentumAdvectionForm::Conservative);
            // Measured: split to 1e-15, conservative 5.5e-3 (closed) and
            // 7.6e-4 (periodic) of the dissipation short, energy the
            // aliasing made
            assert!(dissipation > 0.0, "test regime: no face jumps");
            assert!(
                (split + dissipation).abs() < 1e-10 * dissipation,
                "split form: kinetic energy rate {split:.6e} against the faces' −{dissipation:.6e}"
            );
            assert!(
                (conservative + dissipation).abs() > 1e-4 * dissipation,
                "test regime: the conservative form ({conservative:.6e}) has no aliasing"
            );
        }
    }

    /// Open faces: the column momentum changes by exactly what the layer
    /// fluxes carry out, `−∮ Σ_l F_l u_l`, with the interior's velocity both
    /// ways (zero gradient).
    #[test]
    fn open_boundaries_change_the_momentum_by_the_boundary_flux() {
        let case = Case::open();
        let transport = case.transport();
        let (nn, nfn, nl) = (
            case.ops.n_nodes,
            case.ops.n_face_nodes,
            case.sigma.n_levels(),
        );
        let (rhs_u, rhs_v) = case.momentum_rhs(
            &transport,
            &case.state.u,
            &case.state.v,
            MomentumAdvectionForm::default(),
        );
        for (rhs, field) in [(&rhs_u, &case.state.u), (&rhs_v, &case.state.v)] {
            let tendency = case.integral(rhs);
            let (mut outflow, mut scale) = (0.0, 0.0);
            for k in 0..case.mesh.n_elements {
                for f in 0..4 {
                    let exterior = case
                        .boundaries
                        .exterior(&case.mesh, ElementIndex::new(k), f);
                    if !matches!(exterior, FaceExterior::Open(_)) {
                        continue;
                    }
                    for (fi, &node) in case.ops.face_nodes[f].iter().enumerate() {
                        let slot = (k * 4 + f) * nfn + fi;
                        let weight = case.ops.weights_1d[fi] * case.geom.surface_jacobian(k, f, fi);
                        for l in 0..nl {
                            let carried = weight
                                * transport.face[slot * nl + l]
                                * field[(k * nn + node) * nl + l];
                            outflow += carried;
                            scale += carried.abs();
                        }
                    }
                }
            }
            assert!(
                (tendency + outflow).abs() < 1e-12 * scale,
                "momentum tendency {tendency:.6e} vs boundary outflow {outflow:.6e}"
            );
        }
    }

    /// Vertically the velocity is centred at the σ-surfaces, in inventory
    /// form: `−(Ω_{l+1/2} u_{l+1/2} − Ω_{l−1/2} u_{l−1/2})`, not divided by
    /// `H_z`. Linear `Ω(σ)` and `u(σ)` make the centred values exact.
    #[test]
    fn vertical_momentum_flux_is_centred() {
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 1, 1);
        let ops = DGOperators2D::new(1);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let sigma = SigmaGrid::uniform(4);
        let nl = sigma.n_levels();
        let omega = |s: f64| 0.03 * (s + 1.0) * s;
        let u = |s: f64| 0.4 - 0.5 * s;
        let mut transport = LayerTransport::new(1, &ops, nl);
        for column in transport.omega.chunks_exact_mut(nl + 1) {
            for (w, &s) in column.iter_mut().zip(sigma.sigma_w()) {
                *w = omega(s);
            }
        }
        let column: Vec<f64> = sigma.sigma_rho().iter().map(|&s| u(s)).collect();
        let field: Vec<f64> = (0..ops.n_nodes).flat_map(|_| column.clone()).collect();
        let (mut rhs_u, mut rhs_v) = (vec![0.0; field.len()], vec![0.0; field.len()]);
        apply_momentum_transport_3d(
            &mut rhs_u,
            &mut rhs_v,
            &field,
            &field,
            &transport,
            &mesh,
            &ops,
            &geom,
            &Boundaries3D::default(),
            None,
            MomentumAdvectionForm::default(),
            VerticalAdvection::Centred,
        );
        let sw = sigma.sigma_w();
        // Ω vanishes at the bed and the surface
        let flux = |f: usize| omega(sw[f]) * u(sw[f]);
        for rhs in [&rhs_u, &rhs_v] {
            for (i, column) in rhs.chunks_exact(nl).enumerate() {
                for (l, got) in column.iter().enumerate() {
                    let expected = -(flux(l + 1) - flux(l));
                    assert!(
                        (got - expected).abs() < 1e-15,
                        "node {i}, layer {l}: {got} vs {expected}"
                    );
                }
            }
        }
    }

    /// `None`: the state's own layer transports, `H_z u` uncorrected, with `Ω`
    /// closed at the surface by the free-surface rate they imply.
    #[test]
    fn own_layer_transports_close_omega() {
        let case = Case::closed();
        let mut transport =
            LayerTransport::new(case.mesh.n_elements, &case.ops, case.sigma.n_levels());
        transport.compute(
            &case.state,
            None,
            &case.mesh,
            &case.ops,
            &case.geom,
            &case.sigma,
            &case.bathymetry,
            &case.boundaries,
            &Exterior3D::default(),
            0.0,
        );
        let (nn, nl) = (case.ops.n_nodes, case.sigma.n_levels());
        for idx in 0..case.mesh.n_elements * nn {
            let depth = case.state.eta.data[idx] - case.bathymetry.data[idx];
            for l in 0..nl {
                let expected = depth * case.sigma.d_sigma()[l] * case.state.u[idx * nl + l];
                assert!((transport.hu[idx * nl + l] - expected).abs() < 1e-14 * expected.abs());
            }
        }
        let scale = max_abs(transport.omega.iter().copied());
        assert!(scale > 1e-4, "test flow should drive a non-trivial Ω");
        assert!(
            transport.surface_residual < 1e-12 * scale,
            "surface residual {:.2e}",
            transport.surface_residual
        );
    }

    /// Ω of the own transports is stored at the w-points and satisfies layer
    /// continuity (the regression of the old `compute_vertical_velocity`,
    /// whose layer-centre storage mixed neighbouring layers). A zero-mean
    /// shear `u_l = U_l sin(2πx)` over a flat bed has no column divergence,
    /// so Ω closes at the surface by itself and each layer's increment
    /// `Ω_{l+1/2} − Ω_{l−1/2} = −Δσ_l ∇·(D u_l)` scales with `U_l`.
    #[test]
    fn own_omega_of_a_zero_mean_shear_has_layer_continuity() {
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 1);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let sigma = SigmaGrid::uniform(4);
        let (nn, nl) = (ops.n_nodes, sigma.n_levels());
        let bathymetry = Bathymetry2D::constant(mesh.n_elements, nn, -10.0);
        let shear: Vec<f64> = sigma.sigma_rho().iter().map(|&s| s + 0.5).collect();
        let mut state = Solution3D::new(mesh.n_elements, nn, nl);
        for k in 0..mesh.n_elements {
            let el = ElementIndex::new(k);
            for i in 0..nn {
                let [x, _] = mesh.reference_to_physical(el, ops.nodes_r[i], ops.nodes_s[i]);
                for (u, &shear_l) in state.u_column_mut(el, i).iter_mut().zip(&shear) {
                    *u = shear_l * (2.0 * PI * x).sin();
                }
            }
        }
        let mut transport = LayerTransport::new(mesh.n_elements, &ops, nl);
        transport.compute(
            &state,
            None,
            &mesh,
            &ops,
            &geom,
            &sigma,
            &bathymetry,
            &Boundaries3D::default(),
            &Exterior3D::default(),
            0.0,
        );
        let scale = max_abs(transport.omega.iter().copied());
        assert!(scale > 1e-3, "test flow should drive a non-trivial Ω");
        let tol = 1e-12 * scale;
        let d_sigma = sigma.d_sigma();
        for col in transport.omega.chunks_exact(nl + 1) {
            assert!(col[0].abs() < tol && col[nl].abs() < tol, "{col:?}");
            let rate = (col[1] - col[0]) / (shear[0] * d_sigma[0]);
            for l in 0..nl {
                let increment = col[l + 1] - col[l];
                let expected = rate * shear[l] * d_sigma[l];
                assert!(
                    (increment - expected).abs() < tol,
                    "layer {l}: Ω increment {increment}, continuity requires {expected}"
                );
            }
        }
    }

    /// A nesting parent at every node of the open case: a sheared velocity
    /// unlike the interior's, and `temp`, as `[u, v, temp]` per node and
    /// layer (slot = node).
    struct Parent {
        tags: Vec<BoundaryTag>,
        slot_of_node: Vec<u32>,
        fields: [Vec<f64>; 3],
        n_levels: usize,
    }

    impl Parent {
        fn new(case: &Case, temp: impl Fn(f64) -> f64) -> Self {
            let nl = case.sigma.n_levels();
            let n = case.state.u.len() / nl;
            let mut fields: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::with_capacity(n * nl));
            for _ in 0..n {
                for &s in case.sigma.sigma_rho() {
                    fields[0].push(0.3 - 0.4 * s);
                    fields[1].push(0.2 * s);
                    fields[2].push(temp(s));
                }
            }
            Self {
                tags: vec![BoundaryTag::Open],
                slot_of_node: (0..n as u32).collect(),
                fields,
                n_levels: nl,
            }
        }

        fn field(&self, i: usize) -> ExteriorField<'_> {
            ExteriorField {
                tags: &self.tags,
                slot_of_node: &self.slot_of_node,
                n_levels: self.n_levels,
                values: &self.fields[i],
            }
        }

        fn exterior(&self) -> Exterior3D<'_> {
            Exterior3D {
                velocity: Some([self.field(0), self.field(1)]),
                temp: Some(self.field(2)),
                salt: None,
            }
        }
    }

    /// With a nesting parent, an open face's layer fluxes are the central
    /// average of the interior's and the parent's `H_z u`, corrected to the
    /// 2D flux, which they still add up to.
    #[test]
    fn open_faces_average_the_parent_profile() {
        let case = Case::open();
        let parent = Parent::new(&case, |s| 5.0 + s);
        let transport = case.transport_with(&parent.exterior());
        let (nn, nfn, nl) = (
            case.ops.n_nodes,
            case.ops.n_face_nodes,
            case.sigma.n_levels(),
        );
        let d_sigma = case.sigma.d_sigma();
        let scale = max_abs(transport.face.iter().copied());
        let mut open_faces = 0;
        for k in 0..case.mesh.n_elements {
            for f in 0..4 {
                let el = ElementIndex::new(k);
                if case.boundaries.exterior(&case.mesh, el, f)
                    != FaceExterior::Open(BoundaryTag::Open)
                {
                    continue;
                }
                open_faces += 1;
                for (fi, &node) in case.ops.face_nodes[f].iter().enumerate() {
                    let (nx, ny) = case.geom.normal(k, f, fi);
                    let flat = k * nn + node;
                    let depth = case.state.eta.data[flat] - case.bathymetry.data[flat];
                    let central: Vec<f64> = (0..nl)
                        .map(|l| {
                            let (i, p) = (flat * nl + l, flat * nl + l);
                            let q_in = nx * transport.hu[i] + ny * transport.hv[i];
                            let q_ex = depth
                                * d_sigma[l]
                                * (nx * parent.fields[0][p] + ny * parent.fields[1][p]);
                            0.5 * (q_in + q_ex)
                        })
                        .collect();
                    let slot = (k * 4 + f) * nfn + fi;
                    let correction = case.du_face[slot] - central.iter().sum::<f64>();
                    let fluxes = &transport.face[slot * nl..(slot + 1) * nl];
                    for l in 0..nl {
                        let expected = central[l] + d_sigma[l] * correction;
                        assert!((fluxes[l] - expected).abs() < 1e-14 * scale);
                    }
                    let sum: f64 = fluxes.iter().sum();
                    assert!((sum - case.du_face[slot]).abs() < 1e-14 * scale);
                }
            }
        }
        assert!(open_faces > 0);
    }

    /// Water flowing in through a nested open face brings the parent's
    /// tracer and velocity (and the tracer boundary condition is not
    /// consulted): the inventories change by exactly `−∮ Σ_l F_l φ↑`, with
    /// `φ↑` the parent's on inflow and the interior's on outflow.
    #[test]
    fn nested_inflow_brings_the_parent_values() {
        let case = Case::open();
        let parent = Parent::new(&case, |s| 5.0 + s);
        let exterior = parent.exterior();
        let transport = case.transport_with(&exterior);
        let (nn, nfn, nl) = (
            case.ops.n_nodes,
            case.ops.n_face_nodes,
            case.sigma.n_levels(),
        );
        // Tracer: a boundary condition that would give nonsense
        let mut rhs_t = vec![0.0; case.state.temp.len()];
        apply_tracer_transport_3d(
            [AdvectedTracer {
                rhs: &mut rhs_t,
                tracer: &case.state.temp,
                bc: &FixedTracerBC3D::new(-1e6),
                exterior: exterior.temp,
                reference: None,
                range: UNBOUNDED,
            }],
            &transport,
            &case.mesh,
            &case.ops,
            &case.geom,
            &case.boundaries,
            VerticalAdvection::default(),
        );
        let (mut rhs_u, mut rhs_v) = (vec![0.0; rhs_t.len()], vec![0.0; rhs_t.len()]);
        apply_momentum_transport_3d(
            &mut rhs_u,
            &mut rhs_v,
            &case.state.u,
            &case.state.v,
            &transport,
            &case.mesh,
            &case.ops,
            &case.geom,
            &case.boundaries,
            exterior.velocity,
            MomentumAdvectionForm::default(),
            VerticalAdvection::Centred,
        );
        let fields = [
            (&rhs_t, &case.state.temp, 2),
            (&rhs_u, &case.state.u, 0),
            (&rhs_v, &case.state.v, 1),
        ];
        for (rhs, field, p) in fields {
            let tendency = case.integral(rhs);
            let (mut outflow, mut scale, mut inflows) = (0.0, 0.0, 0);
            for k in 0..case.mesh.n_elements {
                for f in 0..4 {
                    let el = ElementIndex::new(k);
                    if !matches!(
                        case.boundaries.exterior(&case.mesh, el, f),
                        FaceExterior::Open(_)
                    ) {
                        continue;
                    }
                    for (fi, &node) in case.ops.face_nodes[f].iter().enumerate() {
                        let slot = (k * 4 + f) * nfn + fi;
                        let weight = case.ops.weights_1d[fi] * case.geom.surface_jacobian(k, f, fi);
                        for l in 0..nl {
                            let flux = transport.face[slot * nl + l];
                            let idx = (k * nn + node) * nl + l;
                            let value = if flux >= 0.0 {
                                field[idx]
                            } else {
                                inflows += 1;
                                parent.fields[p][idx]
                            };
                            outflow += weight * flux * value;
                            scale += (weight * flux * value).abs();
                        }
                    }
                }
            }
            assert!(inflows > 0, "test regime: no inflow");
            assert!(
                (tendency + outflow).abs() < 1e-12 * scale,
                "tendency {tendency:.6e} vs boundary outflow {outflow:.6e}"
            );
        }
    }

    /// Constancy with a nesting parent: a uniform tracer, and a parent of the
    /// same tracer, stays uniform however the parent shears the boundary
    /// fluxes (the layers still carry exactly the water the free surface
    /// moved).
    #[test]
    fn uniform_tracer_stays_uniform_with_a_parent() {
        let case = Case::open();
        let c = 34.7;
        let parent = Parent::new(&case, |_| c);
        let exterior = parent.exterior();
        let transport = case.transport_with(&exterior);
        let nl = case.sigma.n_levels();
        let tracer = vec![c; case.state.temp.len()];
        let mut rhs = vec![0.0; tracer.len()];
        apply_tracer_transport_3d(
            [AdvectedTracer {
                rhs: &mut rhs,
                tracer: &tracer,
                bc: &ExtrapolationTracerBC3D,
                exterior: exterior.temp,
                reference: None,
                range: UNBOUNDED,
            }],
            &transport,
            &case.mesh,
            &case.ops,
            &case.geom,
            &case.boundaries,
            VerticalAdvection::default(),
        );
        let scale = c * max_abs(case.eta_rate.iter().copied());
        for (idx, column) in rhs.chunks_exact(nl).enumerate() {
            for (l, got) in column.iter().enumerate() {
                let expected = c * case.sigma.d_sigma()[l] * case.eta_rate[idx];
                assert!(
                    (got - expected).abs() < 1e-12 * scale,
                    "{got} vs {expected}"
                );
            }
        }
    }

    /// Inventory ↔ concentration: nodal in unmarked elements; in marked
    /// elements the element's inventory over its volume per level plus each
    /// node's departure from the mean of its last value, which converts back
    /// to the same element inventory and keeps the element's structure; a
    /// dry node, or a dry marked element, keeps its last concentration.
    #[test]
    fn inventory_conversion_keeps_element_inventories() {
        let case = Case::closed();
        let (nn, nl) = (case.ops.n_nodes, case.sigma.n_levels());
        let n_el = case.mesh.n_elements;
        // Element 0 dry at one node, element 1 dry everywhere, element 2
        // marked, element 3 marked and dry at one node
        let mut eta = case.state.eta.data.clone();
        eta[0] = case.bathymetry.data[0];
        eta[nn..2 * nn].copy_from_slice(&case.bathymetry.data[nn..2 * nn]);
        eta[3 * nn] = case.bathymetry.data[3 * nn];
        let mut marked = vec![false; n_el];
        marked[1] = true;
        marked[2] = true;
        marked[3] = true;
        let concentration = case.state.temp.clone();
        let mut inventory = concentration.clone();
        to_inventory(&mut inventory, &eta, case.sigma.d_sigma(), &case.bathymetry);
        let last = vec![-1.0; concentration.len()];
        let mut out = last.clone();
        from_inventory(
            &inventory,
            &eta,
            case.sigma.d_sigma(),
            &case.bathymetry,
            &case.geom,
            &marked,
            &mut out,
        );
        for k in 0..n_el {
            for i in 0..nn {
                let idx = k * nn + i;
                let dry = eta[idx] - case.bathymetry.data[idx] < DRY_DEPTH;
                for l in 0..nl {
                    let (c, got) = (concentration[idx * nl + l], out[idx * nl + l]);
                    match (k, dry) {
                        // Dry: the last value; the other nodes nodal.
                        // Regression (Frøya, 2026-10-16): a marked element's
                        // dry node took every step's shift of the level mean
                        // and drifted (33 °C on a dry shore)
                        (0 | 3, true) | (1, _) => {
                            assert_eq!(got, -1.0, "element {k} node {i}")
                        }
                        (2 | 3, _) => {}
                        _ => assert!((got - c).abs() < 1e-12 * c.abs(), "{got} vs {c}"),
                    }
                }
            }
        }
        // Element 2: one value per level, and its inventory per level kept
        let mut back = out.clone();
        to_inventory(&mut back, &eta, case.sigma.d_sigma(), &case.bathymetry);
        for l in 0..nl {
            let level = |field: &[f64]| -> f64 {
                (0..nn)
                    .map(|i| case.geom.node_mass(2, i) * field[(2 * nn + i) * nl + l])
                    .sum()
            };
            let value = out[(2 * nn) * nl + l];
            for i in 0..nn {
                assert_eq!(out[(2 * nn + i) * nl + l], value);
            }
            let (before, after) = (level(&inventory), level(&back));
            assert!(
                (before - after).abs() < 1e-13 * before.abs(),
                "level {l}: {before} vs {after}"
            );
            // Element 3 too, with its dry node
            let level_3 = |field: &[f64]| -> f64 {
                (0..nn)
                    .map(|i| case.geom.node_mass(3, i) * field[(3 * nn + i) * nl + l])
                    .sum()
            };
            let (before, after) = (level_3(&inventory), level_3(&back));
            assert!(
                (before - after).abs() < 1e-13 * before.abs(),
                "element 3, level {l}: {before} vs {after}"
            );
        }

        // Regression (Frøya, 2026-10-03): a marked element keeps each node's
        // departure from the level's mean, so a stratified element converts
        // back to itself. With its last values the stratification that made
        // its inventory, it is unchanged; the mean alone flattened it, and
        // over a steep shoreline that drove 0.1–1 m/s at rest
        let mut kept = vec![-1.0; concentration.len()];
        let element_2 = 2 * nn * nl..3 * nn * nl;
        kept[element_2.clone()].copy_from_slice(&concentration[element_2.clone()]);
        from_inventory(
            &inventory,
            &eta,
            case.sigma.d_sigma(),
            &case.bathymetry,
            &case.geom,
            &marked,
            &mut kept,
        );
        let spread = concentration[element_2.clone()]
            .iter()
            .fold((f64::MAX, f64::MIN), |(a, b), &c| (a.min(c), b.max(c)));
        assert!(
            spread.1 - spread.0 > 1e-3,
            "test regime: element 2 is stratified"
        );
        for (got, c) in kept[element_2.clone()]
            .iter()
            .zip(&concentration[element_2])
        {
            assert!((got - c).abs() < 1e-12 * c.abs(), "{got} vs {c}");
        }
    }

    /// The w-cells' σ-thicknesses: half the end layers, the means between,
    /// summing to one on stretched levels.
    #[test]
    fn w_cells_span_the_column() {
        let sigma = SigmaGrid::new(
            6,
            crate::vertical::SongHaidvogelStretching::new(5.0, 0.4, 10.0),
        );
        let d = sigma.d_sigma();
        let mut w = vec![0.0; 7];
        w_cell_thicknesses(d, &mut w);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        assert_eq!(w[0], 0.5 * d[0]);
        assert_eq!(w[3], 0.5 * (d[2] + d[3]));
        assert_eq!(w[6], 0.5 * d[5]);
    }

    /// The w-cells of the layer transports ([`LayerTransport::stagger_from`])
    /// carry the layers' water: their fluxes add up to the layers', `Ω`
    /// vanishes at the bed and the surface, a uniform field at the w-points
    /// changes its inventory exactly as its w-cell's thickness,
    /// `C Δσ_w ∂η/∂t` (constancy), and a varying one keeps its inventory in
    /// a closed basin and on a periodic mesh (conservation).
    #[test]
    fn w_cells_keep_constancy_and_conservation() {
        for (case, closed) in [
            (Case::closed(), true),
            (Case::open(), false),
            (Case::periodic(), true),
        ] {
            let layers = case.transport();
            let (ne, nn, nl) = (
                case.mesh.n_elements,
                case.ops.n_nodes,
                case.sigma.n_levels(),
            );
            let nw = nl + 1;
            let mut w_cells = LayerTransport::new(ne, &case.ops, nw);
            w_cells.stagger_from(&layers);
            for (cells, layer) in [
                (&w_cells.hu, &layers.hu),
                (&w_cells.hv, &layers.hv),
                (&w_cells.face, &layers.face),
            ] {
                for (w, l) in cells.chunks_exact(nw).zip(layer.chunks_exact(nl)) {
                    let (sum_w, sum_l) = (w.iter().sum::<f64>(), l.iter().sum::<f64>());
                    assert!((sum_w - sum_l).abs() <= 1e-14 * sum_l.abs().max(1.0));
                }
            }
            for omega in w_cells.omega.chunks_exact(nw + 1) {
                assert_eq!((omega[0], omega[nw]), (0.0, 0.0));
            }

            let c = 0.37;
            let n = ne * nn * nw;
            let rhs = case.tracer_rhs(&w_cells, &vec![c; n], &ExtrapolationTracerBC3D);
            let scale = c * max_abs(case.eta_rate.iter().copied());
            for idx in 0..ne * nn {
                for j in 0..nw {
                    let expected = c * w_cells.d_sigma[j] * case.eta_rate[idx];
                    let got = rhs[idx * nw + j];
                    assert!(
                        (got - expected).abs() < 1e-12 * scale,
                        "node {idx}, w-point {j}: {got:.6e} vs C·Δσ_w·∂η/∂t = {expected:.6e}"
                    );
                }
            }

            if !closed {
                continue;
            }
            // A field that varies in x, y and z, upwind horizontally and
            // limited-Akima vertically: the inventory tendency integrates to
            // zero
            let field: Vec<f64> = (0..n)
                .map(|m| {
                    let (idx, j) = (m / nw, m % nw);
                    case.state.temp[idx * nl + j.min(nl - 1)] + 0.5 * j as f64
                })
                .collect();
            let rhs = case.tracer_rhs(&w_cells, &field, &ExtrapolationTracerBC3D);
            let integral = |values: &[f64]| -> f64 {
                (0..ne)
                    .map(|k| {
                        let sums: Vec<f64> = (0..nn)
                            .map(|i| values[(k * nn + i) * nw..][..nw].iter().sum())
                            .collect();
                        case.geom.integrate_element(k, &sums)
                    })
                    .sum()
            };
            let hu_scale = max_abs(w_cells.hu.iter().copied());
            let scale = integral(&vec![hu_scale / 1000.0; n]) * max_abs(field.iter().copied());
            let tendency = integral(&rhs);
            assert!(
                tendency.abs() < 1e-12 * scale,
                "inventory tendency {tendency:.3e} (advective scale {scale:.3e})"
            );
        }
    }

    /// The all-levels kernels are the per-level ones to the bit, on a general
    /// quadrilateral (the averaged metric's pair terms) and a parallelogram,
    /// in both metric forms, the advective one for each field of a pair.
    #[test]
    fn all_levels_kernels_are_the_per_level_kernels_bit_for_bit() {
        let mut distorted = Mesh2D::uniform_rectangle(0.0, LX, 0.0, LY, 3, 2);
        for v in &mut distorted.vertices {
            let [x, y] = *v;
            let bump = 60.0 * (PI * x / LX).sin() * (PI * y / LY).sin();
            *v = [x + bump, y - 0.7 * bump];
        }
        let affine = Mesh2D::uniform_rectangle(0.0, LX, 0.0, LY, 3, 2);
        let (nl, order) = (7, 2);
        let ops = DGOperators2D::new(order);
        let (nn, nfn) = (ops.n_nodes, ops.n_face_nodes);
        // Values of no pattern, of either sign
        let values = |n: usize, seed: f64| -> Vec<f64> {
            (0..n)
                .map(|j| ((j as f64 + seed) * 12.9898).sin() * 43.758_5 % 3.0)
                .collect()
        };
        let (hu, hv) = (values(nn * nl, 0.1), values(nn * nl, 0.2));
        // A pair of fields, as the kernels take u and v, T and S
        let phi = [values(nn * nl, 0.3), values(nn * nl, 0.5)];
        let face = [values(4 * nfn * nl, 0.4), values(4 * nfn * nl, 0.6)];
        // One level's values out of a `[node][level]` block
        let level = |block: &[f64], l: usize| -> Vec<f64> {
            block.iter().skip(l).step_by(nl).copied().collect()
        };
        let mut scratch = LevelScratch::new(nn, nl);
        let mut checked_averaged = false;
        for mesh in [&distorted, &affine] {
            let geom = GeometricFactors2D::compute(mesh, &ops);
            for metric in [MetricForm::Conservative, MetricForm::Averaged] {
                for k in 0..mesh.n_elements {
                    checked_averaged |= metric.averaged_on(&geom, k);
                    let mut transport = vec![0.0; nn * nl];
                    let mut advective = [vec![0.0; nn * nl], vec![0.0; nn * nl]];
                    transport_divergence_levels(
                        &ops,
                        &geom,
                        metric,
                        k,
                        &hu,
                        &hv,
                        &face[0],
                        &mut transport,
                        &mut scratch,
                    );
                    advective_divergence_levels(
                        &ops,
                        &geom,
                        metric,
                        k,
                        &hu,
                        &hv,
                        phi.each_ref().map(|phi| &phi[..]),
                        face.each_ref().map(|face| &face[..]),
                        advective.each_mut().map(|div| &mut div[..]),
                        &mut scratch,
                    );
                    for l in 0..nl {
                        let (hu_l, hv_l) = (level(&hu, l), level(&hv, l));
                        let face_l = level(&face[0], l);
                        let (mut div, mut fr, mut fs) =
                            (vec![0.0; nn], vec![0.0; nn], vec![0.0; nn]);
                        transport_divergence_element(
                            &ops, &geom, metric, k, &hu_l, &hv_l, &face_l, &mut div,
                        );
                        assert_eq!(
                            level(&transport, l)
                                .iter()
                                .map(|x| x.to_bits())
                                .collect::<Vec<_>>(),
                            div.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                            "transport divergence, element {k}, level {l}, {metric:?}"
                        );
                        for t in 0..2 {
                            let (phi_l, face_l) = (level(&phi[t], l), level(&face[t], l));
                            advective_divergence_element(
                                &ops, &geom, metric, k, &hu_l, &hv_l, &phi_l, &face_l, &mut fr,
                                &mut fs, &mut div,
                            );
                            assert_eq!(
                                level(&advective[t], l)
                                    .iter()
                                    .map(|x| x.to_bits())
                                    .collect::<Vec<_>>(),
                                div.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                "advective divergence of field {t}, element {k}, level {l}, {metric:?}"
                            );
                        }
                    }
                }
            }
        }
        assert!(
            checked_averaged,
            "the distorted mesh takes the averaged pair terms"
        );
    }
}
