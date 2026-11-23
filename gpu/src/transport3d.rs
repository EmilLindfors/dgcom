//! The 3D transport gate (P2.7 Phase 4, `notes/gpu.md`): the layer
//! transports and Ω ([`Transport3D::compute`]), the advection of u and v
//! (split form, centred in the vertical), of T and S (LimitedAkima within
//! their ranges, about a reference stratification or not, and then the
//! reference's own advection) and of the turbulence's k and ψ at the
//! w-points, the water no transport carried and the rivers', on the GPU
//! against dgcom on the CPU, in the order
//! `Hydrostatic3D::compute_transport_rhs_into` takes them
//! (`LayerTransport::compute`, `ReferenceFields::fill`,
//! `apply_momentum_transport_3d`, `apply_tracer_transport_3d`,
//! `add_reference_advection`, `LayerTransport::add_volume_sources`,
//! `apply_w_point_transport_3d`, the rivers' `RiverInflow`), value for
//! value:
//!
//! - fully wet, periodic and walled (Phase 4.0);
//! - a walled beach with flats, films and dry columns, where subcell
//!   elements, thin columns, fallback and pooled elements meet (Phase 4.1a);
//! - a channel open at both ends, nested in a parent at one (its velocity,
//!   T and S outside), with fixed and upwind tracer conditions at the other,
//!   and rivers, two in one element (Phase 4.1b);
//! - the reference stratification from a profile (evenly spaced and linear,
//!   cubic, unevenly spaced) at the columns' levels, and its advection
//!   (Phase 4.2);
//!
//! then timed at Chiloé's size (96,000 P2 elements × 20 levels).

use std::cell::RefCell;
use std::sync::Arc;
use std::time::Instant;

use cuda_core::DeviceBuffer;
use dgcom::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::physics::{EquationOfState, LinearEOS};
use dgcom::solver::rhs::subcells::{
    line_metric, line_node, subcell_interfaces, subcell_slot, telescoped_interfaces,
};
use dgcom::solver::rhs::{
    AdvectedTracer, BarotropicFlux, Boundaries3D, DeviceTracerBC3D, Exterior3D, ExteriorField,
    ExtrapolationTracerBC3D, FaceExterior, FixedTracerBC3D, LayerTransport, MetricForm,
    MomentumAdvectionForm, ReferenceFields, TracerBoundaryCondition3D, UNBOUNDED, UpwindTracerBC3D,
    VerticalAdvection, add_reference_advection, apply_momentum_transport_3d,
    apply_tracer_transport_3d, apply_w_point_transport_3d, w_cell_thicknesses,
};
use dgcom::solver::state::Solution3D;
use dgcom::solver::{Blocks, TracerReferenceProfile};
use dgcom::source::{
    CoriolisSource2D, River, RiverInflow, RiverProfile, RiverSeries, RiverSources,
};
use dgcom::types::ElementIndex;
use dgcom::vertical::{ROMSVstretching4, SigmaGrid};

use crate::baroclinic::{Baroclinic3D, BaroclinicSetup};
use crate::phase0::{DX, Lcg, quad_mesh, ulps};
use crate::swe2d::{Gpu, N1, NN, ORDER};
use crate::three_d::{
    Advected, BarotropicDevice, ExteriorDevice, Interface, RiversDevice, Statics3D, Transport3D,
};

/// Gravity and the Boussinesq density of the fixtures.
pub(crate) const G: f64 = 9.81;
pub(crate) const RHO0: f64 = 1025.0;

/// σ-levels of the fixtures (Chiloé's 3D case has 20).
pub(crate) const LEVELS: usize = 20;
/// The tracers' ranges, narrower than the water's in places so that the end
/// layers' bound (`in_range`) acts.
const T_RANGE: (f64, f64) = (5.5, 11.0);
const S_RANGE: (f64, f64) = (31.6, 34.4);
/// The baroclinic step of the barotropic flux (s).
pub(crate) const DT: f64 = 30.0;
/// The tracers' conditions at the channel's open faces without a parent.
const T_BC: f64 = 9.25;
const S_BC: f64 = 33.4;
/// The fields compared, in the order of [`Cpu`]'s.
const NAMES: [&str; 6] = ["T", "S", "u", "v", "k", "psi"];

/// A fixture's shape.
#[derive(Clone, Copy)]
pub(crate) struct Shape {
    pub(crate) nx: usize,
    pub(crate) ny: usize,
    /// Interior vertices moved by up to this share of the element size
    pub(crate) jitter: f64,
    /// Doubly periodic, else walled
    pub(crate) periodic: bool,
    /// A walled beach with flats, films and dry columns (else a bed
    /// 12–28 m deep): subcell elements, thin columns, the pass's subcell
    /// fluxes in some, fallback and pooled elements
    pub(crate) beach: bool,
    /// A channel open at both ends (x = 0 and x = L), nested in a parent at
    /// x = L, with fixed and upwind tracer conditions at x = 0, and rivers
    pub(crate) channel: bool,
    /// ROMS's Vstretching 4 (θ_s 5, θ_b 0.4), else uniform levels
    pub(crate) stretched: bool,
    /// A reference stratification (T and S advected about it, and it by the
    /// pressure gradient's adjoint): evenly spaced and linear between its
    /// samples, cubic on the beaches, unevenly spaced in the channel
    pub(crate) reference: bool,
    /// Every element's forward-Euler step (s): long enough, and end layers
    /// shed more than half their water in it (`in_range`'s share)
    pub(crate) euler_dt: f64,
}

impl Shape {
    /// The wet fixture of 480 elements the others vary.
    pub(crate) const WET: Self = Self {
        nx: 24,
        ny: 20,
        jitter: 0.0,
        periodic: true,
        beach: false,
        channel: false,
        stretched: false,
        reference: false,
        euler_dt: 0.05,
    };
    /// The walled beach of 720 elements.
    pub(crate) const BEACH: Self = Self {
        nx: 36,
        ny: 20,
        periodic: false,
        beach: true,
        reference: true,
        ..Self::WET
    };
    /// The channel of 360 elements, open at both ends.
    pub(crate) const CHANNEL: Self = Self {
        nx: 30,
        ny: 12,
        jitter: 0.15,
        periodic: false,
        channel: true,
        stretched: true,
        reference: true,
        euler_dt: 40.0,
        ..Self::WET
    };

    /// The reference stratification of a fixture of this shape: a smooth
    /// pycnocline about 3 m (beach) or 10 m deep (the fixtures' water's,
    /// less its front and noise), sampled evenly and linear between the
    /// samples (`from_fn`, as `from_mean` is), cubic (`from_smooth_fn`) on
    /// the beaches, at uneven heights in the channel.
    fn profile(self) -> Option<TracerReferenceProfile> {
        if !self.reference {
            return None;
        }
        let z0 = if self.beach { 3.0 } else { 10.0 };
        let smooth = move |z: f64| {
            let t = ((z + z0) / 3.0).tanh();
            (8.0 + 3.0 * t, 33.0 - 1.5 * t)
        };
        Some(if self.beach {
            TracerReferenceProfile::from_smooth_fn(-10.0, 3.0, 27, smooth)
        } else if self.channel {
            let heights = vec![
                -33.0, -27.0, -22.0, -18.0, -15.0, -13.0, -11.5, -10.6, -10.0, -9.3, -8.5, -7.2,
                -5.5, -3.0, -1.0, 0.6, 2.0,
            ];
            let (temp, salt) = heights.iter().map(|&z| smooth(z)).unzip();
            TracerReferenceProfile::new(heights, temp, salt)
        } else {
            TracerReferenceProfile::from_fn(-33.0, 2.0, 71, smooth)
        })
    }
}

/// A nesting parent's values outside the open faces it covers.
struct Parent {
    tags: Vec<BoundaryTag>,
    slot_of_node: Vec<u32>,
    /// u, v, T, S of every slot, `[slot][level]`
    values: [Vec<f64>; 4],
}

impl Parent {
    fn field(&self, f: usize) -> ExteriorField<'_> {
        ExteriorField {
            tags: &self.tags,
            slot_of_node: &self.slot_of_node,
            n_levels: LEVELS,
            values: &self.values[f],
        }
    }
}

/// Rivers with their step's discharge.
struct Rivers {
    sources: RiverSources,
    discharge: Vec<f64>,
    /// Each river's water's T and S at the stage
    conc: [Vec<f64>; 2],
}

/// A mesh, its σ-levels, a stratified, sheared state on it and a
/// barotropic pass that its layers do not sum to.
pub(crate) struct Fixture {
    pub(crate) mesh: Mesh2D,
    pub(crate) ops: DGOperators2D,
    pub(crate) geom: GeometricFactors2D,
    pub(crate) bathymetry: Bathymetry2D,
    pub(crate) sigma: SigmaGrid,
    pub(crate) boundaries: Boundaries3D,
    /// Columns thinner than this exchange no baroclinic transport
    pub(crate) min_column_depth: f64,
    /// The thin columns at rest, as the mode splitter's stage values are;
    /// k and ψ at the w-points; the density of the linear equation of state
    pub(crate) state: Solution3D,
    /// The reference stratification, if any
    pub(crate) profile: Option<TracerReferenceProfile>,
    /// It at the state's levels, with the elements' displacements (the CPU's
    /// `ReferenceFields::fill`, `set_displacements`)
    pub(crate) fields: Option<ReferenceFields>,
    bar_hu: Vec<f64>,
    bar_hv: Vec<f64>,
    /// The pass's mass flux out of every face node
    pub(crate) bar_face: Vec<f64>,
    /// `∂η/∂t` over the step
    pub(crate) eta_rate: Vec<f64>,
    /// Every element's forward-Euler step
    pub(crate) euler_dt: Vec<f64>,
    /// The pass's subcell interface fluxes: NaN in the elements it did not
    /// move on the subcells throughout
    pub(crate) subcell: Vec<f64>,
    /// The elements the pass balanced only as a whole (empty: none)
    pooled: Vec<bool>,
    parent: Option<Parent>,
    /// T's and S's conditions at open faces without a parent
    bcs: [Box<dyn TracerBoundaryCondition3D>; 2],
    rivers: Option<Rivers>,
    /// The momentum's tendencies before the advection adds to them
    rhs_u0: Vec<f64>,
    rhs_v0: Vec<f64>,
}

impl Fixture {
    pub(crate) fn new(shape: Shape) -> Self {
        let Shape {
            nx,
            ny,
            jitter,
            periodic,
            beach,
            channel,
            stretched,
            reference: _,
            euler_dt,
        } = shape;
        let mut mesh = quad_mesh(nx, ny, jitter, periodic && !beach && !channel);
        if channel {
            // Open at x = 0 and x = L (faces 3 and 1)
            for edge in mesh.edges.iter_mut() {
                if edge.right.is_none() && edge.left.face % 2 == 1 {
                    edge.boundary_tag = Some(BoundaryTag::Open);
                }
            }
        }
        let ops = DGOperators2D::new(ORDER);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let (lx, ly) = (nx as f64 * DX, ny as f64 * DX);
        let (kx, ky) = (std::f64::consts::TAU / lx, std::f64::consts::TAU / ly);
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
            if beach {
                // −8 m to +2 m along x, with bumps
                -8.0 + 10.0 * x / lx + 0.6 * (0.01 * y).sin() * (0.004 * x).cos()
            } else {
                -20.0 - 8.0 * (kx * x).sin() * (ky * y).cos()
            }
        });
        let min_column_depth = if beach { 0.5 } else { 1.0 };
        let sigma = if stretched {
            SigmaGrid::new(LEVELS, ROMSVstretching4::default())
        } else {
            SigmaGrid::uniform(LEVELS)
        };
        let boundaries = Boundaries3D::default();
        let ne = mesh.n_elements;
        let (nl, nw) = (LEVELS, LEVELS + 1);
        let mut state = Solution3D::new(ne, NN, nl);
        state.tke = vec![0.0; ne * NN * nw];
        state.gls = vec![0.0; ne * NN * nw];
        let mut rng = Lcg(20_261_123);
        let mut bar_hu = vec![0.0; ne * NN];
        let mut bar_hv = vec![0.0; ne * NN];
        let mut eta_rate = vec![0.0; ne * NN];
        let mut rhs_u0 = vec![0.0; ne * NN * nl];
        let mut rhs_v0 = vec![0.0; ne * NN * nl];
        let sigma_rho = sigma.sigma_rho().to_vec();
        let sigma_w = sigma.sigma_w().to_vec();
        for k in 0..ne {
            let ki = ElementIndex::new(k);
            for i in 0..NN {
                let idx = k * NN + i;
                let [x, y] = mesh.reference_to_physical(ki, ops.nodes_r[i], ops.nodes_s[i]);
                let b = bathymetry.data[idx];
                let depth = if beach {
                    // A shoreline band, and films and dry columns anywhere
                    let r = rng.next();
                    let wet = (0.1 + 0.25 * (0.003 * x + 0.002 * y).sin() - b).max(0.0);
                    if r < 0.02 {
                        0.0
                    } else if r < 0.04 {
                        0.3 * rng.next()
                    } else {
                        wet
                    }
                } else {
                    0.3 * (2.0 * kx * x).cos() * (ky * y).sin() - b
                };
                let eta = b + depth;
                state.eta.data[idx] = eta;
                let at_rest = depth < min_column_depth;
                // A pycnocline about 3 m (beach) or 10 m deep, tilted by a
                // front along x
                let (z0, front) = if beach {
                    (3.0, 0.8 * (0.004 * y).sin())
                } else {
                    (10.0, 2.0 * (kx * x).sin())
                };
                for (l, &s) in sigma_rho.iter().enumerate() {
                    let at = idx * nl + l;
                    let z = eta + s * depth;
                    let (u, v) = (
                        0.6 * (ky * y).sin() + 0.25 * (s + 0.5) * (kx * x).cos(),
                        0.4 * (kx * x).cos() - 0.2 * (s + 0.5) * (ky * y).sin(),
                    );
                    let (du, dv) = (0.02 * (rng.next() - 0.5), 0.02 * (rng.next() - 0.5));
                    // The thin columns at rest: the stage values' mask
                    state.u[at] = if at_rest { 0.0 } else { u + du };
                    state.v[at] = if at_rest { 0.0 } else { v + dv };
                    let interface = ((z + z0 + front) / 1.5).tanh();
                    state.temp[at] = 8.0 + 3.0 * interface + 0.05 * (rng.next() - 0.5);
                    state.salt[at] = 33.0 - 1.5 * interface + 0.02 * (rng.next() - 0.5);
                    rhs_u0[at] = 1e-3 * (rng.next() - 0.5);
                    rhs_v0[at] = 1e-3 * (rng.next() - 0.5);
                }
                // The turbulence: largest at mid-depth and near the bed
                for (j, &s) in sigma_w.iter().enumerate() {
                    let at = idx * nw + j;
                    let shape = 1.0 + (std::f64::consts::PI * s).sin().powi(2);
                    state.tke[at] = 1e-4 * shape * (1.0 + 0.3 * rng.next());
                    state.gls[at] = 2e-5 * shape * (1.0 + 0.3 * rng.next()) + 1e-6 * (x / lx);
                }
                bar_hu[idx] = depth * (0.62 * (ky * y).sin() + 0.01 * (kx * x).cos());
                bar_hv[idx] = depth * (0.38 * (kx * x).cos() - 0.01 * (ky * y).sin());
                eta_rate[idx] = 1e-4 * (kx * x + ky * y).sin();
            }
        }
        // The pass's face fluxes: the central average of DU_avg2 · n, the
        // interior's at an open face (inflow and outflow along it), none at
        // a wall
        let mut bar_face = vec![0.0; ne * 4 * N1];
        for k in 0..ne {
            for f in 0..4 {
                let exterior = boundaries.exterior(&mesh, ElementIndex::new(k), f);
                for fi in 0..N1 {
                    let flat = k * NN + ops.face_nodes[f][fi];
                    let (nx, ny) = geom.normal(k, f, fi);
                    let q = |n: usize| nx * bar_hu[n] + ny * bar_hv[n];
                    bar_face[(k * 4 + f) * N1 + fi] = match exterior {
                        FaceExterior::Element(nb) => {
                            let other = nb.element * NN + ops.face_nodes[nb.face][N1 - 1 - fi];
                            0.5 * (q(flat) + q(other))
                        }
                        FaceExterior::Open(_) => q(flat),
                        FaceExterior::Wall => 0.0,
                    };
                }
            }
        }
        // The pass's subcell fluxes (the beach): in 70 % of the elements
        // with a thin column and 5 % of the others, the central DU_avg2
        // along each interface, perturbed, none beside a dry column; NaN in
        // the rest
        let n_sub = subcell_interfaces(ops.n_1d);
        let mut subcell = vec![f64::NAN; ne * n_sub];
        let mut pooled = Vec::new();
        if beach {
            let depth = |flat: usize| state.eta.data[flat] - bathymetry.data[flat];
            let mut metric = [(0.0, 0.0); N1];
            let mut interfaces = [(0.0, 0.0); N1 + 1];
            for k in 0..ne {
                let thin = (k * NN..(k + 1) * NN).any(|flat| depth(flat) < min_column_depth);
                if rng.next() >= if thin { 0.7 } else { 0.05 } {
                    continue;
                }
                for dir in 0..2 {
                    for line in 0..N1 {
                        let affine = line_metric(&ops, &geom, k, dir, line, &mut metric);
                        telescoped_interfaces(
                            &ops.weights_1d,
                            &ops.dr_1d_row_major,
                            &metric,
                            affine,
                            &mut interfaces,
                        );
                        for a in 0..N1 - 1 {
                            let m = interfaces[a + 1];
                            let (ia, ib) = (
                                k * NN + line_node(N1, dir, line, a),
                                k * NN + line_node(N1, dir, line, a + 1),
                            );
                            let q = |n: usize| m.0 * bar_hu[n] + m.1 * bar_hv[n];
                            subcell[k * n_sub + subcell_slot(N1, dir, line, a)] =
                                if depth(ia) == 0.0 || depth(ib) == 0.0 {
                                    0.0
                                } else {
                                    0.5 * (q(ia) + q(ib)) * (1.0 + 0.2 * (rng.next() - 0.5))
                                };
                        }
                    }
                }
            }
            pooled = (0..ne).map(|_| rng.next() < 0.06).collect();
        }
        // The channel's parent (outside x = L only) and rivers
        let (mut parent, mut rivers) = (None, None);
        let bcs: [Box<dyn TracerBoundaryCondition3D>; 2] = if channel {
            [
                Box::new(FixedTracerBC3D::new(T_BC)),
                Box::new(UpwindTracerBC3D::new(S_BC)),
            ]
        } else {
            [
                Box::new(ExtrapolationTracerBC3D),
                Box::new(ExtrapolationTracerBC3D),
            ]
        };
        if channel {
            let mut slot_of_node = vec![u32::MAX; ne * NN];
            let mut n_slots = 0;
            for edge in mesh.edges.iter().filter(|e| e.right.is_none()) {
                let (k, f) = (edge.left.element, edge.left.face);
                if f != 1 {
                    continue;
                }
                for &node in &ops.face_nodes[f] {
                    slot_of_node[k * NN + node] = n_slots;
                    n_slots += 1;
                }
            }
            let mut values: [Vec<f64>; 4] = Default::default();
            for slot in 0..n_slots as usize {
                for (l, &s) in sigma_rho.iter().enumerate() {
                    let wave = (0.7 * slot as f64).sin();
                    values[0].push(-0.3 + 0.1 * s + 0.05 * wave);
                    values[1].push(0.05 + 0.02 * wave);
                    values[2].push(7.0 - 2.0 * s + 0.2 * wave + 0.01 * l as f64);
                    values[3].push(34.0 + 0.5 * s - 0.1 * wave);
                }
            }
            parent = Some(Parent {
                tags: vec![BoundaryTag::Open],
                slot_of_node,
                values,
            });
            // Three rivers, two of them in one element
            let at = |fx: f64, fy: f64| [fx * lx, fy * ly];
            let list = vec![
                River::new("a", at(0.31, 0.47), 60.0)
                    .with_temperature(RiverSeries::constant(4.0))
                    .with_salinity(RiverSeries::constant(0.5)),
                River::new("b", at(0.312, 0.472), 25.0)
                    .with_temperature(RiverSeries::constant(5.5))
                    .with_salinity(RiverSeries::constant(0.0))
                    .with_profile(RiverProfile::TopFraction(0.3)),
                River::new("c", at(0.73, 0.21), 8.0)
                    .with_temperature(RiverSeries::constant(6.0))
                    .with_salinity(RiverSeries::constant(2.0))
                    .with_profile(RiverProfile::TopMetres(4.0)),
            ];
            let mut sources = RiverSources::new(list, &mesh, &geom, &bathymetry, 500.0, &[])
                .expect("the rivers are in the channel");
            sources
                .set_levels(&sigma, &bathymetry)
                .expect("the rivers' levels");
            assert_eq!(
                sources.element(0),
                sources.element(1),
                "two rivers share an element"
            );
            let conc = [
                (0..sources.len())
                    .map(|i| sources.river(i).temperature.at(0.0))
                    .collect(),
                (0..sources.len())
                    .map(|i| sources.river(i).salinity.at(0.0))
                    .collect(),
            ];
            rivers = Some(Rivers {
                discharge: vec![58.0, 26.0, 7.5],
                sources,
                conc,
            });
        }
        // The density, and the reference at the state's levels
        let eos = LinearEOS::default();
        eos.update_density(&mut state);
        let profile = shape.profile();
        let fields = profile.as_ref().map(|profile| {
            let mut fields = ReferenceFields::default();
            let all = Blocks::All(ne);
            fields.fill(
                profile,
                |t, s| eos.compute_density(t, s, 0.0),
                |t, s| eos.density_partials(t, s),
                &state,
                &bathymetry,
                &sigma,
                all,
            );
            fields.set_displacements(&state, &geom, &bathymetry, &sigma, min_column_depth, all);
            fields
        });
        Self {
            subcell,
            pooled,
            euler_dt: vec![euler_dt; ne],
            mesh,
            ops,
            geom,
            bathymetry,
            sigma,
            boundaries,
            min_column_depth,
            state,
            profile,
            fields,
            bar_hu,
            bar_hv,
            bar_face,
            eta_rate,
            parent,
            bcs,
            rivers,
            rhs_u0,
            rhs_v0,
        }
    }

    fn inflow(&self) -> Option<RiverInflow<'_>> {
        self.rivers.as_ref().map(|r| RiverInflow {
            rivers: &r.sources,
            discharge: &r.discharge,
        })
    }

    fn flux(&self) -> BarotropicFlux<'_> {
        BarotropicFlux {
            dt: DT,
            euler_dt: &self.euler_dt,
            hu: &self.bar_hu,
            hv: &self.bar_hv,
            face: &self.bar_face,
            subcell: &self.subcell,
            eta_rate: &self.eta_rate,
            rivers: self.inflow(),
            pooled: &self.pooled,
        }
    }

    fn exterior(&self) -> Exterior3D<'_> {
        match &self.parent {
            Some(p) => Exterior3D {
                velocity: Some([p.field(0), p.field(1)]),
                temp: Some(p.field(2)),
                salt: Some(p.field(3)),
            },
            None => Exterior3D::default(),
        }
    }

    pub(crate) fn n_elements(&self) -> usize {
        self.mesh.n_elements
    }

    /// The GPU's momentum terms of this fixture over `statics`, with
    /// `coriolis`, the linear equation of state, the fixture's reference
    /// (if any) and, if `balanced`, the CPU's balanced reference state.
    pub(crate) fn baroclinic(
        &self,
        gpu: &Arc<Gpu>,
        statics: &Statics3D,
        coriolis: &CoriolisSource2D,
        balanced: Option<&dgcom::solver::rhs::BalancedReference>,
    ) -> Result<Baroclinic3D, Box<dyn std::error::Error>> {
        Baroclinic3D::new(
            gpu,
            statics,
            &self.mesh,
            &self.ops,
            &self.sigma,
            BaroclinicSetup {
                coriolis,
                eos: LinearEOS::default(),
                profile: self.profile.as_ref(),
                balanced,
                g: G,
                rho0: RHO0,
                min_column_depth: self.min_column_depth,
            },
        )
    }

    fn rhs(&self) -> [Vec<f64>; 6] {
        let (n, nw) = (self.n_elements() * NN * LEVELS, self.state.tke.len());
        [
            vec![0.0; n],
            vec![0.0; n],
            self.rhs_u0.clone(),
            self.rhs_v0.clone(),
            vec![0.0; nw],
            vec![0.0; nw],
        ]
    }

    pub(crate) fn statics(&self, gpu: &Gpu) -> Result<Statics3D, Box<dyn std::error::Error>> {
        self.statics_in(gpu, MetricForm::Averaged)
    }

    /// The fixture's tables with the volume terms in `metric`'s form.
    pub(crate) fn statics_in(
        &self,
        gpu: &Gpu,
        metric: MetricForm,
    ) -> Result<Statics3D, Box<dyn std::error::Error>> {
        Statics3D::upload(
            gpu,
            &self.mesh,
            &self.ops,
            &self.geom,
            &self.bathymetry,
            &self.sigma,
            &self.boundaries,
            metric,
        )
    }
}

/// The CPU's results: the layer transports, the tendencies of T, S, u, v,
/// k and ψ after the advection and after the sources.
struct Cpu {
    layers: LayerTransport,
    advected: [Vec<f64>; 6],
    sourced: [Vec<f64>; 6],
}

/// The CPU's layer transports of `fx`.
fn cpu_layers(fx: &Fixture, layers: &mut LayerTransport) {
    layers.compute(
        &fx.state,
        Some(fx.flux()),
        &fx.mesh,
        &fx.ops,
        &fx.geom,
        &fx.sigma,
        &fx.bathymetry,
        &fx.boundaries,
        &fx.exterior(),
        fx.min_column_depth,
        Blocks::All(fx.n_elements()),
    );
}

/// The CPU's advection by `layers`: u, v added to `rhs[2..4]`, T, S (within
/// `ranges`) into `rhs[..2]`, k, ψ into `rhs[4..]`.
fn cpu_advect(
    fx: &Fixture,
    layers: &LayerTransport,
    rhs: &mut [Vec<f64>; 6],
    ranges: [(f64, f64); 2],
) {
    let ne = fx.n_elements();
    let exterior = fx.exterior();
    let [rhs_t, rhs_s, rhs_u, rhs_v, rhs_k, rhs_psi] = rhs;
    apply_momentum_transport_3d(
        rhs_u,
        rhs_v,
        &fx.state.u,
        &fx.state.v,
        layers,
        &fx.mesh,
        &fx.ops,
        &fx.geom,
        &fx.boundaries,
        exterior.velocity,
        MomentumAdvectionForm::Split,
        VerticalAdvection::Centred,
        Blocks::All(ne),
    );
    let reference = |t: usize| fx.fields.as_ref().map(|r| &r.tracers[t][..]);
    apply_tracer_transport_3d(
        [
            AdvectedTracer {
                rhs: rhs_t,
                tracer: &fx.state.temp,
                bc: &*fx.bcs[0],
                exterior: exterior.temp,
                reference: reference(0),
                range: ranges[0],
            },
            AdvectedTracer {
                rhs: rhs_s,
                tracer: &fx.state.salt,
                bc: &*fx.bcs[1],
                exterior: exterior.salt,
                reference: reference(1),
                range: ranges[1],
            },
        ],
        layers,
        &fx.mesh,
        &fx.ops,
        &fx.geom,
        &fx.boundaries,
        VerticalAdvection::LimitedAkima,
        Blocks::All(ne),
    );
    // The reference's own advection, as `compute_transport_rhs_3d` adds it
    if let Some(fields) = &fx.fields {
        add_reference_advection(
            [rhs_t, rhs_s],
            [&fx.state.temp, &fx.state.salt],
            fields,
            &fx.state,
            layers,
            &fx.ops,
            &fx.geom,
            &fx.sigma,
            &fx.bathymetry,
            fx.min_column_depth,
            Blocks::All(ne),
        );
    }
    let bc = ExtrapolationTracerBC3D;
    apply_w_point_transport_3d(
        [
            AdvectedTracer::new(rhs_k, &fx.state.tke, &bc),
            AdvectedTracer::new(rhs_psi, &fx.state.gls, &bc),
        ],
        layers,
        &fx.mesh,
        &fx.ops,
        &fx.geom,
        &fx.boundaries,
        VerticalAdvection::LimitedAkima,
        Blocks::All(ne),
    );
}

/// The CPU's sources added to `rhs`, as `Hydrostatic3D` adds them: the
/// water no transport carried to T, S, u, v; the rivers' water at the own
/// value, then the water no transport carried, to k and ψ; the rivers'
/// T and S. `pooled` in place of the fixture's (the CPU's check).
fn cpu_sources(fx: &Fixture, layers: &LayerTransport, rhs: &mut [Vec<f64>; 6], pooled: &[bool]) {
    let elements = Blocks::All(fx.n_elements());
    let s = &fx.state;
    let fields = [&s.temp, &s.salt, &s.u, &s.v];
    for (out, field) in rhs.iter_mut().zip(fields) {
        layers.add_volume_sources(out, field, fx.sigma.d_sigma(), &fx.geom, pooled, elements);
    }
    let mut d_sigma_w = vec![0.0; LEVELS + 1];
    w_cell_thicknesses(&layers.d_sigma, &mut d_sigma_w);
    for (out, field) in rhs[4..].iter_mut().zip([&s.tke, &s.gls]) {
        if let Some(rivers) = fx.inflow() {
            rivers.add_w_point_sources_at_own_value(out, field, NN, elements);
        }
        layers.add_volume_sources(out, field, &d_sigma_w, &fx.geom, pooled, elements);
    }
    if let Some(rivers) = fx.inflow() {
        rivers.add_tracer_sources(&mut rhs[0], NN, elements, |r, _| r.temperature.at(0.0));
        rivers.add_tracer_sources(&mut rhs[1], NN, elements, |r, _| r.salinity.at(0.0));
    }
}

fn cpu(fx: &Fixture) -> Cpu {
    let ne = fx.n_elements();
    let mut layers = LayerTransport::new(ne, &fx.ops, LEVELS).with_metric(MetricForm::Averaged);
    cpu_layers(fx, &mut layers);
    let mut rhs = fx.rhs();
    cpu_advect(fx, &layers, &mut rhs, [T_RANGE, S_RANGE]);
    let advected = rhs.clone();
    cpu_sources(fx, &layers, &mut rhs, &fx.pooled);
    Cpu {
        layers,
        advected,
        sourced: rhs,
    }
}

/// The fixture's fields on the device.
struct Device {
    eta: DeviceBuffer<f64>,
    /// The density, which the displacements read (kept while the device may)
    _rho: DeviceBuffer<f64>,
    /// T, S, u, v, k, ψ
    fields: [DeviceBuffer<f64>; 6],
    /// The reference at the state's levels, filled on the device, and its
    /// advection
    baroclinic: RefCell<Baroclinic3D>,
    barotropic: BarotropicDevice,
    /// The parent's u, v, T, S outside the open faces
    parent: Option<[ExteriorDevice; 4]>,
    /// T's and S's conditions at the open faces
    bcs: [DeviceTracerBC3D; 2],
    /// The rivers' T and S
    river_conc: Option<[DeviceBuffer<f64>; 2]>,
    rhs0: [Vec<f64>; 6],
}

/// What the device gets wrong on purpose, for the gate to catch.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mutant {
    None,
    /// T and S without their reference
    NoReference,
    /// T and S about their reference, but without its own advection
    NoReferenceAdvection,
    /// T and S at the central value beside thin columns, not the donor's
    CentralTracers,
    /// T's boundary condition the interior's value
    ExtrapolatedT,
    /// No river water in T
    NoRiverT,
    /// No parent's T outside the open faces (T's boundary condition there)
    NoParentT,
}

impl Device {
    /// The fixture's fields and its transports' statics (with the rivers),
    /// on the device.
    fn upload(
        gpu: &Arc<Gpu>,
        fx: &Fixture,
    ) -> Result<(Self, Transport3D), Box<dyn std::error::Error>> {
        let up = |v: &[f64]| DeviceBuffer::from_host(&gpu.stream, v);
        let s = &fx.state;
        let pooled: Vec<u32> = fx.pooled.iter().map(|&p| u32::from(p)).collect();
        let statics = fx.statics(gpu)?;
        let rivers = match &fx.rivers {
            Some(r) => {
                let mut device = RiversDevice::new(gpu, &statics, &r.sources)?;
                device.set_discharge(gpu, &r.discharge)?;
                Some(device)
            }
            None => None,
        };
        let parent = match &fx.parent {
            Some(p) => Some([
                ExteriorDevice::new(gpu, &statics, &p.field(0))?,
                ExteriorDevice::new(gpu, &statics, &p.field(1))?,
                ExteriorDevice::new(gpu, &statics, &p.field(2))?,
                ExteriorDevice::new(gpu, &statics, &p.field(3))?,
            ]),
            None => None,
        };
        let form = |bc: &dyn TracerBoundaryCondition3D| bc.device_form().expect("a device form");
        let (eta, rho) = (up(&s.eta.data)?, up(&s.rho)?);
        // The reference at the state's levels, filled on the device
        let mut baroclinic = fx.baroclinic(gpu, &statics, &CoriolisSource2D::f_plane(0.0), None)?;
        baroclinic.reference(&statics, &eta, &rho)?;
        let device = Self {
            eta,
            _rho: rho,
            fields: [
                up(&s.temp)?,
                up(&s.salt)?,
                up(&s.u)?,
                up(&s.v)?,
                up(&s.tke)?,
                up(&s.gls)?,
            ],
            baroclinic: RefCell::new(baroclinic),
            barotropic: BarotropicDevice {
                hu: up(&fx.bar_hu)?,
                hv: up(&fx.bar_hv)?,
                face: up(&fx.bar_face)?,
                subcell: up(&fx.subcell)?,
                eta_rate: up(&fx.eta_rate)?,
                euler_dt: up(&fx.euler_dt)?,
                pooled: DeviceBuffer::from_host(&gpu.stream, &pooled)?,
            },
            parent,
            bcs: [form(&*fx.bcs[0]), form(&*fx.bcs[1])],
            river_conc: match &fx.rivers {
                Some(r) => Some([up(&r.conc[0])?, up(&r.conc[1])?]),
                None => None,
            },
            rhs0: fx.rhs(),
        };
        let transport = Transport3D::new(gpu, statics, fx.min_column_depth, rivers)?;
        Ok((device, transport))
    }

    /// The layer transports on the device.
    fn layers(&self, transport: &mut Transport3D) -> Result<(), Box<dyn std::error::Error>> {
        let [_, _, u, v, ..] = &self.fields;
        let parent = self.parent.as_ref().map(|p| [&p[0], &p[1]]);
        transport.compute(&self.eta, u, v, &self.barotropic, parent)
    }

    /// The tendencies' buffers at their starting values.
    fn rhs(&self, gpu: &Gpu) -> Result<[DeviceBuffer<f64>; 6], Box<dyn std::error::Error>> {
        let up = |v: &Vec<f64>| DeviceBuffer::from_host(&gpu.stream, v);
        let r = &self.rhs0;
        Ok([
            up(&r[0])?,
            up(&r[1])?,
            up(&r[2])?,
            up(&r[3])?,
            up(&r[4])?,
            up(&r[5])?,
        ])
    }

    /// The advection of u, v, T, S, k and ψ on the device, as
    /// [`cpu_advect`], and with `sources` the sources, as [`cpu_sources`];
    /// or a `mutant` of it.
    fn advect(
        &self,
        transport: &Transport3D,
        rhs: &mut [DeviceBuffer<f64>; 6],
        mutant: Mutant,
        sources: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let [t, s, u, v, k, psi] = &self.fields;
        let [rhs_t, rhs_s, rhs_u, rhs_v, rhs_k, rhs_psi] = rhs;
        let parent = |f: usize| self.parent.as_ref().map(|p| &p[f]);
        let velocity = |field, f| Advected {
            field,
            reference: None,
            range: UNBOUNDED,
            exterior: parent(f),
            bc: DeviceTracerBC3D::Extrapolation,
        };
        transport.advect(
            [&velocity(u, 0), &velocity(v, 1)],
            [&mut *rhs_u, &mut *rhs_v],
            VerticalAdvection::Centred,
            Interface::Central,
            false,
            sources,
        )?;
        let t_bc = if mutant == Mutant::ExtrapolatedT {
            DeviceTracerBC3D::Extrapolation
        } else {
            self.bcs[0]
        };
        // About the reference (then its advection adds the sources after
        // its own), or without one
        let referenced = mutant != Mutant::NoReference && self.baroclinic.borrow().fields.is_some();
        {
            let baroclinic = self.baroclinic.borrow();
            let reference = |i: usize| {
                baroclinic
                    .fields
                    .as_ref()
                    .filter(|_| referenced)
                    .map(|f| &f.tracers[i])
            };
            transport.advect(
                [
                    &Advected {
                        field: t,
                        reference: reference(0),
                        range: T_RANGE,
                        exterior: parent(2).filter(|_| mutant != Mutant::NoParentT),
                        bc: t_bc,
                    },
                    &Advected {
                        field: s,
                        reference: reference(1),
                        range: S_RANGE,
                        exterior: parent(3),
                        bc: self.bcs[1],
                    },
                ],
                [&mut *rhs_t, &mut *rhs_s],
                VerticalAdvection::LimitedAkima,
                if mutant == Mutant::CentralTracers {
                    Interface::Central
                } else {
                    Interface::DonorBesideThin
                },
                true,
                sources && !referenced,
            )?;
        }
        if referenced && mutant != Mutant::NoReferenceAdvection {
            self.baroclinic.borrow_mut().advect_reference(
                transport,
                &self.eta,
                t,
                s,
                &mut *rhs_t,
                &mut *rhs_s,
                sources,
            )?;
        }
        transport.advect_w(
            [k, psi],
            [rhs_k, rhs_psi],
            VerticalAdvection::LimitedAkima,
            sources,
        )?;
        if sources && let Some([conc_t, conc_s]) = &self.river_conc {
            if mutant != Mutant::NoRiverT {
                transport.river_sources(rhs_t, conc_t)?;
            }
            transport.river_sources(rhs_s, conc_s)?;
        }
        Ok(())
    }
}

/// Agreement of one field: its values identical to the bit, the largest
/// difference in ulp and relative to the field's largest magnitude.
pub(crate) struct Agreement {
    pub(crate) identical: usize,
    pub(crate) len: usize,
    worst_ulps: u64,
    /// The largest difference over the field's largest value
    pub(crate) worst_rel: f64,
    first_difference: Option<usize>,
}

pub(crate) fn agreement(cpu: &[f64], gpu: &[f64]) -> Agreement {
    assert_eq!(cpu.len(), gpu.len(), "the field's length");
    let scale = cpu.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let mut a = Agreement {
        identical: 0,
        len: cpu.len(),
        worst_ulps: 0,
        worst_rel: 0.0,
        first_difference: None,
    };
    for (n, (&c, &g)) in cpu.iter().zip(gpu).enumerate() {
        if c.to_bits() == g.to_bits() {
            a.identical += 1;
            continue;
        }
        a.first_difference.get_or_insert(n);
        a.worst_ulps = a.worst_ulps.max(ulps(c, g));
        a.worst_rel = a
            .worst_rel
            .max((c - g).abs() / scale.max(f64::MIN_POSITIVE));
    }
    a
}

/// Compare `gpu` (downloaded) with `cpu`, print the line, and whether they
/// are the same to the bit.
pub(crate) fn report(
    gpu: &Gpu,
    name: &str,
    cpu: &[f64],
    device: &DeviceBuffer<f64>,
) -> Result<bool, Box<dyn std::error::Error>> {
    let values = device.to_host_vec(&gpu.stream)?;
    let a = agreement(cpu, &values);
    let same = a.identical == a.len;
    if same {
        println!("    {name:<16} {} of {} bit-identical", a.identical, a.len);
    } else {
        let at = a.first_difference.expect("a difference");
        println!(
            "    {name:<16} {} of {} bit-identical, largest difference {} ulp, {:.1e} of the \
             field's largest; first at {at}: CPU {:e}, GPU {:e}",
            a.identical, a.len, a.worst_ulps, a.worst_rel, cpu[at], values[at]
        );
    }
    Ok(same)
}

/// The device's reference fields (`baroclinic`'s) against the CPU's
/// `fields`, with the elements' displacements if `displacements`. Whether
/// all are the same to the bit.
pub(crate) fn report_reference(
    gpu: &Gpu,
    fields: &ReferenceFields,
    baroclinic: &Baroclinic3D,
    displacements: bool,
) -> Result<bool, Box<dyn std::error::Error>> {
    let device = baroclinic
        .fields
        .as_ref()
        .expect("a reference on the device");
    let mut same = true;
    for (name, cpu, gpu_values) in [
        ("reference T", &fields.tracers[0], &device.tracers[0]),
        ("reference S", &fields.tracers[1], &device.tracers[1]),
        ("gradient T", &fields.gradients[0], &device.gradients[0]),
        ("gradient S", &fields.gradients[1], &device.gradients[1]),
        ("reference rho", &fields.density, &device.density),
        (
            "stratification",
            &fields.stratification,
            &device.stratification,
        ),
        ("residual", &fields.residual, &device.residual),
    ] {
        same &= report(gpu, name, cpu, gpu_values)?;
    }
    if displacements {
        same &= report(
            gpu,
            "displacement",
            &fields.displacement,
            &device.displacement,
        )?;
    }
    Ok(same)
}

/// The device's element kinds and thin columns against the CPU's
/// (`subcell_elements`, `fallback`, `thin` in the subcell elements, where
/// the CPU sets it); prints the counts. Whether they agree.
fn report_kinds(
    gpu: &Gpu,
    fx: &Fixture,
    layers: &LayerTransport,
    transport: &Transport3D,
) -> Result<bool, Box<dyn std::error::Error>> {
    let kinds = transport.kinds.to_host_vec(&gpu.stream)?;
    let thin = transport.thin.to_host_vec(&gpu.stream)?;
    let ne = fx.n_elements();
    let mut wrong = 0;
    for (k, &kind) in kinds.iter().enumerate() {
        let subcell = layers.subcell_elements[k];
        let fallback = layers.fallback[k];
        let pooling = fallback || fx.pooled.get(k).copied().unwrap_or(false);
        let device = (kind & 1 != 0, kind & 2 != 0, kind & 4 != 0);
        if device != (subcell, fallback, pooling) {
            wrong += 1;
        }
        if subcell {
            wrong += (k * NN..(k + 1) * NN)
                .filter(|&flat| (thin[flat] != 0) != layers.thin[flat])
                .count();
        }
    }
    let count = |f: &dyn Fn(usize) -> bool| (0..ne).filter(|&k| f(k)).count();
    let thin_columns = (0..ne * NN)
        .filter(|&flat| fx.state.eta.data[flat] - fx.bathymetry.data[flat] < fx.min_column_depth)
        .count();
    println!(
        "    {:<16} subcell {}, fallback {}, pooled {}, thin columns {}: {}",
        "elements",
        count(&|k| layers.subcell_elements[k]),
        count(&|k| layers.fallback[k]),
        count(&|k| fx.pooled.get(k).copied().unwrap_or(false)),
        thin_columns,
        if wrong == 0 {
            "the device's kinds and thin columns agree".to_string()
        } else {
            format!("{wrong} kinds or thin columns DIFFER")
        }
    );
    Ok(wrong == 0)
}

/// How many of `field`'s values a mutant of the device changes against
/// the CPU's `cpu`; prints the line. Whether the gate catches it.
fn report_mutant(
    gpu: &Gpu,
    device: &Device,
    transport: &Transport3D,
    mutant: Mutant,
    what: &str,
    field: usize,
    cpu: &Cpu,
) -> Result<bool, Box<dyn std::error::Error>> {
    let sources = mutant == Mutant::NoRiverT;
    let mut rhs = device.rhs(gpu)?;
    device.advect(transport, &mut rhs, mutant, sources)?;
    let values = rhs[field].to_host_vec(&gpu.stream)?;
    let expected = if sources {
        &cpu.sourced[field]
    } else {
        &cpu.advected[field]
    };
    let a = agreement(expected, &values);
    let caught = a.identical < a.len;
    println!(
        "    mutant ({what}): {} {} of {} differ{}",
        NAMES[field],
        a.len - a.identical,
        a.len,
        if caught { ", caught" } else { ": NOT CAUGHT" }
    );
    Ok(caught)
}

/// One fixture on both: every output compared. Whether all are the same to
/// the bit.
fn gate(gpu: &Arc<Gpu>, label: &str, shape: Shape) -> Result<bool, Box<dyn std::error::Error>> {
    let fx = Fixture::new(shape);
    let ne = fx.n_elements();
    let affine = (0..ne).filter(|&k| fx.geom.element_is_affine(k)).count();
    println!(
        "{label}: {ne} P{ORDER} elements ({affine} parallelograms) × {LEVELS} levels, {}{}",
        if shape.stretched {
            "stretched levels"
        } else {
            "uniform levels"
        },
        if shape.reference {
            ", T and S about a reference"
        } else {
            ""
        }
    );
    let cpu = cpu(&fx);
    let (device, mut transport) = Device::upload(gpu, &fx)?;
    device.layers(&mut transport)?;
    let layers = &cpu.layers;
    let mut same = report_kinds(gpu, &fx, layers, &transport)?;
    // The reference at the levels, filled on the device
    if let Some(fields) = &fx.fields {
        same &= report_reference(gpu, fields, &device.baroclinic.borrow(), false)?;
    }
    if !shape.beach {
        let any = layers.subcell_elements.iter().any(|&s| s);
        same &= !any;
        if any {
            println!("    a wet fixture has subcell elements");
        }
    }
    for (name, cpu, gpu_values) in [
        ("layer hu", &layers.hu, &transport.hu),
        ("layer hv", &layers.hv, &transport.hv),
        ("layer face", &layers.face, &transport.face),
        ("layer subcell", &layers.subcell, &transport.subcell),
        ("omega", &layers.omega, &transport.omega),
        (
            "volume source",
            &layers.volume_source,
            &transport.volume_source,
        ),
        (
            "dt over depth",
            &layers.dt_over_depth,
            &transport.dt_over_depth,
        ),
    ] {
        same &= report(gpu, name, cpu, gpu_values)?;
    }
    let residual = transport.surface_residual()?;
    let residual_same = residual.to_bits() == layers.surface_residual.to_bits();
    println!(
        "    {:<16} CPU {:e}, GPU {residual:e}{}",
        "surface residual",
        layers.surface_residual,
        if residual_same { ", bit-identical" } else { "" }
    );
    same &= residual_same;

    for (sources, results, prefix) in [
        (false, &cpu.advected, "advected"),
        (true, &cpu.sourced, "+ sources"),
    ] {
        let mut rhs = device.rhs(gpu)?;
        device.advect(&transport, &mut rhs, Mutant::None, sources)?;
        for ((name, cpu), gpu_values) in NAMES.iter().zip(results).zip(&rhs) {
            same &= report(gpu, &format!("{prefix} {name}"), cpu, gpu_values)?;
        }
    }

    // The mutants: the gate must catch each
    let mutants = [
        (
            shape.reference,
            Mutant::NoReference,
            "no reference on the device",
        ),
        (
            shape.reference,
            Mutant::NoReferenceAdvection,
            "no advection of the reference",
        ),
        (
            shape.beach,
            Mutant::CentralTracers,
            "T at the central value beside thin columns",
        ),
        (
            shape.channel,
            Mutant::ExtrapolatedT,
            "T's open boundary extrapolated, not fixed",
        ),
        (shape.channel, Mutant::NoRiverT, "no river water in T"),
        (
            shape.channel,
            Mutant::NoParentT,
            "no parent's T outside x = L",
        ),
    ];
    for (applies, mutant, what) in mutants {
        if applies {
            same &= report_mutant(gpu, &device, &transport, mutant, what, 0, &cpu)?;
        }
    }
    // And the layer transports without the parent's velocity
    if device.parent.is_some() {
        let [_, _, u, v, ..] = &device.fields;
        let mut without = Device::upload(gpu, &fx)?.1;
        without.compute(&device.eta, u, v, &device.barotropic, None)?;
        let faces = without.face.to_host_vec(&gpu.stream)?;
        let a = agreement(&layers.face, &faces);
        let caught = a.identical < a.len;
        println!(
            "    mutant (no parent's velocity outside x = L): layer face {} of {} differ{}",
            a.len - a.identical,
            a.len,
            if caught { ", caught" } else { ": NOT CAUGHT" }
        );
        same &= caught;
    }
    // That the fixture reaches the pooled sources: without the pass's pooled
    // elements (the fallback ones still pool) the sources differ (on the
    // CPU alone)
    if !fx.pooled.is_empty() {
        let mut unpooled = cpu.advected.clone();
        cpu_sources(&fx, layers, &mut unpooled, &[]);
        let differ = |t: usize| {
            let a = agreement(&cpu.sourced[t], &unpooled[t]);
            a.len - a.identical
        };
        println!(
            "    the pooled elements act (CPU, none pooled): T {}, u {} and k {} values differ",
            differ(0),
            differ(2),
            differ(4)
        );
    }
    // That the fixture reaches the end layers' bound: unbounded ranges give
    // other tendencies (on the CPU alone)
    let mut unbounded = fx.rhs();
    cpu_advect(&fx, layers, &mut unbounded, [UNBOUNDED; 2]);
    let differ = |t: usize| {
        let a = agreement(&cpu.advected[t], &unbounded[t]);
        a.len - a.identical
    };
    println!(
        "    the ranges act (CPU, unbounded): T {} and S {} of {} values differ",
        differ(0),
        differ(1),
        ne * NN * LEVELS
    );
    println!(
        "  {}",
        if same {
            "bit for bit"
        } else {
            "NOT bit for bit"
        }
    );
    Ok(same)
}

/// Time the CPU and the GPU on `shape`, per element and level, by part.
fn time(
    gpu: &Arc<Gpu>,
    label: &str,
    shape: Shape,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let fx = Fixture::new(shape);
    let ne = fx.n_elements();
    let points = (ne * LEVELS) as f64;
    let mut layers = LayerTransport::new(ne, &fx.ops, LEVELS).with_metric(MetricForm::Averaged);
    let mut rhs = fx.rhs();
    // CPU: warm up, then each part on its own
    let mut cpu_parts = [0.0; 2];
    for it in 0..iterations + 2 {
        let t0 = Instant::now();
        cpu_layers(&fx, &mut layers);
        let t1 = Instant::now();
        cpu_advect(&fx, &layers, &mut rhs, [T_RANGE, S_RANGE]);
        cpu_sources(&fx, &layers, &mut rhs, &fx.pooled);
        let t2 = Instant::now();
        if it >= 2 {
            cpu_parts[0] += (t1 - t0).as_secs_f64();
            cpu_parts[1] += (t2 - t1).as_secs_f64();
        }
    }
    let subcell = layers.subcell_elements.iter().filter(|&&s| s).count();
    println!(
        "timing, {label}: {ne} P{ORDER} elements × {LEVELS} levels ({:.1} M points; {subcell} \
         subcell elements, {:.1} %), {iterations} evaluations",
        (ne * NN * LEVELS) as f64 / 1e6,
        100.0 * subcell as f64 / ne as f64
    );
    let (device, mut transport) = Device::upload(gpu, &fx)?;
    let mut rhs = device.rhs(gpu)?;
    let gpu_iterations = 5 * iterations;
    let mut gpu_parts = [0.0; 2];
    for it in 0..gpu_iterations + 2 {
        gpu.stream.synchronize()?;
        let t0 = Instant::now();
        device.layers(&mut transport)?;
        gpu.stream.synchronize()?;
        let t1 = Instant::now();
        device.advect(&transport, &mut rhs, Mutant::None, true)?;
        gpu.stream.synchronize()?;
        let t2 = Instant::now();
        if it >= 2 {
            gpu_parts[0] += (t1 - t0).as_secs_f64();
            gpu_parts[1] += (t2 - t1).as_secs_f64();
        }
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per = |total: f64, count: usize| total * 1e9 / (count as f64 * points);
    println!("  per element and level (CPU {threads} threads | GPU, ×):");
    let names = [
        "layer transports and Ω",
        "advection (u, v, T, S, k, ψ), sources",
    ];
    let (mut cpu_all, mut gpu_all) = (0.0, 0.0);
    for ((name, c), g) in names.iter().zip(cpu_parts).zip(gpu_parts) {
        let (c, g) = (per(c, iterations), per(g, gpu_iterations));
        cpu_all += c;
        gpu_all += g;
        println!("    {name:<38} {c:7.2} ns | {g:6.2} ns ({:.1}×)", c / g);
    }
    println!(
        "    {:<38} {cpu_all:7.2} ns | {gpu_all:6.2} ns ({:.1}×); per stage {:.1} ms | {:.2} ms",
        "all",
        cpu_all / gpu_all,
        cpu_all * points / 1e6,
        gpu_all * points / 1e6
    );
    Ok(())
}

pub fn run(gpu: &Arc<Gpu>) -> Result<(), Box<dyn std::error::Error>> {
    println!("== The 3D transport: layer transports, Ω, advection (Phase 4) ==");
    let wet = Shape {
        nx: 24,
        ny: 20,
        jitter: 0.0,
        periodic: true,
        beach: false,
        channel: false,
        stretched: false,
        reference: false,
        euler_dt: 0.05,
    };
    let beach = Shape {
        nx: 36,
        ny: 20,
        periodic: false,
        beach: true,
        reference: true,
        ..wet
    };
    let channel = Shape {
        nx: 30,
        ny: 12,
        jitter: 0.15,
        periodic: false,
        channel: true,
        stretched: true,
        reference: true,
        euler_dt: 40.0,
        ..wet
    };
    let fixtures = [
        ("periodic, parallelograms", wet),
        (
            "periodic, jittered, stretched",
            Shape {
                jitter: 0.15,
                stretched: true,
                ..wet
            },
        ),
        (
            "periodic, jittered, stretched, reference",
            Shape {
                jitter: 0.15,
                stretched: true,
                reference: true,
                ..wet
            },
        ),
        (
            "walled, jittered, reference, long Euler step",
            Shape {
                jitter: 0.15,
                periodic: false,
                reference: true,
                euler_dt: 40.0,
                ..wet
            },
        ),
        ("beach, parallelograms, reference", beach),
        (
            "beach, jittered, stretched, reference, long Euler step",
            Shape {
                jitter: 0.2,
                stretched: true,
                euler_dt: 40.0,
                ..beach
            },
        ),
        (
            "channel, open ends, a parent, rivers, jittered, stretched, reference",
            channel,
        ),
    ];
    let mut failed = Vec::new();
    for (label, shape) in fixtures {
        if !gate(gpu, label, shape)? {
            failed.push(label);
        }
    }
    let large = Shape {
        nx: 320,
        ny: 300,
        jitter: 0.15,
        stretched: true,
        reference: true,
        ..wet
    };
    time(gpu, "wet", large, 10)?;
    time(
        gpu,
        "beach",
        Shape {
            beach: true,
            periodic: false,
            jitter: 0.2,
            ..large
        },
        10,
    )?;
    if !failed.is_empty() {
        println!(
            "(not bit for bit: {}; expected only in a build with multiply–add contraction)",
            failed.join("; ")
        );
    }
    Ok(())
}
