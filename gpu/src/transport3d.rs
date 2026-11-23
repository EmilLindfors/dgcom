//! The 3D transport gate (P2.7 Phase 4, `notes/gpu.md`): the layer
//! transports and Ω ([`Transport3D::compute`]), the advection of u and v
//! (split form, centred in the vertical) and of T and S (LimitedAkima within
//! their ranges, about a reference stratification or not), and the water no
//! transport carried, on the GPU against dgcom's `LayerTransport::compute`,
//! `apply_momentum_transport_3d`, `apply_tracer_transport_3d` and
//! `LayerTransport::add_volume_sources` on the CPU, value for value: fully
//! wet (Phase 4.0) and on a walled beach with flats, films and dry columns,
//! where subcell elements, thin columns, fallback and pooled elements meet
//! (Phase 4.1); then timed at Chiloé's size (96,000 P2 elements × 20
//! levels).

use std::sync::Arc;
use std::time::Instant;

use cuda_core::DeviceBuffer;
use dgcom::mesh::{Bathymetry2D, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::solver::Blocks;
use dgcom::solver::rhs::subcells::{
    line_metric, line_node, subcell_interfaces, subcell_slot, telescoped_interfaces,
};
use dgcom::solver::rhs::{
    AdvectedTracer, BarotropicFlux, Boundaries3D, Exterior3D, ExtrapolationTracerBC3D,
    FaceExterior, LayerTransport, MetricForm, MomentumAdvectionForm, UNBOUNDED, VerticalAdvection,
    apply_momentum_transport_3d, apply_tracer_transport_3d,
};
use dgcom::solver::state::Solution3D;
use dgcom::types::ElementIndex;
use dgcom::vertical::{ROMSVstretching4, SigmaGrid};

use crate::phase0::{DX, Lcg, quad_mesh, ulps};
use crate::swe2d::{Gpu, N1, NN, ORDER};
use crate::three_d::{Advected, BarotropicDevice, Interface, Statics3D, Transport3D};

/// σ-levels of the fixtures (Chiloé's 3D case has 20).
const LEVELS: usize = 20;
/// The tracers' ranges, narrower than the water's in places so that the end
/// layers' bound (`in_range`) acts.
const T_RANGE: (f64, f64) = (5.5, 11.0);
const S_RANGE: (f64, f64) = (31.6, 34.4);
/// The baroclinic step of the barotropic flux (s).
const DT: f64 = 30.0;

/// A fixture's shape.
#[derive(Clone, Copy)]
struct Shape {
    nx: usize,
    ny: usize,
    /// Interior vertices moved by up to this share of the element size
    jitter: f64,
    /// Doubly periodic, else walled
    periodic: bool,
    /// A walled beach with flats, films and dry columns (else a bed
    /// 12–28 m deep): subcell elements, thin columns, the pass's subcell
    /// fluxes in some, fallback and pooled elements
    beach: bool,
    /// ROMS's Vstretching 4 (θ_s 5, θ_b 0.4), else uniform levels
    stretched: bool,
    /// T and S advected about a reference stratification
    reference: bool,
    /// Every element's forward-Euler step (s): long enough, and end layers
    /// shed more than half their water in it (`in_range`'s share)
    euler_dt: f64,
}

/// A mesh, its σ-levels, a stratified, sheared state on it and a
/// barotropic pass that its layers do not sum to.
struct Fixture {
    mesh: Mesh2D,
    ops: DGOperators2D,
    geom: GeometricFactors2D,
    bathymetry: Bathymetry2D,
    sigma: SigmaGrid,
    boundaries: Boundaries3D,
    /// Columns thinner than this exchange no baroclinic transport
    min_column_depth: f64,
    /// The thin columns at rest, as the mode splitter's stage values are
    state: Solution3D,
    /// The reference stratification of T and S at each column's levels
    reference: Option<[Vec<f64>; 2]>,
    bar_hu: Vec<f64>,
    bar_hv: Vec<f64>,
    bar_face: Vec<f64>,
    eta_rate: Vec<f64>,
    euler_dt: Vec<f64>,
    /// The pass's subcell interface fluxes: NaN in the elements it did not
    /// move on the subcells throughout
    subcell: Vec<f64>,
    /// The elements the pass balanced only as a whole (empty: none)
    pooled: Vec<bool>,
    /// The momentum's tendencies before the advection adds to them
    rhs_u0: Vec<f64>,
    rhs_v0: Vec<f64>,
}

impl Fixture {
    fn new(shape: Shape) -> Self {
        let Shape {
            nx,
            ny,
            jitter,
            periodic,
            beach,
            stretched,
            reference,
            euler_dt,
        } = shape;
        let mesh = quad_mesh(nx, ny, jitter, periodic && !beach);
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
        let nl = LEVELS;
        let mut state = Solution3D::new(ne, NN, nl);
        let mut references = [vec![0.0; ne * NN * nl], vec![0.0; ne * NN * nl]];
        let mut rng = Lcg(20_261_123);
        let mut bar_hu = vec![0.0; ne * NN];
        let mut bar_hv = vec![0.0; ne * NN];
        let mut eta_rate = vec![0.0; ne * NN];
        let mut rhs_u0 = vec![0.0; ne * NN * nl];
        let mut rhs_v0 = vec![0.0; ne * NN * nl];
        let sigma_rho = sigma.sigma_rho().to_vec();
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
                    let smooth = ((z + z0) / 3.0).tanh();
                    references[0][at] = 8.0 + 3.0 * smooth;
                    references[1][at] = 33.0 - 1.5 * smooth;
                    rhs_u0[at] = 1e-3 * (rng.next() - 0.5);
                    rhs_v0[at] = 1e-3 * (rng.next() - 0.5);
                }
                bar_hu[idx] = depth * (0.62 * (ky * y).sin() + 0.01 * (kx * x).cos());
                bar_hv[idx] = depth * (0.38 * (kx * x).cos() - 0.01 * (ky * y).sin());
                eta_rate[idx] = 1e-4 * (kx * x + ky * y).sin();
            }
        }
        // The pass's face fluxes: the central average of DU_avg2 · n, none
        // at a wall
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
                        _ => 0.0,
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
            reference: reference.then_some(references),
            bar_hu,
            bar_hv,
            bar_face,
            eta_rate,
            rhs_u0,
            rhs_v0,
        }
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
            rivers: None,
            pooled: &self.pooled,
        }
    }

    fn n_elements(&self) -> usize {
        self.mesh.n_elements
    }

    fn statics(&self, gpu: &Gpu) -> Result<Statics3D, Box<dyn std::error::Error>> {
        Statics3D::upload(
            gpu,
            &self.mesh,
            &self.ops,
            &self.geom,
            &self.bathymetry,
            &self.sigma,
            &self.boundaries,
            MetricForm::Averaged,
        )
    }
}

/// The CPU's results: the layer transports, the tendencies of T, S, u, v
/// after the advection and after the volume sources.
struct Cpu {
    layers: LayerTransport,
    advected: [Vec<f64>; 4],
    sourced: [Vec<f64>; 4],
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
        &Exterior3D::default(),
        fx.min_column_depth,
        Blocks::All(fx.n_elements()),
    );
}

/// The CPU's advection of u, v (added to `rhs[2..]`) and T, S (into
/// `rhs[..2]`, within `ranges`) by `layers`.
fn cpu_advect(
    fx: &Fixture,
    layers: &LayerTransport,
    rhs: &mut [Vec<f64>; 4],
    ranges: [(f64, f64); 2],
) {
    let ne = fx.n_elements();
    let [rhs_t, rhs_s, rhs_u, rhs_v] = rhs;
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
        None,
        MomentumAdvectionForm::Split,
        VerticalAdvection::Centred,
        Blocks::All(ne),
    );
    let bc = ExtrapolationTracerBC3D;
    let reference = |t: usize| fx.reference.as_ref().map(|r| &r[t][..]);
    apply_tracer_transport_3d(
        [
            AdvectedTracer {
                rhs: rhs_t,
                tracer: &fx.state.temp,
                bc: &bc,
                exterior: None,
                reference: reference(0),
                range: ranges[0],
            },
            AdvectedTracer {
                rhs: rhs_s,
                tracer: &fx.state.salt,
                bc: &bc,
                exterior: None,
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
}

/// The CPU's volume sources added to `rhs`.
fn cpu_sources(fx: &Fixture, layers: &LayerTransport, rhs: &mut [Vec<f64>; 4]) {
    let fields = [&fx.state.temp, &fx.state.salt, &fx.state.u, &fx.state.v];
    for (out, field) in rhs.iter_mut().zip(fields) {
        layers.add_volume_sources(
            out,
            field,
            fx.sigma.d_sigma(),
            &fx.geom,
            &fx.pooled,
            Blocks::All(fx.n_elements()),
        );
    }
}

fn cpu(fx: &Fixture) -> Cpu {
    let ne = fx.n_elements();
    let mut layers = LayerTransport::new(ne, &fx.ops, LEVELS).with_metric(MetricForm::Averaged);
    cpu_layers(fx, &mut layers);
    let n = ne * NN * LEVELS;
    let mut rhs = [
        vec![0.0; n],
        vec![0.0; n],
        fx.rhs_u0.clone(),
        fx.rhs_v0.clone(),
    ];
    cpu_advect(fx, &layers, &mut rhs, [T_RANGE, S_RANGE]);
    let advected = rhs.clone();
    cpu_sources(fx, &layers, &mut rhs);
    Cpu {
        layers,
        advected,
        sourced: rhs,
    }
}

/// The fixture's fields on the device.
struct Device {
    eta: DeviceBuffer<f64>,
    fields: [DeviceBuffer<f64>; 4],
    reference: Option<[DeviceBuffer<f64>; 2]>,
    barotropic: BarotropicDevice,
    rhs_u0: Vec<f64>,
    rhs_v0: Vec<f64>,
}

impl Device {
    fn upload(gpu: &Gpu, fx: &Fixture) -> Result<Self, Box<dyn std::error::Error>> {
        let up = |v: &[f64]| DeviceBuffer::from_host(&gpu.stream, v);
        let s = &fx.state;
        let pooled: Vec<u32> = fx.pooled.iter().map(|&p| u32::from(p)).collect();
        Ok(Self {
            eta: up(&s.eta.data)?,
            fields: [up(&s.temp)?, up(&s.salt)?, up(&s.u)?, up(&s.v)?],
            reference: match &fx.reference {
                Some([t, s]) => Some([up(t)?, up(s)?]),
                None => None,
            },
            barotropic: BarotropicDevice {
                hu: up(&fx.bar_hu)?,
                hv: up(&fx.bar_hv)?,
                face: up(&fx.bar_face)?,
                subcell: up(&fx.subcell)?,
                eta_rate: up(&fx.eta_rate)?,
                euler_dt: up(&fx.euler_dt)?,
                pooled: DeviceBuffer::from_host(&gpu.stream, &pooled)?,
            },
            rhs_u0: fx.rhs_u0.clone(),
            rhs_v0: fx.rhs_v0.clone(),
        })
    }

    /// The tendencies' buffers at their starting values (T and S zero, u
    /// and v the fixture's).
    fn rhs(&self, gpu: &Gpu) -> Result<[DeviceBuffer<f64>; 4], Box<dyn std::error::Error>> {
        let st = &gpu.stream;
        let n = self.rhs_u0.len();
        Ok([
            DeviceBuffer::zeroed(st, n)?,
            DeviceBuffer::zeroed(st, n)?,
            DeviceBuffer::from_host(st, &self.rhs_u0)?,
            DeviceBuffer::from_host(st, &self.rhs_v0)?,
        ])
    }

    /// The advection of u, v and T, S on the device, as [`cpu_advect`], and
    /// with `sources` the volume sources, as [`cpu_sources`]; or a
    /// `mutant` of it.
    fn advect(
        &self,
        transport: &Transport3D,
        rhs: &mut [DeviceBuffer<f64>; 4],
        mutant: Mutant,
        sources: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let [t, s, u, v] = &self.fields;
        let [rhs_t, rhs_s, rhs_u, rhs_v] = rhs;
        let unbounded = |field| Advected {
            field,
            reference: None,
            range: UNBOUNDED,
        };
        transport.advect(
            [&unbounded(u), &unbounded(v)],
            [rhs_u, rhs_v],
            VerticalAdvection::Centred,
            Interface::Central,
            false,
            sources,
        )?;
        let reference = |i: usize| {
            self.reference
                .as_ref()
                .filter(|_| mutant != Mutant::NoReference)
                .map(|r| &r[i])
        };
        transport.advect(
            [
                &Advected {
                    field: t,
                    reference: reference(0),
                    range: T_RANGE,
                },
                &Advected {
                    field: s,
                    reference: reference(1),
                    range: S_RANGE,
                },
            ],
            [rhs_t, rhs_s],
            VerticalAdvection::LimitedAkima,
            if mutant == Mutant::CentralTracers {
                Interface::Central
            } else {
                Interface::DonorBesideThin
            },
            true,
            sources,
        )
    }
}

/// What the device gets wrong on purpose, for the gate to catch.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mutant {
    None,
    /// T and S without their reference
    NoReference,
    /// T and S at the central value beside thin columns, not the donor's
    CentralTracers,
}

/// Agreement of one field: its values identical to the bit, the largest
/// difference in ulp and relative to the field's largest magnitude.
struct Agreement {
    identical: usize,
    len: usize,
    worst_ulps: u64,
    worst_rel: f64,
    first_difference: Option<usize>,
}

fn agreement(cpu: &[f64], gpu: &[f64]) -> Agreement {
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
fn report(
    gpu: &Gpu,
    name: &str,
    cpu: &[f64],
    device: &DeviceBuffer<f64>,
) -> Result<bool, Box<dyn std::error::Error>> {
    let values = device.to_host_vec(&gpu.stream)?;
    Ok(report_values(name, cpu, &values))
}

fn report_values(name: &str, cpu: &[f64], values: &[f64]) -> bool {
    let a = agreement(cpu, values);
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
    same
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
    let mut transport = Transport3D::new(gpu, fx.statics(gpu)?, fx.min_column_depth)?;
    let device = Device::upload(gpu, &fx)?;
    let [_, _, u, v] = &device.fields;
    transport.compute(&device.eta, u, v, &device.barotropic)?;
    let layers = &cpu.layers;
    let mut same = report_kinds(gpu, &fx, layers, &transport)?;
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

    let names = ["T", "S", "u", "v"];
    for (sources, results, prefix) in [
        (false, &cpu.advected, "advected"),
        (true, &cpu.sourced, "+ sources"),
    ] {
        let mut rhs = device.rhs(gpu)?;
        device.advect(&transport, &mut rhs, Mutant::None, sources)?;
        for ((name, cpu), gpu_values) in names.iter().zip(results).zip(&rhs) {
            same &= report(gpu, &format!("{prefix} {name}"), cpu, gpu_values)?;
        }
    }

    // The mutants: the gate must see each
    let mutants = [
        (
            shape.reference,
            Mutant::NoReference,
            "no reference on the device",
        ),
        (
            shape.beach,
            Mutant::CentralTracers,
            "T at the central value beside thin columns",
        ),
    ];
    for (applies, mutant, what) in mutants {
        if !applies {
            continue;
        }
        let mut rhs = device.rhs(gpu)?;
        device.advect(&transport, &mut rhs, mutant, false)?;
        let t = rhs[0].to_host_vec(&gpu.stream)?;
        let a = agreement(&cpu.advected[0], &t);
        println!(
            "    mutant ({what}): T {} of {} differ{}",
            a.len - a.identical,
            a.len,
            if a.identical < a.len {
                ", caught"
            } else {
                ": NOT CAUGHT"
            }
        );
        same &= a.identical < a.len;
    }
    // That the fixture reaches the pooled sources: without the pass's pooled
    // elements (the fallback ones still pool) the sources differ (on the
    // CPU alone)
    if !fx.pooled.is_empty() {
        let mut unpooled = cpu.advected.clone();
        let fields = [&fx.state.temp, &fx.state.salt, &fx.state.u, &fx.state.v];
        for (out, field) in unpooled.iter_mut().zip(fields) {
            layers.add_volume_sources(
                out,
                field,
                fx.sigma.d_sigma(),
                &fx.geom,
                &[],
                Blocks::All(ne),
            );
        }
        let differ = |t: usize| {
            let a = agreement(&cpu.sourced[t], &unpooled[t]);
            a.len - a.identical
        };
        println!(
            "    the pooled elements act (CPU, none pooled): T {} and u {} of {} values differ",
            differ(0),
            differ(2),
            ne * NN * LEVELS
        );
    }
    // That the fixture reaches the end layers' bound: unbounded ranges give
    // other tendencies (on the CPU alone)
    let n = ne * NN * LEVELS;
    let mut unbounded = [
        vec![0.0; n],
        vec![0.0; n],
        fx.rhs_u0.clone(),
        fx.rhs_v0.clone(),
    ];
    cpu_advect(&fx, layers, &mut unbounded, [UNBOUNDED; 2]);
    let differ = |t: usize| {
        let a = agreement(&cpu.advected[t], &unbounded[t]);
        a.len - a.identical
    };
    println!(
        "    the ranges act (CPU, unbounded): T {} and S {} of {n} values differ",
        differ(0),
        differ(1)
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
    let n = ne * NN * LEVELS;
    let mut layers = LayerTransport::new(ne, &fx.ops, LEVELS).with_metric(MetricForm::Averaged);
    let mut rhs = [
        vec![0.0; n],
        vec![0.0; n],
        fx.rhs_u0.clone(),
        fx.rhs_v0.clone(),
    ];
    // CPU: warm up, then each part on its own
    let mut cpu_parts = [0.0; 2];
    for it in 0..iterations + 2 {
        let t0 = Instant::now();
        cpu_layers(&fx, &mut layers);
        let t1 = Instant::now();
        cpu_advect(&fx, &layers, &mut rhs, [T_RANGE, S_RANGE]);
        cpu_sources(&fx, &layers, &mut rhs);
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
    let mut transport = Transport3D::new(gpu, fx.statics(gpu)?, fx.min_column_depth)?;
    let device = Device::upload(gpu, &fx)?;
    let mut rhs = device.rhs(gpu)?;
    let [_, _, u, v] = &device.fields;
    let gpu_iterations = 5 * iterations;
    let mut gpu_parts = [0.0; 2];
    for it in 0..gpu_iterations + 2 {
        gpu.stream.synchronize()?;
        let t0 = Instant::now();
        transport.compute(&device.eta, u, v, &device.barotropic)?;
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
    let names = ["layer transports and Ω", "advection, volume sources"];
    let (mut cpu_all, mut gpu_all) = (0.0, 0.0);
    for ((name, c), g) in names.iter().zip(cpu_parts).zip(gpu_parts) {
        let (c, g) = (per(c, iterations), per(g, gpu_iterations));
        cpu_all += c;
        gpu_all += g;
        println!("    {name:<26} {c:7.2} ns | {g:6.2} ns ({:.1}×)", c / g);
    }
    println!(
        "    {:<26} {cpu_all:7.2} ns | {gpu_all:6.2} ns ({:.1}×); per stage {:.1} ms | {:.2} ms",
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
        stretched: false,
        reference: false,
        euler_dt: 0.05,
    };
    let beach = Shape {
        nx: 36,
        ny: 20,
        jitter: 0.0,
        periodic: false,
        beach: true,
        stretched: false,
        reference: true,
        euler_dt: 0.05,
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
