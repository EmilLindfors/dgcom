//! The 3D transport gate (P2.7 Phase 4.0, `notes/gpu.md`): the layer
//! transports and Ω ([`Transport3D::compute`]), the advection of u and v
//! (split form, centred in the vertical) and of T and S (LimitedAkima within
//! their ranges, about a reference stratification or not), and the water no
//! transport carried, on the GPU against dgcom's `LayerTransport::compute`,
//! `apply_momentum_transport_3d`, `apply_tracer_transport_3d` and
//! `LayerTransport::add_volume_sources` on the CPU, value for value; then
//! timed at Chiloé's size (96,000 P2 elements × 20 levels).

use std::sync::Arc;
use std::time::Instant;

use cuda_core::DeviceBuffer;
use dgcom::mesh::{Bathymetry2D, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::solver::Blocks;
use dgcom::solver::rhs::subcells::subcell_interfaces;
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
use crate::three_d::{Advected, BarotropicDevice, Statics3D, Transport3D};

/// σ-levels of the fixtures (Chiloé's 3D case has 20).
const LEVELS: usize = 20;
/// Columns thinner than this exchange no baroclinic transport (none is).
const MIN_COLUMN_DEPTH: f64 = 1.0;
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
    state: Solution3D,
    /// The reference stratification of T and S at each column's levels
    reference: Option<[Vec<f64>; 2]>,
    bar_hu: Vec<f64>,
    bar_hv: Vec<f64>,
    bar_face: Vec<f64>,
    eta_rate: Vec<f64>,
    euler_dt: Vec<f64>,
    /// NaN: the pass moved no element's mass on the subcells
    subcell: Vec<f64>,
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
            stretched,
            reference,
            euler_dt,
        } = shape;
        let mesh = quad_mesh(nx, ny, jitter, periodic);
        let ops = DGOperators2D::new(ORDER);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let (kx, ky) = (
            std::f64::consts::TAU / (nx as f64 * DX),
            std::f64::consts::TAU / (ny as f64 * DX),
        );
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
            -20.0 - 8.0 * (kx * x).sin() * (ky * y).cos()
        });
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
        let mut noise = |a: f64| a * (rng.next() - 0.5);
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
                let eta = 0.3 * (2.0 * kx * x).cos() * (ky * y).sin();
                state.eta.data[idx] = eta;
                let depth = eta - bathymetry.data[idx];
                // A pycnocline about 10 m deep, tilted by a front along x
                let front = 2.0 * (kx * x).sin();
                for (l, &s) in sigma_rho.iter().enumerate() {
                    let at = idx * nl + l;
                    let z = eta + s * depth;
                    state.u[at] =
                        0.6 * (ky * y).sin() + 0.25 * (s + 0.5) * (kx * x).cos() + noise(0.02);
                    state.v[at] =
                        0.4 * (kx * x).cos() - 0.2 * (s + 0.5) * (ky * y).sin() + noise(0.02);
                    let interface = ((z + 10.0 + front) / 1.5).tanh();
                    state.temp[at] = 8.0 + 3.0 * interface + noise(0.05);
                    state.salt[at] = 33.0 - 1.5 * interface + noise(0.02);
                    let smooth = ((z + 10.0) / 3.0).tanh();
                    references[0][at] = 8.0 + 3.0 * smooth;
                    references[1][at] = 33.0 - 1.5 * smooth;
                    rhs_u0[at] = noise(1e-3);
                    rhs_v0[at] = noise(1e-3);
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
        Self {
            subcell: vec![f64::NAN; ne * subcell_interfaces(ops.n_1d)],
            euler_dt: vec![euler_dt; ne],
            mesh,
            ops,
            geom,
            bathymetry,
            sigma,
            boundaries,
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
            pooled: &[],
        }
    }

    fn n_elements(&self) -> usize {
        self.mesh.n_elements
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
        MIN_COLUMN_DEPTH,
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
            &[],
            Blocks::All(fx.n_elements()),
        );
    }
}

fn cpu(fx: &Fixture) -> Cpu {
    let ne = fx.n_elements();
    let mut layers = LayerTransport::new(ne, &fx.ops, LEVELS).with_metric(MetricForm::Averaged);
    cpu_layers(fx, &mut layers);
    assert!(
        layers.subcell_elements.iter().all(|&s| !s),
        "the fixture is fully wet"
    );
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
                eta_rate: up(&fx.eta_rate)?,
                euler_dt: up(&fx.euler_dt)?,
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
    /// with `sources` the volume sources, as [`cpu_sources`]; `reference`
    /// false drops T's and S's reference (the mutant).
    fn advect(
        &self,
        transport: &Transport3D,
        rhs: &mut [DeviceBuffer<f64>; 4],
        reference: bool,
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
            false,
            sources,
        )?;
        let reference = |i: usize| self.reference.as_ref().filter(|_| reference).map(|r| &r[i]);
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
            true,
            sources,
        )
    }
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
    let statics = Statics3D::upload(
        gpu,
        &fx.mesh,
        &fx.ops,
        &fx.geom,
        &fx.bathymetry,
        &fx.sigma,
        &fx.boundaries,
        MetricForm::Averaged,
    )?;
    let mut transport = Transport3D::new(gpu, statics, MIN_COLUMN_DEPTH)?;
    let device = Device::upload(gpu, &fx)?;
    let [_, _, u, v] = &device.fields;
    transport.compute(&device.eta, u, v, &device.barotropic)?;
    let layers = &cpu.layers;
    let mut same = true;
    for (name, cpu, gpu_values) in [
        ("layer hu", &layers.hu, &transport.hu),
        ("layer hv", &layers.hv, &transport.hv),
        ("layer face", &layers.face, &transport.face),
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
        device.advect(&transport, &mut rhs, true, sources)?;
        for ((name, cpu), gpu_values) in names.iter().zip(results).zip(&rhs) {
            same &= report(gpu, &format!("{prefix} {name}"), cpu, gpu_values)?;
        }
    }

    // The mutant: the device drops the reference; the gate must see it
    if shape.reference {
        let mut rhs = device.rhs(gpu)?;
        device.advect(&transport, &mut rhs, false, false)?;
        let t = rhs[0].to_host_vec(&gpu.stream)?;
        let a = agreement(&cpu.advected[0], &t);
        println!(
            "    mutant (no reference on the device): T {} of {} differ{}",
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
fn time(gpu: &Arc<Gpu>, shape: Shape, iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
    let fx = Fixture::new(shape);
    let ne = fx.n_elements();
    let points = (ne * LEVELS) as f64;
    println!(
        "timing: {ne} P{ORDER} elements × {LEVELS} levels ({:.1} M points), {iterations} \
         evaluations",
        (ne * NN * LEVELS) as f64 / 1e6
    );
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
    let statics = Statics3D::upload(
        gpu,
        &fx.mesh,
        &fx.ops,
        &fx.geom,
        &fx.bathymetry,
        &fx.sigma,
        &fx.boundaries,
        MetricForm::Averaged,
    )?;
    let mut transport = Transport3D::new(gpu, statics, MIN_COLUMN_DEPTH)?;
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
        device.advect(&transport, &mut rhs, true, true)?;
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
    println!("== The 3D transport: layer transports, Ω, advection (Phase 4.0) ==");
    let base = Shape {
        nx: 24,
        ny: 20,
        jitter: 0.0,
        periodic: true,
        stretched: false,
        reference: false,
        euler_dt: 0.05,
    };
    let fixtures = [
        ("periodic, parallelograms", base),
        (
            "periodic, jittered, stretched",
            Shape {
                jitter: 0.15,
                stretched: true,
                ..base
            },
        ),
        (
            "periodic, jittered, stretched, reference",
            Shape {
                jitter: 0.15,
                stretched: true,
                reference: true,
                ..base
            },
        ),
        (
            "walled, jittered, reference, long Euler step",
            Shape {
                jitter: 0.15,
                periodic: false,
                reference: true,
                euler_dt: 40.0,
                ..base
            },
        ),
    ];
    let mut failed = Vec::new();
    for (label, shape) in fixtures {
        if !gate(gpu, label, shape)? {
            failed.push(label);
        }
    }
    time(
        gpu,
        Shape {
            nx: 320,
            ny: 300,
            jitter: 0.15,
            stretched: true,
            reference: true,
            ..base
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
