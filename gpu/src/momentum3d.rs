//! The 3D momentum gate (P2.7 Phase 4.2, `notes/gpu.md`): the reference
//! stratification at the columns' levels and the elements' displacements
//! about it, then the baroclinic pressure gradient in the σ-pairs form with
//! Coriolis ([`Baroclinic3D::reference`], [`Baroclinic3D::momentum`]) and
//! the horizontal viscosity of the shear ([`Viscosity3D::apply`]) on the
//! GPU, against dgcom's `ReferenceFields::fill`, `set_displacements`,
//! `compute_momentum_rhs_3d` and `apply_horizontal_viscosity_3d` on the CPU,
//! value for value, on the transport gate's fixtures: periodic and walled, a
//! walled beach with flats, films and dry columns (the subcell interfaces'
//! depth-mean force, thin columns without shear), a channel open at both
//! ends; about a reference profile (evenly spaced and linear, cubic,
//! unevenly spaced), without one (the anomaly about ρ₀ as a monotone
//! Hermite cubic), with a balanced reference state; in both metric forms;
//! f- and β-planes; a constant viscosity and Smagorinsky's. Mutants it must
//! catch, and timed at Chiloé's size (96,000 P2 elements × 20 levels).

use std::sync::Arc;
use std::time::Instant;

use cuda_core::DeviceBuffer;
use dgcom::physics::{EquationOfState, LinearEOS};
use dgcom::solver::Blocks;
use dgcom::solver::rhs::{
    BalancedReference, Boundaries3D, Exterior3D, ExtrapolationTracerBC3D, HorizontalViscosity3D,
    MetricForm, MomentumAdvectionForm, PressureGradientForm, ReferenceFields, Rhs3DConfig,
    UNBOUNDED, VerticalAdvection, ViscosityScratch3D, apply_horizontal_viscosity_3d,
    compute_momentum_rhs_3d,
};
use dgcom::solver::state::Solution3D;
use dgcom::source::CoriolisSource2D;

use crate::baroclinic::Baroclinic3D;
use crate::swe2d::{Gpu, NN, ORDER};
use crate::three_d::Statics3D;
use crate::transport3d::{Fixture, G, LEVELS, RHO0, Shape, agreement, report, report_reference};
use crate::viscosity::Viscosity3D;

/// No viscosity.
const INVISCID: HorizontalViscosity3D = HorizontalViscosity3D {
    background: 0.0,
    smagorinsky: 0.0,
};

/// One fixture of the gate: its shape, the volume terms' metric form, the
/// Coriolis parameter, whether a balanced reference state is set, and the
/// viscosity of the shear.
#[derive(Clone, Copy)]
struct Case {
    label: &'static str,
    shape: Shape,
    metric: MetricForm,
    /// β (1/(m s)) about f₀ = 1.3e-4 at y = 0: zero for an f-plane
    beta: f64,
    balanced: bool,
    viscosity: HorizontalViscosity3D,
}

impl Case {
    fn coriolis(&self) -> CoriolisSource2D {
        if self.beta == 0.0 {
            CoriolisSource2D::f_plane(1.3e-4)
        } else {
            CoriolisSource2D::beta_plane(1.3e-4, self.beta, 0.0)
        }
    }
}

/// The CPU's balanced reference state of `fx` (its own stratification).
fn balanced(fx: &Fixture, metric: MetricForm) -> BalancedReference {
    BalancedReference::new(
        &fx.state,
        &fx.mesh,
        &fx.bathymetry,
        &fx.sigma,
        &fx.ops,
        &fx.geom,
        G,
        RHO0,
        RHO0,
        fx.min_column_depth,
        metric,
        fx.fields.as_ref(),
    )
}

/// The CPU's `compute_momentum_rhs_3d` of `state` (`fields` its reference
/// at its levels): the velocity tendencies.
fn cpu_momentum(
    fx: &Fixture,
    state: &Solution3D,
    fields: Option<&ReferenceFields>,
    metric: MetricForm,
    coriolis: &CoriolisSource2D,
    balanced: Option<&BalancedReference>,
    rhs: &mut Solution3D,
) {
    let bc = ExtrapolationTracerBC3D;
    let config = Rhs3DConfig {
        mesh: &fx.mesh,
        ops: &fx.ops,
        geom: &fx.geom,
        bathymetry: &fx.bathymetry,
        sigma: &fx.sigma,
        coriolis,
        temp_bc: &bc,
        salt_bc: &bc,
        boundaries: &fx.boundaries,
        exterior: Exterior3D::default(),
        g: G,
        rho0: RHO0,
        min_column_depth: fx.min_column_depth,
        vertical_advection: VerticalAdvection::LimitedAkima,
        momentum_advection_form: MomentumAdvectionForm::Split,
        momentum_vertical_advection: VerticalAdvection::Centred,
        pressure_gradient: PressureGradientForm::SigmaPairs,
        balanced_reference: balanced,
        reference: fields,
        metric,
        tracer_range: [UNBOUNDED; 2],
        elements: Blocks::All(fx.n_elements()),
    };
    compute_momentum_rhs_3d(rhs, state, &config);
}

/// The CPU's `apply_horizontal_viscosity_3d` of `state` added to `rhs_u`,
/// `rhs_v`, on `fx`'s boundaries.
fn cpu_viscosity(
    fx: &Fixture,
    state: &Solution3D,
    viscosity: HorizontalViscosity3D,
    scratch: &mut ViscosityScratch3D,
    rhs_u: &mut [f64],
    rhs_v: &mut [f64],
) {
    apply_horizontal_viscosity_3d(
        rhs_u,
        rhs_v,
        state,
        viscosity,
        &fx.mesh,
        &fx.ops,
        &fx.geom,
        &fx.bathymetry,
        &fx.sigma,
        &fx.boundaries,
        fx.min_column_depth,
        scratch,
        Blocks::All(fx.n_elements()),
    );
}

/// The state's fields on the device.
struct Device {
    eta: DeviceBuffer<f64>,
    rho: DeviceBuffer<f64>,
    u: DeviceBuffer<f64>,
    v: DeviceBuffer<f64>,
    rhs_u: DeviceBuffer<f64>,
    rhs_v: DeviceBuffer<f64>,
}

impl Device {
    fn upload(gpu: &Gpu, state: &Solution3D) -> Result<Self, Box<dyn std::error::Error>> {
        let up = |v: &[f64]| DeviceBuffer::from_host(&gpu.stream, v);
        let n = state.u.len();
        Ok(Self {
            eta: up(&state.eta.data)?,
            rho: up(&state.rho)?,
            u: up(&state.u)?,
            v: up(&state.v)?,
            rhs_u: DeviceBuffer::zeroed(&gpu.stream, n)?,
            rhs_v: DeviceBuffer::zeroed(&gpu.stream, n)?,
        })
    }

    /// The reference and the momentum terms by `baroclinic`.
    fn momentum(
        &mut self,
        baroclinic: &mut Baroclinic3D,
        statics: &Statics3D,
    ) -> Result<(), Box<dyn std::error::Error>> {
        baroclinic.reference(statics, &self.eta, &self.rho)?;
        baroclinic.momentum(
            statics,
            &self.eta,
            &self.rho,
            &self.u,
            &self.v,
            &mut self.rhs_u,
            &mut self.rhs_v,
        )
    }

    /// The viscosity's tendencies alone by `viscosity` (on zero tendencies),
    /// into `rhs_u`, `rhs_v`.
    fn viscosity(
        &mut self,
        gpu: &Gpu,
        viscosity: &mut Viscosity3D,
        statics: &Statics3D,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.zero(gpu)?;
        viscosity.apply(
            statics,
            &self.eta,
            &self.u,
            &self.v,
            &mut self.rhs_u,
            &mut self.rhs_v,
        )
    }

    fn zero(&mut self, gpu: &Gpu) -> Result<(), Box<dyn std::error::Error>> {
        let n = self.u.len();
        self.rhs_u = DeviceBuffer::zeroed(&gpu.stream, n)?;
        self.rhs_v = DeviceBuffer::zeroed(&gpu.stream, n)?;
        Ok(())
    }
}

/// How many of `cpu`'s u values the device's `values` differ from; prints
/// the mutant's line. Whether the gate catches it.
fn report_mutant(
    gpu: &Gpu,
    what: &str,
    cpu: &[f64],
    device: &Device,
) -> Result<bool, Box<dyn std::error::Error>> {
    let values = device.rhs_u.to_host_vec(&gpu.stream)?;
    let a = agreement(cpu, &values);
    let caught = a.identical < a.len;
    println!(
        "    mutant ({what}): u {} of {} differ{}",
        a.len - a.identical,
        a.len,
        if caught { ", caught" } else { ": NOT CAUGHT" }
    );
    Ok(caught)
}

/// One case on both: every output compared, and the mutants. Whether all
/// are the same to the bit (and every mutant caught).
fn gate(gpu: &Arc<Gpu>, case: Case) -> Result<bool, Box<dyn std::error::Error>> {
    let fx = Fixture::new(case.shape);
    let ne = fx.n_elements();
    let coriolis = case.coriolis();
    let thin = (0..ne)
        .filter(|&k| {
            (k * NN..(k + 1) * NN)
                .any(|idx| fx.state.eta.data[idx] - fx.bathymetry.data[idx] < fx.min_column_depth)
        })
        .count();
    println!(
        "{}: {ne} P{ORDER} elements ({thin} with a thin column) × {LEVELS} levels, {:?} metric, \
         {}{}",
        case.label,
        case.metric,
        match (&fx.profile, case.balanced) {
            (Some(_), _) => "about a reference profile",
            (None, true) => "about ρ₀, with a balanced reference state",
            (None, false) => "about ρ₀",
        },
        match case.viscosity {
            v if v.is_zero() => String::new(),
            v => format!(
                ", the shear's viscosity ν₀ {} m²/s, C_s {}",
                v.background, v.smagorinsky
            ),
        }
    );
    let balanced = case.balanced.then(|| balanced(&fx, case.metric));
    let mut rhs = Solution3D::new(ne, NN, LEVELS);
    cpu_momentum(
        &fx,
        &fx.state,
        fx.fields.as_ref(),
        case.metric,
        &coriolis,
        balanced.as_ref(),
        &mut rhs,
    );
    let statics = fx.statics_in(gpu, case.metric)?;
    let mut baroclinic = fx.baroclinic(gpu, &statics, &coriolis, balanced.as_ref())?;
    let mut device = Device::upload(gpu, &fx.state)?;
    device.momentum(&mut baroclinic, &statics)?;
    let mut same = true;
    if let Some(fields) = &fx.fields {
        same &= report_reference(gpu, fields, &baroclinic, true)?;
    }
    same &= report(gpu, "pgf + coriolis u", &rhs.u, &device.rhs_u)?;
    same &= report(gpu, "pgf + coriolis v", &rhs.v, &device.rhs_v)?;

    // The mutants: the gate must catch each
    if let Some(fields) = &mut baroclinic.fields {
        // No displacement term (the momentum without a new reference)
        let kept = std::mem::replace(
            &mut fields.displacement,
            DeviceBuffer::zeroed(&gpu.stream, ne)?,
        );
        baroclinic.momentum(
            &statics,
            &device.eta,
            &device.rho,
            &device.u,
            &device.v,
            &mut device.rhs_u,
            &mut device.rhs_v,
        )?;
        same &= report_mutant(gpu, "no displacement term", &rhs.u, &device)?;
        if let Some(fields) = &mut baroclinic.fields {
            fields.displacement = kept;
        }
    }
    if fx.profile.is_some() && thin > 0 {
        // No depth-mean force along the subcell interfaces
        baroclinic.pressures(&statics, &device.eta, &device.rho)?;
        baroclinic.sub_mean = DeviceBuffer::zeroed(&gpu.stream, ne * crate::three_d::NSUB)?;
        baroclinic.gradient(
            &statics,
            &device.eta,
            &device.rho,
            &device.u,
            &device.v,
            &mut device.rhs_u,
            &mut device.rhs_v,
        )?;
        same &= report_mutant(
            gpu,
            "no subcell interfaces' depth-mean force",
            &rhs.u,
            &device,
        )?;
    }
    if case.beta != 0.0 {
        let mut f_plane = fx.baroclinic(
            gpu,
            &statics,
            &CoriolisSource2D::f_plane(1.3e-4),
            balanced.as_ref(),
        )?;
        device.momentum(&mut f_plane, &statics)?;
        same &= report_mutant(gpu, "an f-plane, not the β-plane", &rhs.u, &device)?;
    }
    if case.balanced {
        let mut without = fx.baroclinic(gpu, &statics, &coriolis, None)?;
        device.momentum(&mut without, &statics)?;
        same &= report_mutant(gpu, "no balanced reference state", &rhs.u, &device)?;
    }

    // The viscosity of the shear: alone, then after the force, as
    // `Hydrostatic3D::momentum_rhs` adds it
    let viscosity = case.viscosity;
    if !viscosity.is_zero() {
        let n = ne * NN * LEVELS;
        let mut scratch = ViscosityScratch3D::new(ne, &fx.ops);
        let (mut alone_u, mut alone_v) = (vec![0.0; n], vec![0.0; n]);
        cpu_viscosity(
            &fx,
            &fx.state,
            viscosity,
            &mut scratch,
            &mut alone_u,
            &mut alone_v,
        );
        let mut full = Solution3D::new(ne, NN, LEVELS);
        cpu_momentum(
            &fx,
            &fx.state,
            fx.fields.as_ref(),
            case.metric,
            &coriolis,
            balanced.as_ref(),
            &mut full,
        );
        cpu_viscosity(
            &fx,
            &fx.state,
            viscosity,
            &mut scratch,
            &mut full.u,
            &mut full.v,
        );
        let new = |statics: &Statics3D| {
            Viscosity3D::new(
                gpu,
                statics,
                &fx.ops,
                &fx.geom,
                viscosity,
                fx.min_column_depth,
            )
        };
        let mut device_viscosity = new(&statics)?;
        device.viscosity(gpu, &mut device_viscosity, &statics)?;
        same &= report(gpu, "viscosity u", &alone_u, &device.rhs_u)?;
        same &= report(gpu, "viscosity v", &alone_v, &device.rhs_v)?;
        device.momentum(&mut baroclinic, &statics)?;
        device_viscosity.apply(
            &statics,
            &device.eta,
            &device.u,
            &device.v,
            &mut device.rhs_u,
            &mut device.rhs_v,
        )?;
        same &= report(gpu, "momentum u", &full.u, &device.rhs_u)?;
        same &= report(gpu, "momentum v", &full.v, &device.rhs_v)?;

        // Its mutants, on the viscosity alone
        let layers = |device: &mut Device, v: &mut Viscosity3D| {
            v.layers(
                &statics,
                &device.u,
                &device.v,
                &mut device.rhs_u,
                &mut device.rhs_v,
            )
        };
        device.zero(gpu)?;
        device_viscosity.means(&statics, &device.eta, &device.u, &device.v)?;
        device_viscosity.mean = DeviceBuffer::zeroed(&gpu.stream, ne * NN)?;
        layers(&mut device, &mut device_viscosity)?;
        same &= report_mutant(gpu, "viscosity of u, not of the shear", &alone_u, &device)?;
        if viscosity.smagorinsky > 0.0 {
            device.zero(gpu)?;
            device_viscosity.means(&statics, &device.eta, &device.u, &device.v)?;
            device_viscosity.mean_gradient = DeviceBuffer::zeroed(&gpu.stream, ne * NN)?;
            layers(&mut device, &mut device_viscosity)?;
            same &= report_mutant(gpu, "the strain without the mean's", &alone_u, &device)?;
        }
        if !case.shape.periodic {
            // The walls as open faces: copied, not mirrored, and no stress
            let open = Statics3D::upload(
                gpu,
                &fx.mesh,
                &fx.ops,
                &fx.geom,
                &fx.bathymetry,
                &fx.sigma,
                &Boundaries3D::with_walls([]),
                case.metric,
            )?;
            let mut through = new(&open)?;
            device.viscosity(gpu, &mut through, &open)?;
            same &= report_mutant(gpu, "walls as open faces", &alone_u, &device)?;
        }
    }
    // The metric form acts on the jittered fixtures (on the CPU alone)
    if case.shape.jitter > 0.0 {
        let other = match case.metric {
            MetricForm::Averaged => MetricForm::Conservative,
            MetricForm::Conservative => MetricForm::Averaged,
        };
        let mut rhs_other = Solution3D::new(ne, NN, LEVELS);
        cpu_momentum(
            &fx,
            &fx.state,
            fx.fields.as_ref(),
            other,
            &coriolis,
            balanced.as_ref(),
            &mut rhs_other,
        );
        let a = agreement(&rhs.u, &rhs_other.u);
        println!(
            "    the metric form acts (CPU, {other:?}): u {} of {} differ",
            a.len - a.identical,
            a.len
        );
    }
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

/// Time the CPU and the GPU on `shape` (a reference, a β-plane, the averaged
/// metric, the case runner's Smagorinsky viscosity of the shear), per
/// element and level, by part: the reference at a stage's levels with the
/// displacements, the force with Coriolis, the viscosity. Every evaluation
/// is at another free surface (two alternating), so that the reference is
/// filled every time, as at every stage.
fn time(
    gpu: &Arc<Gpu>,
    label: &str,
    shape: Shape,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let fx = Fixture::new(shape);
    let ne = fx.n_elements();
    let points = (ne * LEVELS) as f64;
    let coriolis = CoriolisSource2D::beta_plane(1.3e-4, 2e-11, 0.0);
    let metric = MetricForm::Averaged;
    let eos = LinearEOS::default();
    let profile = fx.profile.as_ref().expect("a reference");
    // A second state, its free surface 1 mm higher
    let mut raised = fx.state.clone();
    raised.eta.data.iter_mut().for_each(|e| *e += 1e-3);
    let states = [&fx.state, &raised];
    let all = Blocks::All(ne);
    let viscosity = HorizontalViscosity3D::smagorinsky(0.1);
    let mut scratch = ViscosityScratch3D::new(ne, &fx.ops);
    let mut fields = ReferenceFields::default();
    let mut rhs = Solution3D::new(ne, NN, LEVELS);
    let mut cpu_parts = [0.0; 3];
    for it in 0..iterations + 2 {
        let state = states[it % 2];
        let t0 = Instant::now();
        fields.fill(
            profile,
            |t, s| eos.compute_density(t, s, 0.0),
            |t, s| eos.density_partials(t, s),
            state,
            &fx.bathymetry,
            &fx.sigma,
            all,
        );
        fields.set_displacements(
            state,
            &fx.geom,
            &fx.bathymetry,
            &fx.sigma,
            fx.min_column_depth,
            all,
        );
        let t1 = Instant::now();
        cpu_momentum(&fx, state, Some(&fields), metric, &coriolis, None, &mut rhs);
        let t2 = Instant::now();
        cpu_viscosity(&fx, state, viscosity, &mut scratch, &mut rhs.u, &mut rhs.v);
        let t3 = Instant::now();
        if it >= 2 {
            cpu_parts[0] += (t1 - t0).as_secs_f64();
            cpu_parts[1] += (t2 - t1).as_secs_f64();
            cpu_parts[2] += (t3 - t2).as_secs_f64();
        }
    }
    println!(
        "timing, {label}: {ne} P{ORDER} elements × {LEVELS} levels ({:.1} M points), \
         {iterations} evaluations",
        (ne * NN * LEVELS) as f64 / 1e6
    );
    let statics = fx.statics_in(gpu, metric)?;
    let mut baroclinic = fx.baroclinic(gpu, &statics, &coriolis, None)?;
    let mut device_viscosity = Viscosity3D::new(
        gpu,
        &statics,
        &fx.ops,
        &fx.geom,
        viscosity,
        fx.min_column_depth,
    )?;
    let mut devices = [
        Device::upload(gpu, &fx.state)?,
        Device::upload(gpu, &raised)?,
    ];
    let gpu_iterations = 5 * iterations;
    let mut gpu_parts = [0.0; 3];
    for it in 0..gpu_iterations + 2 {
        let d = &mut devices[it % 2];
        gpu.stream.synchronize()?;
        let t0 = Instant::now();
        baroclinic.reference(&statics, &d.eta, &d.rho)?;
        gpu.stream.synchronize()?;
        let t1 = Instant::now();
        baroclinic.momentum(
            &statics,
            &d.eta,
            &d.rho,
            &d.u,
            &d.v,
            &mut d.rhs_u,
            &mut d.rhs_v,
        )?;
        gpu.stream.synchronize()?;
        let t2 = Instant::now();
        device_viscosity.apply(&statics, &d.eta, &d.u, &d.v, &mut d.rhs_u, &mut d.rhs_v)?;
        gpu.stream.synchronize()?;
        let t3 = Instant::now();
        if it >= 2 {
            gpu_parts[0] += (t1 - t0).as_secs_f64();
            gpu_parts[1] += (t2 - t1).as_secs_f64();
            gpu_parts[2] += (t3 - t2).as_secs_f64();
        }
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per = |total: f64, count: usize| total * 1e9 / (count as f64 * points);
    println!("  per element and level (CPU {threads} threads | GPU, ×):");
    let names = [
        "reference at the levels, displacements",
        "pressure gradient, Coriolis",
        "viscosity of the shear (Smagorinsky 0.1)",
    ];
    let (mut cpu_all, mut gpu_all) = (0.0, 0.0);
    for ((name, c), g) in names.iter().zip(cpu_parts).zip(gpu_parts) {
        let (c, g) = (per(c, iterations), per(g, gpu_iterations));
        cpu_all += c;
        gpu_all += g;
        println!("    {name:<40} {c:7.2} ns | {g:6.2} ns ({:.1}×)", c / g);
    }
    println!(
        "    {:<40} {cpu_all:7.2} ns | {gpu_all:6.2} ns ({:.1}×); per stage {:.1} ms | {:.2} ms",
        "all",
        cpu_all / gpu_all,
        cpu_all * points / 1e6,
        gpu_all * points / 1e6
    );
    Ok(())
}

pub fn run(gpu: &Arc<Gpu>) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "== The 3D momentum terms: the reference, the pressure gradient, Coriolis, the viscosity (Phase 4.2) =="
    );
    let wet = Shape::WET;
    let cases = [
        Case {
            label: "periodic, parallelograms, reference",
            shape: Shape {
                reference: true,
                ..wet
            },
            metric: MetricForm::Averaged,
            beta: 0.0,
            balanced: false,
            viscosity: INVISCID,
        },
        Case {
            label: "periodic, jittered, stretched, reference",
            shape: Shape {
                jitter: 0.15,
                stretched: true,
                reference: true,
                ..wet
            },
            metric: MetricForm::Averaged,
            beta: 2e-11,
            balanced: false,
            viscosity: HorizontalViscosity3D {
                background: 0.1,
                smagorinsky: 0.1,
            },
        },
        Case {
            label: "walled, jittered, reference, conservative metric",
            shape: Shape {
                jitter: 0.15,
                periodic: false,
                reference: true,
                ..wet
            },
            metric: MetricForm::Conservative,
            beta: 2e-11,
            balanced: false,
            viscosity: HorizontalViscosity3D::constant(5.0),
        },
        Case {
            label: "walled, jittered, stretched, no reference",
            shape: Shape {
                jitter: 0.15,
                periodic: false,
                stretched: true,
                ..wet
            },
            metric: MetricForm::Averaged,
            beta: 0.0,
            balanced: false,
            viscosity: HorizontalViscosity3D::smagorinsky(0.1),
        },
        Case {
            label: "walled, jittered, no reference, balanced reference state",
            shape: Shape {
                jitter: 0.15,
                periodic: false,
                ..wet
            },
            metric: MetricForm::Conservative,
            beta: 2e-11,
            balanced: true,
            viscosity: INVISCID,
        },
        Case {
            label: "beach, parallelograms, reference",
            shape: Shape::BEACH,
            metric: MetricForm::Averaged,
            beta: 0.0,
            balanced: false,
            viscosity: HorizontalViscosity3D::smagorinsky(0.1),
        },
        Case {
            label: "beach, jittered, stretched, reference",
            shape: Shape {
                jitter: 0.2,
                stretched: true,
                ..Shape::BEACH
            },
            metric: MetricForm::Averaged,
            beta: 2e-11,
            balanced: false,
            viscosity: HorizontalViscosity3D {
                background: 0.1,
                smagorinsky: 0.1,
            },
        },
        Case {
            label: "beach, jittered, no reference",
            shape: Shape {
                jitter: 0.2,
                reference: false,
                ..Shape::BEACH
            },
            metric: MetricForm::Averaged,
            beta: 0.0,
            balanced: false,
            viscosity: HorizontalViscosity3D::constant(2.0),
        },
        Case {
            label: "channel, open ends, jittered, stretched, reference",
            shape: Shape::CHANNEL,
            metric: MetricForm::Averaged,
            beta: 2e-11,
            balanced: false,
            viscosity: HorizontalViscosity3D::smagorinsky(0.1),
        },
    ];
    let mut failed = Vec::new();
    for case in cases {
        if !gate(gpu, case)? {
            failed.push(case.label);
        }
    }
    time(
        gpu,
        "wet",
        Shape {
            nx: 320,
            ny: 300,
            jitter: 0.15,
            stretched: true,
            reference: true,
            ..wet
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
