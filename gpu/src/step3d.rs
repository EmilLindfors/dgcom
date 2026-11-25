//! The 3D step's gate (P2.7 Phase 4.5b, `notes/gpu.md`): whole steps of
//! dgcom's own `ModeSplitIntegrator` with the device's model ([`Model3D`])
//! against the same steps with a `Hydrostatic3D` on the CPU, every field
//! after every step.
//!
//! The model has the case runner's terms: the wet/dry 2D module of a 3D run
//! (Coriolis on a β-plane, walls), k-ε, the log-layer bottom drag, a farm
//! of net cages, uniform surface stresses and a wind stress field, surface
//! heat and fresh-water fluxes over each column's sea surface, rivers,
//! the reference stratification with its pressure gradient, its advection
//! and the Kuzmin limiter about it, implicit vertical advection, the
//! shear's Smagorinsky viscosity. The integrator is the case runner's: the
//! pass on two levels of local time stepping, SSP-RK(4,3) in both, and the
//! 3D stages on global steps or on local ones (Phase 4.5d: two and three
//! levels, each side assigning the levels of its own state before every
//! step as `Simulation3D` does). The walled beach (films, dry and thin
//! columns, merged cells) starts with no turbulence, as a case does, so the
//! first step allocates it.
//!
//! The mutants give the device's side another input than the CPU's: a term
//! left out of its physics, or another integrator.

use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

use dgcom::Reflective2D;
use dgcom::physics::vertical_mixing::Forcing;
use dgcom::physics::{
    BottomDrag3D, GlsMixing, Hydrostatic3D, ImplicitVerticalAdvection, JerlovWater, LinearEOS,
    SeaSurface, SurfaceFluxFields, SurfaceFluxes3D, TimeStepLimit, TimeStepLimits,
};
use dgcom::simulation::{NonFinite, RunModel3D, non_finite_field};
use dgcom::solver::state::Solution3D;
use dgcom::solver::{TracerLimiter3DConfig, TracerLimiterType3D};
use dgcom::time::{ModeSplitIntegrator, ModeSplitPhysics, SplitModel, SplitWork, SspScheme};

use crate::gate::{self, Bar};
use crate::forcing3d::{cages, coriolis, wind};
use crate::hooks::State3D;
use crate::model3d::{Model3D, Work3D};
use crate::pass3d::{beach_physics, beach_rivers};
use crate::stage3d::{Snapshots, report, report_mutant, start_state};
use crate::swe2d::Gpu;
use crate::transport3d::{Fixture, G, RHO0, Shape};

pub(crate) type Physics = Hydrostatic3D<LinearEOS, GlsMixing, Reflective2D>;

/// The steps compared.
const STEPS: usize = 4;
/// The case runner's 3D CFL number with SSP-RK(4,3), and its cap (s).
pub(crate) const CFL: f64 = 4.0;
pub(crate) const DT_MAX: f64 = 60.0;

/// The terms of a model (a mutant leaves one out).
#[derive(Clone, Copy)]
pub(crate) struct Terms {
    pub(crate) wind: bool,
    pub(crate) rivers: bool,
    pub(crate) cages: bool,
    pub(crate) implicit: bool,
    pub(crate) limiter: bool,
    pub(crate) reference: bool,
    pub(crate) smagorinsky: f64,
    pub(crate) roughness: f64,
    /// The surface heat and fresh-water fluxes with their shortwave's water,
    /// and whether they read the sea surface's T
    pub(crate) fluxes: Option<JerlovWater>,
    pub(crate) flux_sst: bool,
    /// The shear's viscosity once per step (`with_viscosity_once_per_step`)
    pub(crate) viscosity_once: bool,
}

impl Terms {
    pub(crate) const CASE_RUNNER: Self = Self {
        wind: true,
        rivers: true,
        cages: true,
        implicit: true,
        limiter: true,
        reference: true,
        smagorinsky: 0.1,
        roughness: 0.003,
        fluxes: Some(JerlovWater::III),
        flux_sst: true,
        viscosity_once: false,
    };
}

/// Surface fluxes of each column's own sea surface and the time, as a
/// weather model's heat budget (`GriddedSurfaceFluxes`) depends on them: a
/// non-solar loss growing with the SST, a shortwave varying from node to
/// node, an E − P growing with the SSS. A device that evaluated them over
/// another surface or at another time would give other values. Without
/// `sst`, the non-solar flux is a uniform 10 °C surface's.
struct SeaSurfaceFluxes {
    sst: bool,
}

impl SurfaceFluxes3D for SeaSurfaceFluxes {
    fn surface_fluxes_into(&self, t: f64, surface: SeaSurface<'_>, fluxes: SurfaceFluxFields<'_>) {
        let hour = (t / 3600.0).sin();
        for (i, (&sst, sss)) in surface.temperature.iter().zip(surface.salinity).enumerate() {
            let sst = if self.sst { sst } else { 10.0 };
            fluxes.nonsolar[i] = -80.0 - 15.0 * (sst - 10.0) + 20.0 * hour;
            fluxes.shortwave[i] = 300.0 * (1.0 + hour) * (i % 9) as f64 / 8.0;
            fluxes.freshwater[i] = 2e-5 * (sss - 33.0);
        }
    }
}

/// A `Hydrostatic3D` of `fx` with `terms` (its boundaries the fixture's): the
/// gate's uniform stresses, k-ε and the beach's rivers.
pub(crate) fn physics(fx: &Fixture, terms: Terms) -> Physics {
    physics_with(
        fx,
        terms,
        Forcing {
            surface_stress: [0.03, -0.01],
            bottom_stress: [0.0, 0.0],
            surface_heat_flux: 0.0,
        },
        GlsMixing::k_epsilon(),
        terms.rivers.then(|| beach_rivers(fx)),
    )
}

/// [`physics`] with the uniform stresses `forcing`, the closure `mixing` and
/// `rivers` (the device's own judges choose them, `judges.rs`).
pub(crate) fn physics_with(
    fx: &Fixture,
    terms: Terms,
    forcing: Forcing,
    mixing: GlsMixing,
    rivers: Option<dgcom::source::RiverSources>,
) -> Physics {
    let mut limiter = TracerLimiter3DConfig {
        limiter_type: TracerLimiterType3D::HorizontalKuzmin { relaxation: 1.0 },
        ..TracerLimiter3DConfig::default()
    };
    if let Some(profile) = &fx.profile {
        limiter = limiter.with_reference_profile(profile.clone());
    }
    if !terms.limiter {
        limiter = TracerLimiter3DConfig::none();
    }
    let mut physics = Hydrostatic3D::new(
        Arc::new(fx.mesh.clone()),
        Arc::new(fx.ops.clone()),
        Arc::new(fx.geom.clone()),
        Arc::new(fx.sigma.clone()),
        Arc::new(fx.bathymetry.clone()),
        Arc::new(coriolis()),
        LinearEOS::default(),
        mixing,
        beach_physics(fx),
        forcing,
        G,
        RHO0,
    )
    .with_bottom_drag(BottomDrag3D::log_layer(terms.roughness))
    .with_min_column_depth(fx.min_column_depth)
    .with_smagorinsky_viscosity(terms.smagorinsky)
    .with_tracer_limiter(limiter);
    if terms.cages {
        physics = physics.with_cage_drag(cages(fx));
    }
    if terms.wind {
        physics = physics.with_surface_stress(wind(fx));
    }
    if let Some(rivers) = rivers {
        physics = physics.with_rivers(rivers);
    }
    if terms.implicit {
        physics = physics.with_implicit_vertical_advection(ImplicitVerticalAdvection::default());
    }
    if let Some(water) = terms.fluxes {
        physics = physics
            .with_surface_fluxes(SeaSurfaceFluxes {
                sst: terms.flux_sst,
            })
            .with_shortwave_absorption(water);
    }
    if terms.viscosity_once {
        physics = physics.with_viscosity_once_per_step();
    }
    physics.boundaries = fx.boundaries.clone();
    match &fx.profile {
        Some(profile) if terms.reference => physics.with_reference_profile(profile.clone()),
        _ => physics.without_reference_profile(),
    }
}

/// The integrator of a run (a mutant changes it).
#[derive(Clone, Copy)]
pub(crate) struct Run {
    pub(crate) barotropic_levels: usize,
    pub(crate) scheme: SspScheme,
    pub(crate) min_substeps: usize,
    /// Levels of the 3D stages' local time stepping (0: global steps)
    pub(crate) baroclinic_levels: usize,
}

impl Run {
    /// The case runner's: the pass on two levels, SSP-RK(4,3) in both,
    /// global 3D steps.
    pub(crate) const CASE_RUNNER: Self = Self {
        barotropic_levels: 2,
        scheme: SspScheme::Rk43,
        min_substeps: dgcom::time::MIN_BAROTROPIC_SUBSTEPS,
        baroclinic_levels: 0,
    };

    pub(crate) fn integrator(self) -> ModeSplitIntegrator {
        ModeSplitIntegrator::new()
            .with_barotropic_cfl(2.0)
            .with_barotropic_scheme(SspScheme::Rk43)
            .with_baroclinic_scheme(self.scheme)
            .with_barotropic_levels(self.barotropic_levels)
            .with_min_substeps(self.min_substeps)
            .with_baroclinic_levels(self.baroclinic_levels)
    }

    /// The next step: `dt` on global 3D steps; on local ones the coarse
    /// step of the levels `state`'s element steps give (assigned as
    /// `Simulation3D` assigns them, at most `DT_MAX`).
    fn step<W: SplitWork>(
        self,
        integrator: &mut ModeSplitIntegrator<W>,
        physics: &Physics,
        state: &Solution3D,
        dt: f64,
    ) -> f64 {
        if self.baroclinic_levels == 0 {
            return dt;
        }
        let mut element_dt = vec![0.0; physics.mesh.n_elements];
        physics.element_steps_into(state, CFL, self.scheme, &mut element_dt);
        let stencils = ModeSplitPhysics::stage_stencils(physics);
        integrator.assign_baroclinic_levels(
            &element_dt,
            &physics.mesh,
            &stencils,
            Some(DT_MAX),
            None,
        )
    }
}

/// `state`'s fields after step `n` of `dt`, with the splitter's counts (the
/// cells merged and kept, the barotropic substeps, the 3D stages' element
/// evaluations and finest level).
fn take<W: SplitWork>(
    out: &mut Snapshots,
    n: usize,
    state: &Solution3D,
    integrator: &ModeSplitIntegrator<W>,
    dt: f64,
) {
    let at = |field: &str| format!("step {n} {field}");
    let merged = integrator.merged_cells();
    let local = integrator.baroclinic_multirate_stats().unwrap_or_default();
    out.push((
        at("counts"),
        vec![
            merged[0] as f64,
            merged[1] as f64,
            integrator.last_substeps() as f64,
            dt,
            local.element_evaluations as f64,
            local.finest_level as f64,
        ],
    ));
    for (name, field) in [
        ("eta", &state.eta.data),
        ("ubar", &state.ubar.data),
        ("vbar", &state.vbar.data),
        ("u", &state.u),
        ("v", &state.v),
        ("T", &state.temp),
        ("S", &state.salt),
        ("k", &state.tke),
        ("psi", &state.gls),
        ("rho", &state.rho),
    ] {
        out.push((at(name), field.clone()));
    }
}

/// `STEPS` steps from `start` on the CPU (of `dt`, or on local 3D steps
/// their coarse steps): the snapshots and the time (s).
fn on_cpu(physics: &Physics, run: Run, start: &Solution3D, dt: f64) -> (Snapshots, f64) {
    let mut state = start.clone();
    let mut integrator = run.integrator();
    let mut snapshots = Snapshots::new();
    let clock = Instant::now();
    let mut t = 0.0;
    for n in 0..STEPS {
        physics.update_density(&mut state);
        let dt = run.step(&mut integrator, physics, &state, dt);
        integrator.step(&mut state, physics, dt, t);
        t += dt;
        take(&mut snapshots, n, &state, &integrator, dt);
    }
    (snapshots, clock.elapsed().as_secs_f64())
}

/// The same steps on the device, the levels of local 3D steps from its own
/// state: the snapshots and the time (s, the snapshots' downloads and the
/// levels' assignment not counted).
fn on_device(
    gpu: &Arc<Gpu>,
    physics: &Physics,
    run: Run,
    start: &Solution3D,
    dt: f64,
) -> Result<(Snapshots, f64), Box<dyn Error>> {
    let model = Model3D::new(gpu, physics)?;
    let mut state = State3D::upload(gpu, start)?;
    let mut integrator = run.integrator().on::<Work3D>();
    let mut snapshots = Snapshots::new();
    // The state on the host; its `w` stays the start's, as the CPU's does
    let mut host = start.clone();
    let (mut time, mut t) = (0.0, 0.0);
    for n in 0..STEPS {
        model.update_density(&mut state);
        if run.baroclinic_levels > 0 {
            state.download_into(gpu, &mut host)?;
        }
        let dt = run.step(&mut integrator, physics, &host, dt);
        let clock = Instant::now();
        integrator.step(&mut state, &model, dt, t);
        gpu.stream.synchronize()?;
        time += clock.elapsed().as_secs_f64();
        t += dt;
        state.download_into(gpu, &mut host)?;
        take(&mut snapshots, n, &host, &integrator, dt);
    }
    Ok((snapshots, time))
}

/// One fixture on both sides, and if `mutants` the mutants; whether every
/// field of every step is the same to the bit (and every mutant caught).
fn gate(
    gpu: &Arc<Gpu>,
    label: &str,
    shape: Shape,
    base: Terms,
    run: Run,
    mutants: bool,
) -> Result<bool, Box<dyn Error>> {
    let fx = Fixture::new(shape);
    let physics = physics(&fx, base);
    // As a case starts: no turbulence (the closure allocates it at the
    // first step's end), the density of the start's T and S
    let mut start = start_state(&fx);
    start.tke.clear();
    start.gls.clear();
    physics.update_density(&mut start);
    let dt = physics
        .compute_dt_ssp(&start, CFL, SspScheme::Rk43)
        .min(DT_MAX);
    if run.baroclinic_levels == 0 {
        println!(
            "  {label}: {} elements × {} levels, {STEPS} steps of {dt:.2} s",
            fx.n_elements(),
            fx.sigma.n_levels()
        );
    } else {
        println!(
            "  {label}: {} elements × {} levels, {STEPS} coarse steps of the 3D stages on up to \
             {} levels (global: {dt:.2} s)",
            fx.n_elements(),
            fx.sigma.n_levels(),
            run.baroclinic_levels
        );
    }
    let (cpu, cpu_time) = on_cpu(&physics, run, &start, dt);
    let (device, device_time) = on_device(gpu, &physics, run, &start, dt)?;
    let mut same = report(&cpu, &device);
    let last = &cpu[cpu.len() - 11].1;
    println!(
        "    after {STEPS} steps: {} cells merged and {} kept so far, {} barotropic substeps",
        last[0], last[1], last[2]
    );
    if run.baroclinic_levels > 0 {
        let steps: Vec<String> = (0..STEPS)
            .map(|n| format!("{:.2}", cpu[n * 11].1[3]))
            .collect();
        println!(
            "    coarse steps {} s; finest level {}, {} element evaluations of the 3D stages",
            steps.join(", "),
            last[5],
            last[4]
        );
    }
    println!(
        "    {:.1} ms a step on the CPU, {:.1} ms on the GPU",
        1e3 * cpu_time / STEPS as f64,
        1e3 * device_time / STEPS as f64
    );
    if base.viscosity_once {
        // The viscosity's own tendency at every stage
        let mutant = self::physics(
            &fx,
            Terms {
                viscosity_once: false,
                ..base
            },
        );
        let (mutant, _) = on_device(gpu, &mutant, run, &start, dt)?;
        same &= report_mutant("the viscosity at every stage", &cpu, &mutant);
    }
    if !mutants {
        return Ok(same);
    }
    let terms: [(&str, Terms); 11] = [
        (
            "no wind stress field",
            Terms {
                wind: false,
                ..base
            },
        ),
        (
            "no rivers",
            Terms {
                rivers: false,
                ..base
            },
        ),
        (
            "no cages",
            Terms {
                cages: false,
                ..base
            },
        ),
        (
            "explicit vertical advection",
            Terms {
                implicit: false,
                ..base
            },
        ),
        (
            "no tracer limiter",
            Terms {
                limiter: false,
                ..base
            },
        ),
        (
            "no reference",
            Terms {
                reference: false,
                ..base
            },
        ),
        (
            "Smagorinsky 0.12",
            Terms {
                smagorinsky: 0.12,
                ..base
            },
        ),
        (
            "bed roughness 4 mm",
            Terms {
                roughness: 0.004,
                ..base
            },
        ),
        (
            "no surface fluxes",
            Terms {
                fluxes: None,
                ..base
            },
        ),
        (
            "Jerlov I's shortwave",
            Terms {
                fluxes: Some(JerlovWater::I),
                ..base
            },
        ),
        (
            "the fluxes of a 10 °C sea surface",
            Terms {
                flux_sst: false,
                ..base
            },
        ),
    ];
    for (what, terms) in terms {
        let (mutant, _) = on_device(gpu, &self::physics(&fx, terms), run, &start, dt)?;
        same &= report_mutant(what, &cpu, &mutant);
    }
    let mut runs = vec![
        (
            "global substeps",
            Run {
                barotropic_levels: 0,
                ..run
            },
        ),
        (
            "SSP-RK3 stages",
            Run {
                scheme: SspScheme::Rk3,
                ..run
            },
        ),
        (
            "at least 40 substeps",
            Run {
                min_substeps: 40,
                ..run
            },
        ),
    ];
    if run.baroclinic_levels > 0 {
        runs.push((
            "the 3D stages on a level fewer",
            Run {
                baroclinic_levels: run.baroclinic_levels - 1,
                ..run
            },
        ));
    }
    for (what, mutated) in runs {
        let (mutant, _) = on_device(gpu, &physics, mutated, &start, dt)?;
        same &= report_mutant(what, &cpu, &mutant);
    }
    println!("    {}", gate::verdict(label, same));
    Ok(same)
}

/// `limits`' bounds, by name.
fn named(limits: &TimeStepLimits) -> [(&'static str, TimeStepLimit); 5] {
    [
        ("advection", limits.advection),
        ("vertical", limits.vertical),
        ("Coriolis", limits.coriolis),
        ("sloping levels", limits.sloping_levels),
        ("viscosity", limits.viscosity),
    ]
}

/// Whether two sets of limits are the same to the bit, each with its
/// element; the differences printed under `label` if `print`.
fn same_limits(label: &str, cpu: &TimeStepLimits, device: &TimeStepLimits, print: bool) -> bool {
    let mut same = true;
    for ((what, a), (_, b)) in named(cpu).into_iter().zip(named(device)) {
        // Its bits and its element; or, where the gate judges by a tolerance,
        // the step within it (two elements may then tie)
        let agree = if gate::exact() {
            a.dt.to_bits() == b.dt.to_bits() && a.element == b.element
        } else {
            a.dt == b.dt
                || (a.dt - b.dt).abs() <= gate::tolerance() * a.dt.abs().max(b.dt.abs())
        };
        if !agree {
            same = false;
            if print {
                println!(
                    "    {label}: {what} {:e} s at element {} on the CPU, {:e} s at {} on the GPU",
                    a.dt, a.element, b.dt, b.element
                );
            }
        }
    }
    same
}

/// The step's bound of every state of `states` on the device against
/// `Hydrostatic3D::time_step_limits` on the CPU, at the case runner's CFL
/// and at one a hundred times it (more elements' sloping levels measured),
/// with `physics` on the CPU and `device` on the device; whether all are
/// the same to the bit, and how many elements had their frequency measured.
fn bounds_agree(
    gpu: &Arc<Gpu>,
    label: &str,
    physics: &Physics,
    device: &Physics,
    states: &[Solution3D],
    print: bool,
) -> Result<(bool, usize), Box<dyn Error>> {
    let mut model = Model3D::new(gpu, device)?;
    let (mut same, mut measured) = (true, 0);
    for (n, state) in states.iter().enumerate() {
        // The state's `w` comes with the work (`new_work`: the host's)
        model.start_from(state);
        let s = State3D::upload(gpu, state)?;
        let mut work = model.new_work(&s);
        for cfl in [CFL, 100.0 * CFL] {
            let scheme = SspScheme::Rk43;
            let cpu = physics.time_step_limits(state, cfl, scheme);
            let (limits, listed) = model.time_step_limits(&s, &mut work, cfl, scheme);
            measured += listed;
            same &= same_limits(
                &format!("{label}, state {n}, CFL {cfl}"),
                &cpu,
                &limits,
                print,
            );
            let explicit = physics.implicit_vertical_advection.is_none();
            let (a, b) = (cpu.step(explicit, scheme), limits.step(explicit, scheme));
            same &= a.to_bits() == b.to_bits()
                || (!gate::exact() && (a - b).abs() <= gate::tolerance() * a.abs().max(b.abs()));
        }
    }
    Ok((same, measured))
}

/// The bound's gate on one fixture: the start and the CPU's states after
/// each of the steps, and the mutants (the device's side with another
/// input); whether all agree to the bit and every mutant is caught.
fn bound_gate(gpu: &Arc<Gpu>, label: &str, shape: Shape) -> Result<bool, Box<dyn Error>> {
    let fx = Fixture::new(shape);
    let base = Terms::CASE_RUNNER;
    let physics = physics(&fx, base);
    let mut state = start_state(&fx);
    state.tke.clear();
    state.gls.clear();
    physics.update_density(&mut state);
    let dt = physics
        .compute_dt_ssp(&state, CFL, SspScheme::Rk43)
        .min(DT_MAX);
    let mut integrator = Run::CASE_RUNNER.integrator();
    let mut states = vec![state.clone()];
    for n in 0..STEPS {
        physics.update_density(&mut state);
        integrator.step(&mut state, &physics, dt, n as f64 * dt);
        physics.post_process(&mut state);
        states.push(state.clone());
    }
    let (mut same, measured) = bounds_agree(gpu, label, &physics, &physics, &states, true)?;
    println!(
        "  {label}: the bound of {} states at CFL {CFL} and {}: {} ({measured} sloping-level \
         frequencies measured)",
        states.len(),
        100.0 * CFL,
        gate::verdict(&format!("{label}: the bound"), same)
    );
    // The mutants: another input on the device's side
    let mut thicker = self::physics(&fx, base);
    thicker.min_column_depth *= 1.5;
    let mutants: [(&str, Physics); 3] = [
        (
            "no reference",
            self::physics(
                &fx,
                Terms {
                    reference: false,
                    ..base
                },
            ),
        ),
        (
            "Smagorinsky 0.12",
            self::physics(
                &fx,
                Terms {
                    smagorinsky: 0.12,
                    ..base
                },
            ),
        ),
        ("thin columns to 1.5× the depth", thicker),
    ];
    for (what, mutant) in mutants {
        // (Bit for bit a bound that agrees is one not caught; within a
        // tolerance it may differ under it, which the other build judges)
        let (agree, _) = bounds_agree(gpu, label, &physics, &mutant, &states, false)?;
        let (caught, verdict) = gate::mutant(!agree, !gate::exact());
        println!("    mutant {what}{verdict}");
        same &= caught;
    }
    // `w` at rest on the device's side
    let rest: Vec<Solution3D> = states
        .iter()
        .map(|s| {
            let mut s = s.clone();
            s.w.iter_mut().for_each(|w| *w = 0.0);
            s
        })
        .collect();
    let mut model = Model3D::new(gpu, &physics)?;
    let mut agree = true;
    for (state, at_rest) in states.iter().zip(&rest) {
        model.start_from(at_rest);
        let s = State3D::upload(gpu, state)?;
        let mut work = model.new_work(&s);
        let cpu = physics.time_step_limits(state, CFL, SspScheme::Rk43);
        let (limits, _) = model.time_step_limits(&s, &mut work, CFL, SspScheme::Rk43);
        agree &= same_limits(label, &cpu, &limits, false);
    }
    let (caught, verdict) = gate::mutant(!agree, !gate::exact());
    println!("    mutant w at rest{verdict}");
    same &= caught;
    Ok(same)
}

/// The blow-up check's gate on one fixture: `non_finite_field` of states
/// with values that are not finite planted in them, on the CPU and on the
/// device (`RunModel3D::non_finite`): the same field, first element and
/// number of elements, or none for the finite start.
fn non_finite_gate(gpu: &Arc<Gpu>, label: &str, shape: Shape) -> Result<bool, Box<dyn Error>> {
    let fx = Fixture::new(shape);
    let physics = physics(&fx, Terms::CASE_RUNNER);
    let mut start = start_state(&fx);
    physics.update_density(&mut start);
    let (ne, nn, layer) = (
        start.n_elements,
        start.n_nodes,
        start.n_nodes * start.n_levels,
    );
    // The values planted: (field as NON_FINITE_FIELDS numbers it, element,
    // value within the element's), as NaN or an infinity
    let mixed = vec![
        (3, 3, 2),
        (3, 5, 0),
        (3, 9, 1),
        (4, 7, 0),
        (4, 2, layer - 1),
        (2, 8, layer - 1),
        (2, 4, 3),
        (1, 6, 0),
        (1, 1, 5),
        (1, 0, 0),
    ];
    let with_eta = [mixed.clone(), vec![(0, 5, 1)]].concat();
    type Planted = Vec<(usize, usize, usize)>;
    let cases: [(&str, Planted); 6] = [
        ("finite", vec![]),
        ("T in 3 elements, S and v in 2, u in 3", mixed),
        ("and η in 1", with_eta),
        ("S in the last value", vec![(4, ne - 1, layer - 1)]),
        ("η in the last value", vec![(0, ne - 1, nn - 1)]),
        ("T everywhere", (0..ne).map(|k| (3, k, k % layer)).collect()),
    ];
    let mut model = Model3D::new(gpu, &physics)?;
    let mut same = true;
    for (what, planted) in cases {
        let mut state = start.clone();
        for (n, &(field, k, i)) in planted.iter().enumerate() {
            let bad = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY][n % 3];
            match field {
                0 => state.eta.data[k * nn + i] = bad,
                1 => state.u[k * layer + i] = bad,
                2 => state.v[k * layer + i] = bad,
                3 => state.temp[k * layer + i] = bad,
                _ => state.salt[k * layer + i] = bad,
            }
        }
        model.start_from(&state);
        let device = State3D::upload(gpu, &state)?;
        let mut work = model.new_work(&device);
        let cpu = non_finite_field(&state);
        let on_gpu = model.non_finite(&device, &mut work);
        let key = |bad: &Option<NonFinite>| bad.as_ref().map(|b| (b.field, b.first, b.elements));
        let agree = key(&cpu) == key(&on_gpu);
        same &= agree;
        let show = |bad: &Option<NonFinite>| match bad {
            Some(b) => format!("{} in {} elements from {}", b.field, b.elements, b.first),
            None => "none".into(),
        };
        println!(
            "  {label}, {what}: CPU {}, GPU {}{}",
            show(&cpu),
            show(&on_gpu),
            if agree { "" } else { ": DIFFERENT" }
        );
    }
    Ok(same)
}

/// Whole steps of the case runner's model on `fx`, after one untimed step
/// on each side: the CPU's 24 threads against the device.
fn time(gpu: &Arc<Gpu>, fx: &Fixture, steps: usize) -> Result<(), Box<dyn Error>> {
    let physics = physics(fx, Terms::CASE_RUNNER);
    let mut start = start_state(fx);
    start.tke.clear();
    start.gls.clear();
    physics.update_density(&mut start);
    let dt = physics
        .compute_dt_ssp(&start, CFL, SspScheme::Rk43)
        .min(DT_MAX);
    let run = Run::CASE_RUNNER;
    let mut cpu = start.clone();
    let mut integrator = run.integrator();
    let mut cpu_time = 0.0;
    for n in 0..=steps {
        let clock = Instant::now();
        physics.update_density(&mut cpu);
        integrator.step(&mut cpu, &physics, dt, n as f64 * dt);
        if n > 0 {
            cpu_time += clock.elapsed().as_secs_f64();
        }
    }
    let substeps = integrator.last_substeps();
    drop(cpu);
    let model = Model3D::new(gpu, &physics)?;
    let mut state = State3D::upload(gpu, &start)?;
    let mut integrator = run.integrator().on::<Work3D>();
    let mut gpu_time = 0.0;
    for n in 0..=steps {
        gpu.stream.synchronize()?;
        let clock = Instant::now();
        model.update_density(&mut state);
        integrator.step(&mut state, &model, dt, n as f64 * dt);
        gpu.stream.synchronize()?;
        if n > 0 {
            gpu_time += clock.elapsed().as_secs_f64();
        }
    }
    let (cpu_ms, gpu_ms) = (1e3 * cpu_time / steps as f64, 1e3 * gpu_time / steps as f64);
    println!(
        "timing, the whole 3D step: {} elements × {} levels, steps of {dt:.1} s with \
         {substeps} barotropic substeps: {cpu_ms:.0} ms a step on the CPU's threads, \
         {gpu_ms:.0} ms on the GPU ({:.1}×)",
        fx.n_elements(),
        fx.sigma.n_levels(),
        cpu_ms / gpu_ms
    );
    Ok(())
}

pub fn run(gpu: &Arc<Gpu>) -> Result<(), Box<dyn Error>> {
    println!("3D steps, whole: ModeSplitIntegrator with the device's model against the CPU's");
    // Four whole steps. Within a tolerance the first is judged: contraction
    // moves 0.4 % of T's values by more than 1e-9 in it (a merged cell more
    // or fewer), and from the second step on that cell's pass differs at
    // every node, as a mutant's does
    gate::start("step3d", Bar::within(1e-6).with_outliers(0.02));
    gate::scope(Some("step 0"));
    let base = Terms::CASE_RUNNER;
    let run = Run::CASE_RUNNER;
    let jittered = Shape {
        jitter: 0.2,
        stretched: true,
        ..Shape::BEACH
    };
    let mut same = gate(gpu, "beach", Shape::BEACH, base, run, true)?;
    same &= gate(
        gpu,
        "beach, jittered and stretched",
        jittered,
        base,
        run,
        false,
    )?;
    same &= gate(
        gpu,
        "beach, the viscosity once per step",
        Shape::BEACH,
        Terms {
            viscosity_once: true,
            ..base
        },
        run,
        false,
    )?;
    // The 3D stages on local time steps (P2.7 4.5d)
    let local = |levels| Run {
        baroclinic_levels: levels,
        ..run
    };
    same &= gate(
        gpu,
        "beach, local 3D steps",
        Shape::BEACH,
        base,
        local(2),
        true,
    )?;
    same &= gate(
        gpu,
        "beach, jittered and stretched, local 3D steps",
        jittered,
        base,
        local(3),
        false,
    )?;
    // One state's bound and its blow-up check: a single evaluation's bar
    gate::set(Bar::MIRRORED);
    gate::scope(None);
    println!("3D blow-up check: the device's against non_finite_field");
    same &= non_finite_gate(gpu, "beach", Shape::BEACH)?;
    println!("3D step's bound: the device's against Hydrostatic3D::time_step_limits");
    same &= bound_gate(gpu, "beach", Shape::BEACH)?;
    same &= bound_gate(
        gpu,
        "beach, jittered and stretched",
        Shape {
            jitter: 0.2,
            stretched: true,
            ..Shape::BEACH
        },
    )?;
    // 48,000 elements by default: the model takes ≈ 136 kB an element
    // (6.5 GB at 48,000), and a card that cannot hold it pages to the
    // host's memory (Chiloé's 96,000 took 53 s a step on the laptop's 8 GB);
    // `DGCOM_STEP3D_ELEMENTS` sets the size (`time.sh`: 96,000)
    let elements: usize = std::env::var("DGCOM_STEP3D_ELEMENTS")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(48_000);
    // The beach's shape at any size (96,000: the other gates' 320 × 300)
    let ny = ((elements as f64 * 300.0 / 320.0).sqrt().round() as usize).max(4);
    let large = Fixture::new(Shape {
        nx: (elements / ny).max(4),
        ny,
        jitter: 0.15,
        stretched: true,
        ..Shape::BEACH
    });
    time(gpu, &large, 3)?;
    if !same {
        return Err("the 3D steps differ, or a mutant went uncaught".into());
    }
    Ok(())
}
