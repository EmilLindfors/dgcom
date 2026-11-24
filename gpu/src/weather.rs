//! The weather gate (P2.7 Phase 4.5i, `notes/gpu.md`): a run under a weather
//! model's air pressure (`GriddedAtmosphere2D`, its pressure gradient on the
//! depth mean) with its open boundary raised by the inverse barometer
//! (`InverseBarometer`), on the CPU and on the GPU, with global steps and
//! local time stepping, compared bit for bit.
//!
//! The weather is made up and held in memory (no NetCDF needed): a low of
//! 15 hPa crossing `nest`'s channel eastward at 10 m/s over a pressure
//! gradient of 3 hPa per 40 km, hourly on a 4 km grid, ramped up over half
//! an hour. Two configurations, as the case runner builds a weather case:
//!
//! - an M2 tide at the open boundary raised by the full inverse barometer,
//!   the Coriolis force and the pressure;
//! - `nest`'s parent at the open boundary raised by a quarter of it (as the
//!   case runner raises NorKyst's), its relaxation band, then the pressure:
//!   the device adds the band's and the weather's sources in the CPU's order.
//!
//! The mutants give the device another input: no pressure source, another
//! factor of the inverse barometer, the weather ramped up over an hour, no
//! inverse barometer.

use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

use dgcom::boundary::{
    CharacteristicOBC, HarmonicTide, InverseBarometer, MultiBoundaryCondition2D, Reflective2D,
};
use dgcom::equations::ShallowWater2D;
use dgcom::io::{
    AtmosphereField, AtmosphereReader, CoordinateProjection, FieldSeries, GeoGrid, LocalProjection,
    P_REFERENCE,
};
use dgcom::physics::{PhysicsBuilder, SWEPhysics2D};
use dgcom::simulation::Simulation;
use dgcom::solver::{SWESolution2D, SWEState2D, StandardLimiter2D, WetDryConfig};
use dgcom::source::{CoriolisSource2D, GriddedAtmosphere2D, ManningFriction2D, coriolis_parameter};
use dgcom::time::{ModelClock, Multirate, SSPRK43};
use dgcom::types::ElementIndex;

use crate::nest::{Domain, Nest};
use crate::phase0::{agreement, cpu_kernel};
use crate::swe2d::{DeviceState, Gpu, GpuSWE2D, NN};

const G: f64 = 9.81;
/// The weather's first snapshot (Unix s), the run's time 0 (`nest`'s).
const T0: f64 = 1_750_000_000.0;
/// The run (hours).
const HOURS: f64 = 6.0;
/// The CFL number of both runs.
const CFL: f64 = 0.5;

/// The made-up weather: sea-level pressure over a grid around the channel,
/// hourly over the run.
fn weather_reader(projection: &LocalProjection) -> Arc<AtmosphereReader> {
    let xs: Vec<f64> = (0..21).map(|i| -20_000.0 + 4000.0 * i as f64).collect();
    let ys: Vec<f64> = (0..12).map(|j| -20_000.0 + 4000.0 * j as f64).collect();
    let lon: Vec<f64> = xs.iter().map(|&x| projection.xy_to_geo(x, 0.0).1).collect();
    let lat: Vec<f64> = ys.iter().map(|&y| projection.xy_to_geo(0.0, y).0).collect();
    let grid = GeoGrid::regular(lon, lat).expect("a regular grid");
    let m = grid.len();
    let times: Vec<f64> = (0..=HOURS as usize + 1)
        .map(|k| T0 + 3600.0 * k as f64)
        .collect();
    let mut pressure = Vec::with_capacity(times.len() * m);
    for &t in &times {
        let centre = -30_000.0 + 10.0 * (t - T0);
        for &y in &ys {
            for &x in &xs {
                let r2 = ((x - centre).powi(2) + (y - 2000.0).powi(2)) / 30_000.0_f64.powi(2);
                pressure.push(P_REFERENCE + 300.0 * x / 40_000.0 - 1500.0 * (-r2).exp());
            }
        }
    }
    Arc::new(
        AtmosphereReader::new(grid, times)
            .expect("a weather")
            .with_field(
                AtmosphereField::Pressure,
                FieldSeries::with_offset(m, &pressure, P_REFERENCE),
            ),
    )
}

/// The weather on `domain`'s nodes, ramped up over `ramp` seconds.
fn atmosphere(domain: &Domain, ramp: f64) -> Result<GriddedAtmosphere2D, Box<dyn Error>> {
    let projection = LocalProjection::new(63.5, 8.0);
    Ok(GriddedAtmosphere2D::new(
        weather_reader(&projection),
        &domain.mesh,
        &domain.ops,
        projection,
        ModelClock::new(T0),
    )?
    .with_ramp_up(ramp))
}

/// The physics as the case runner builds a weather case on `bc`: the
/// Coriolis force, the band towards `nest`'s parent if `band`, then the
/// weather if any.
fn physics<'a>(
    domain: &Domain,
    band: Option<&Nest>,
    weather: Option<&GriddedAtmosphere2D>,
    bc: MultiBoundaryCondition2D<'a>,
) -> SWEPhysics2D<MultiBoundaryCondition2D<'a>> {
    let mut builder = PhysicsBuilder::swe_2d(
        domain.mesh.clone(),
        domain.ops.clone(),
        domain.geom.clone(),
        ShallowWater2D::new(G),
        bc,
    )
    .with_bathymetry(domain.bed.clone())
    .with_limiter(StandardLimiter2D::Positivity(WetDryConfig::DEFAULT_H_DRY))
    .with_wet_dry(WetDryConfig::default())
    .with_source(CoriolisSource2D::f_plane(coriolis_parameter(63.5)));
    if let Some(nest) = band {
        builder = builder.with_source(
            domain
                .parent
                .relaxation(nest.timescale)
                .with_momentum_only(nest.momentum_only),
        );
    }
    if let Some(weather) = weather {
        builder = builder.with_source(weather.clone());
    }
    builder
        .with_implicit_friction(ManningFriction2D::new(G, 0.025))
        .build()
}

/// Still water at mean sea level.
fn still_water(domain: &Domain) -> SWESolution2D {
    let mut q = SWESolution2D::new(domain.mesh.n_elements, domain.ops.n_nodes);
    for k in ElementIndex::iter(domain.mesh.n_elements) {
        for i in 0..domain.ops.n_nodes {
            let h = (0.0 - domain.bed.get(k, i)).max(0.0);
            q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
        }
    }
    q
}

/// `physics` on the CPU with `levels` of local time stepping (0: global
/// steps) to the run's end: the state and the steps.
fn on_cpu(
    physics: SWEPhysics2D<MultiBoundaryCondition2D<'_>>,
    q0: &SWESolution2D,
    levels: usize,
) -> (SWESolution2D, usize) {
    let mut q = q0.clone();
    let t_end = 3600.0 * HOURS;
    let result = if levels > 0 {
        Simulation::new(physics, Multirate::with_base(SSPRK43, levels))
            .with_cfl(CFL)
            .run(&mut q, 0.0, t_end)
    } else {
        Simulation::new(physics, SSPRK43)
            .with_cfl(CFL)
            .run(&mut q, 0.0, t_end)
    };
    assert!(result.success, "CPU run: {result:?}");
    (q, result.n_steps)
}

/// The same on the device, from `physics`' device form.
fn on_device(
    gpu: &Arc<Gpu>,
    physics: &SWEPhysics2D<MultiBoundaryCondition2D<'_>>,
    q0: &SWESolution2D,
    levels: usize,
) -> Result<(SWESolution2D, usize), Box<dyn Error>> {
    let form = physics.device_form()?;
    let module = GpuSWE2D::from_form(gpu, &form)?;
    let mut state = DeviceState::upload(gpu, q0);
    let t_end = 3600.0 * HOURS;
    let result = if levels > 0 {
        Simulation::new(module, Multirate::with_base(SSPRK43, levels))
            .with_cfl(CFL)
            .run(&mut state, 0.0, t_end)
    } else {
        Simulation::new(module, SSPRK43)
            .with_cfl(CFL)
            .run(&mut state, 0.0, t_end)
    };
    gpu.stream.synchronize()?;
    assert!(result.success, "GPU run: {result:?}");
    Ok((state.download(), result.n_steps))
}

/// One configuration and its mutants with `levels`; whether the GPU's run
/// is the CPU's bit for bit and every mutant differs.
#[allow(clippy::too_many_arguments)]
fn compare(
    gpu: &Arc<Gpu>,
    label: &str,
    levels: usize,
    domain: &Domain,
    q0: &SWESolution2D,
    cpu: SWEPhysics2D<MultiBoundaryCondition2D<'_>>,
    device: &SWEPhysics2D<MultiBoundaryCondition2D<'_>>,
    mutants: &[(&str, SWEPhysics2D<MultiBoundaryCondition2D<'_>>)],
) -> Result<bool, Box<dyn Error>> {
    let start = Instant::now();
    let (cpu, cpu_steps) = on_cpu(cpu, q0, levels);
    let cpu_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let (on_gpu, gpu_steps) = on_device(gpu, device, q0, levels)?;
    let gpu_seconds = start.elapsed().as_secs_f64();
    let n = domain.mesh.n_elements * NN;
    let (identical, worst_ulps, worst_rel) = agreement(&cpu, &on_gpu);
    println!(
        "{label}, {}: {} {} on the CPU ({cpu_seconds:.1} s) and {gpu_steps} on the GPU \
         ({gpu_seconds:.1} s) to {HOURS} h; {identical} of {n} nodes bit-identical to the {} CPU \
         kernel, largest difference {worst_ulps} ulp, {worst_rel:.1e} of the field's largest",
        if levels > 0 {
            format!("{levels} levels")
        } else {
            "global steps".into()
        },
        cpu_steps,
        if levels > 0 { "coarse steps" } else { "steps" },
        cpu_kernel()
    );
    let mut same = cpu_steps == gpu_steps && identical == n;
    for (what, mutant) in mutants {
        let (q, _) = on_device(gpu, mutant, q0, levels)?;
        let (identical, ..) = agreement(&cpu, &q);
        let caught = identical < n;
        println!(
            "  mutant {what}: {} of {n} nodes differ, {}",
            n - identical,
            if caught { "caught" } else { "NOT CAUGHT" }
        );
        same &= caught;
    }
    Ok(same)
}

pub fn run(gpu: &Arc<Gpu>) -> Result<(), Box<dyn Error>> {
    println!("== Phase 4.5i: a run under a weather model's air pressure ==");
    let nest = Nest::cubic();
    let domain = Domain::new(&nest)?;
    let q0 = still_water(&domain);
    let wall = Reflective2D::new();
    let weather = atmosphere(&domain, 1800.0)?;
    let slow = atmosphere(&domain, 3600.0)?;
    let mut all = true;

    // An M2 tide raised by the full inverse barometer
    let tide = || HarmonicTide::m2(0.8, 0.0).with_ramp_up(1800.0);
    let raised = |w: &GriddedAtmosphere2D, factor: f64| {
        CharacteristicOBC::new(InverseBarometer::new(tide(), w.clone()).with_factor(factor))
    };
    let (full, half, full_slow) = (
        raised(&weather, 1.0),
        raised(&weather, 0.5),
        raised(&slow, 1.0),
    );
    for levels in [0, 3] {
        let with = |open| MultiBoundaryCondition2D::new(&wall).with_open(open);
        let mutants = if levels == 0 {
            vec![
                (
                    "no pressure source",
                    physics(&domain, None, None, with(&full)),
                ),
                (
                    "the inverse barometer at half",
                    physics(&domain, None, Some(&weather), with(&half)),
                ),
                (
                    "the weather ramped over an hour",
                    physics(&domain, None, Some(&slow), with(&full_slow)),
                ),
            ]
        } else {
            Vec::new()
        };
        all &= compare(
            gpu,
            "tide raised by the inverse barometer, pressure",
            levels,
            &domain,
            &q0,
            physics(&domain, None, Some(&weather), with(&full)),
            &physics(&domain, None, Some(&weather), with(&full)),
            &mutants,
        )?;
    }

    // Nested: the parent raised by a quarter of it, the band, the pressure
    let parent = CharacteristicOBC::new(
        InverseBarometer::new(domain.parent.clone(), weather.clone()).with_factor(0.25),
    );
    let bare = CharacteristicOBC::new(domain.parent.clone());
    for levels in [0, 3] {
        let with = |open| MultiBoundaryCondition2D::new(&wall).with_open(open);
        let mutants = if levels == 0 {
            vec![(
                "no inverse barometer",
                physics(&domain, Some(&nest), Some(&weather), with(&bare)),
            )]
        } else {
            Vec::new()
        };
        all &= compare(
            gpu,
            "nested, the parent raised by a quarter, band, pressure",
            levels,
            &domain,
            &q0,
            physics(&domain, Some(&nest), Some(&weather), with(&parent)),
            &physics(&domain, Some(&nest), Some(&weather), with(&parent)),
            &mutants,
        )?;
    }
    if all {
        Ok(())
    } else {
        Err("a run under the weather on the GPU differs from the CPU's".into())
    }
}
