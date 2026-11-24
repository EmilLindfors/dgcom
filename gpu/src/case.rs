//! A case file on the GPU: dgcom's case runner with the CUDA backend
//! ([`Cuda`], `[compute] device`; a 2D run and, since Phase 4.5c, a 3D
//! one), and the gate of a whole case: the same case run on the CPU and on
//! the GPU, the stations' records and the 3D snapshots compared bit for
//! bit.

use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use dgcom::boundary::MultiBoundaryCondition2D;
use dgcom::case::{
    CaseDevice, CaseFile, DeviceCallback3D, DeviceRun, DeviceRun3D, DeviceRunReport,
    DeviceRunReport3D, RunReport, StateView3D, StationSeries,
};
use dgcom::physics::{GlsMixing, Hydrostatic3D, LinearEOS, SWEDeviceForm2D};
use dgcom::simulation::{Simulation, Simulation3D, SimulationResult};
use dgcom::solver::SWESolution2D;
use dgcom::solver::state::Solution3D;
use dgcom::time::{IntegratorInfo, ModeSplitIntegrator, Multirate, TimeIntegrator};

use crate::hooks::State3D;
use crate::model3d::{Model3D, Work3D};
use crate::swe2d::{DeviceState, Gpu, GpuSWE2D, MAX_DEVICE_LEVELS};

/// The CUDA backend of the case runner: a 2D run's state and physics on a
/// CUDA GPU, stepped by dgcom's own `Simulation`, the state back on the host
/// at every output; a 3D run's on the device as [`Model3D`], stepped by
/// dgcom's own `Simulation3D`, the depth means back on the host after every
/// step (the stations) and the whole state at outputs.
pub struct Cuda;

impl CaseDevice for Cuda {
    fn run_2d(
        &self,
        ordinal: usize,
        form: &SWEDeviceForm2D<'_>,
        q: &mut SWESolution2D,
        run: &DeviceRun<'_>,
        callback: &mut dyn FnMut(&SWESolution2D, f64),
    ) -> Result<DeviceRunReport, String> {
        if run.levels > MAX_DEVICE_LEVELS {
            return Err(format!(
                "{} levels of local time stepping: the device takes at most {MAX_DEVICE_LEVELS}",
                run.levels
            ));
        }
        let gpu = Gpu::on(ordinal).map_err(|e| e.to_string())?;
        let physics = GpuSWE2D::from_form(&gpu, form)?;
        let mut state = DeviceState::upload(&gpu, q);
        let (result, clips) = if run.levels > 0 {
            let integrator = Multirate::with_base(run.integrator, run.levels);
            drive(
                Simulation::new(physics, integrator),
                run,
                &mut state,
                callback,
            )
        } else {
            drive(
                Simulation::new(physics, run.integrator),
                run,
                &mut state,
                callback,
            )
        };
        *q = state.download();
        Ok(DeviceRunReport {
            result,
            negative_depth_clips: clips,
            device: gpu.name.clone(),
        })
    }

    fn run_3d(
        &self,
        ordinal: usize,
        physics: &Hydrostatic3D<LinearEOS, GlsMixing, MultiBoundaryCondition2D<'_>>,
        integrator: ModeSplitIntegrator,
        state: &mut Solution3D,
        run: &DeviceRun3D,
        callback: &mut DeviceCallback3D<'_>,
    ) -> Result<DeviceRunReport3D, String> {
        if integrator.baroclinic_levels() > 0 {
            return Err(
                "local time steps of the 3D stages (three_d.levels_lts) are not on the device \
                 yet (P2.7 4.5d)"
                    .into(),
            );
        }
        let gpu = Gpu::on(ordinal).map_err(|e| e.to_string())?;
        let mut model = Model3D::new(&gpu, physics)?;
        model.start_from(state);
        let mut device = State3D::upload(&gpu, state).map_err(|e| e.to_string())?;
        let mut sim = Simulation3D::new(model, integrator.on::<Work3D>())
            .with_cfl(run.cfl)
            .with_dt_max(run.dt_max)
            .with_landing_interval(run.landing_interval);
        let (t_start, t_end) = run.span;
        let result = sim.run_with_context_callback(&mut device, t_start, t_end, |s, t, context| {
            let mut view = context.physics.view(s, context.integrator().work());
            callback(&mut view, t, &|| context.integrator().slow_forcing_record());
        });
        // The end state, whole (the last callback's, if an output asked)
        let integrator = sim.integrator();
        *state = sim
            .physics()
            .view(&device, integrator.work())
            .full()
            .clone();
        Ok(DeviceRunReport3D {
            result,
            last_substeps: integrator.last_substeps(),
            negative_depth_clips: sim.physics().negative_depth_clips(),
            barotropic: integrator.barotropic_multirate_stats(),
            baroclinic: integrator.baroclinic_multirate_stats(),
            device: gpu.name.clone(),
        })
    }
}

/// `sim` from the state over the run's span with its settings, the state to
/// the host at every callback; the result and the wet/dry clips.
fn drive<I: TimeIntegrator<DeviceState> + IntegratorInfo>(
    sim: Simulation<DeviceState, GpuSWE2D<'_>, I>,
    run: &DeviceRun<'_>,
    state: &mut DeviceState,
    callback: &mut dyn FnMut(&SWESolution2D, f64),
) -> (SimulationResult, usize) {
    let mut sim = sim
        .with_cfl(run.cfl)
        .with_callback_interval(run.interval)
        .with_stop(Arc::clone(run.stop));
    if let Some(dt) = run.dt_max {
        sim = sim.with_dt_max(dt);
    }
    let result = sim.run_with_callback(state, run.span.0, run.span.1, |s, t| {
        callback(&s.download(), t)
    });
    (result, sim.physics().negative_depth_clips())
}

/// The case gate: the case at `path` (with `overrides`) on the CPU and on
/// CUDA device 0, each into its own directory under `output/gpu_gate/`,
/// their steps, volumes, clips, every station's every record and a 3D
/// run's snapshot compared bit for bit (a build without multiply–add
/// contraction).
pub fn gate(path: &Path, overrides: &[String]) -> Result<(), Box<dyn Error>> {
    let run = |device: &str| -> Result<(RunReport, f64, std::path::PathBuf), Box<dyn Error>> {
        let mut all = overrides.to_vec();
        all.push(format!("compute.device={device}"));
        all.push(format!(
            "output.directory=output/gpu_gate/{}",
            device.replace(':', "")
        ));
        let file = CaseFile::read(path, &all)?;
        let domain = file.build_domain()?;
        let start = Instant::now();
        let report = file.run_on_with(&domain, Some(&Cuda), &mut |_| {})?;
        // The run's own snapshot (other runs' share the directory)
        let snapshot = file
            .output_dir()
            .join(format!("{}.dgsnap", file.case.run.name));
        Ok((report, start.elapsed().as_secs_f64(), snapshot))
    };
    let (cpu, cpu_seconds, cpu_snapshot) = run("cpu")?;
    let (gpu, gpu_seconds, gpu_snapshot) = run("cuda:0")?;

    println!(
        "{}: {} steps on the CPU ({cpu_seconds:.1} s) and {} on the GPU ({gpu_seconds:.1} s), \
         dt {:.3}–{:.3} s and {:.3}–{:.3} s",
        path.display(),
        cpu.n_steps,
        gpu.n_steps,
        cpu.dt_range_s.0,
        cpu.dt_range_s.1,
        gpu.dt_range_s.0,
        gpu.dt_range_s.1
    );
    let same_volume = cpu.volume_m3.1.to_bits() == gpu.volume_m3.1.to_bits();
    println!(
        "  volume at the end {:.9e} and {:.9e} m³: {}",
        cpu.volume_m3.1,
        gpu.volume_m3.1,
        if same_volume { "identical" } else { "differ" }
    );
    for note in gpu.notes.iter().filter(|n| n.starts_with("computed on")) {
        println!("  {note}");
    }
    let clips = |r: &RunReport| {
        r.notes
            .iter()
            .find(|n| n.ends_with("negative-depth clips"))
            .cloned()
            .unwrap_or_default()
    };
    println!("  CPU: {}; GPU: {}", clips(&cpu), clips(&gpu));

    let mut all_identical = same_volume && cpu.n_steps == gpu.n_steps && clips(&cpu) == clips(&gpu);
    for (c, g) in cpu.stations.iter().zip(&gpu.stations) {
        let (identical, n, worst) = compare_station(c, g);
        all_identical &= identical == n;
        println!(
            "  station {}: {identical} of {n} records bit-identical, largest difference {worst:.1e}",
            c.name
        );
    }
    // A 3D run's snapshot (written by the same build: the same stamp)
    if let (Ok(a), Ok(b)) = (std::fs::read(&cpu_snapshot), std::fs::read(&gpu_snapshot)) {
        let differ = a.len().max(b.len()) - a.iter().zip(&b).filter(|(x, y)| x == y).count();
        all_identical &= differ == 0;
        println!(
            "  snapshot {}: {differ} of {} bytes differ",
            cpu_snapshot
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            a.len().max(b.len())
        );
    }
    if all_identical {
        println!("  the GPU's run is the CPU's bit for bit");
        Ok(())
    } else {
        Err("the GPU's run differs from the CPU's".into())
    }
}

/// Of two stations' records: how many values are bit-identical, of how
/// many, and the largest absolute difference.
fn compare_station(a: &StationSeries, b: &StationSeries) -> (usize, usize, f64) {
    let fields = |s: &StationSeries| {
        [&s.time_s, &s.eta_m, &s.u_m_s, &s.v_m_s, &s.depth_m]
            .into_iter()
            .flatten()
            .copied()
            .collect::<Vec<f64>>()
    };
    let (a, b) = (fields(a), fields(b));
    let n = a.len().max(b.len());
    let identical = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() == y.to_bits())
        .count();
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    (identical, n, worst)
}
