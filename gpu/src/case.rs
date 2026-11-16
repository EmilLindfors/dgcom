//! A case file on the GPU: dgcom's case runner with the CUDA backend
//! ([`Cuda`], `[compute] device`), and the gate of a whole case: the same
//! case run on the CPU and on the GPU, the stations' records compared bit
//! for bit.

use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use dgcom::case::{CaseDevice, CaseFile, DeviceRun, DeviceRunReport, RunReport, StationSeries};
use dgcom::physics::SWEDeviceForm2D;
use dgcom::simulation::Simulation;
use dgcom::solver::SWESolution2D;

use crate::swe2d::{DeviceState, Gpu, GpuSWE2D};

/// The CUDA backend of the case runner: a 2D run's state and physics on a
/// CUDA GPU, stepped by dgcom's own `Simulation`; the state comes back to
/// the host at every output.
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
        let gpu = Gpu::on(ordinal).map_err(|e| e.to_string())?;
        let physics = GpuSWE2D::from_form(&gpu, form)?;
        let mut state = DeviceState::upload(&gpu, q);
        let mut sim = Simulation::new(physics, run.integrator)
            .with_cfl(run.cfl)
            .with_callback_interval(run.interval)
            .with_stop(Arc::clone(run.stop));
        if let Some(dt) = run.dt_max {
            sim = sim.with_dt_max(dt);
        }
        let result = sim.run_with_callback(&mut state, run.span.0, run.span.1, |s, t| {
            callback(&s.download(), t)
        });
        *q = state.download();
        Ok(DeviceRunReport {
            result,
            negative_depth_clips: sim.physics().negative_depth_clips(),
            device: gpu.name.clone(),
        })
    }
}

/// The case gate: the case at `path` (with `overrides`) on the CPU and on
/// CUDA device 0, each into its own directory under `output/gpu_gate/`,
/// their steps, volumes, clips and every station's every record compared
/// bit for bit (a build without multiply–add contraction).
pub fn gate(path: &Path, overrides: &[String]) -> Result<(), Box<dyn Error>> {
    let run = |device: &str| -> Result<(RunReport, f64), Box<dyn Error>> {
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
        Ok((report, start.elapsed().as_secs_f64()))
    };
    let (cpu, cpu_seconds) = run("cpu")?;
    let (gpu, gpu_seconds) = run("cuda:0")?;

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
