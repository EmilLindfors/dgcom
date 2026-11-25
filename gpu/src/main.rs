//! dgcom's GPU kernels in Rust on NVIDIA's cuda-oxide (P2.7, `notes/gpu.md`),
//! their gates against the CPU, and dgcom's command line with the CUDA
//! backend.
//!
//! ```text
//! cargo oxide run --no-fmad                   # the gates, against the scalar CPU kernel
//! cargo oxide run --no-fmad --features simd   # ... against the SIMD batch
//! cargo oxide run --features simd             # multiply–adds fused: the timing
//! target\release\dgcom-gpu.exe phase1         # one gate (phase0, phase1, nest, weather, beach,
//!                                             # transport3d, momentum3d, stage3d,
//!                                             # limiter3d, vertical3d, implicit3d,
//!                                             # forcing3d, pass3d, step3d, parent3d;
//!                                             # all by default)
//! target\release\dgcom-gpu.exe case <case.toml> [--set key.path=value]...
//!                                             # a case on the CPU and the GPU, compared
//! target\release\dgcom-gpu.exe run <case.toml> [--set ...]
//!                                             # dgcom's command line (run, check, mesh,
//!                                             # score), [compute] device on CUDA
//! ```
//!
//! - **Phase 0:** the fully wet split-form RHS, node for node.
//! - **Phase 1:** dgcom's `Simulation` stepping the device state with the GPU
//!   physics module, against the same run on the CPU; a case file through
//!   dgcom's case runner (`case`).
//! - **Phase 2:** a run nested in a parent model, its boundary and
//!   relaxation band, with global steps and local time stepping (`nest`);
//!   under a weather model's air pressure, the open boundary raised by the
//!   inverse barometer (`weather`, Phase 4.5i).
//! - **Phase 4:** the 3D stages' layer transports, Ω and the advection of
//!   u, v, T and S with the reference's (`transport3d`); the reference at
//!   the levels, the baroclinic pressure gradient, Coriolis and the shear's
//!   viscosity (`momentum3d`); the stage values the mode splitter makes of
//!   a stage's inventories, with the merged cells, the thin columns' mask,
//!   the tracer limiter and the density, and the nesting band's relaxation
//!   (`stage3d`); the tracer limiter's configurations (`limiter3d`); the
//!   step's vertical terms, GLS and the implicit diffusion with the drags
//!   and the surface fluxes, and `dgcom::math` itself (`vertical3d`); the
//!   implicit vertical advection, the split of Ω and the relaxation
//!   (`implicit3d`); the slow forcing `G` of the barotropic pass
//!   (`forcing3d`); the barotropic pass on global substeps and on local
//!   time steps with its `DU_avg2`, filter, rivers and the 2D module's
//!   volume sources, the step's rates and balance, and its merged cells
//!   from what the pass moved (`pass3d`); a nesting parent model's columns
//!   at the band's slots, sampled on the device (`parent3d`); value for
//!   value.
//!
//! Built without multiply–add contraction (`--no-fmad`), the GPU's results
//! must be the CPU's bit for bit; that is the gate.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

mod baroclinic;
mod bound3d;
mod case;
mod forcing;
mod forcing3d;
mod gate;
mod hooks;
mod implicit3d;
mod judges;
mod limiter;
mod limiter3d;
mod listed;
mod model3d;
mod momentum3d;
mod nest;
mod parent;
mod parent3d;
mod pass;
mod pass3d;
mod phase0;
mod phase1;
mod stage3d;
mod step3d;
mod swe2d;
mod three_d;
mod transport3d;
mod vertical;
mod vertical3d;
mod viscosity;
mod weather;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let which = args.first().map_or("all", String::as_str);
    // dgcom's own command line, with the CUDA backend
    if matches!(
        which,
        "run" | "check" | "mesh" | "score" | "--version" | "-V"
    ) {
        return dgcom::case::cli::main(args, Some(&case::Cuda));
    }
    match gates(which, args.get(1..).unwrap_or_default()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// A gate: its name on the command line, what it runs, and whether `all`
/// runs it.
type Gate = (
    &'static str,
    fn(&Arc<swe2d::Gpu>) -> Result<(), Box<dyn std::error::Error>>,
    bool,
);

/// The gates in `all`'s order; the device's own judges last.
const GATES: [Gate; 16] = [
    ("phase0", phase0::run, true),
    ("phase1", phase1::run, true),
    ("nest", nest::run_gates, true),
    ("weather", weather::run, true),
    ("beach", phase1::run_beach, false),
    ("transport3d", transport3d::run, true),
    ("momentum3d", momentum3d::run, true),
    ("stage3d", stage3d::run, true),
    ("limiter3d", limiter3d::run, true),
    ("vertical3d", vertical3d::run, true),
    ("implicit3d", implicit3d::run, true),
    ("forcing3d", forcing3d::run, true),
    ("pass3d", pass3d::run, true),
    ("step3d", step3d::run, true),
    ("parent3d", parent3d::run, true),
    ("judges", judges::run, true),
];

fn gates(which: &str, rest: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let gpu = swe2d::Gpu::on(0)?;
    // How the gates judge: bit for bit where a build without multiply–add
    // contraction still mirrors the CPU, within a tolerance otherwise
    let fused = gate::probe(&gpu)?;
    println!(
        "dgcom-gpu on {}: {}",
        gpu.name,
        if fused {
            "a build with multiply–add contraction (the one that runs), judged within the \
             gates' tolerances"
        } else {
            "a build without multiply–add contraction, judged bit for bit where the kernels \
             mirror the CPU"
        }
    );
    if which == "case" {
        let mut path = None;
        let mut overrides = Vec::new();
        // A case's bar where it is not judged bit for bit: whole runs
        // A day of the 2D channel with contraction: 2.2e-8 of the cross-channel
        // current's largest value (4.0e-8 on local time steps)
        let mut bar = gate::Bar::within(1e-7);
        let mut rest = rest.iter();
        while let Some(arg) = rest.next() {
            match arg.as_str() {
                "--set" => overrides.push(rest.next().ok_or("--set needs key.path=value")?.clone()),
                "--tolerance" => {
                    bar.tolerance = rest.next().ok_or("--tolerance needs a number")?.parse()?;
                }
                "--outliers" => {
                    bar.outliers = rest.next().ok_or("--outliers needs a share")?.parse()?;
                }
                _ => path = Some(PathBuf::from(arg)),
            }
        }
        let path = path.ok_or("case needs a case file")?;
        // The case runner opens the device itself
        drop(gpu);
        gate::start("case", bar);
        return case::gate(&path, &overrides);
    }
    if which != "all" && !GATES.iter().any(|&(name, ..)| name == which) {
        let names: Vec<&str> = GATES.iter().map(|&(name, ..)| name).collect();
        return Err(format!(
            "no gate {which:?}: all, case or one of {}",
            names.join(", ")
        )
        .into());
    }
    // Every gate asked for runs, whatever the others find
    let mut stopped = Vec::new();
    for (name, run, in_all) in GATES {
        if (which == name || (which == "all" && in_all))
            && let Err(e) = run(&gpu)
        {
            println!("{name}: {e}");
            stopped.push(format!("{name}: {e}"));
        }
    }
    let (held, failed, unjudged) = gate::summary();
    if unjudged > 0 {
        println!(
            "({unjudged} mutants under their gate's tolerance were not judged by this build: \
             the build without multiply–add contraction judges them)"
        );
    }
    println!(
        "== {held} fixtures held against the CPU{} ==",
        if failed.is_empty() && stopped.is_empty() {
            ", none failed".to_string()
        } else {
            format!(
                "; FAILED: {}",
                failed
                    .iter()
                    .chain(&stopped)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("; ")
            )
        }
    );
    if failed.is_empty() && stopped.is_empty() {
        Ok(())
    } else {
        Err(format!("{} gates failed", failed.len().max(stopped.len())).into())
    }
}
