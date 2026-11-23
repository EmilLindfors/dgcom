//! dgcom's GPU kernels in Rust on NVIDIA's cuda-oxide (P2.7, `notes/gpu.md`),
//! their gates against the CPU, and dgcom's command line with the CUDA
//! backend.
//!
//! ```text
//! cargo oxide run --no-fmad                   # the gates, against the scalar CPU kernel
//! cargo oxide run --no-fmad --features simd   # ... against the SIMD batch
//! cargo oxide run --features simd             # multiply–adds fused: the timing
//! target\release\dgcom-gpu.exe phase1         # one gate (phase0, phase1, nest, beach,
//!                                             # transport3d, momentum3d; all by default)
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
//!   relaxation band, with global steps and local time stepping (`nest`).
//! - **Phase 4:** the 3D stages' layer transports, Ω and the advection of
//!   u, v, T and S with the reference's (`transport3d`); the reference at
//!   the levels, the baroclinic pressure gradient and Coriolis
//!   (`momentum3d`); value for value.
//!
//! Built without multiply–add contraction (`--no-fmad`), the GPU's results
//! must be the CPU's bit for bit; that is the gate.

use std::path::PathBuf;
use std::process::ExitCode;

mod baroclinic;
mod case;
mod momentum3d;
mod nest;
mod phase0;
mod phase1;
mod swe2d;
mod three_d;
mod transport3d;

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

fn gates(which: &str, rest: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if which == "case" {
        let mut path = None;
        let mut overrides = Vec::new();
        let mut rest = rest.iter();
        while let Some(arg) = rest.next() {
            match arg.as_str() {
                "--set" => overrides.push(rest.next().ok_or("--set needs key.path=value")?.clone()),
                _ => path = Some(PathBuf::from(arg)),
            }
        }
        let path = path.ok_or("case needs a case file")?;
        return case::gate(&path, &overrides);
    }
    let gpu = swe2d::Gpu::on(0)?;
    if which == "all" || which == "phase0" {
        phase0::run(&gpu)?;
    }
    if which == "all" || which == "phase1" {
        phase1::run(&gpu)?;
    }
    if which == "all" || which == "nest" {
        nest::run_gates(&gpu)?;
    }
    if which == "beach" {
        phase1::run_beach(&gpu)?;
    }
    if which == "all" || which == "transport3d" {
        transport3d::run(&gpu)?;
    }
    if which == "all" || which == "momentum3d" {
        momentum3d::run(&gpu)?;
    }
    Ok(())
}
