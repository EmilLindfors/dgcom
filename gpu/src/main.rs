//! dgcom's GPU kernels in Rust on NVIDIA's cuda-oxide (P2.7, `notes/gpu.md`),
//! and their gates against the CPU.
//!
//! ```text
//! cargo oxide run --no-fmad                   # the gates, against the scalar CPU kernel
//! cargo oxide run --no-fmad --features simd   # ... against the SIMD batch
//! cargo oxide run --features simd             # multiply–adds fused: the timing
//! target\release\dgcom-gpu.exe phase1         # one phase (phase0, phase1; all by default)
//! ```
//!
//! - **Phase 0:** the fully wet split-form RHS, node for node.
//! - **Phase 1:** dgcom's `Simulation` stepping the device state with the GPU
//!   physics module, against the same run on the CPU.
//!
//! Built without multiply–add contraction (`--no-fmad`), the GPU's results
//! must be the CPU's bit for bit; that is the gate.

mod phase0;
mod phase1;
mod swe2d;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = swe2d::Gpu::new()?;
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    if which == "all" || which == "phase0" {
        phase0::run(&gpu)?;
    }
    if which == "all" || which == "phase1" {
        phase1::run(&gpu)?;
    }
    if which == "beach" {
        phase1::run_beach(&gpu)?;
    }
    Ok(())
}
