# GPU port (P2.7)

The GPU port: the hardware it runs on, the toolchain, the kernels' design, the phases and their gates. The history before 2026-11-15 (the Burn prototype, the sizing) is in `performance.md` P2.7 and `correctness-bugs.md` P0.11.

## Decisions (2026-11-15)

- **CUDA first**, on the cards we have: an RTX 3090 (local) and an RTX PRO 6000 Blackwell (work). The production card is open: a CUDA card with full-rate FP64 (H100/H200/GH200, B200) or AMD's MI300X. Neither the B300 (≈ 1.3 TF FP64) nor the RTX PRO 6000 (≈ 1.9 TF) is a production FP64 card.
- **Kernels in Rust, with NVIDIA's cuda-oxide** (the SIMT track of "CUDA Rust", announced 2026-09-08; we use 0.4, git `348895e`, 2026-11-13). It compiles ordinary Rust to PTX: structs, enums, generics, closures and device functions from other crates, core-only (no `std`, no heap), f64 with libdevice's math (`sqrt`, `cbrt`, …), shared memory, f64 atomics and shuffles, launch contracts. So the GPU kernels call **the same pointwise physics as the CPU's scalar kernel** (fluxes, reconstruction, friction, sources), checked bit for bit (spike below). The GPU adds no copy of the physics, but it does not remove the one there is: real runs (`simd`, on by default) spend their time in the 8-wide batch (`swe_2d_split_form_batch.rs`), which has its own `hydrostatic_hll`, `limited_slope`, `reconstruct_subcell` and two-point fluxes on `f64x8`, held to the scalar kernel bit for bit by its own gates (`test_batched_kernels_are_bit_for_bit`). A physics change still goes into both CPU copies; see "One source for the physics" below.
- Not Burn (tensor ops; f64 fusion off, ≈ 13× the bytes per node-stage; P0.11), not cutile-rs (tile model: our kernels gather across faces and branch on wet/dry; f64 not documented), not CubeCL (its `#[cube]` DSL cannot call ordinary Rust, so the physics would exist twice; API breaks each release), not CUDA C++ through NVRTC (a second language).
- **Risk:** cuda-oxide is alpha (0.1 in May 2026, 0.4 now; API breaks expected), and its toolchain pin moves with its releases. Pin its git revision; keep the shared physics plain Rust with no cuda-oxide dependency (it ports to any later backend, MI300X's included); gate every kernel bit for bit against the CPU.

## Hardware

| | FP64 peak | Bandwidth | Memory | FP64 flop/byte |
|---|---|---|---|---|
| RTX 4060 Laptop (this laptop, sm_89) | ≈ 0.24 TF (1:64) | 0.26 TB/s | 8 GB | 0.9 |
| RTX 3090 (sm_86) | 0.56 TF (1:64) | 0.94 TB/s | 24 GB | 0.6 |
| RTX PRO 6000 Blackwell (sm_120) | ≈ 1.9 TF (1:64) | 1.8 TB/s | 96 GB | 1.1 |
| H100/H200, GH200 (Sigma2's Olivia) | 34 TF | 3.4–4.8 TB/s | 80–141 GB | 7–10 |
| B200 | ≈ 37–40 TF | 8 TB/s | 180 GB | 5 |
| B300 | ≈ 1.3 TF | 8 TB/s | 288 GB | 0.2 |
| MI300X | 81.7 TF | 5.3 TB/s | 192 GB (+ 256 MB cache) | 15 |
| HX 370 laptop CPU, for scale | ≈ 0.6 TF | 0.12 TB/s | | |

- The 2D stage needs an estimated 300–500 FP64 operations and ≈ 126 B per node and stage (`review-2026-09-25.md` §5.2): ≈ 3 flop/byte. **On the 1:64 cards it is FP64-bound**, on the data-centre cards bandwidth-bound. On the 1:64 cards load what is static (geometry, 1/J, normals, friction factors) rather than recompute it, compute 1/h once per node, and spare divisions, square and cube roots (multi-instruction sequences on the scarce FP64 units).
- The 3090's FP64 peak is about the laptop CPU's. What the GPU wins is running near its peak, where the CPU code runs far below its own (gathers, branches).
- **Predictions** (Phase 0 checks them; the FLOP count of step 1 sharpens them): a fully wet P2 element-stage takes 10–30 ns on the 3090 and 3–9 ns on the PRO 6000, against 71 ns on the laptop's 24 threads (Chiloé 2D, whole stage). A Chiloé model hour: 102.5 s on the laptop, 10–30 s on the 3090, 4–10 s on the PRO 6000.
- Small meshes do not fill a GPU (Kattholmen's 3,584 elements, `performance.md` P2.7 sizing): the GPU is for meshes of 10⁴–10⁵ elements and up, and for several runs at once.

## Toolchain (2026-11-15, this laptop)

- Windows 11, RTX 4060 Laptop GPU (correctness only), driver 616.92, CUDA 13.1, Visual Studio 2022, LLVM 22 in `C:\Program Files\LLVM` (libclang).
- Rust 1.99.1 with `rust-src`, `rustc-dev`, `llvm-tools` (cuda-oxide's pin); `cargo-oxide` 0.4.0 built from `C:\Users\emil\dev\cuda-rust` (`cargo +1.99.1 install --locked --path crates/cargo-oxide`); the codegen backend (`cargo oxide setup`) cached in `~/.cargo/cuda-oxide/`. `cargo oxide doctor` passes. The Windows guide's `libffi` from vcpkg was not needed.
- Per shell (PowerShell): `$env:CUDA_TOOLKIT_PATH = $env:CUDA_PATH; $env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"; $env:PATH = "$env:CUDA_PATH\bin\x64;$env:CUDA_PATH\bin;$env:LIBCLANG_PATH;$env:PATH"`.
- **Two toolchains.** The crate builds on stable (1.100 since 2026-11-09; CI's `dtolnay/rust-toolchain@stable`); device code builds on cuda-oxide's pin (1.99.1). Code both compile, the shared physics, must build on the pin: no `rust-version` above it there, and a CI job checks it with the pinned toolchain. Prefer cuda-oxide's device-crate layout (`[[package.metadata.cuda-oxide.device-crates]]`: the kernels' crate builds to PTX/cubin, the host builds normally), so `cargo build` of the CPU code never needs the CUDA toolchain.
- Flags that matter: `--no-fmad` (no multiply–add contraction: the bit-for-bit gates), `--lineinfo` (Nsight source lines), `cargo oxide sanitize` (compute-sanitizer; on WDDM run `EnableDebuggerInterface.bat` once as administrator), `--materialize-cubin --arch sm_XX` (no JIT at start-up).

## Spike: dgcom's own flux on the GPU (2026-11-15)

A cuda-oxide kernel calling `dgcom::flux::wintermeyer_flux_2d` and `entropy_stable_dissipation_2d` unchanged (dgcom as a path dependency, `--no-default-features --features parallel`, built by the pinned toolchain in about a minute), one thread per face node, 2²⁰ random node pairs: dry, below `h_min`, films and 0–200 m deep, ±2 m/s, beds to −50 m.

- Default build (FMA contraction): 30 % bit-identical to the CPU, the rest within 1.1e-12 of max(|F|, 1).
- `--no-fmad`: **1,048,576 of 1,048,576 bit-identical.**
- So the GPU's gates can be bit for bit against the CPU (a `--no-fmad` build); production builds keep FMA and tolerance gates.

## Architecture

- **One source for the physics (to decide in Phase 1).** Today the physics has two CPU copies, scalar and batch, and the GPU shares the scalar one. To make it one source, write the pointwise functions generic over the lane type: a small numeric trait (arithmetic, `sqrt`, compare-and-select, `max`/`min`) with an `f64` implementation in the shared crate and an `f64x8` one in dgcom behind `simd`. The scalar kernel and the GPU then instantiate them with `f64`, the batch with `f64x8`. Branches become selects, as the batch already writes them, and on a GPU that also avoids divergent warps. The same bits throughout: the batch's gates and the GPU's hold the three to the scalar order. What stays per target is what an element reads: the batch derives its geometry from the corners (64 B against ≈ 1 kB at P2, P2.8), the GPU on the 1:64 cards loads it. Measure the batch before and after (the alternating Chiloé half hours of the 2D kernel bench): a generic `f64` path must not slow the scalar kernel either.
- **Crates.** The pointwise physics (node states, interface fluxes, the subcells' reconstruction, the wall ram pressure, friction rates, Coriolis and the other sources) moves to a small crate of `#[inline]` functions on `Copy` structs that both the CPU kernels and the GPU kernels call. The kernels' crate (cuda-oxide) and the host driver (a `cuda` feature, replacing `burn*`/`gpu`) build on it.
- **Host side unchanged.** `Simulation<S, P, I>` is generic over the state: a device-resident `SWESolution2D` (SoA as on the CPU, `[var][k·n_nodes + i]`) implements `Integrable` (scale, axpy, combine, is_finite as kernels), a `GpuSWEPhysics2D` built from the same `SWEPhysics2D` implements `PhysicsModule` and `LocalTimeStepping`, and the runner, `SSPRK43` and `Multirate` drive it from the host. A case picks it with `[compute] device = "cuda:0"`.
- **A stage** is a face kernel (each face once, as `SubsetEdges::owns` on the CPU, into a face buffer) and an element kernel (volume term: flux differencing, or the subcells for an element with a dry node, sorted so a warp does not mix them; lifting the face buffer; sources; the Shu–Osher combination; implicit friction; wet/dry correction; positivity limiter). No atomics: deterministic and conservative to round-off, as the CPU. Alternative, one kernel per pass: each element computes its own faces' fluxes with the lower-numbered element as the minus side, so both sides compute the same bits; worth it where launches dominate (the data-centre cards).
- **Local time stepping.** Levels, element lists and rates stay on the host and go up once per coarse step (`element_dt` comes down). One trait change: `LocalTimeStepping::stage_where` takes `plan: &dyn Fn(usize) -> ElementStage`, which `Multirate::schedule` builds as a formula of the element's level and rate and the pass's substep, stage, time and step. Pass that as data; the CPU evaluates the same formula. A Chiloé coarse step is ≈ 1,280 passes (2⁸ finest substeps × 5): ≈ 2,500 launches. Measure the overhead; CUDA graphs per coarse step if it passes 10 %; fewer levels may pay on a GPU than on the CPU.
- **Forcing.** Boundary conditions, external-state providers and sources are trait objects running arbitrary Rust per node; the device needs a closed set. A `device_form()` returning plain data (a run with a term without one refuses the GPU and names it). The case runner's set: walls; `CharacteristicOBC` over `StillWater`, `HarmonicTide` and `BoundaryTides` (harmonic constants per boundary node, evaluated on the device), `OceanModelState` (parent snapshots time-interpolated on the device, uploaded when the parent advances), `InverseBarometer`; Coriolis (f- and β-plane); Manning/Chézy and `CageDrag2D` (implicit); `GriddedAtmosphere2D` (interpolation weights precomputed, two snapshots); `RiverSources`; the nesting relaxation; `HorizontalViscosity2D` (BR1, the largest kernel, Kattholmen's only: last).
- **Host–device traffic** only at outputs (stations, snapshots), when forcing snapshots change, and once per coarse step.

## Phases and gates

- **Phase 0, measure and one kernel (≈ 1–2 weeks).** (1) FP64 operations and bytes per node and stage of today's 2D stage, from the CPU (uProf's FLOP counters, one thread), by part; sharpen the predictions. (2) Delete Burn: `src/solver/burn/`, `burn_ssp_rk3.rs`, `profile_gpu`, `MeshGPUData`, the `burn*` features (P6 has it). (3) The fully wet RHS (flux differencing, faces) in a test crate (`gpu/`) whose kernels call dgcom's pointwise physics directly; the shared physics crate and the host driver wait for Phase 1, once the kernels show which functions they share. Gate: bit for bit against the CPU under `--no-fmad`. (4) Time it on the 3090 (and the PRO 6000). Go: ≤ 30 ns per P2 element-stage on the 3090 or ≤ 10 on the PRO 6000; otherwise rethink the layout (a thread per node or per element, shared memory) before building on it.
- **Phase 1, the 2D stage, global steps.** The shared physics crate (the per-line volume term and the per-face surface term as functions over arrays, which `SplitFormSWE2D` and the kernels both call: the kernels mirror its loops today), the device state (`Integrable`) and the host driver behind a `cuda` feature. Subcells, hydrostatic HLL with the ram pressure, positivity limiter and wet/dry correction, implicit friction, Coriolis, walls and a harmonic-tide `CharacteristicOBC`; `compute_dt` and `is_finite` as reductions; readback for stations and snapshots. Gates: each kernel bit for bit (`--no-fmad`); the convergence, conservation and lake-at-rest gates on the device backend (a `gpu` nextest profile run locally: GitHub's runners have no GPU); `cases/tidal_channel` stations matching the CPU run.
- **Phase 2, local time stepping and the real cases.** `Multirate` on the device (the trait change), the launch overhead, the forcing's device forms, `[compute] device`. Judge: Chiloé and the Mausund/Bodø nests through `dgcom score` match the CPU runs to the printed digits; time per model hour against the predictions on the 3090 and the PRO 6000.
- **Phase 3, the production card.** A full-FP64 CUDA card is Phase 2 on new hardware (bandwidth-bound there: one kernel per pass, graphs, fewer levels). MI300X would need a second kernel backend over the same shared physics.
- **Phase 4, 3D**, after P4.5 settles the 3D defaults. The 3D stages are 64 % of a Chiloé 3D step and the barotropic pass 13 % (`performance.md` P2.9; Phase 2 covers the pass); the vertical implicit solves are one thread per column. Particles (F.2) ride along once velocities live on the device.

## Phase 0 results (2026-11-15)

- **Burn deleted** (P0.11 closed): 2,233 lines, the `burn*`/`gpu` features and their dependency tree (1,983 lines of `Cargo.lock`), `MeshGPUData`.
- **`gpu/`**, a workspace of its own on cuda-oxide's toolchain (as `viz/`): `edge_fluxes` (a thread per interior edge, both sides' F* into a face buffer, as `SplitFormSWE2D::edge_fluxes`) and `element_rhs` (a thread per element: the flux-differencing volume term with the bed, the lifted face terms, 1/J, as `element_rhs` without subcells), P2, on doubly periodic meshes of 100 m quadrilaterals (parallelograms, and interior vertices jittered by 0.15 of the size: no element affine), a smooth bed 12–28 m deep and a smooth flow. The kernels call `SWENodeState2D::new`, `wintermeyer_flux_2d`, `entropy_stable_dissipation_2d`, `wintermeyer_bed_interface_term_2d` and `SWEState2D`'s arithmetic from dgcom unchanged, and mirror the loops around them.
- **The gate holds:** under `--no-fmad` every node of the RHS is the CPU's (`compute_rhs_swe_2d_parallel_into`) bit for bit, against both CPU copies of the physics: the scalar kernel (`gpu/` without its `simd` feature) and the 8-wide batch that real runs use (`--features simd`). 17,280 nodes on 1,920 elements and 864,000 on 96,000, parallelograms and jittered. With FMA contraction: within 1.3e-12 (small meshes) and 1.2e-11 (large) of each field's largest value.
- **Time per element evaluation of the RHS**, 96,000 P2 elements, RTX 4060 Laptop GPU (FP64 at 1:64, ≈ 0.2 TF) against the HX 370's 24 threads (20 CPU, 100 GPU evaluations; the CPU varies ±5 % between runs):

  | | CPU | GPU, `--no-fmad` | GPU, FMA |
  |---|---|---|---|
  | jittered | 49–54 ns | 31.5 ns | 24.9–25.7 ns |
  | parallelograms | 45–50 ns | 22.7 ns | 17.7 ns |

  The laptop's own GPU is 2–2.8× its 24 threads on this term. (The CPU column is the batch, `--features simd`; the scalar kernel takes 73 ns jittered and 54 ns on parallelograms: the batch's geometry from the corners pays on curved elements.)
- **Registers, not local memory.** The first version kept the element's nodes and RHS in local memory (130 loads and 173 stores in the PTX): its loops did not unroll, and the face nodes came from a table. `#[unroll]` on every loop (on `#[device]` helpers, with constant trip counts: `(a + 1)..N1` as `0..N1` with `continue`), the face nodes as a formula (`face_node`, checked against `ops.face_nodes` on the host) and the face buffer read per access instead of copied: no local memory left, 35.8 → 31.5 ns (jittered, `--no-fmad`), the same bits.
- **FP64-bound, by a rough count:** ≈ 3,100 FP64 instructions per element evaluation in the PTX (both kernels, the non-affine path, divisions expanded) at 31.5 ns is ≈ 100 G per second, about the 4060 Laptop's FP64 peak. If so the time scales with the card's FP64 rate: ≈ 10 ns (jittered) and 7 (parallelograms) on the 3090, ≈ 3 and 2 on the PRO 6000, with FMA; inside the predictions (10–30 and 3–9 for the whole stage). Nsight Compute counts the instructions exactly once the GPU's performance counters are open to the user (NVIDIA Control Panel → Developer → Manage GPU Performance Counters → all users; or an administrator prompt): `ERR_NVGPUCTRPERM` without.
- **Left of Phase 0:** the timing on the 3090 and the PRO 6000 (`cargo oxide run` in `gpu/`), the Nsight counts, and the stage combination (with Phase 1's device state).

## Out of TODO.md

### Parked
- **P2.7** Emulated double precision (double-float on the FP32 units) could beat native FP64 a few times on the 1:64 cards, at f32's exponent range: only if the PRO 6000's measured numbers disappoint.
- **P2.7** Several runs in one set of launches (scenarios, ensembles, the 15-day runs side by side) for meshes too small to fill a GPU.
- **P2.7** MI300X: a HIP or CubeCL backend over the shared physics crate, if production moves to AMD.
