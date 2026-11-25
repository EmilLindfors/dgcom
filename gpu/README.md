# dgcom-gpu

dgcom's GPU kernels, in Rust on NVIDIA's [cuda-oxide](https://github.com/NVIDIA/cuda-rust) (P2.7; the plan, the measurements and the toolchain are in [`../notes/gpu.md`](../notes/gpu.md)).

- `src/swe2d.rs`: the kernels, a device-resident SWE state (`DeviceState`, dgcom's `Integrable`) and the GPU physics module (`GpuSWE2D`, dgcom's `PhysicsModule`), built from dgcom's own module (`SWEPhysics2D::device_form`): the split forms `EntropyStable` and `WetDry`, walls, characteristic open boundaries with a uniform external state (a harmonic tide, still water), a tidal atlas's tides or a parent model's state node by node (nesting, with its relaxation band), Coriolis, the wet/dry correction, Manning friction, the positivity-bounded step. dgcom's own `Simulation` and SSP-RK steppers drive them, with global steps or local time stepping (`Multirate`, up to 8 levels: `GpuSWE2D` is dgcom's `LocalTimeStepping`).
- `src/case.rs`: the case runner's CUDA backend (`Cuda`, dgcom's `CaseDevice`, for `[compute] device`: a 2D run, and since Phase 4.5c a 3D one, dgcom's own `Simulation3D` over `Model3D` with the state on the host after every step) and the case gate: a case file on the CPU and on the GPU, every station record and a 3D run's snapshot compared.
- `src/gate.rs`: how a gate holds the device to the CPU (the rule of 2026-11-25, `notes/gpu.md`): bit for bit where its kernels still mirror the CPU's loops and the build does not contract multiply–adds (a probe kernel finds that), otherwise within its stated tolerance of each field's largest value, with a share of outliers where a threshold decides (a merged cell, a pooled element); a mutant is caught if the gate fails it by the same rule.
- `src/judges.rs`: the device's own judges (`judges`, also in `all`), which ask the scheme's properties of the device and nothing of the CPU: third order of both split forms on a steady vortex over a hill on distorted meshes (and the second-order subcells under that bar), a lake at rest with a shoreline, mass to round-off and positivity on the drying beach; in 3D rest about the reference on the wet/dry beach, uniform tracers under a sheared flow with rivers, and the tracers' inventories; each with a mutant it must catch.
- `src/phase0.rs`: the RHS against `compute_rhs_swe_2d_parallel_into`, node for node, on periodic P2 meshes (parallelograms and jittered quadrilaterals) and walled drying beaches, and timed.
- `src/phase1.rs`: `Simulation` with SSP-RK(4,3) on the GPU against the same run on the CPU (`SWEPhysics2D`), the final states node for node, and the time per step.
- `src/nest.rs`: a run nested in a made-up parent held in memory (no NetCDF), its boundary and relaxation band, on the CPU and the GPU with global steps and local time stepping, the final states node for node (`nest`, also in `all`).
- `src/three_d.rs`: the 3D stages' transport (Phase 4): the layer transports and Ω (dgcom's `LayerTransport`, with subcell elements and thin columns), the advection of u, v, T, S and the turbulence's k and ψ by them, the water no transport carried and the rivers' (`Transport3D`, `RiversDevice`); open faces with a nesting parent's values (`ExteriorDevice`) or the tracers' conditions (`DeviceTracerBC3D`); dgcom's own column kernel (`column_vertical_flux`).
- `src/baroclinic.rs`: the 3D stages' momentum terms (Phase 4.2): the reference stratification at a stage's levels from its profile and the linear equation of state, every element's displacement about it, the σ-pairs baroclinic pressure gradient with Coriolis (`Baroclinic3D`), and the reference's advection of T and S; dgcom's own column and pointwise functions (`reference_column`, `column_pressure`, `sigma_pair_difference`, `reference_edge`, …).
- `src/transport3d.rs`: those against dgcom's `LayerTransport::compute`, `ReferenceFields::fill`, `apply_momentum_transport_3d`, `apply_tracer_transport_3d`, `add_reference_advection`, `apply_w_point_transport_3d`, `add_volume_sources` and `RiverInflow`, value for value, on periodic and walled P2 meshes, walled beaches with flats and a channel open at both ends with a parent and rivers, under 20 σ-levels, with mutants it must catch, and timed at 96,000 elements (`transport3d`, also in `all`).
- `src/viscosity.rs`: the horizontal viscosity of the 3D shear (Phase 4.2b): BR1 per layer with a constant background and Smagorinsky's from the layer's strain, walls mirrored, open faces copied (`Viscosity3D`); dgcom's `mirrored_velocity`, `wall_stress`, `depth_average`.
- `src/momentum3d.rs`: the reference, the displacements, the pressure gradient with Coriolis and the shear's viscosity against dgcom's `ReferenceFields`, `compute_momentum_rhs_3d` and `apply_horizontal_viscosity_3d`, value for value, on the same meshes, about a reference (linear, cubic and unevenly spaced profiles) or ρ₀, with a balanced reference state, in both metric forms, with a constant and Smagorinsky's viscosity, with mutants, and timed at 96,000 elements (`momentum3d`, also in `all`).
- `src/hooks.rs`: what the mode splitter does to a 3D stage's inventories to make the next stage's values (Phase 4.3, `StageHooks3D`): inventories ↔ values with the step's merged cells (`Merged3D`), the thin columns' mask, `zero_thin_momentum`, the density, and the nesting band's relaxation (`Band3D`).
- `src/limiter.rs`: the 3D tracer limiter (Phase 4.3b, `Limiter3D`): the horizontal Kuzmin limiter about the reference stratification over the vertex patches (`VertexPatches`), then the bounds; dgcom's `HeightBracket`, `relaxed_bounds`, `kuzmin_layer_alpha`, `compute_theta`.
- `src/stage3d.rs`: those against dgcom's functions composed in `ModeSplitIntegrator::step`'s order, value for value: the step's start, a stage from another buffer's values (with the case runner's tracer limiter) and in place, the step's end, on walled beaches whose stages merge cells and flush films; the nesting band's relaxation on the open channel; with mutants, and timed at 96,000 elements (`stage3d`, also in `all`).
- `src/limiter3d.rs`: `Limiter3D` against `apply_tracer_limiters_3d` in three configurations on the beach, the open channel and the doubly periodic mesh, with mutants, and timed at 96,000 elements (`limiter3d`, also in `all`).
- `src/vertical.rs`: the 3D step's vertical terms (Phase 4.4a, `Vertical3D`): the bottom and cage drag rates, then a thread per column stepping the GLS turbulence and solving the implicit diffusion of u, v, T, S with the surface stress and fluxes; dgcom's `GlsMixing::step_column`, `diffuse_column`, `thin_column_fluxes`, on `dgcom::math`. Implicit vertical advection (Phase 4.4b, `Implicit3D`): a stage's Ω split into its explicit and implicit parts, and the columns with an implicit part relaxed; dgcom's `split_water_column`, `relax_column`.
- `src/forcing.rs`: the 3D step's slow forcing (Phase 4.4c1, `SlowForcing3D`): `G` from the layers' advection by their own transports, the depth mean's with its Coriolis, the stresses and the drags' shear parts.
- `src/forcing3d.rs`: `SlowForcing3D` against `Hydrostatic3D::slow_forcing_into` with the case runner's terms on the beach and the channel, with mutants, and timed at 96,000 elements (`forcing3d`, also in `all`).
- `src/pass.rs`: the 3D step's barotropic pass (Phase 4.4c2–c3, `BarotropicPass3D`), on global substeps and on local time steps (`LocalPass3D`: dgcom's `MultirateStepper` over the device's state): every stage the 2D module's RHS with its subcell fluxes and the bands' volume sources, the rivers, and one kernel for the stage with `G`, the drags, the sums of `DU_avg2` and of the filter's average; then the step's rates, the divergence of `DU_avg2` and the pass's balance. dgcom's `barotropic_stages`, `element_stage`, `barotropic_node_rates`, `pooled_element`.
- `src/pass3d.rs`: `BarotropicPass3D` against dgcom's `barotropic_pass` and `barotropic_pass_local` and what follows them on the beach with rivers (global SSP-RK3 and SSP-RK(4,3), the case runner's local time steps, three levels) and the nesting gate's channel with its band, then the step's merged cells from each side's pass (`Merged3D::find` against `MergedCells::find`), with mutants, and timed at 96,000 elements (`pass3d`, also in `all`).
- `src/bound3d.rs`: the 3D step's bound on the device (Phase 4.5e, `Bound3D`): dgcom's element rates (`element_column_rates`, `sloping_level_rate_of`) a thread per element, the sloping levels' frequency measured on the listed elements (`sloping_level_frequency_of`), Smagorinsky's largest viscosity from `Viscosity3D`; the host takes the smallest step.
- `src/model3d.rs`: the 3D model on the device (Phase 4.5b, `Model3D`): dgcom's `SplitModel` over the modules above, so that dgcom's own `ModeSplitIntegrator` steps a 3D state that stays on the device (`Work3D` holds the modules and the step's buffers); kernels for a stage's combination (`shu_osher`), the column sums of its momentum tendency, the filtered barotropic state at the step's end and the depth means.
- `src/step3d.rs`: whole steps of `ModeSplitIntegrator` with `Model3D` against `Hydrostatic3D` on the CPU, with the case runner's terms and integrator on the beach (from no turbulence, as a case starts), every field after every step; mutants that leave a term out of the device's physics or change its integrator; the step's bound (`Bound3D`) against `Hydrostatic3D::time_step_limits` on the start and the steps' states at two Courant numbers, with mutants; timed at 48,000 elements in the beach's shape (`step3d`, also in `all`; `DGCOM_STEP3D_ELEMENTS` sets the size: the model takes ≈ 136 kB an element, and `time.sh` asks for 96,000).
- `src/implicit3d.rs`: `Implicit3D` against `split_vertical_velocity` and `apply_implicit_vertical_advection` on beaches whose 40 s steps reach Courant numbers of 2.8–4.8 and on the channel, with mutants, and timed at 96,000 elements (`implicit3d`, also in `all`).
- `src/vertical3d.rs`: `dgcom::math` on the device against the CPU, then `Vertical3D` against `apply_vertical_diffusion` and the drag rates in three configurations (k-ε with every forcing the case runner takes; the generic model with breaking waves; k-ω alone) on the beach and the channel, with mutants, and timed at 96,000 elements (`vertical3d`, also in `all`).
- `time.sh`: the gates and the timing of a card in one go ([Timing a card](#timing-a-card)).

It is a workspace of its own, as `viz/`: it builds with cuda-oxide's pinned toolchain (`rust-toolchain.toml`, Rust 1.99.1), dgcom included (without its NetCDF feature unless built with `--features netcdf`, so no HDF5 or netCDF-C by default). CI does not build it (no GPU or CUDA toolkit on the runners).

## Setting up a machine

### Ubuntu (the GPU server)

```bash
# 1. An R580+ driver (nvidia-smi's "CUDA Version" 13.x) and the CUDA 13 toolkit with
#    cuRAND's headers, from NVIDIA's apt repository (developer.nvidia.com/cuda-downloads:
#    Linux, x86_64, Ubuntu, deb (network)):
sudo apt install cuda-toolkit-13-1       # or the driver's own 13.x: 13-0 on an R580 driver
export PATH=/usr/local/cuda/bin:$PATH        # nvcc, and nsys if the toolkit brought it
# If /usr/local/cuda stays on another version (12.x for other software), point at 13 instead,
# or cargo-oxide mixes 12.x's libNVVM and nvJitLink with 13's headers (and so does time.sh):
#   export CUDA_TOOLKIT_PATH=/usr/local/cuda-13 PATH=/usr/local/cuda-13/bin:$PATH

# 2. Build tools and libclang for bindgen (cuda-oxide's host bindings). clang-21 comes
#    from apt.llvm.org (`wget https://apt.llvm.org/llvm.sh && sudo bash llvm.sh 21`)
#    where the release has no clang-21.
sudo apt install build-essential git pkg-config clang-21 libclang-common-21-dev

# 3. Rust, and cuda-oxide's pinned toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install 1.99.1 --component rust-src,rustc-dev,llvm-tools,clippy,rustfmt

# 4. cargo-oxide and its codegen backend, at the revision gpu/Cargo.toml pins
git clone https://github.com/NVIDIA/cuda-rust.git ~/cuda-rust
git -C ~/cuda-rust checkout 348895e421a92d13a2c3a843ccb20ed1fb8b7f8b
cd ~/cuda-rust/cuda-oxide
cargo +1.99.1 install --locked --path crates/cargo-oxide
cargo oxide setup       # builds the backend into ~/.cargo/cuda-oxide/
cargo oxide doctor      # must pass: Rust, CUDA, LLVM, the backend

# 5. dgcom
git clone https://github.com/EmilLindfors/dgcom.git ~/dgcom

# 6. For cases that read NetCDF (a nested boundary's parent, weather, NetCDF output):
#    netCDF-C and HDF5 (1.x), then build with `--features simd,netcdf`
sudo apt install libnetcdf-dev libhdf5-dev
```

When gpu/Cargo.toml moves to another cuda-oxide revision, check that revision out in `~/cuda-rust` and run steps 4's `install` and `setup` again: the backend must be built from the revision the crates are.

### Windows (the laptop)

CUDA 13.0+, Visual Studio 2022, LLVM for libclang (`C:\Program Files\LLVM`), Rust 1.99.1 as above; `cargo-oxide` installed and set up from a checkout of cuda-rust at the pinned revision as in step 4 (the Windows guide's `libffi` from vcpkg is not needed). Per PowerShell:

```powershell
$env:CUDA_TOOLKIT_PATH = $env:CUDA_PATH
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
$env:PATH = "$env:CUDA_PATH\bin\x64;$env:CUDA_PATH\bin;$env:LIBCLANG_PATH;$env:PATH"
```

## Running

In `gpu/` (on Windows the binary is `target\release\dgcom-gpu.exe`):

```bash
cargo oxide run --no-fmad                   # the gates against the scalar CPU kernel
cargo oxide run --no-fmad --features simd   # ... and against the 8-wide SIMD batch
cargo oxide run --features simd             # multiply–adds fused: within ~1e-11; the timing
target/release/dgcom-gpu phase1             # one gate (phase0, phase1, nest, weather, beach,
                                            # transport3d, momentum3d, stage3d, limiter3d,
                                            # vertical3d, implicit3d, forcing3d, pass3d, step3d,
                                            # parent3d, judges; all by default)
target/release/dgcom-gpu judges             # the device's own judges alone (a minute)
target/release/dgcom-gpu case ../cases/tidal_channel/case.toml [--tolerance 1e-8] [--outliers 0]
                                            # the case on the CPU and the GPU, compared (the bar
                                            # where it is not judged bit for bit)
target/release/dgcom-gpu run ../cases/tidal_channel/case.toml --set compute.device=cuda
                                            # dgcom's command line, the 2D run on the GPU
```

The CPU has two copies of the physics: the scalar kernel, whose pointwise functions the GPU kernels call, and the SIMD batch that real runs use (`simd`), held to the scalar one bit for bit by its own gates. The gate runs against either.

`--no-fmad` keeps each multiply and add separate, as the CPU code does: in that build a gate whose kernels mirror the CPU's loops must agree with it bit for bit. The default build (the one that runs) is faster and agrees within rounding: its gates judge within their tolerances (`src/gate.rs`; `DGCOM_GATE=tolerance` asks the same of a `--no-fmad` build). Either way the command fails unless every fixture holds, every mutant is caught and the device's own judges hold; its last line counts them. A mutant whose effect is round-off's size is caught bit for bit only, and the default build says so.

## Timing a card

```bash
gpu/time.sh                                  # about 15 min on the laptop
SIZES="800x80 1600x160 3200x320" REPEATS=5 gpu/time.sh
```

1. Refuses a GPU that is more than 10 % busy (the cards are shared; `FORCE=1` overrides).
2. Builds with `--no-fmad` and runs the gates against the SIMD batch: `all` (every gate's fixtures and mutants, and the device's own judges), `beach`, and `case` on `cases/tidal_channel` (25 h, every station record), global steps and local time stepping at 3 levels, `case_atlas` on `cases/tidal_channel/atlas.toml` (the same channel forced by a tidal atlas node by node) and `case_3d` (the 3D channel, its snapshot too). On any card each must hold: bit for bit where the kernels mirror the CPU, within its tolerance where they have left its order (`SKIP_GATES=1` skips them).
3. Builds with FMA, runs the same gates with the build that runs (each within its tolerance), then `dgcom-gpu case` on the tidal channel refined to each of `SIZES` (`<nx>x<ny>` elements: 64,000 and 256,000 by default, about 1,150 steps each), on the CPU (the SIMD batch on all threads) and on the GPU, `REPEATS` times (3); then `dgcom-gpu transport3d`, `momentum3d`, `stage3d`, `limiter3d`, `vertical3d`, `implicit3d`, `forcing3d`, `pass3d` and `step3d`, the 3D transport, momentum terms, stage hooks, vertical step, implicit vertical advection, slow forcing (per element and level at 96,000 P2 elements × 20 levels), barotropic pass with the merged cells' search (per pass at 96,000 elements) and the whole 3D step (per step).
4. Profiles the first size with Nsight Systems, if `nsys` is found (`SKIP_PROFILE=1` skips it): each kernel's time per step.

Everything goes to `output/gpu_timing/<gpu>_<date>/` (git-ignored): each command's log and `summary.md`. The time per step is the run's wall time over its steps, start-up included; a build's first GPU run also compiles the PTX for the card, so judge by the later runs.

**What to record** (`notes/gpu.md`, a section "Phase 1 on the <card> (<date>)"; the P2.7 line of `TODO.md`):

- Whether the gates held on the card, in both builds (the two tables). A gate that fails is the finding, before any timing: the same PTX on another architecture should give the same bits under `--no-fmad`, and the device's own judges (convergence, rest, conservation, positivity) must hold on any card.
- The time per step against the predictions (`notes/gpu.md`, Hardware and "Phase 1: a case file on the GPU"): from the laptop's 14.5–15.9 ms at 64,000 elements, by FP64 rate, ≈ 6 ms on the RTX 3090 and ≈ 1.7 ms on the RTX PRO 6000; the CPU it ran against (the laptop's 24 threads take 15.3–16.9 ms).
- The kernel shares against the laptop's (`edge_fluxes_hll` 28 %, `element_rhs_volume` and `element_rhs_subcells` 22 % each, `implicit_damping` 14 %): if they hold, the card is FP64-bound as the laptop is.

**FP64 counts** (Nsight Compute, not yet tried; it needs profiling rights: root, or the driver option `NVreg_RestrictProfilingToAdminUsers=0`):

```bash
sudo $(command -v ncu) --kernel-name regex:edge_fluxes_hll --launch-skip 100 --launch-count 3 \
    --metrics smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    gpu/target/release/dgcom-gpu run cases/tidal_channel/case.toml --set compute.device=cuda \
    --set mesh.rectangle.nx=800 --set mesh.rectangle.ny=80 --set run.duration_hours=0.01
```
