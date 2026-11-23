# dgcom-gpu

dgcom's GPU kernels, in Rust on NVIDIA's [cuda-oxide](https://github.com/NVIDIA/cuda-rust) (P2.7; the plan, the measurements and the toolchain are in [`../notes/gpu.md`](../notes/gpu.md)).

- `src/swe2d.rs`: the kernels, a device-resident SWE state (`DeviceState`, dgcom's `Integrable`) and the GPU physics module (`GpuSWE2D`, dgcom's `PhysicsModule`), built from dgcom's own module (`SWEPhysics2D::device_form`): the split forms `EntropyStable` and `WetDry`, walls, characteristic open boundaries with a uniform external state (a harmonic tide, still water), a tidal atlas's tides or a parent model's state node by node (nesting, with its relaxation band), Coriolis, the wet/dry correction, Manning friction, the positivity-bounded step. dgcom's own `Simulation` and SSP-RK steppers drive them, with global steps or local time stepping (`Multirate`, up to 8 levels: `GpuSWE2D` is dgcom's `LocalTimeStepping`).
- `src/case.rs`: the case runner's CUDA backend (`Cuda`, dgcom's `CaseDevice`, for `[compute] device`) and the case gate: a case file on the CPU and on the GPU, every station record compared.
- `src/phase0.rs`: the RHS against `compute_rhs_swe_2d_parallel_into`, node for node, on periodic P2 meshes (parallelograms and jittered quadrilaterals) and walled drying beaches, and timed.
- `src/phase1.rs`: `Simulation` with SSP-RK(4,3) on the GPU against the same run on the CPU (`SWEPhysics2D`), the final states node for node, and the time per step.
- `src/nest.rs`: a run nested in a made-up parent held in memory (no NetCDF), its boundary and relaxation band, on the CPU and the GPU with global steps and local time stepping, the final states node for node (`nest`, also in `all`).
- `src/three_d.rs`: the 3D stages' transport (Phase 4): the layer transports and Ω (dgcom's `LayerTransport`, fully wet), the advection of u, v, T and S by them, and the water no transport carried (`Transport3D`), with dgcom's own column kernel (`column_vertical_flux`).
- `src/transport3d.rs`: those against dgcom's `LayerTransport::compute`, `apply_momentum_transport_3d`, `apply_tracer_transport_3d` and `add_volume_sources`, value for value, on periodic and walled P2 meshes under 20 σ-levels, and timed at 96,000 elements (`transport3d`, also in `all`).
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
target/release/dgcom-gpu phase1             # one gate (phase0, phase1, nest, beach, transport3d;
                                            # all by default)
target/release/dgcom-gpu case ../cases/tidal_channel/case.toml
                                            # the case on the CPU and the GPU, compared
target/release/dgcom-gpu run ../cases/tidal_channel/case.toml --set compute.device=cuda
                                            # dgcom's command line, the 2D run on the GPU
```

The CPU has two copies of the physics: the scalar kernel, whose pointwise functions the GPU kernels call, and the SIMD batch that real runs use (`simd`), held to the scalar one bit for bit by its own gates. The gate runs against either.

`--no-fmad` keeps each multiply and add separate, as the CPU code does; that build must agree with the CPU bit for bit. The default build is faster and agrees within rounding, so its gates print "differs" by about 1e-13: expected.

## Timing a card

```bash
gpu/time.sh                                  # about 15 min on the laptop
SIZES="800x80 1600x160 3200x320" REPEATS=5 gpu/time.sh
```

1. Refuses a GPU that is more than 10 % busy (the cards are shared; `FORCE=1` overrides).
2. Builds with `--no-fmad` and runs the gates against the SIMD batch: `all` (the RHS and `Simulation` runs, the nested runs and the 3D transport), `beach`, and `case` on `cases/tidal_channel` (25 h, every station record), global steps and local time stepping at 3 levels, and `case_atlas` on `cases/tidal_channel/atlas.toml` (the same channel forced by a tidal atlas node by node). On any card each must be bit for bit (`SKIP_GATES=1` skips them).
3. Builds with FMA and runs `dgcom-gpu case` on the tidal channel refined to each of `SIZES` (`<nx>x<ny>` elements: 64,000 and 256,000 by default, about 1,150 steps each), on the CPU (the SIMD batch on all threads) and on the GPU, `REPEATS` times (3); then `dgcom-gpu transport3d`, the 3D transport per element and level at 96,000 P2 elements × 20 levels.
4. Profiles the first size with Nsight Systems, if `nsys` is found (`SKIP_PROFILE=1` skips it): each kernel's time per step.

Everything goes to `output/gpu_timing/<gpu>_<date>/` (git-ignored): each command's log and `summary.md`. The time per step is the run's wall time over its steps, start-up included; a build's first GPU run also compiles the PTX for the card, so judge by the later runs.

**What to record** (`notes/gpu.md`, a section "Phase 1 on the <card> (<date>)"; the P2.7 line of `TODO.md`):

- Whether the gates held on the card. A gate that differs is the finding, before any timing: the same PTX on another architecture should give the same bits under `--no-fmad`.
- The time per step against the predictions (`notes/gpu.md`, Hardware and "Phase 1: a case file on the GPU"): from the laptop's 14.5–15.9 ms at 64,000 elements, by FP64 rate, ≈ 6 ms on the RTX 3090 and ≈ 1.7 ms on the RTX PRO 6000; the CPU it ran against (the laptop's 24 threads take 15.3–16.9 ms).
- The kernel shares against the laptop's (`edge_fluxes_hll` 28 %, `element_rhs_volume` and `element_rhs_subcells` 22 % each, `implicit_damping` 14 %): if they hold, the card is FP64-bound as the laptop is.

**FP64 counts** (Nsight Compute, not yet tried; it needs profiling rights: root, or the driver option `NVreg_RestrictProfilingToAdminUsers=0`):

```bash
sudo $(command -v ncu) --kernel-name regex:edge_fluxes_hll --launch-skip 100 --launch-count 3 \
    --metrics smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    gpu/target/release/dgcom-gpu run cases/tidal_channel/case.toml --set compute.device=cuda \
    --set mesh.rectangle.nx=800 --set mesh.rectangle.ny=80 --set run.duration_hours=0.01
```
