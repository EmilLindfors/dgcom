# dgcom-gpu

dgcom's GPU kernels, in Rust on NVIDIA's [cuda-oxide](https://github.com/NVIDIA/cuda-rust) (P2.7; the plan, the measurements and the toolchain are in [`../notes/gpu.md`](../notes/gpu.md)).

- `src/swe2d.rs`: the kernels, a device-resident SWE state (`DeviceState`, dgcom's `Integrable`) and the GPU physics module (`GpuSWE2D`, dgcom's `PhysicsModule`), built from dgcom's own module (`SWEPhysics2D::device_form`): the split forms `EntropyStable` and `WetDry`, walls, characteristic open boundaries with a uniform external state (a harmonic tide, still water), Coriolis, the wet/dry correction, Manning friction, the positivity-bounded step. dgcom's own `Simulation` and SSP-RK steppers drive them.
- `src/case.rs`: the case runner's CUDA backend (`Cuda`, dgcom's `CaseDevice`, for `[compute] device`) and the case gate: a case file on the CPU and on the GPU, every station record compared.
- `src/phase0.rs`: the RHS against `compute_rhs_swe_2d_parallel_into`, node for node, on periodic P2 meshes (parallelograms and jittered quadrilaterals) and walled drying beaches, and timed.
- `src/phase1.rs`: `Simulation` with SSP-RK(4,3) on the GPU against the same run on the CPU (`SWEPhysics2D`), the final states node for node, and the time per step.

It is a workspace of its own, as `viz/`: it builds with cuda-oxide's pinned toolchain (`rust-toolchain.toml`), and the crate above with stable. CI does not build it (no GPU or CUDA toolkit on the runners).

## Running

Windows, PowerShell (CUDA 13.0+, Visual Studio 2022, LLVM for libclang; `cargo-oxide` and its backend installed as in `notes/gpu.md`):

```powershell
$env:CUDA_TOOLKIT_PATH = $env:CUDA_PATH
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
$env:PATH = "$env:CUDA_PATH\bin\x64;$env:CUDA_PATH\bin;$env:LIBCLANG_PATH;$env:PATH"
cargo oxide run --no-fmad                   # the gate against the scalar CPU kernel
cargo oxide run --no-fmad --features simd   # ... and against the 8-wide SIMD batch
cargo oxide run --features simd             # multiply–adds fused: within ~1e-11; the timing
target\release\dgcom-gpu.exe phase1         # one gate (phase0, phase1, beach; all by default)
target\release\dgcom-gpu.exe case ..\cases\tidal_channel\case.toml
                                            # the case on the CPU and the GPU, compared
target\release\dgcom-gpu.exe run ..\cases\tidal_channel\case.toml --set compute.device=cuda
                                            # dgcom's command line, the 2D run on the GPU
```

The CPU has two copies of the physics: the scalar kernel, whose pointwise functions the GPU kernels call, and the SIMD batch that real runs use (`simd`), held to the scalar one bit for bit by its own gates. The gate runs against either.

`--no-fmad` keeps each multiply and add separate, as the CPU code does; that build must agree with the CPU bit for bit. The default build is faster and agrees within rounding.
