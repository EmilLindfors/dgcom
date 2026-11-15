# dgcom-gpu

dgcom's GPU kernels, in Rust on NVIDIA's [cuda-oxide](https://github.com/NVIDIA/cuda-rust) (P2.7; the plan, the measurements and the toolchain are in [`../notes/gpu.md`](../notes/gpu.md)).

Phase 0 is a test bed: the fully wet split-form 2D RHS (`SWEFormulation2D::EntropyStable`) on the GPU, calling dgcom's own pointwise physics, checked against `compute_rhs_swe_2d_parallel_into` on periodic P2 meshes (parallelograms and jittered quadrilaterals) and timed against it.

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
```

The CPU has two copies of the physics: the scalar kernel, whose pointwise functions the GPU kernels call, and the SIMD batch that real runs use (`simd`), held to the scalar one bit for bit by its own gates. The gate runs against either.

`--no-fmad` keeps each multiply and add separate, as the CPU code does; that build must agree with the CPU bit for bit. The default build is faster and agrees within rounding.
