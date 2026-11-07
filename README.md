# DGCOM

**DGCOM, the Discontinuous Galerkin Coastal Ocean Model**, is a highly experimental coastal ocean model written in Rust. It uses nodal DG on unstructured quadrilateral meshes for the 2D shallow-water equations, and a ROMS-style 3D layer: sigma coordinates, mode splitting and GLS mixing. It was developed for currents along the Norwegian coast, nested in NorKyst-800. Running any coastal domain from a case file is the next step (`TODO.md` P8).

DGCOM is not a port of ROMS. It follows ROMS's recipe for the 3D physics and is validated against ROMS-based NorKyst-800.

## Windows HDF5/netCDF-C Setup

The default feature set enables NetCDF I/O, so Windows builds need native HDF5 and netCDF-C libraries available to Cargo.

Use the Miniforge/conda-forge setup in [docs/windows-native-deps.md](docs/windows-native-deps.md). The important detail is to pin HDF5 to the supported 1.14 line:

```powershell
conda create -y -n roms-rs -c conda-forge "hdf5=1.14.4" "libnetcdf<4.10" pkg-config
conda env config vars set -n roms-rs HDF5_DIR="$env:USERPROFILE\miniforge3\envs\roms-rs" NETCDF_DIR="$env:USERPROFILE\miniforge3\envs\roms-rs"
conda activate roms-rs
cargo check
```

For work that does not need NetCDF I/O:

```powershell
cargo check --no-default-features --features parallel,simd
```

## Changelog

Notable changes are tracked in [CHANGELOG.md](CHANGELOG.md).
