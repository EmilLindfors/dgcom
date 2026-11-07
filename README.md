# DGCOM

**DGCOM, the Discontinuous Galerkin Coastal Ocean Model**, is a highly experimental coastal ocean model written in Rust. It uses nodal DG on unstructured quadrilateral meshes for the 2D shallow-water equations, and a ROMS-style 3D layer: sigma coordinates, mode splitting and GLS mixing. It was developed for currents along the Norwegian coast, nested in NorKyst-800. Running any coastal domain from a case file is the next step (`TODO.md` P8).

DGCOM is not a port of ROMS. It follows ROMS's recipe for the 3D physics and is validated against ROMS-based NorKyst-800.

## Quick start

A run is described by a case file: the mesh, the bed, the physics, the open boundaries and the output, in TOML. The example is an idealized estuary that needs no input data:

```sh
cargo build --release --bin dgcom --no-default-features --features parallel,simd   # no native NetCDF needed
./target/release/dgcom check cases/tidal_channel/case.toml   # validate it, print it with every default
./target/release/dgcom run   cases/tidal_channel/case.toml   # 25 h of tide in a few seconds
```

The run writes to `cases/tidal_channel/output/tidal_channel/`:
- `case.resolved.toml`: the case as run, with every default filled in;
- `stations.csv`: the surface and the current at the case's stations;
- `tidal_channel.dgsnap`: the whole state, for the viewer in `viz/`.

`output.vtk = true` adds ParaView files, and `output.netcdf = true` adds CF NetCDF (the default build, which needs HDF5/netCDF-C: see below). `--set key.path=value` overrides any key for a run, e.g. `--set numerics.cfl=0.5`. Unknown keys are errors. Every key and its default is documented in the `dgcom::case::schema` module (`cargo doc --open`).

Case files cover the 2D model:
- meshes: rectangles, Gmsh meshes, and longitude/latitude boxes clipped to the water;
- beds: uniform, sloping, or from a GeoTIFF;
- open boundaries: still water, a harmonic tide, or a tidal atlas.

`cases/froya/case.toml` and `cases/bodo/case.toml` run Frøya–Smøla–Hitra and Bodø on Kartverket's topobathy model, with NorKyst-800's tide along the open boundary from a tidal atlas. Their headers say how to fetch the inputs. `cases/froya/nested.toml` and `cases/bodo/nested.toml` nest the same domains in NorKyst-800 instead (`[boundary.nesting]`: the parent's level and current at the boundary, a relaxation band, the bed blended to the parent's). Atmospheric forcing is next (`TODO.md` P8).

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
