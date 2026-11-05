//! Inspect a NorKyst/ROMS file as nesting data.
//!
//! Reads a parent-model file (`OceanModelReader::from_file`: local NetCDF or
//! an OPeNDAP URL, e.g. one written by `norkyst_nesting_subset`), prints what
//! it found, the state at a point, and the open-boundary state of a 10 km
//! square nested around that point (`OceanModelState`).
//!
//! ```bash
//! cargo run --release --example test_norkyst -- data/froya_norkyst.nc [lon=8.5] [lat=63.75]
//! ```

#[cfg(feature = "netcdf")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::HashMap;
    use std::sync::Arc;

    use dg_rs::boundary::{
        BCContext2D, CharacteristicOBC, ExternalStateProvider, NestingOptions, OceanModelState,
        ParentDatum, SWEBoundaryCondition2D,
    };
    use dg_rs::io::{LocalProjection, OceanModelReader};
    use dg_rs::mesh::{BoundaryTag, Mesh2D};
    use dg_rs::operators::DGOperators2D;
    use dg_rs::solver::SWEState2D;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args
        .iter()
        .find(|a| !a.contains('='))
        .ok_or("usage: test_norkyst <file or URL> [lon=…] [lat=…]")?;
    let options: HashMap<&str, f64> = args
        .iter()
        .filter_map(|a| a.split_once('='))
        .filter_map(|(k, v)| Some((k, v.parse().ok()?)))
        .collect();
    let lon = options.get("lon").copied().unwrap_or(8.5);
    let lat = options.get("lat").copied().unwrap_or(63.75);

    println!("Loading {path}");
    let reader = Arc::new(OceanModelReader::from_file(path)?);
    println!("{}", reader.summary());

    match reader.get_state(lon, lat, 0) {
        Some(state) => {
            println!("\nFirst snapshot at ({lon}, {lat}):");
            println!("  SSH:      {:?} m", state.ssh);
            println!("  velocity: {:?} m/s (east, north)", state.velocity);
            println!("  depth:    {:?} m", state.depth);
            println!("  T, S:     {:?}, {:?}", state.temperature, state.salinity);
        }
        None => println!("\nNo parent data at ({lon}, {lat})"),
    }

    // A 10 km square around the point, open on every side
    let projection = LocalProjection::new(lat, lon);
    let mesh = Mesh2D::uniform_rectangle_with_bc(
        -5000.0,
        5000.0,
        -5000.0,
        5000.0,
        10,
        10,
        BoundaryTag::Open,
    );
    let ops = DGOperators2D::new(2);
    let clock = OceanModelState::first_snapshot(&reader);
    let parent = OceanModelState::new(
        Arc::clone(&reader),
        &mesh,
        &ops,
        &projection,
        BoundaryTag::Open,
        clock,
        // ζ as it is: a check of the coupling, and a subset of a few days
        // is too short for the default record-mean datum
        &NestingOptions::default()
            .with_band(2000.0)
            .with_datum(ParentDatum::Shift(0.0)),
    )?;
    let (s0, s1) = parent.simulation_time_range();
    println!(
        "\nNested 10 km square: {} boundary nodes ({} snapped to wet parent points), {} band nodes; \
         simulation times {s0:.0}–{s1:.0} s from {}",
        parent.n_boundary_nodes(),
        parent.n_snapped(),
        parent.n_band_nodes(),
        clock.format(0.0)
    );

    let bc = CharacteristicOBC::new(parent.clone());
    for (name, position, normal) in [
        ("east", (5000.0, 0.0), (1.0, 0.0)),
        ("north", (0.0, 5000.0), (0.0, 1.0)),
    ] {
        let bed = -50.0;
        let ctx = BCContext2D::new(
            0.0,
            position,
            SWEState2D::new(-bed, 0.0, 0.0),
            bed,
            normal,
            9.81,
            1e-6,
        );
        let external = parent.external_state(&ctx);
        let q = bc.boundary_state(&ctx);
        println!("  {name} side, 50 m bed: parent {external:?} → boundary state {q:?}");
    }
    Ok(())
}

#[cfg(not(feature = "netcdf"))]
fn main() {
    eprintln!("This example requires the `netcdf` feature.");
}
