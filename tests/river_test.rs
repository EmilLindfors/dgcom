//! Rivers as volume sources in the 2D model (TODO P1.6/P5.1).
//!
//! The production path, `Simulation` + `SSPRK3` + `SWEPhysics2D`, in a closed
//! basin with a dry beach: one river into open water, one with its mouth on
//! the dry beach (river databases put mouths on the coastline, often above
//! the water), which enters the nearest water. The basin's volume must grow
//! by exactly `∫ Q dt`; SSP-RK3 integrates a discharge linear in time
//! exactly. The beach stays dry: the water enters the sea, not the land
//! (TODO P1.6).

use std::sync::Arc;

use dgcom::boundary::Reflective2D;
use dgcom::equations::ShallowWater2D;
use dgcom::mesh::{Bathymetry2D, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::physics::PhysicsBuilder;
use dgcom::simulation::Simulation;
use dgcom::solver::{SWEFormulation2D, SWESolution2D, SWEState2D, StandardLimiter2D, WetDryConfig};
use dgcom::source::{BathymetrySource2D, River, RiverSeries, RiverSources};
use dgcom::time::SSPRK3;
use dgcom::types::{Depth, ElementIndex};

const G: f64 = 9.81;
const LENGTH: f64 = 2000.0;

/// Bed from −5 m at x = 0 up to +2 m at the far end: dry beyond 1429 m.
fn bed(x: f64, _y: f64) -> f64 {
    -5.0 + 7.0 * x / LENGTH
}

#[test]
fn rivers_add_exactly_their_discharge_to_the_volume() {
    let mesh = Arc::new(Mesh2D::uniform_rectangle(0.0, LENGTH, 0.0, 400.0, 10, 2));
    let ops = Arc::new(DGOperators2D::new(2));
    let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
    let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, bed));
    let t_end = 600.0;
    let ramp = RiverSeries::new(vec![0.0, t_end], vec![5.0, 20.0]).unwrap();
    let rivers = vec![
        River::new("open water", [300.0, 150.0], 0.0).with_discharge(ramp),
        River::new("beach", [1900.0, 250.0], 3.0),
    ];
    // ∫ Q dt: the ramp's mean is 12.5 m³/s
    let added = 12.5 * t_end + 3.0 * t_end;

    for formulation in [SWEFormulation2D::Standard, SWEFormulation2D::WetDry] {
        let sources =
            RiverSources::new(rivers.clone(), &mesh, &geom, &bathymetry, SNAP, &[]).unwrap();
        let builder = PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            ShallowWater2D::new(G),
            Reflective2D::new(),
        )
        .with_bathymetry(bathymetry.clone())
        .with_formulation(formulation)
        .with_limiter(StandardLimiter2D::Positivity(WetDryConfig::DEFAULT_H_DRY))
        .with_wet_dry(WetDryConfig::new(
            Depth::new(WetDryConfig::DEFAULT_H_DRY),
            G,
        ))
        .with_source(sources);
        let physics = match formulation {
            SWEFormulation2D::Standard => builder
                .with_well_balanced(true)
                .with_source(BathymetrySource2D::new(G)),
            _ => builder,
        }
        .build();

        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let b = bathymetry.get(k, i);
                q.set_state(k, i, SWEState2D::new((-b).max(0.0), 0.0, 0.0));
            }
        }
        let volume0 = q.integrate_depth(&ops, &geom);

        let mut lowest = f64::INFINITY;
        let sim = Simulation::new(physics, SSPRK3).with_cfl(0.5);
        let result = sim.run_with_callback(&mut q, 0.0, t_end, |q, _| {
            lowest = q.h_data().iter().copied().fold(lowest, f64::min);
        });
        assert!(result.success, "{formulation:?}: {result:?}");

        let volume = q.integrate_depth(&ops, &geom);
        let error = (volume - volume0 - added).abs();
        // Measured 1.8e-8 / 2.1e-8 m³ (Standard / WetDry) against 9300 m³
        // added to 1.4e6 m³: round-off
        assert!(
            error < 1e-12 * volume0,
            "{formulation:?}: volume grew by {:.6} m³, the rivers added {added} m³",
            volume - volume0
        );
        assert!(lowest >= 0.0, "{formulation:?}: negative depth {lowest:e}");
        assert_eq!(sim.physics().negative_depth_clips(), 0, "{formulation:?}");

        // The beach river's water went into the sea: the beach, 0.6 m and
        // more above it, stayed dry (the river in its mouth's beach element,
        // the rate shared evenly, left 7.0 mm on it). Only where the
        // formulation keeps a shore at rest: `Standard` leaks 5.8 mm onto the
        // beach in these 600 s with no rivers at all (it is to be retired,
        // TODO P1.2)
        if formulation != SWEFormulation2D::WetDry {
            continue;
        }
        let mut beach = 0.0_f64;
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                if bathymetry.get(k, i) > 0.5 {
                    beach = beach.max(q.get_state(k, i).h);
                }
            }
        }
        assert!(
            beach < WetDryConfig::DEFAULT_H_DRY,
            "{formulation:?}: {beach} m of water on the beach"
        );
    }
}

/// How far the beach river's mouth may move: from 1900 m to the element of
/// 1200–1400 m, the nearest under water (the shore is at 1429 m, in the
/// element of 1400–1600 m).
const SNAP: f64 = 500.0;

/// A mouth on the beach enters the nearest element under water, shared by
/// depth.
#[test]
fn a_mouth_on_the_beach_enters_the_nearest_water() {
    let mesh = Mesh2D::uniform_rectangle(0.0, LENGTH, 0.0, 400.0, 10, 2);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, bed);
    let beach = vec![River::new("beach", [1900.0, 250.0], 3.0)];
    assert!(RiverSources::new(beach.clone(), &mesh, &geom, &bathymetry, SNAP - 1.0, &[]).is_err());
    let sources = RiverSources::new(beach, &mesh, &geom, &bathymetry, SNAP, &[]).unwrap();
    let k = sources.element(0);
    let locator = dgcom::mesh::PointLocator2D::new(&mesh);
    assert_eq!(k, locator.locate([1300.0, 250.0]).unwrap().element);
    assert!((sources.snap_distance(0) - SNAP).abs() < 1e-9);
    for i in 0..ops.n_nodes {
        let (b, s) = (bathymetry.get(k, i), sources.share(0)[i]);
        assert!(b < 0.0 && s > 0.0, "{s} at the bed {b} m");
        let dilution = s / -b;
        let first = sources.share(0)[0] / -bathymetry.get(k, 0);
        assert!((dilution - first).abs() < 1e-14 * first);
    }
}
