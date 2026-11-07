//! Rivers as volume sources in the 2D model (TODO P1.6/P5.1).
//!
//! The production path, `Simulation` + `SSPRK3` + `SWEPhysics2D`, in a closed
//! basin with a dry beach: one river into open water, one onto the dry beach
//! (river databases put mouths on the coastline, often above the water). The
//! basin's volume must grow by exactly `∫ Q dt`; SSP-RK3 integrates a
//! discharge linear in time exactly.

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
        let sources = RiverSources::new(rivers.clone(), &mesh, &geom, 0.0).unwrap();
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

        // The beach river wetted its dry element
        let k = RiverSources::new(rivers.clone(), &mesh, &geom, 0.0)
            .unwrap()
            .element(1);
        let wet: f64 = (0..ops.n_nodes).map(|i| q.get_state(k, i).h).sum();
        assert!(wet > 0.0, "{formulation:?}: the beach river left no water");
    }
}
