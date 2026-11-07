//! Gate for station sampling and current validation (P3.1): an M2 tide
//! travelling through a channel, sampled at stations between the nodes
//! (`PointLocator2D` + `Probe2D`), fitted for reference constants of η and
//! tidal ellipses of the depth-averaged current (`HarmonicAnalysis::fit_ellipses`), and
//! compared with the linear progressive wave
//!
//! ```text
//! η = H f cos(χ − G₀ − ωx/c),   u = √(g/h) η,   v = 0,
//! ```
//!
//! where `χ = ω t + V₀ + u_nodal` is the astronomical argument of the model
//! clock. So the whole chain is checked: point location and evaluation of
//! the DG polynomial, the transport-over-depth velocity, the astronomy of
//! the fit (the lag must grow by ωx/c along the channel from the G₀ the
//! boundary is forced with), and the ellipse conversion (rectilinear along
//! the channel with semi-major axis H√(g/h) and the lag of η).

use std::f64::consts::PI;

use dgcom::analysis::HarmonicAnalysis;
use dgcom::boundary::{CharacteristicOBC, ExternalState, MultiBoundaryCondition2D, Reflective2D};
use dgcom::mesh::{Bathymetry2D, BoundaryTag, PointLocator2D};
use dgcom::solver::Probe2D;
use dgcom::tides::constituent_period;
use dgcom::time::{ModelClock, SSPRK3, TimeIntegrator};
use dgcom::{
    DGOperators2D, GeometricFactors2D, Mesh2D, SWE2DRhsConfig, SWEFormulation2D, SWESolution2D,
    SWEState2D, ShallowWater2D, compute_rhs_swe_2d,
};

const G: f64 = 9.81;
/// Still-water depth (m)
const DEPTH: f64 = 10.0;
/// M2 amplitude at the west end (m): a/h = 0.005, nearly linear
const AMPLITUDE: f64 = 0.05;
/// Greenwich lag of M2 at the west end (degrees)
const LAG_WEST: f64 = 120.0;
/// Channel length and width (m), elements along it, order
const LENGTH: f64 = 40_000.0;
const WIDTH: f64 = 2_000.0;
const NX: usize = 10;
const ORDER: usize = 3;
/// Time step (s) and sampling interval (s)
const DT: f64 = 20.0;
const SAMPLE: f64 = 600.0;

#[test]
fn stations_recover_the_progressive_tide_and_its_current_ellipse() {
    let mesh = Mesh2D::uniform_rectangle_with_sides(
        0.0,
        LENGTH,
        0.0,
        WIDTH,
        NX,
        1,
        [
            BoundaryTag::Wall,
            BoundaryTag::Open,
            BoundaryTag::Wall,
            BoundaryTag::TidalForcing,
        ],
    );
    let ops = DGOperators2D::new(ORDER);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let bathymetry = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -DEPTH);
    let equation = ShallowWater2D::new(G);

    // M2 at the west end on the model clock, as an incoming wave
    let clock = ModelClock::at_datetime(2025, 6, 1, 0, 0);
    let (spin_up, record) = (0.5 * 86_400.0, 1.5 * 86_400.0);
    let t_end = spin_up + record;
    let nodal = clock.nodal_correction("M2", 0.5 * t_end).unwrap();
    let omega = 2.0 * PI / constituent_period("M2").unwrap();
    let celerity = (G * DEPTH).sqrt();
    let forcing = CharacteristicOBC::new(move |_: f64, _: f64, t: f64| {
        let eta = nodal.f
            * AMPLITUDE
            * (omega * t + nodal.phase_offset_rad() - LAG_WEST.to_radians()).cos();
        ExternalState::new(eta, (G / DEPTH).sqrt() * eta, 0.0)
    });
    let wall = Reflective2D::new();
    let radiating = CharacteristicOBC::still_water();
    let bc = MultiBoundaryCondition2D::new(&wall)
        .with_tidal(&forcing)
        .with_open(&radiating);
    let config = SWE2DRhsConfig::new(&equation, &bc)
        .with_coriolis(false)
        .with_bathymetry(&bathymetry)
        .with_formulation(SWEFormulation2D::WetDry);

    // Stations between the nodes, away from the ends
    let locator = PointLocator2D::new(&mesh);
    let stations: Vec<Probe2D> = [[7_333.3, 1_234.5], [18_111.1, 321.0], [29_876.5, 1_777.7]]
        .into_iter()
        .map(|p| Probe2D::at(&locator, &ops, p).expect("station in the channel"))
        .collect();
    let mut series = vec![(Vec::new(), Vec::new(), Vec::new(), Vec::new()); stations.len()];

    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    for k in dgcom::types::ElementIndex::iter(mesh.n_elements) {
        for i in 0..ops.n_nodes {
            q.set_state(k, i, SWEState2D::new(DEPTH, 0.0, 0.0));
        }
    }
    let steps_per_sample = (SAMPLE / DT).round() as usize;
    let n_steps = (t_end / DT).round() as usize;
    for step in 1..=n_steps {
        let t = (step - 1) as f64 * DT;
        SSPRK3.step(&mut q, DT, t, |s, stage_t| {
            compute_rhs_swe_2d(s, &mesh, &ops, &geom, &config, stage_t)
        });
        let t = step as f64 * DT;
        if step % steps_per_sample == 0 && t > spin_up {
            for (probe, (times, eta, u, v)) in stations.iter().zip(&mut series) {
                let sample = probe.sample_swe(&q, Some(&bathymetry), 1e-6);
                times.push(clock.unix(t));
                eta.push(sample.eta);
                u.push(sample.u);
                v.push(sample.v);
            }
        }
    }

    for (probe, (times, eta, u, v)) in stations.iter().zip(&series) {
        let x = probe.position()[0];
        let expected_lag = (LAG_WEST + (omega * x / celerity).to_degrees()).rem_euclid(360.0);
        let lag_error = |g: f64| ((g - expected_lag + 180.0).rem_euclid(360.0) - 180.0).abs();

        let analysis = HarmonicAnalysis::new(&["M2", "M4"]);
        let surface = analysis.fit(times, eta).unwrap();
        let m2 = surface.get("M2").unwrap();
        let ellipses = analysis.fit_ellipses(times, u, v).unwrap();
        let e = ellipses.get("M2").unwrap();
        let major = AMPLITUDE * (G / DEPTH).sqrt();
        println!(
            "x = {x:7.0} m: η M2 {:.5} m, G {:.2}° (expected {expected_lag:.2}°); \
             ellipse {:.5}/{:+.2e} m/s at {:.2}°, G {:.2}° (expected {major:.5} m/s)",
            m2.amplitude, m2.lag_deg, e.major, e.minor, e.inclination_deg, e.lag_deg
        );
        assert!(
            (m2.amplitude / AMPLITUDE - 1.0).abs() < 2e-3,
            "η amplitude {}",
            m2.amplitude
        );
        assert!(lag_error(m2.lag_deg) < 0.1, "η lag {}", m2.lag_deg);
        assert!((e.major / major - 1.0).abs() < 2e-3, "major {}", e.major);
        assert!(e.minor.abs() < 1e-3 * major, "minor {}", e.minor);
        let inclination = e.inclination_deg.min(180.0 - e.inclination_deg);
        assert!(inclination < 0.1, "inclination {}", e.inclination_deg);
        assert!(lag_error(e.lag_deg) < 0.1, "current lag {}", e.lag_deg);
    }
}
