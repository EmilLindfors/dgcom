//! Tide Gauge Validation Example
//!
//! Demonstrates validation of model output against tide gauge observations,
//! here three Trøndelag gauges.
//!
//! This example:
//! 1. Reads the gauges' positions from a station list
//!    (`scripts/norkyst_tide_gauges.csv`, `io::read_tide_gauge_stations`) and
//!    generates synthetic observations from tidal constants given here
//! 2. Runs a 2D SWE simulation with tidal forcing
//! 3. Extracts model water levels at gauge locations
//! 4. Compares model vs observations using multiple metrics
//! 5. Performs harmonic analysis comparison
//!
//! ## Run
//!
//! ```bash
//! cargo run --release --example tide_gauge_validation
//! ```

use std::collections::HashMap;
use std::f64::consts::PI;

use dgcom::SWEFluxType2D;
use dgcom::analysis::{
    HarmonicAnalysis, Inference, PrecomputedExtractor, StabilityMonitor, StabilityThresholds,
    TideGaugeStation, TimeSeries,
};
use dgcom::boundary::{
    CharacteristicOBC, HarmonicTidal2D, MultiBoundaryCondition2D, Reflective2D, TidalConstituent,
};
use dgcom::equations::ShallowWater2D;
use dgcom::io::read_tide_gauge_stations;
use dgcom::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::solver::{
    SWE2DRhsConfig, SWESolution2D, compute_dt_swe_2d, compute_rhs_swe_2d, swe_positivity_limiter_2d,
};
use dgcom::source::CoriolisSource2D;
use dgcom::types::ElementIndex;

/// Helper for typed element indices
fn k(idx: usize) -> ElementIndex {
    ElementIndex::new(idx)
}

// ============================================================================
// Physical Constants
// ============================================================================

const G: f64 = 9.81;
const H_MIN: f64 = 0.01;

/// Coriolis parameter at 63°N (Trøndelag)
const F_CORIOLIS: f64 = 1.26e-4;

// ============================================================================
// Tidal constants of three Trøndelag gauges
// ============================================================================

/// The gauge list the positions come from (Kartverket's permanent gauges).
const STATION_LIST: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/scripts/norkyst_tide_gauges.csv"
);

/// Illustrative tidal constants for a gauge, named as in [`STATION_LIST`].
/// Synthetic, in the range of Kartverket's harmonic constants there.
struct StationTidalData {
    name: &'static str,
    m2_amplitude: f64,
    m2_phase_deg: f64,
    s2_amplitude: f64,
    s2_phase_deg: f64,
    k1_amplitude: f64,
    k1_phase_deg: f64,
    o1_amplitude: f64,
    o1_phase_deg: f64,
}

/// Realistic tidal data for Trøndelag region stations.
fn trondelag_tidal_data() -> Vec<StationTidalData> {
    vec![
        StationTidalData {
            name: "Heimsjøen",
            m2_amplitude: 0.85,
            m2_phase_deg: 125.0,
            s2_amplitude: 0.28,
            s2_phase_deg: 165.0,
            k1_amplitude: 0.06,
            k1_phase_deg: 185.0,
            o1_amplitude: 0.05,
            o1_phase_deg: 175.0,
        },
        StationTidalData {
            name: "Trondheim",
            m2_amplitude: 0.92,
            m2_phase_deg: 132.0,
            s2_amplitude: 0.30,
            s2_phase_deg: 172.0,
            k1_amplitude: 0.07,
            k1_phase_deg: 190.0,
            o1_amplitude: 0.05,
            o1_phase_deg: 180.0,
        },
        StationTidalData {
            name: "Kristiansund",
            m2_amplitude: 0.78,
            m2_phase_deg: 118.0,
            s2_amplitude: 0.26,
            s2_phase_deg: 158.0,
            k1_amplitude: 0.05,
            k1_phase_deg: 178.0,
            o1_amplitude: 0.04,
            o1_phase_deg: 168.0,
        },
    ]
}

/// Generate synthetic tide gauge observations from tidal constituents.
fn generate_observations(data: &StationTidalData, times: &[f64], noise_amplitude: f64) -> Vec<f64> {
    // Tidal periods in seconds
    let m2_period = 12.42 * 3600.0;
    let s2_period = 12.00 * 3600.0;
    let k1_period = 23.93 * 3600.0;
    let o1_period = 25.82 * 3600.0;

    let m2_omega = 2.0 * PI / m2_period;
    let s2_omega = 2.0 * PI / s2_period;
    let k1_omega = 2.0 * PI / k1_period;
    let o1_omega = 2.0 * PI / o1_period;

    let m2_phase = data.m2_phase_deg.to_radians();
    let s2_phase = data.s2_phase_deg.to_radians();
    let k1_phase = data.k1_phase_deg.to_radians();
    let o1_phase = data.o1_phase_deg.to_radians();

    times
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let m2 = data.m2_amplitude * (m2_omega * t - m2_phase).cos();
            let s2 = data.s2_amplitude * (s2_omega * t - s2_phase).cos();
            let k1 = data.k1_amplitude * (k1_omega * t - k1_phase).cos();
            let o1 = data.o1_amplitude * (o1_omega * t - o1_phase).cos();

            // Add small deterministic "noise" for realism
            let noise = if noise_amplitude > 0.0 {
                noise_amplitude * ((i as f64 * 0.1).sin() * 0.7 + (i as f64 * 0.23).cos() * 0.3)
            } else {
                0.0
            };

            m2 + s2 + k1 + o1 + noise
        })
        .collect()
}

/// SSP-RK3 time stepping for SWE with positivity limiter.
fn ssp_rk3_step<BC: dgcom::SWEBoundaryCondition2D>(
    q: &mut SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    dt: f64,
    time: f64,
    h_min: f64,
) {
    // Stage 1
    let rhs1 = compute_rhs_swe_2d(q, mesh, ops, geom, config, time);
    let mut q1 = q.clone();
    q1.axpy(dt, &rhs1);
    swe_positivity_limiter_2d(&mut q1, geom, h_min);

    // Stage 2
    let rhs2 = compute_rhs_swe_2d(&q1, mesh, ops, geom, config, time + dt);
    let mut q2 = q.clone();
    q2.axpy(0.25 * dt, &rhs1);
    q2.axpy(0.25 * dt, &rhs2);
    swe_positivity_limiter_2d(&mut q2, geom, h_min);

    // Stage 3
    let rhs3 = compute_rhs_swe_2d(&q2, mesh, ops, geom, config, time + 0.5 * dt);
    q.scale(1.0 / 3.0);
    q.axpy(2.0 / 3.0, &q2);
    q.axpy(2.0 / 3.0 * dt, &rhs3);
    swe_positivity_limiter_2d(q, geom, h_min);
}

fn main() {
    println!("==============================================");
    println!("Tide Gauge Validation - Trøndelag gauges");
    println!("==============================================\n");

    // ========================================================================
    // 1. Setup simulation parameters
    // ========================================================================

    let h0 = 50.0; // Mean water depth (m)
    let domain_size = 100_000.0; // 100 km domain
    let nx = 5;
    let ny = 5;
    let n_poly = 2;

    // Simulation time: 24 hours (covers ~2 M2 tidal cycles)
    let t_end = 24.0 * 3600.0;
    let output_interval = 600.0; // 10 minutes
    let cfl = 0.5;

    println!("Simulation Setup:");
    println!(
        "  Domain: {:.0} km x {:.0} km",
        domain_size / 1000.0,
        domain_size / 1000.0
    );
    println!("  Elements: {} x {}", nx, ny);
    println!("  Polynomial order: {}", n_poly);
    println!("  Mean depth: {:.0} m", h0);
    println!("  Duration: {:.1} hours", t_end / 3600.0);
    println!();

    // ========================================================================
    // 2. Generate synthetic tide gauge observations
    // ========================================================================

    println!("Generating synthetic tide gauge observations...");

    let tidal_data = trondelag_tidal_data();
    let gauges = read_tide_gauge_stations(std::path::Path::new(STATION_LIST))
        .unwrap_or_else(|e| panic!("{STATION_LIST}: {e}"));
    let n_outputs = (t_end / output_interval) as usize + 1;
    let times: Vec<f64> = (0..n_outputs).map(|i| i as f64 * output_interval).collect();

    let mut observations: HashMap<String, TimeSeries> = HashMap::new();
    let mut stations: Vec<TideGaugeStation> = Vec::new();

    // Scale factor for observations to match reduced forcing amplitude
    let amp_scale = 0.1;

    for data in &tidal_data {
        // Generate scaled observations matching the reduced forcing amplitude
        let scaled_data = StationTidalData {
            name: data.name,
            m2_amplitude: data.m2_amplitude * amp_scale,
            m2_phase_deg: data.m2_phase_deg,
            s2_amplitude: data.s2_amplitude * amp_scale,
            s2_phase_deg: data.s2_phase_deg,
            k1_amplitude: 0.0, // Only M2+S2 for simplicity
            k1_phase_deg: data.k1_phase_deg,
            o1_amplitude: 0.0,
            o1_phase_deg: data.o1_phase_deg,
        };
        let values = generate_observations(&scaled_data, &times, 0.002);
        let ts = TimeSeries::new(&times, &values).with_name(data.name);
        observations.insert(data.name.to_string(), ts);

        // The gauge from the list, its geographic position mapped into the
        // model domain's interior
        let gauge: TideGaugeStation = gauges
            .iter()
            .find(|g| g.name == data.name)
            .unwrap_or_else(|| panic!("{} is not in {STATION_LIST}", data.name))
            .clone();
        let x = domain_size * 0.2 + (gauge.longitude - 7.0) / 4.0 * domain_size * 0.6;
        let y = domain_size * 0.2 + (gauge.latitude - 63.0) / 1.0 * domain_size * 0.6;
        let station = gauge.with_local_coords(
            x.clamp(5000.0, domain_size - 5000.0),
            y.clamp(5000.0, domain_size - 5000.0),
        );
        stations.push(station);

        println!(
            "  {}: M2={:.2}m @ {:.0}°, local=({:.1}km, {:.1}km)",
            data.name,
            data.m2_amplitude,
            data.m2_phase_deg,
            x / 1000.0,
            y / 1000.0
        );
    }
    println!();

    // ========================================================================
    // 3. Run 2D SWE simulation
    // ========================================================================

    println!("Running 2D SWE simulation...");

    // Create mesh with:
    // - West (left): Tidal forcing boundary
    // - East (right): Open/Radiation boundary
    // - North/South: Wall boundaries
    let bc_tags = [
        BoundaryTag::Wall,         // South
        BoundaryTag::Open,         // East (radiation)
        BoundaryTag::Wall,         // North
        BoundaryTag::TidalForcing, // West (tidal inflow)
    ];
    let mesh =
        Mesh2D::uniform_rectangle_with_sides(0.0, domain_size, 0.0, domain_size, nx, ny, bc_tags);
    let ops = DGOperators2D::new(n_poly);
    let geom = GeometricFactors2D::compute(&mesh, &ops);

    println!(
        "  Mesh: {} elements, {} boundary edges",
        mesh.n_elements, mesh.n_boundary_edges
    );

    // Use average tidal characteristics for boundary forcing
    let avg_m2_amp =
        tidal_data.iter().map(|d| d.m2_amplitude).sum::<f64>() / tidal_data.len() as f64;
    let avg_s2_amp =
        tidal_data.iter().map(|d| d.s2_amplitude).sum::<f64>() / tidal_data.len() as f64;
    let _avg_k1_amp =
        tidal_data.iter().map(|d| d.k1_amplitude).sum::<f64>() / tidal_data.len() as f64;
    let _avg_o1_amp =
        tidal_data.iter().map(|d| d.o1_amplitude).sum::<f64>() / tidal_data.len() as f64;
    let avg_m2_phase =
        tidal_data.iter().map(|d| d.m2_phase_deg).sum::<f64>() / tidal_data.len() as f64;
    let avg_s2_phase =
        tidal_data.iter().map(|d| d.s2_phase_deg).sum::<f64>() / tidal_data.len() as f64;
    let _avg_k1_phase =
        tidal_data.iter().map(|d| d.k1_phase_deg).sum::<f64>() / tidal_data.len() as f64;
    let _avg_o1_phase =
        tidal_data.iter().map(|d| d.o1_phase_deg).sum::<f64>() / tidal_data.len() as f64;

    println!(
        "  Tidal forcing (scaled): M2={:.3}m @ {:.0}°, S2={:.3}m @ {:.0}°",
        avg_m2_amp * amp_scale,
        avg_m2_phase,
        avg_s2_amp * amp_scale,
        avg_s2_phase
    );

    // Setup boundary conditions
    //
    // NOTE: HarmonicTidal2D clamps the elevation (Dirichlet). For open
    // boundaries that must let outgoing waves leave, use
    // CharacteristicOBC::new(HarmonicTide::new(...)) instead.
    //
    // Alternatively, use TidalSimulationBuilder for recommended presets:
    //   let builder = TidalSimulationBuilder::closed_basin_stable(amp, h0, sponge_width);
    //   let bc = builder.build_bc();
    let wall_bc = Reflective2D::new();
    let radiation_bc = CharacteristicOBC::still_water();

    let tidal_bc = HarmonicTidal2D::new(vec![
        TidalConstituent::m2(avg_m2_amp * amp_scale, avg_m2_phase.to_radians()),
        TidalConstituent::s2(avg_s2_amp * amp_scale, avg_s2_phase.to_radians()),
    ])
    .with_ramp_up(6.0 * 3600.0); // 6-hour ramp (half M2 period)

    // Combine boundary conditions
    let multi_bc = MultiBoundaryCondition2D::new(&wall_bc)
        .with_wall(&wall_bc)
        .with_open(&radiation_bc)
        .with_tidal(&tidal_bc);

    let equation = ShallowWater2D::with_coriolis(G, F_CORIOLIS);
    let coriolis = CoriolisSource2D::f_plane(F_CORIOLIS);

    // Create bathymetry: bottom at -h0 (so surface elevation = h + B = 0 at rest)
    // This is CRITICAL for Flather BC to work correctly!
    let bathymetry = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -h0);

    let config = SWE2DRhsConfig::new(&equation, &multi_bc)
        .with_flux_type(SWEFluxType2D::Roe)
        .with_source_terms(&coriolis)
        .with_bathymetry(&bathymetry);

    // Initialize solution at rest (h = h0 means surface at z = 0)
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(&mesh, &ops, |_, _| h0, |_, _| 0.0, |_, _| 0.0);

    // Storage for model output at stations
    let mut model_at_stations: HashMap<String, Vec<f64>> = HashMap::new();
    for station in &stations {
        model_at_stations.insert(station.name.clone(), Vec::with_capacity(n_outputs));
    }

    // Time stepping with stability monitoring
    let mut time = 0.0;
    let mut step = 0;
    let mut next_output = 0.0;

    // Use StabilityMonitor for detecting numerical issues
    let mut stability_monitor = StabilityMonitor::new(StabilityThresholds::tidal_default());

    while time < t_end {
        // Adaptive time step
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);

        // Check for stability issues using StabilityMonitor
        let status = stability_monitor.check(&q, dt);
        if !status.is_stable {
            stability_monitor.print_report(time, step);

            if stability_monitor.should_stop() {
                println!("  FATAL: Solution has blown up. Stopping.");
                for suggestion in stability_monitor.suggest_remediation() {
                    println!("    * {}", suggestion);
                }
                break;
            }
        }

        let min_dt = stability_monitor.thresholds().min_dt;
        let dt = dt.max(min_dt).min(t_end - time);
        let dt = if time < next_output && time + dt > next_output {
            next_output - time
        } else {
            dt
        };

        // Record output at stations
        if time >= next_output - 1e-6 {
            for station in &stations {
                let (x, y) = station.local_coords();
                let eta = extract_surface_elevation(&q, &mesh, &ops, x, y, h0);
                model_at_stations.get_mut(&station.name).unwrap().push(eta);
            }
            next_output += output_interval;
        }

        // Take time step
        ssp_rk3_step(&mut q, &mesh, &ops, &geom, &config, dt, time, H_MIN);
        time += dt;
        step += 1;

        // Progress report
        if step % 1000 == 0 {
            let hours = time / 3600.0;
            let pct = 100.0 * time / t_end;
            print!("\r  t = {:.1} h ({:.0}%), step {}    ", hours, pct, step);
        }
    }
    println!(
        "\r  Completed: {} steps, {:.1} hours simulated",
        step,
        time / 3600.0
    );
    println!();

    // ========================================================================
    // 4. Build model time series and validate
    // ========================================================================

    println!("Validating model against observations...\n");

    // Debug: check model output
    for station in &stations {
        let values = model_at_stations.get(&station.name).unwrap();
        let (min, max) = values
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &v| {
                (min.min(v), max.max(v))
            });
        println!(
            "  Model at {}: {} points, range [{:.4}, {:.4}] m",
            station.name,
            values.len(),
            min,
            max
        );
    }

    let mut extractor = PrecomputedExtractor::new();
    for station in &stations {
        let values = model_at_stations.get(&station.name).unwrap();
        // Ensure same length as observations
        let values: Vec<f64> = values.iter().take(times.len()).copied().collect();
        if values.is_empty() {
            println!("  WARNING: No model data for {}", station.name);
            continue;
        }
        let ts = TimeSeries::new(&times[..values.len()], &values).with_name(&station.name);
        extractor.add_series(&station.name, ts);
    }

    // Trim observations to match
    let mut obs_trimmed: HashMap<String, TimeSeries> = HashMap::new();
    for (name, ts) in &observations {
        let model_len = model_at_stations.get(name).unwrap().len();
        if model_len == 0 {
            println!("  WARNING: No model data for obs {}", name);
            continue;
        }
        let values: Vec<f64> = ts.values().iter().take(model_len).copied().collect();
        let times_trimmed: Vec<f64> = times.iter().take(model_len).copied().collect();
        obs_trimmed.insert(name.clone(), TimeSeries::new(&times_trimmed, &values));
    }

    // Check data lengths
    let model_len = model_at_stations.values().next().unwrap().len();
    println!("  Total model output points: {}", model_len);
    println!("  Observation time points: {}", times.len());

    if model_len == 0 {
        println!("\nERROR: No model output was recorded!");
        return;
    }

    // Run validation with harmonic analysis: the constituents the run
    // resolves, the equilibrium inferences for the rest
    let analysis = HarmonicAnalysis::resolvable(
        &["M2", "S2", "K1", "O1", "N2", "M4"],
        times[model_len - 1] - times[0],
        1.0,
        &Inference::EQUILIBRIUM,
    );
    println!("  Harmonic analysis: {:?}", analysis.names());
    let (results, summary) = dgcom::validate_stations(
        &stations,
        &extractor,
        &obs_trimmed,
        &times[..model_len],
        Some(&analysis),
    );

    // ========================================================================
    // 5. Print validation results
    // ========================================================================

    if results.is_empty() {
        println!("ERROR: No validation results produced!");
        println!("This may indicate a mismatch between station names or missing data.");
        return;
    }

    println!("Station-by-Station Results:");
    println!("----------------------------");

    for result in &results {
        let result = match result {
            Ok(result) => result,
            Err(e) => {
                println!("\n  not validated: {e}");
                continue;
            }
        };
        println!("\n{}", result.summary());

        // Print constituent comparison if available
        if !result.constituent_comparisons.is_empty() {
            println!("Constituent Comparison:");
            for comp in &result.constituent_comparisons {
                println!(
                    "  {}: Amp ratio={:.3}, Phase error={:.1}°",
                    comp.name,
                    comp.amplitude_ratio,
                    comp.phase_error_degrees()
                );
            }
        }
    }

    println!("\n{}", summary);

    // ========================================================================
    // 6. Assess validation status
    // ========================================================================

    println!("\nValidation Assessment:");
    println!("----------------------");

    if summary.n_passing_strict == summary.n_stations {
        println!("✓ EXCELLENT: All stations pass strict validation criteria");
    } else if summary.n_passing_basic == summary.n_stations {
        println!("✓ GOOD: All stations pass basic validation criteria");
    } else if summary.n_passing_basic > 0 {
        println!(
            "⚠ PARTIAL: {}/{} stations pass basic validation",
            summary.n_passing_basic, summary.n_stations
        );
    } else {
        println!("✗ POOR: No stations pass validation criteria");
    }

    println!("\nKey Metrics:");
    println!("  Mean RMSE: {:.3} m (target: < 0.1 m)", summary.mean_rmse);
    println!(
        "  Mean Correlation: {:.3} (target: > 0.9)",
        summary.mean_correlation
    );
    println!(
        "  Mean Skill: {:.3} (target: > 0.8)",
        summary.mean_skill_score
    );

    // Expected result: Model should capture tidal signal but with some phase/amplitude
    // errors due to simplified domain (no real bathymetry, uniform depth)
    println!("\nNote: This is a simplified demonstration with uniform bathymetry.");
    println!("Real simulations with actual bathymetry should show better agreement.");
}

/// Extract surface elevation at a point by finding nearest element.
fn extract_surface_elevation(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    x: f64,
    y: f64,
    h0: f64,
) -> f64 {
    // Find element containing point (simple search)
    for ki in 0..mesh.n_elements {
        let [x0, y0] = mesh.reference_to_physical(k(ki), -1.0, -1.0);
        let [x1, y1] = mesh.reference_to_physical(k(ki), 1.0, 1.0);

        if x >= x0.min(x1) && x <= x0.max(x1) && y >= y0.min(y1) && y <= y0.max(y1) {
            // Found element - get average h
            let mut h_sum = 0.0;
            for i in 0..ops.n_nodes {
                h_sum += q.get_state(k(ki), i).h;
            }
            let h_avg = h_sum / ops.n_nodes as f64;
            return h_avg - h0; // Return surface elevation (h - h0)
        }
    }

    // Fallback: use first element
    let mut h_sum = 0.0;
    for i in 0..ops.n_nodes {
        h_sum += q.get_state(k(0), i).h;
    }
    h_sum / ops.n_nodes as f64 - h0
}
