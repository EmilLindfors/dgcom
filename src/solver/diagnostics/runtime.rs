//! Runtime diagnostics for 2D shallow water simulations.
//!
//! Provides functions and types for computing and tracking:
//! - Conservation quantities (mass, momentum, energy)
//! - CFL number
//! - Solution bounds (max velocity, min depth)
//! - Progress reporting for long simulations
//!
//! # Example
//!
//! ```ignore
//! use dgcom::solver::{SWEDiagnostics2D, DiagnosticsTracker};
//!
//! // Compute diagnostics at current time
//! let diag = SWEDiagnostics2D::compute(&q, &mesh, &ops, &geom, g, dt);
//! println!("Mass: {:.6}, Energy: {:.6}", diag.total_mass, diag.total_energy);
//!
//! // Track diagnostics over simulation
//! let mut tracker = DiagnosticsTracker::new(initial_diag);
//! // ... simulation loop ...
//! tracker.update(time, &current_diag);
//! tracker.print_summary();
//! ```

use crate::mesh::Mesh2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::SystemSolution2D;
use crate::types::ElementIndex;

/// Diagnostic quantities for 2D shallow water equations.
#[derive(Clone, Debug)]
pub struct SWEDiagnostics2D {
    /// Total water mass (integral of h over domain)
    pub total_mass: f64,
    /// Total x-momentum (integral of hu over domain)
    pub momentum_x: f64,
    /// Total y-momentum (integral of hv over domain)
    pub momentum_y: f64,
    /// Total energy (integral of kinetic + potential energy)
    pub total_energy: f64,
    /// Kinetic energy component (integral of 0.5 * h * (u² + v²))
    pub kinetic_energy: f64,
    /// Potential energy component (integral of 0.5 * g * h²)
    pub potential_energy: f64,
    /// Maximum velocity magnitude in domain
    pub max_velocity: f64,
    /// Minimum water depth in domain
    pub min_depth: f64,
    /// Maximum water depth in domain
    pub max_depth: f64,
    /// Current CFL number (max wave speed * dt / min element size)
    pub cfl_number: f64,
    /// Maximum Froude number (|u|/sqrt(gh))
    pub max_froude: f64,
}

/// Compute 2D quadrature weights from 1D weights (tensor product).
#[inline]
fn compute_weight_2d(ops: &DGOperators2D, i: usize) -> f64 {
    let n_1d = ops.order + 1;
    let i_r = i % n_1d;
    let i_s = i / n_1d;
    ops.weights_1d[i_r] * ops.weights_1d[i_s]
}

impl SWEDiagnostics2D {
    /// Compute all diagnostics from current solution.
    ///
    /// # Arguments
    /// * `q` - Current SWE solution
    /// * `mesh` - 2D mesh
    /// * `ops` - DG operators
    /// * `geom` - Geometric factors
    /// * `g` - Gravitational acceleration
    /// * `dt` - Current timestep (for CFL computation)
    pub fn compute(
        q: &SystemSolution2D<3>,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        g: f64,
        dt: f64,
    ) -> Self {
        let h_min_threshold = 1e-10;
        let n_nodes = ops.n_nodes;

        let mut total_mass = 0.0;
        let mut momentum_x = 0.0;
        let mut momentum_y = 0.0;
        let mut kinetic_energy = 0.0;
        let mut potential_energy = 0.0;
        let mut max_velocity = 0.0;
        let mut min_depth = f64::MAX;
        let mut max_depth = 0.0;
        let mut max_wave_speed = 0.0;
        let mut max_froude = 0.0;

        for k in ElementIndex::iter(q.n_elements) {
            for i in 0..n_nodes {
                let det_j = geom.jacobian(k.as_usize(), i);
                let state = q.get_state(k, i);
                let h = state.h;
                let hu = state.hu;
                let hv = state.hv;

                // Quadrature weight
                let w = compute_weight_2d(ops, i);
                let area_factor = w * det_j;

                // Conservation quantities
                total_mass += h * area_factor;
                momentum_x += hu * area_factor;
                momentum_y += hv * area_factor;

                // Track depth bounds
                if h < min_depth {
                    min_depth = h;
                }
                if h > max_depth {
                    max_depth = h;
                }

                // Energy and velocity (only for wet cells)
                if h > h_min_threshold {
                    let u = hu / h;
                    let v = hv / h;
                    let speed_sq = u * u + v * v;
                    let speed = speed_sq.sqrt();

                    // Energy
                    kinetic_energy += 0.5 * h * speed_sq * area_factor;
                    potential_energy += 0.5 * g * h * h * area_factor;

                    // Velocity bounds
                    if speed > max_velocity {
                        max_velocity = speed;
                    }

                    // Wave speed for CFL
                    let c = (g * h).sqrt();
                    let wave_speed = speed + c;
                    if wave_speed > max_wave_speed {
                        max_wave_speed = wave_speed;
                    }

                    // Froude number
                    let froude = speed / c;
                    if froude > max_froude {
                        max_froude = froude;
                    }
                }
            }
        }

        // CFL number
        let min_h_elem = mesh.h_min();
        let dg_factor = (2 * ops.order + 1) as f64;
        let cfl_number = if min_h_elem > 0.0 {
            max_wave_speed * dt * dg_factor / min_h_elem
        } else {
            0.0
        };

        // Handle empty/dry domain
        if min_depth == f64::MAX {
            min_depth = 0.0;
        }

        Self {
            total_mass,
            momentum_x,
            momentum_y,
            total_energy: kinetic_energy + potential_energy,
            kinetic_energy,
            potential_energy,
            max_velocity,
            min_depth,
            max_depth,
            cfl_number,
            max_froude,
        }
    }

    /// Format diagnostics as a single-line summary.
    pub fn summary_line(&self) -> String {
        format!(
            "M={:.4e} E={:.4e} |u|_max={:.3} h=[{:.3},{:.3}] CFL={:.3} Fr={:.3}",
            self.total_mass,
            self.total_energy,
            self.max_velocity,
            self.min_depth,
            self.max_depth,
            self.cfl_number,
            self.max_froude
        )
    }

    /// Format diagnostics as detailed multi-line output.
    pub fn detailed(&self) -> String {
        format!(
            "Conservation:\n  Mass:      {:.6e}\n  Momentum:  ({:.6e}, {:.6e})\n  Energy:    {:.6e} (KE={:.6e}, PE={:.6e})\n\
             Bounds:\n  Depth:     [{:.4}, {:.4}] m\n  Velocity:  {:.4} m/s\n\
             Stability:\n  CFL:       {:.4}\n  Froude:    {:.4}",
            self.total_mass,
            self.momentum_x,
            self.momentum_y,
            self.total_energy,
            self.kinetic_energy,
            self.potential_energy,
            self.min_depth,
            self.max_depth,
            self.max_velocity,
            self.cfl_number,
            self.max_froude
        )
    }
}

/// Track diagnostics over time for monitoring conservation and stability.
#[derive(Clone, Debug)]
pub struct DiagnosticsTracker {
    /// Initial diagnostics (for conservation error computation)
    initial: SWEDiagnostics2D,
    /// Most recent diagnostics
    current: SWEDiagnostics2D,
    /// Time of most recent update
    current_time: f64,
    /// Number of updates
    n_updates: usize,
    /// Maximum CFL seen during simulation
    max_cfl_seen: f64,
    /// Maximum velocity seen during simulation
    max_velocity_seen: f64,
    /// Maximum Froude number seen during simulation
    max_froude_seen: f64,
    /// Minimum depth seen during simulation
    min_depth_seen: f64,
}

impl DiagnosticsTracker {
    /// Create a new tracker with initial diagnostics.
    pub fn new(initial: SWEDiagnostics2D) -> Self {
        let max_cfl = initial.cfl_number;
        let max_vel = initial.max_velocity;
        let max_fr = initial.max_froude;
        let min_d = initial.min_depth;

        Self {
            current: initial.clone(),
            initial,
            current_time: 0.0,
            n_updates: 0,
            max_cfl_seen: max_cfl,
            max_velocity_seen: max_vel,
            max_froude_seen: max_fr,
            min_depth_seen: min_d,
        }
    }

    /// Update tracker with new diagnostics.
    pub fn update(&mut self, time: f64, diag: SWEDiagnostics2D) {
        self.current_time = time;
        self.n_updates += 1;

        if diag.cfl_number > self.max_cfl_seen {
            self.max_cfl_seen = diag.cfl_number;
        }
        if diag.max_velocity > self.max_velocity_seen {
            self.max_velocity_seen = diag.max_velocity;
        }
        if diag.max_froude > self.max_froude_seen {
            self.max_froude_seen = diag.max_froude;
        }
        if diag.min_depth < self.min_depth_seen {
            self.min_depth_seen = diag.min_depth;
        }

        self.current = diag;
    }

    /// Get relative mass conservation error.
    pub fn mass_error(&self) -> f64 {
        if self.initial.total_mass.abs() > 1e-14 {
            (self.current.total_mass - self.initial.total_mass).abs()
                / self.initial.total_mass.abs()
        } else {
            0.0
        }
    }

    /// Get relative momentum conservation error (magnitude).
    pub fn momentum_error(&self) -> f64 {
        let initial_mom =
            (self.initial.momentum_x.powi(2) + self.initial.momentum_y.powi(2)).sqrt();
        let current_mom =
            (self.current.momentum_x.powi(2) + self.current.momentum_y.powi(2)).sqrt();

        if initial_mom.abs() > 1e-14 {
            (current_mom - initial_mom).abs() / initial_mom.abs()
        } else {
            (current_mom - initial_mom).abs()
        }
    }

    /// Get relative energy change (can be negative due to dissipation).
    pub fn energy_change(&self) -> f64 {
        if self.initial.total_energy.abs() > 1e-14 {
            (self.current.total_energy - self.initial.total_energy)
                / self.initial.total_energy.abs()
        } else {
            0.0
        }
    }

    /// Get current diagnostics.
    pub fn current(&self) -> &SWEDiagnostics2D {
        &self.current
    }

    /// Get initial diagnostics.
    pub fn initial(&self) -> &SWEDiagnostics2D {
        &self.initial
    }

    /// Check if simulation appears stable (no blow-up indicators).
    pub fn is_stable(&self) -> bool {
        let diag = &self.current;

        // Check for NaN/Inf
        if !diag.total_mass.is_finite()
            || !diag.max_velocity.is_finite()
            || !diag.total_energy.is_finite()
        {
            return false;
        }

        // Check for unreasonable values
        if diag.max_velocity > 100.0 {
            // > 100 m/s is unrealistic for coastal flows
            return false;
        }
        if diag.cfl_number > 2.0 {
            // CFL > 2 is unstable for explicit methods
            return false;
        }
        if diag.max_froude > 10.0 {
            // Fr > 10 suggests numerical issues
            return false;
        }

        true
    }

    /// Print a summary of the simulation diagnostics.
    pub fn print_summary(&self) {
        println!("=== Diagnostics Summary ===");
        println!(
            "Time: {:.2} s ({} updates)",
            self.current_time, self.n_updates
        );
        println!();
        println!("Conservation:");
        println!(
            "  Mass error:     {:.2e} ({:.4}%)",
            self.mass_error(),
            self.mass_error() * 100.0
        );
        println!(
            "  Momentum error: {:.2e} ({:.4}%)",
            self.momentum_error(),
            self.momentum_error() * 100.0
        );
        println!(
            "  Energy change:  {:.2e} ({:.4}%)",
            self.energy_change(),
            self.energy_change() * 100.0
        );
        println!();
        println!("Extrema during simulation:");
        println!("  Max CFL:      {:.4}", self.max_cfl_seen);
        println!("  Max velocity: {:.4} m/s", self.max_velocity_seen);
        println!("  Max Froude:   {:.4}", self.max_froude_seen);
        println!("  Min depth:    {:.4} m", self.min_depth_seen);
        println!();
        println!("Current state:");
        println!("{}", self.current.detailed());
    }
}

/// Progress reporter for long-running simulations.
#[derive(Clone, Debug)]
pub struct ProgressReporter {
    /// Start time of simulation (wall clock)
    start_instant: std::time::Instant,
    /// Total simulation time to reach
    total_sim_time: f64,
    /// Last reported progress percentage
    last_reported_pct: u32,
    /// Report interval in percentage points
    report_interval_pct: u32,
    /// Whether to print diagnostics with progress
    print_diagnostics: bool,
    /// Number of timesteps taken
    n_steps: usize,
}

impl ProgressReporter {
    /// Create a new progress reporter.
    ///
    /// # Arguments
    /// * `total_sim_time` - Total simulation time to reach
    /// * `report_interval_pct` - Report every N percent (e.g., 10 for 10%, 20%, ...)
    pub fn new(total_sim_time: f64, report_interval_pct: u32) -> Self {
        Self {
            start_instant: std::time::Instant::now(),
            total_sim_time,
            last_reported_pct: 0,
            report_interval_pct,
            print_diagnostics: false,
            n_steps: 0,
        }
    }

    /// Enable diagnostic printing with progress reports.
    pub fn with_diagnostics(mut self) -> Self {
        self.print_diagnostics = true;
        self
    }

    /// Record a timestep.
    pub fn step(&mut self) {
        self.n_steps += 1;
    }

    /// Check and report progress if threshold reached.
    ///
    /// Returns true if progress was reported.
    pub fn maybe_report(&mut self, current_time: f64, diag: Option<&SWEDiagnostics2D>) -> bool {
        let pct = ((current_time / self.total_sim_time) * 100.0) as u32;
        let threshold = self.last_reported_pct + self.report_interval_pct;

        if pct >= threshold || (pct == 100 && self.last_reported_pct < 100) {
            self.report(current_time, diag);
            self.last_reported_pct = (pct / self.report_interval_pct) * self.report_interval_pct;
            true
        } else {
            false
        }
    }

    /// Force a progress report.
    pub fn report(&self, current_time: f64, diag: Option<&SWEDiagnostics2D>) {
        let elapsed = self.start_instant.elapsed();
        let pct = (current_time / self.total_sim_time) * 100.0;

        // Estimate remaining time
        let eta = if pct > 0.1 {
            let total_estimated = elapsed.as_secs_f64() * 100.0 / pct;
            let remaining = total_estimated - elapsed.as_secs_f64();
            format_duration(remaining)
        } else {
            "calculating...".to_string()
        };

        let steps_per_sec = if elapsed.as_secs_f64() > 0.0 {
            self.n_steps as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        print!(
            "\r[{:>5.1}%] t={:.1}s | elapsed={} | ETA={} | {:.0} steps/s",
            pct,
            current_time,
            format_duration(elapsed.as_secs_f64()),
            eta,
            steps_per_sec
        );

        if self.print_diagnostics
            && let Some(d) = diag
        {
            print!(" | {}", d.summary_line());
        }

        println!();
    }

    /// Print final summary.
    pub fn finish(&self, final_time: f64, tracker: Option<&DiagnosticsTracker>) {
        let elapsed = self.start_instant.elapsed();
        let steps_per_sec = self.n_steps as f64 / elapsed.as_secs_f64();

        println!();
        println!("=== Simulation Complete ===");
        println!(
            "Final time:    {:.2} s ({:.2} hours)",
            final_time,
            final_time / 3600.0
        );
        println!(
            "Wall time:     {} ({:.1} steps/s)",
            format_duration(elapsed.as_secs_f64()),
            steps_per_sec
        );
        println!("Total steps:   {}", self.n_steps);

        if let Some(t) = tracker {
            println!();
            t.print_summary();
        }
    }
}

/// Format a duration in seconds as human-readable string.
fn format_duration(secs: f64) -> String {
    if secs < 60.0 {
        format!("{:.1}s", secs)
    } else if secs < 3600.0 {
        let mins = (secs / 60.0).floor();
        let s = secs - mins * 60.0;
        format!("{:.0}m{:.0}s", mins, s)
    } else {
        let hours = (secs / 3600.0).floor();
        let mins = ((secs - hours * 3600.0) / 60.0).floor();
        format!("{:.0}h{:.0}m", hours, mins)
    }
}

/// Compute total mass (integral of h) for 2D SWE solution.
pub fn total_mass_2d(
    q: &SystemSolution2D<3>,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
) -> f64 {
    let n_nodes = ops.n_nodes;
    let mut mass = 0.0;

    for k in ElementIndex::iter(q.n_elements) {
        for i in 0..n_nodes {
            let det_j = geom.jacobian(k.as_usize(), i);
            let h = q.get_state(k, i).h;
            let w = compute_weight_2d(ops, i);
            mass += w * h * det_j;
        }
    }

    mass
}

/// Compute total momentum for 2D SWE solution.
///
/// Returns (momentum_x, momentum_y).
pub fn total_momentum_2d(
    q: &SystemSolution2D<3>,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
) -> (f64, f64) {
    let n_nodes = ops.n_nodes;
    let mut mom_x = 0.0;
    let mut mom_y = 0.0;

    for k in ElementIndex::iter(q.n_elements) {
        for i in 0..n_nodes {
            let det_j = geom.jacobian(k.as_usize(), i);
            let state = q.get_state(k, i);
            let w = compute_weight_2d(ops, i);
            mom_x += w * state.hu * det_j;
            mom_y += w * state.hv * det_j;
        }
    }

    (mom_x, mom_y)
}

/// Compute total energy for 2D SWE solution.
///
/// Energy = integral of (0.5 * h * (u² + v²) + 0.5 * g * h²).
pub fn total_energy_2d(
    q: &SystemSolution2D<3>,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    g: f64,
) -> f64 {
    let n_nodes = ops.n_nodes;
    let h_min = 1e-10;
    let mut energy = 0.0;

    for k in ElementIndex::iter(q.n_elements) {
        for i in 0..n_nodes {
            let det_j = geom.jacobian(k.as_usize(), i);
            let state = q.get_state(k, i);
            let h = state.h;
            let w = compute_weight_2d(ops, i);

            if h > h_min {
                let u = state.hu / h;
                let v = state.hv / h;
                let ke = 0.5 * h * (u * u + v * v);
                let pe = 0.5 * g * h * h;
                energy += w * (ke + pe) * det_j;
            }
        }
    }

    energy
}

/// Compute the current CFL number for 2D SWE.
pub fn current_cfl_2d(
    q: &SystemSolution2D<3>,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    g: f64,
    dt: f64,
) -> f64 {
    let n_nodes = ops.n_nodes;
    let h_min = 1e-10;
    let mut max_wave_speed = 0.0;

    for k in ElementIndex::iter(q.n_elements) {
        for i in 0..n_nodes {
            let state = q.get_state(k, i);
            let h = state.h;

            if h > h_min {
                let u = state.hu / h;
                let v = state.hv / h;
                let speed = (u * u + v * v).sqrt();
                let c = (g * h).sqrt();
                let wave_speed = speed + c;
                if wave_speed > max_wave_speed {
                    max_wave_speed = wave_speed;
                }
            }
        }
    }

    let min_h_elem = mesh.h_min();
    let dg_factor = (2 * ops.order + 1) as f64;

    if min_h_elem > 0.0 {
        max_wave_speed * dt * dg_factor / min_h_elem
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh2D;
    use crate::operators::{DGOperators2D, GeometricFactors2D};
    use crate::solver::{SWEState2D, SystemSolution2D};
    use crate::types::ElementIndex;

    fn k(idx: usize) -> ElementIndex {
        ElementIndex::new(idx)
    }

    fn setup() -> (
        Mesh2D,
        DGOperators2D,
        GeometricFactors2D,
        SystemSolution2D<3>,
    ) {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let q = SystemSolution2D::new(mesh.n_elements, ops.n_nodes);
        (mesh, ops, geom, q)
    }

    #[test]
    fn test_diagnostics_still_water() {
        let (mesh, ops, geom, mut q) = setup();
        let g = 9.81;
        let h0 = 10.0;

        // Set uniform still water
        for k in ElementIndex::iter(q.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(h0, 0.0, 0.0));
            }
        }

        let diag = SWEDiagnostics2D::compute(&q, &mesh, &ops, &geom, g, 0.1);

        // Mass should be h0 * area = 10 * 1 = 10
        assert!(
            (diag.total_mass - 10.0).abs() < 0.01,
            "Mass: {}",
            diag.total_mass
        );

        // No velocity
        assert!(diag.max_velocity < 1e-10);

        // No momentum
        assert!(diag.momentum_x.abs() < 1e-10);
        assert!(diag.momentum_y.abs() < 1e-10);

        // Only potential energy
        assert!(diag.kinetic_energy < 1e-10);
        assert!(diag.potential_energy > 0.0);

        // Froude = 0 for still water
        assert!(diag.max_froude < 1e-10);
    }

    #[test]
    fn test_diagnostics_with_flow() {
        let (mesh, ops, geom, mut q) = setup();
        let g = 9.81;
        let h0 = 10.0;
        let u0 = 1.0;

        // Set uniform flow in x-direction
        for k in ElementIndex::iter(q.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(h0, h0 * u0, 0.0));
            }
        }

        let diag = SWEDiagnostics2D::compute(&q, &mesh, &ops, &geom, g, 0.1);

        // Velocity should be 1.0
        assert!(
            (diag.max_velocity - 1.0).abs() < 0.01,
            "Velocity: {}",
            diag.max_velocity
        );

        // X-momentum should be h0 * u0 * area = 10 * 1 * 1 = 10
        assert!(
            (diag.momentum_x - 10.0).abs() < 0.01,
            "Momentum: {}",
            diag.momentum_x
        );

        // Kinetic energy should be positive
        assert!(diag.kinetic_energy > 0.0);

        // Froude = u / sqrt(gh) = 1 / sqrt(9.81 * 10) ≈ 0.1
        let expected_froude = u0 / (g * h0).sqrt();
        assert!(
            (diag.max_froude - expected_froude).abs() < 0.01,
            "Froude: {} vs expected {}",
            diag.max_froude,
            expected_froude
        );
    }

    #[test]
    fn test_tracker_conservation() {
        let (mesh, ops, geom, mut q) = setup();
        let g = 9.81;
        let h0 = 10.0;

        for k in ElementIndex::iter(q.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(h0, 0.0, 0.0));
            }
        }

        let initial = SWEDiagnostics2D::compute(&q, &mesh, &ops, &geom, g, 0.1);
        let mut tracker = DiagnosticsTracker::new(initial);

        // Simulate some small change
        for k in ElementIndex::iter(q.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(h0 * 1.001, 0.0, 0.0)); // 0.1% mass increase
            }
        }

        let current = SWEDiagnostics2D::compute(&q, &mesh, &ops, &geom, g, 0.1);
        tracker.update(100.0, current);

        // Mass error should be about 0.1%
        assert!(
            (tracker.mass_error() - 0.001).abs() < 0.0001,
            "Mass error: {}",
            tracker.mass_error()
        );
    }

    #[test]
    fn test_tracker_stability_check() {
        let (mesh, ops, geom, mut q) = setup();
        let g = 9.81;

        // Normal case - should be stable
        for k in ElementIndex::iter(q.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(10.0, 10.0, 5.0)); // u=1, v=0.5
            }
        }
        let diag = SWEDiagnostics2D::compute(&q, &mesh, &ops, &geom, g, 0.001);
        let tracker = DiagnosticsTracker::new(diag);
        assert!(tracker.is_stable());

        // Unrealistic velocity - should be unstable
        for k in ElementIndex::iter(q.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(10.0, 1500.0, 0.0)); // u=150 m/s
            }
        }
        let diag = SWEDiagnostics2D::compute(&q, &mesh, &ops, &geom, g, 0.001);
        let tracker = DiagnosticsTracker::new(diag);
        assert!(!tracker.is_stable());
    }

    #[test]
    fn test_individual_functions() {
        let (mesh, ops, geom, mut q) = setup();
        let g = 9.81;
        let h0 = 10.0;

        for k in ElementIndex::iter(q.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(h0, h0, h0 * 0.5)); // u=1, v=0.5
            }
        }

        let mass = total_mass_2d(&q, &ops, &geom);
        let (mom_x, mom_y) = total_momentum_2d(&q, &ops, &geom);
        let energy = total_energy_2d(&q, &ops, &geom, g);
        let cfl = current_cfl_2d(&q, &mesh, &ops, g, 0.001);

        assert!((mass - 10.0).abs() < 0.01);
        assert!((mom_x - 10.0).abs() < 0.01);
        assert!((mom_y - 5.0).abs() < 0.01);
        assert!(energy > 0.0);
        assert!(cfl > 0.0 && cfl < 1.0); // Small dt should give small CFL
    }

    #[test]
    fn test_progress_reporter() {
        let reporter = ProgressReporter::new(3600.0, 10);
        assert_eq!(reporter.n_steps, 0);
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30.0), "30.0s");
        assert_eq!(format_duration(90.0), "1m30s");
        assert_eq!(format_duration(3700.0), "1h1m");
    }

    #[test]
    fn test_summary_line() {
        let (mesh, ops, geom, mut q) = setup();
        let g = 9.81;

        for ki in 0..q.n_elements {
            for i in 0..ops.n_nodes {
                q.set_state(k(ki), i, SWEState2D::new(10.0, 10.0, 5.0));
            }
        }

        let diag = SWEDiagnostics2D::compute(&q, &mesh, &ops, &geom, g, 0.01);
        let summary = diag.summary_line();

        assert!(summary.contains("M="));
        assert!(summary.contains("E="));
        assert!(summary.contains("CFL="));
    }
}
