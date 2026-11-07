//! Stability monitoring for shallow water simulations.
//!
//! Provides reusable diagnostics for detecting numerical instability,
//! particularly in tidal simulations where Flather BCs or steep bathymetry
//! can cause blow-up.
//!
//! # Example
//!
//! ```ignore
//! use dgcom::analysis::{StabilityMonitor, StabilityThresholds};
//!
//! let mut monitor = StabilityMonitor::new(StabilityThresholds::tidal_default());
//!
//! // In time stepping loop:
//! let status = monitor.check(&q, dt);
//! if !status.is_stable {
//!     for warning in &status.warnings {
//!         eprintln!("{:?}", warning);
//!     }
//!     if monitor.should_stop() {
//!         break;
//!     }
//! }
//! ```

use crate::solver::SWESolution2D;
use crate::types::ElementIndex;

/// Thresholds for stability monitoring.
#[derive(Debug, Clone, Copy)]
pub struct StabilityThresholds {
    /// Maximum allowed water depth (m).
    pub max_depth: f64,
    /// Minimum allowed water depth (m). Negative values indicate below bed.
    pub min_depth: f64,
    /// Maximum velocity magnitude (m/s).
    pub max_velocity: f64,
    /// Minimum timestep before warning (s).
    pub min_dt: f64,
    /// Maximum consecutive warnings before recommending stop.
    pub max_consecutive_warnings: usize,
}

impl Default for StabilityThresholds {
    fn default() -> Self {
        Self::tidal_default()
    }
}

impl StabilityThresholds {
    /// Default thresholds for tidal simulations.
    ///
    /// - max_depth: 1000 m (reasonable for shelf seas)
    /// - min_depth: 0.0 m (no negative depths)
    /// - max_velocity: 50 m/s (very high, catches blow-up)
    /// - min_dt: 1e-3 s (1 ms)
    /// - max_consecutive_warnings: 10
    pub fn tidal_default() -> Self {
        Self {
            max_depth: 1000.0,
            min_depth: 0.0,
            max_velocity: 50.0,
            min_dt: 1e-3,
            max_consecutive_warnings: 10,
        }
    }

    /// Strict thresholds for detecting issues early.
    ///
    /// - max_depth: 200 m
    /// - min_depth: 0.001 m
    /// - max_velocity: 20 m/s
    /// - min_dt: 0.01 s
    /// - max_consecutive_warnings: 3
    pub fn strict() -> Self {
        Self {
            max_depth: 200.0,
            min_depth: 0.001,
            max_velocity: 20.0,
            min_dt: 0.01,
            max_consecutive_warnings: 3,
        }
    }

    /// Relaxed thresholds for exploratory runs.
    ///
    /// Only catches catastrophic blow-up.
    pub fn relaxed() -> Self {
        Self {
            max_depth: 10000.0,
            min_depth: -1.0,
            max_velocity: 1000.0,
            min_dt: 1e-6,
            max_consecutive_warnings: 100,
        }
    }

    /// Set maximum depth threshold.
    pub fn with_max_depth(mut self, max_depth: f64) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set minimum depth threshold.
    pub fn with_min_depth(mut self, min_depth: f64) -> Self {
        self.min_depth = min_depth;
        self
    }

    /// Set maximum velocity threshold.
    pub fn with_max_velocity(mut self, max_velocity: f64) -> Self {
        self.max_velocity = max_velocity;
        self
    }

    /// Set minimum timestep threshold.
    pub fn with_min_dt(mut self, min_dt: f64) -> Self {
        self.min_dt = min_dt;
        self
    }
}

/// Types of stability warnings.
#[derive(Debug, Clone, PartialEq)]
pub enum StabilityWarning {
    /// Water depth exceeds maximum threshold.
    DepthExceedsMax {
        element: usize,
        node: usize,
        value: f64,
        threshold: f64,
    },
    /// Water depth below minimum threshold.
    DepthBelowMin {
        element: usize,
        node: usize,
        value: f64,
        threshold: f64,
    },
    /// Velocity magnitude exceeds threshold.
    VelocityExceedsMax {
        element: usize,
        node: usize,
        value: f64,
        threshold: f64,
    },
    /// Timestep dropped below minimum.
    TimestepBelowMin { value: f64, threshold: f64 },
    /// Non-finite values detected (NaN or Inf).
    NonFiniteValue { element: usize, node: usize },
    /// Solution has blown up catastrophically.
    SolutionBlowUp,
}

impl std::fmt::Display for StabilityWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DepthExceedsMax {
                element,
                node,
                value,
                threshold,
            } => write!(
                f,
                "Depth exceeds max: h={:.2}m > {:.2}m at element {}, node {}",
                value, threshold, element, node
            ),
            Self::DepthBelowMin {
                element,
                node,
                value,
                threshold,
            } => write!(
                f,
                "Depth below min: h={:.6}m < {:.6}m at element {}, node {}",
                value, threshold, element, node
            ),
            Self::VelocityExceedsMax {
                element,
                node,
                value,
                threshold,
            } => write!(
                f,
                "Velocity exceeds max: |u|={:.2}m/s > {:.2}m/s at element {}, node {}",
                value, threshold, element, node
            ),
            Self::TimestepBelowMin { value, threshold } => {
                write!(
                    f,
                    "Timestep below min: dt={:.2e}s < {:.2e}s",
                    value, threshold
                )
            }
            Self::NonFiniteValue { element, node } => {
                write!(f, "Non-finite value at element {}, node {}", element, node)
            }
            Self::SolutionBlowUp => write!(f, "SOLUTION BLOW-UP DETECTED"),
        }
    }
}

/// Current stability status of the solution.
#[derive(Debug, Clone)]
pub struct StabilityStatus {
    /// Water depth range (min, max).
    pub h_range: (f64, f64),
    /// Maximum velocity magnitude.
    pub max_velocity: f64,
    /// Current timestep.
    pub dt: f64,
    /// Whether the solution is stable.
    pub is_stable: bool,
    /// List of warnings detected.
    pub warnings: Vec<StabilityWarning>,
}

impl StabilityStatus {
    /// Check if any warnings were generated.
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Check if any critical warnings (blow-up, non-finite) were detected.
    pub fn has_critical_warnings(&self) -> bool {
        self.warnings.iter().any(|w| {
            matches!(
                w,
                StabilityWarning::SolutionBlowUp | StabilityWarning::NonFiniteValue { .. }
            )
        })
    }
}

/// Monitor for tracking solution stability.
#[derive(Debug, Clone)]
pub struct StabilityMonitor {
    thresholds: StabilityThresholds,
    consecutive_warnings: usize,
    total_checks: usize,
    total_warnings: usize,
    last_status: Option<StabilityStatus>,
}

impl StabilityMonitor {
    /// Create a new stability monitor with the given thresholds.
    pub fn new(thresholds: StabilityThresholds) -> Self {
        Self {
            thresholds,
            consecutive_warnings: 0,
            total_checks: 0,
            total_warnings: 0,
            last_status: None,
        }
    }

    /// Get the thresholds being used.
    pub fn thresholds(&self) -> &StabilityThresholds {
        &self.thresholds
    }

    /// Get the number of consecutive warnings.
    pub fn consecutive_warnings(&self) -> usize {
        self.consecutive_warnings
    }

    /// Get the total number of checks performed.
    pub fn total_checks(&self) -> usize {
        self.total_checks
    }

    /// Get the total number of warnings generated.
    pub fn total_warnings(&self) -> usize {
        self.total_warnings
    }

    /// Get the last stability status.
    pub fn last_status(&self) -> Option<&StabilityStatus> {
        self.last_status.as_ref()
    }

    /// Check the solution for stability issues.
    ///
    /// Returns a `StabilityStatus` with current diagnostics and any warnings.
    pub fn check(&mut self, q: &SWESolution2D, dt: f64) -> StabilityStatus {
        self.total_checks += 1;

        let mut warnings = Vec::new();
        let mut max_h = f64::NEG_INFINITY;
        let mut min_h = f64::INFINITY;
        let mut max_vel = 0.0_f64;
        let mut found_blow_up = false;

        // Scan solution for issues
        for k in ElementIndex::iter(q.n_elements) {
            let ki = k.as_usize();
            for i in 0..q.n_nodes {
                let state = q.get_state(k, i);

                // Check for non-finite values
                if !state.h.is_finite() || !state.hu.is_finite() || !state.hv.is_finite() {
                    warnings.push(StabilityWarning::NonFiniteValue {
                        element: ki,
                        node: i,
                    });
                    found_blow_up = true;
                    continue;
                }

                // Track depth range
                max_h = max_h.max(state.h);
                min_h = min_h.min(state.h);

                // Check depth thresholds
                if state.h > self.thresholds.max_depth {
                    warnings.push(StabilityWarning::DepthExceedsMax {
                        element: ki,
                        node: i,
                        value: state.h,
                        threshold: self.thresholds.max_depth,
                    });
                }
                if state.h < self.thresholds.min_depth {
                    warnings.push(StabilityWarning::DepthBelowMin {
                        element: ki,
                        node: i,
                        value: state.h,
                        threshold: self.thresholds.min_depth,
                    });
                }

                // Check velocity
                if state.h > 1e-6 {
                    let u = state.hu / state.h;
                    let v = state.hv / state.h;
                    let vel = (u * u + v * v).sqrt();
                    max_vel = max_vel.max(vel);

                    if vel > self.thresholds.max_velocity {
                        warnings.push(StabilityWarning::VelocityExceedsMax {
                            element: ki,
                            node: i,
                            value: vel,
                            threshold: self.thresholds.max_velocity,
                        });
                    }
                }
            }
        }

        // Check timestep
        if !dt.is_finite() {
            warnings.push(StabilityWarning::NonFiniteValue {
                element: 0,
                node: 0,
            });
            found_blow_up = true;
        } else if dt < self.thresholds.min_dt {
            warnings.push(StabilityWarning::TimestepBelowMin {
                value: dt,
                threshold: self.thresholds.min_dt,
            });
        }

        // Check for catastrophic blow-up
        if found_blow_up || max_h > 1e6 || max_vel > 1e6 {
            warnings.push(StabilityWarning::SolutionBlowUp);
        }

        // Update tracking
        let is_stable = warnings.is_empty();
        if is_stable {
            self.consecutive_warnings = 0;
        } else {
            self.consecutive_warnings += 1;
            self.total_warnings += warnings.len();
        }

        let status = StabilityStatus {
            h_range: (min_h, max_h),
            max_velocity: max_vel,
            dt,
            is_stable,
            warnings,
        };

        self.last_status = Some(status.clone());
        status
    }

    /// Check if the simulation should be stopped based on warning history.
    pub fn should_stop(&self) -> bool {
        // Stop if consecutive warnings exceed threshold
        if self.consecutive_warnings >= self.thresholds.max_consecutive_warnings {
            return true;
        }

        // Stop if last status had critical warnings
        if let Some(status) = &self.last_status
            && status.has_critical_warnings()
        {
            return true;
        }

        false
    }

    /// Get suggested remediation actions based on warnings.
    pub fn suggest_remediation(&self) -> Vec<String> {
        let mut suggestions = Vec::new();

        if let Some(status) = &self.last_status {
            for warning in &status.warnings {
                match warning {
                    StabilityWarning::DepthExceedsMax { .. } => {
                        suggestions.push(
                            "Depth exceeds threshold: Check bathymetry convention (should be negative below MSL)".to_string()
                        );
                    }
                    StabilityWarning::VelocityExceedsMax { .. } => {
                        suggestions.push(
                            "Velocity exceeds threshold: Check boundary conditions, consider using sponge layers".to_string()
                        );
                    }
                    StabilityWarning::TimestepBelowMin { .. } => {
                        suggestions.push(
                            "Timestep too small: Solution may be unstable, reduce CFL or increase h_min".to_string()
                        );
                    }
                    StabilityWarning::NonFiniteValue { .. } | StabilityWarning::SolutionBlowUp => {
                        suggestions.push(
                            "Solution blow-up: Check bathymetry convention for Flather BCs"
                                .to_string(),
                        );
                        suggestions.push(
                            "Consider using HarmonicTidal2D instead of HarmonicFlather2D for closed basins".to_string()
                        );
                        suggestions.push("Add sponge layers to absorb reflected waves".to_string());
                    }
                    StabilityWarning::DepthBelowMin { .. } => {
                        suggestions.push(
                            "Depth below threshold: Check wetting/drying treatment".to_string(),
                        );
                    }
                }
            }
        }

        // Deduplicate
        suggestions.sort();
        suggestions.dedup();
        suggestions
    }

    /// Print a formatted stability report.
    pub fn print_report(&self, time: f64, step: usize) {
        if let Some(status) = &self.last_status
            && !status.is_stable
        {
            eprintln!(
                "\nSTABILITY WARNING at t={:.2}h, step {}:",
                time / 3600.0,
                step
            );
            eprintln!("  h: [{:.2}, {:.2}] m", status.h_range.0, status.h_range.1);
            eprintln!("  max |u|: {:.2} m/s", status.max_velocity);
            eprintln!("  dt: {:.2e} s", status.dt);
            eprintln!("  Warnings ({}):", status.warnings.len());
            for warning in &status.warnings {
                eprintln!("    - {}", warning);
            }
            if self.should_stop() {
                eprintln!("\n  RECOMMENDATION: Stop simulation");
                for suggestion in self.suggest_remediation() {
                    eprintln!("    * {}", suggestion);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::SWEState2D;
    use crate::types::ElementIndex;

    fn make_solution(n_elements: usize, n_nodes: usize, h: f64, hu: f64, hv: f64) -> SWESolution2D {
        let mut q = SWESolution2D::new(n_elements, n_nodes);
        for k in ElementIndex::iter(n_elements) {
            for i in 0..n_nodes {
                q.set_state(k, i, SWEState2D::new(h, hu, hv));
            }
        }
        q
    }

    #[test]
    fn test_tidal_thresholds() {
        let t = StabilityThresholds::tidal_default();
        assert!((t.max_depth - 1000.0).abs() < 1e-10);
        assert!((t.max_velocity - 50.0).abs() < 1e-10);
        assert!((t.min_dt - 1e-3).abs() < 1e-10);
    }

    #[test]
    fn test_stable_solution() {
        let q = make_solution(4, 9, 50.0, 5.0, 0.0); // h=50, u=0.1 m/s
        let mut monitor = StabilityMonitor::new(StabilityThresholds::tidal_default());

        let status = monitor.check(&q, 1.0);
        assert!(status.is_stable);
        assert!(status.warnings.is_empty());
        assert!(!monitor.should_stop());
    }

    #[test]
    fn test_detect_velocity_blowup() {
        let q = make_solution(4, 9, 50.0, 5000.0, 0.0); // u = 100 m/s
        let mut monitor = StabilityMonitor::new(StabilityThresholds::tidal_default());

        let status = monitor.check(&q, 1.0);
        assert!(!status.is_stable);
        assert!(
            status
                .warnings
                .iter()
                .any(|w| matches!(w, StabilityWarning::VelocityExceedsMax { .. }))
        );
    }

    #[test]
    fn test_detect_depth_anomaly() {
        let q = make_solution(4, 9, 5000.0, 0.0, 0.0); // h = 5000 m
        let mut monitor = StabilityMonitor::new(StabilityThresholds::tidal_default());

        let status = monitor.check(&q, 1.0);
        assert!(!status.is_stable);
        assert!(
            status
                .warnings
                .iter()
                .any(|w| matches!(w, StabilityWarning::DepthExceedsMax { .. }))
        );
    }

    #[test]
    fn test_suggest_remediation() {
        let q = make_solution(4, 9, 50.0, 5000.0, 0.0); // blowing up
        let mut monitor = StabilityMonitor::new(StabilityThresholds::tidal_default());
        monitor.check(&q, 1.0);

        let suggestions = monitor.suggest_remediation();
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.contains("sponge")));
    }

    #[test]
    fn test_consecutive_warnings() {
        let q_bad = make_solution(4, 9, 50.0, 5000.0, 0.0);
        let mut monitor =
            StabilityMonitor::new(StabilityThresholds::tidal_default().with_max_velocity(10.0));

        // Generate consecutive warnings
        for _ in 0..5 {
            monitor.check(&q_bad, 1.0);
        }
        assert_eq!(monitor.consecutive_warnings(), 5);
        assert!(!monitor.should_stop()); // Need 10 by default

        for _ in 0..5 {
            monitor.check(&q_bad, 1.0);
        }
        assert!(monitor.should_stop()); // Now at 10
    }
}
