//! Simulation runner implementation.
//!
//! Provides a high-level interface for running time-dependent simulations.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::physics::PhysicsModule;
use crate::time::{Integrable, MultirateStats, MultirateStepper, StageWorkspace, TimeIntegrator};

// =============================================================================
// Simulation Configuration
// =============================================================================

/// Configuration for a simulation run.
#[derive(Clone, Debug)]
pub struct SimulationConfig {
    /// CFL number for time step calculation: the linear stability bound. The
    /// physics module applies its forward-Euler bounds times the integrator's
    /// SSP coefficient (`PhysicsModule::compute_dt_ssp`).
    pub cfl: f64,
    /// Maximum time step (overrides CFL if smaller).
    pub dt_max: Option<f64>,
    /// Minimum time step (simulation fails if dt drops below this).
    pub dt_min: Option<f64>,
    /// Interval for calling callbacks (in simulation time units).
    pub callback_interval: Option<f64>,
    /// Maximum number of time steps.
    pub max_steps: Option<usize>,
    /// Whether to print progress to stdout.
    pub verbose: bool,
    /// Run an integrator with an SSP scheme (SSP-RK3, SSP-RK(4,3)) through
    /// the physics module's fused per-element stages when it supports local
    /// time stepping (see `IntegratorInfo::ssp_scheme`). The result is the
    /// same bit for bit; `false` runs the whole-state stages instead.
    pub fused_stages: bool,
    /// Set from another thread to end the run before its next step (e.g. by a
    /// viewer that no longer shows it); the run then returns as stopped.
    pub stop: Option<Arc<AtomicBool>>,
}

impl SimulationConfig {
    /// The result of a run that `stop` ended at `t`, if it did.
    pub(crate) fn stopped(&self, t: f64, n_steps: usize) -> Option<SimulationResult> {
        self.stop
            .as_ref()
            .filter(|stop| stop.load(Ordering::Relaxed))
            .map(|_| SimulationResult::failure(t, n_steps, format!("Stopped at t = {t:.1} s")))
    }
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            cfl: 0.5,
            dt_max: None,
            dt_min: None,
            callback_interval: None,
            max_steps: None,
            verbose: false,
            fused_stages: true,
            stop: None,
        }
    }
}

// =============================================================================
// Simulation Result
// =============================================================================

/// Result of a simulation run.
#[derive(Clone, Debug)]
pub struct SimulationResult {
    /// Final simulation time reached.
    pub final_time: f64,
    /// Total number of time steps taken.
    pub n_steps: usize,
    /// Minimum time step used.
    pub dt_min: f64,
    /// Maximum time step used.
    pub dt_max: f64,
    /// Total wall-clock time in seconds.
    pub wall_time: f64,
    /// Whether the simulation completed successfully.
    pub success: bool,
    /// Error message if simulation failed.
    pub error: Option<String>,
    /// Work counts of local time stepping (a multirate integrator), if used.
    pub local_time_stepping: Option<MultirateStats>,
}

impl SimulationResult {
    /// Create a successful result.
    pub fn success(
        final_time: f64,
        n_steps: usize,
        dt_min: f64,
        dt_max: f64,
        wall_time: f64,
    ) -> Self {
        Self {
            final_time,
            n_steps,
            dt_min,
            dt_max,
            wall_time,
            success: true,
            error: None,
            local_time_stepping: None,
        }
    }

    /// Create a failed result.
    pub fn failure(final_time: f64, n_steps: usize, error: String) -> Self {
        Self {
            final_time,
            n_steps,
            dt_min: f64::INFINITY,
            dt_max: 0.0,
            wall_time: 0.0,
            success: false,
            error: Some(error),
            local_time_stepping: None,
        }
    }
}

// =============================================================================
// Simulation Runner
// =============================================================================

/// Relative slack in landing on a target time: a step that ends within this
/// fraction of itself before the target is stretched to land on it.
pub(crate) const LANDING_SLACK: f64 = 1e-9;

/// High-level simulation runner.
///
/// Ties together physics modules and time integrators into a complete
/// simulation workflow with diagnostics and callbacks.
///
/// # Type Parameters
///
/// * `S` - Solution type (must implement [`Integrable`])
/// * `P` - Physics module (must implement [`PhysicsModule<S>`])
/// * `I` - Time integrator (must implement [`TimeIntegrator<S>`])
pub struct Simulation<S, P, I>
where
    S: Integrable,
    P: PhysicsModule<S>,
    I: TimeIntegrator<S>,
{
    physics: P,
    integrator: I,
    config: SimulationConfig,
    _marker: std::marker::PhantomData<S>,
}

impl<S, P, I> Simulation<S, P, I>
where
    S: Integrable,
    P: PhysicsModule<S>,
    I: TimeIntegrator<S>,
{
    /// Create a new simulation with the given physics and integrator.
    pub fn new(physics: P, integrator: I) -> Self {
        Self {
            physics,
            integrator,
            config: SimulationConfig::default(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the CFL number.
    pub fn with_cfl(mut self, cfl: f64) -> Self {
        self.config.cfl = cfl;
        self
    }

    /// Whether to run SSP schemes through the fused per-element stages (default
    /// `true`; see [`SimulationConfig::fused_stages`]).
    pub fn with_fused_stages(mut self, fused: bool) -> Self {
        self.config.fused_stages = fused;
        self
    }

    /// Set the maximum time step.
    pub fn with_dt_max(mut self, dt_max: f64) -> Self {
        self.config.dt_max = Some(dt_max);
        self
    }

    /// Set the minimum time step (simulation fails if dt drops below).
    pub fn with_dt_min(mut self, dt_min: f64) -> Self {
        self.config.dt_min = Some(dt_min);
        self
    }

    /// Call back at `t_start + k·interval` exactly (the step before each
    /// callback is shortened to land on it), e.g. for regularly sampled
    /// station series and output times. Without an interval, every step.
    pub fn with_callback_interval(mut self, interval: f64) -> Self {
        assert!(interval > 0.0, "callback interval must be positive");
        self.config.callback_interval = Some(interval);
        self
    }

    /// Set the maximum number of steps.
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.config.max_steps = Some(max_steps);
        self
    }

    /// End the run before the next step once `stop` is set (see
    /// [`SimulationConfig::stop`]).
    pub fn with_stop(mut self, stop: Arc<AtomicBool>) -> Self {
        self.config.stop = Some(stop);
        self
    }

    /// Enable verbose output.
    pub fn verbose(mut self) -> Self {
        self.config.verbose = true;
        self
    }

    /// Get a reference to the physics module.
    pub fn physics(&self) -> &P {
        &self.physics
    }

    /// Get a mutable reference to the physics module, e.g. to replace a
    /// forcing between runs (see [`Self::run_with_exchange`]).
    pub fn physics_mut(&mut self) -> &mut P {
        &mut self.physics
    }

    /// Get a reference to the time integrator.
    pub fn integrator(&self) -> &I {
        &self.integrator
    }

    /// Run the simulation from `t_start` to `t_end`.
    ///
    /// # Arguments
    /// * `state` - Initial solution state (modified in place)
    /// * `t_start` - Starting time
    /// * `t_end` - Ending time
    ///
    /// # Returns
    /// Simulation result with timing and step statistics.
    pub fn run(&self, state: &mut S, t_start: f64, t_end: f64) -> SimulationResult {
        self.run_with_callback(state, t_start, t_end, |_, _| {})
    }

    /// Run the simulation with a callback function.
    ///
    /// The callback is called at the configured interval (or every step if not set).
    ///
    /// # Arguments
    /// * `state` - Initial solution state (modified in place)
    /// * `t_start` - Starting time
    /// * `t_end` - Ending time
    /// * `callback` - Function called with (state, time) at each callback point
    pub fn run_with_callback<F>(
        &self,
        state: &mut S,
        t_start: f64,
        t_end: f64,
        mut callback: F,
    ) -> SimulationResult
    where
        F: FnMut(&S, f64),
    {
        let start_wall = std::time::Instant::now();

        let mut t = t_start;
        let mut n_steps = 0;
        let mut dt_min_used = f64::INFINITY;
        let mut dt_max_used: f64 = 0.0;
        // Callbacks fall exactly on t_start + k·interval: the step before
        // one is shortened to land on it (k counts, so no drift accumulates)
        let mut callbacks_done = 0_u64;
        let next_callback = |k: u64| {
            self.config
                .callback_interval
                .map(|interval| t_start + (k + 1) as f64 * interval)
        };
        // Stage buffers reused for the whole run (no per-step allocation)
        let mut stages = StageWorkspace::new();
        // The physics module's forward-Euler bounds (e.g. the positivity
        // bound of a wet/dry scheme) hold up to the SSP coefficient times
        // theirs; the module applies them with the linear CFL
        let cfl = self.config.cfl;
        let scheme = self.integrator.ssp_scheme();
        let ssp_coefficient = self.integrator.ssp_coefficient();
        // Local time stepping: every element at its own power-of-two
        // fraction of the (coarse) step
        let mut multirate = self.integrator.max_local_levels().map(|max_levels| {
            let local = self.physics.local_time_stepping().unwrap_or_else(|| {
                panic!(
                    "the {} integrator needs a physics module with local time stepping; {} has none",
                    self.integrator.name(),
                    self.physics.name()
                )
            });
            let scheme = self.integrator.ssp_scheme().unwrap_or_else(|| {
                panic!("the {} integrator has no SSP scheme", self.integrator.name())
            });
            (local, MultirateStepper::new(max_levels, scheme))
        });
        // Global steps of an SSP scheme as one-level multirate steps: the
        // same result, with the RHS, stage combination, damping and
        // post-processing fused per element (one parallel pass per stage
        // instead of several)
        let mut fused = self
            .integrator
            .ssp_scheme()
            .filter(|_| multirate.is_none() && self.config.fused_stages)
            .and_then(|scheme| {
                let local = self.physics.local_time_stepping()?;
                local.is_element_local().then_some((local, scheme))
            })
            .map(|(local, scheme)| {
                let mut stepper = MultirateStepper::new(0, scheme);
                stepper.assign_one_level(self.physics.mesh().n_elements);
                (local, stepper)
            });

        // Call initial callback
        callback(state, t);

        if self.config.verbose {
            println!(
                "Starting simulation: {} with {} integrator",
                self.physics.name(),
                self.integrator.name()
            );
            println!("  t_start = {:.4}, t_end = {:.4}", t_start, t_end);
            if let Some(max) = self.physics.max_cfl()
                && ssp_coefficient * max < cfl
            {
                println!(
                    "  CFL {cfl} capped at {:.3} by {} where elements may run dry",
                    ssp_coefficient * max,
                    self.physics.name()
                );
            }
        }

        while t < t_end {
            // Check step limit
            if let Some(max_steps) = self.config.max_steps
                && n_steps >= max_steps
            {
                return SimulationResult::failure(
                    t,
                    n_steps,
                    format!("Maximum step limit ({}) reached", max_steps),
                );
            }
            if let Some(stopped) = self.config.stopped(t, n_steps) {
                return stopped;
            }

            // Compute time step (with local time stepping, assign the levels
            // and take the coarse step, which respects dt_max)
            let mut dt = match multirate.as_mut() {
                Some((local, stepper)) => stepper.assign_levels(
                    *local,
                    self.physics.mesh(),
                    state,
                    cfl,
                    self.config.dt_max,
                ),
                None => self.physics.compute_dt_ssp(state, cfl, scheme),
            };

            // Apply dt limits
            if let Some(dt_max) = self.config.dt_max {
                dt = dt.min(dt_max);
            }

            // Check minimum dt
            if let Some(dt_min) = self.config.dt_min
                && dt < dt_min
            {
                return SimulationResult::failure(
                    t,
                    n_steps,
                    format!("Time step ({:.2e}) below minimum ({:.2e})", dt, dt_min),
                );
            }

            // Don't overshoot the end time or the next callback time, and
            // don't leave a sliver of it: steps that should land exactly
            // (ten steps of 0.1 to 1.0) accumulate round-off, and the sliver
            // would cost a whole step. Landing stretches the step by at most
            // LANDING_SLACK of itself.
            let target = next_callback(callbacks_done).map_or(t_end, |c| c.min(t_end));
            let landed = t + dt * (1.0 + LANDING_SLACK) >= target;
            if landed {
                dt = target - t;
            }

            // Track dt statistics
            dt_min_used = dt_min_used.min(dt);
            dt_max_used = dt_max_used.max(dt);

            // Advance the solution. Stiff damping (friction, wet/dry
            // relaxation) is implicit in every RK stage; limiters
            // and wet/dry treatment run after every RK stage, so no RHS
            // evaluation sees an unlimited state.
            match multirate.as_mut().or(fused.as_mut()) {
                Some((local, stepper)) => stepper.step(*local, state, t, dt),
                None => self.integrator.step_with_relaxation(
                    state,
                    dt,
                    t,
                    |s, time, out| self.physics.compute_rhs_into(s, time, out),
                    |stage, from, dt| self.physics.implicit_damping(stage, from, dt),
                    |s| self.physics.post_process(s),
                    &mut stages,
                ),
            }

            // Exact landing: no round-off drift from t + (target − t)
            t = if landed { target } else { t + dt };
            n_steps += 1;

            // A blown-up step leaves NaN or infinities, which every later
            // step would carry on to `t_end` (the step's bound skips NaN)
            if !state.is_finite() {
                return SimulationResult::failure(
                    t,
                    n_steps,
                    format!("The state is not finite at t = {t:.1} s: it has blown up"),
                );
            }

            // Callback at the configured interval, or every step
            match next_callback(callbacks_done) {
                Some(c) if t >= c => {
                    callback(state, t);
                    callbacks_done += 1;
                }
                Some(_) => {}
                None => callback(state, t),
            }

            // Progress output
            if self.config.verbose && n_steps % 100 == 0 {
                println!("  Step {}: t = {:.4}, dt = {:.2e}", n_steps, t, dt);
            }
        }

        let wall_time = start_wall.elapsed().as_secs_f64();

        if self.config.verbose {
            println!("Simulation complete:");
            println!("  Steps: {}", n_steps);
            println!("  Wall time: {:.2}s", wall_time);
            println!("  dt range: [{:.2e}, {:.2e}]", dt_min_used, dt_max_used);
            if let Some((_, stepper)) = &multirate {
                let stats = stepper.stats();
                println!(
                    "  Local time stepping: time-step ratio up to 2^{}, {:.2}x less RHS work than global steps",
                    stats.finest_level,
                    stats.speedup()
                );
            }
        }

        let mut result = SimulationResult::success(t, n_steps, dt_min_used, dt_max_used, wall_time);
        result.local_time_stepping = multirate.map(|(_, stepper)| stepper.stats());
        result
    }

    /// Run from `t_start` to `t_end` in coupling intervals of `interval`,
    /// calling `exchange(physics, state, t, t_next)` at the start of each:
    /// a coupling to another model (the spectral waves,
    /// [`crate::waves::CoupledWaves2D`]) advances it over the interval and
    /// gives this physics module its forcing for the interval. Each interval
    /// is then a [`Self::run_with_callback`] that lands on `t_next`.
    ///
    /// `callback` is called as by [`Self::run_with_callback`] over the whole
    /// run: once at `t_start`, then at `t_start + k·callback_interval`, so the
    /// callback interval must divide `interval`. Without an exchange that
    /// changes the physics the run is [`Self::run_with_callback`]'s, bit for
    /// bit where the callback times are exact in floating point (otherwise
    /// they may differ in the last bit). The result's wall time includes the
    /// exchanges.
    ///
    /// # Panics
    /// If `interval` is not positive, or not a multiple of the callback
    /// interval.
    pub fn run_with_exchange<E, F>(
        &mut self,
        state: &mut S,
        t_start: f64,
        t_end: f64,
        interval: f64,
        mut exchange: E,
        mut callback: F,
    ) -> SimulationResult
    where
        E: FnMut(&mut P, &S, f64, f64),
        F: FnMut(&S, f64),
    {
        assert!(interval > 0.0, "coupling interval must be positive");
        if let Some(c) = self.config.callback_interval {
            let ratio = interval / c;
            assert!(
                ratio.round() >= 1.0 && (ratio - ratio.round()).abs() <= 1e-9 * ratio,
                "the callback interval {c} must divide the coupling interval {interval}"
            );
        }
        let start_wall = std::time::Instant::now();
        let mut total: Option<SimulationResult> = None;
        let mut t = t_start;
        let mut k = 0_u64;
        while t < t_end {
            k += 1;
            let mut t_next = t_start + k as f64 * interval;
            if t_next + LANDING_SLACK * interval >= t_end {
                t_next = t_end;
            }
            exchange(&mut self.physics, state, t, t_next);
            // Every interval but the first starts where the last one called
            // back already
            let mut skip = total.is_some();
            let result = self.run_with_callback(state, t, t_next, |s, time| {
                if !std::mem::take(&mut skip) {
                    callback(s, time);
                }
            });
            t = result.final_time;
            let success = result.success;
            total = Some(match total {
                None => result,
                Some(total) => total.then(result),
            });
            if !success {
                break;
            }
        }
        let mut total =
            total.unwrap_or_else(|| self.run_with_callback(state, t_start, t_end, &mut callback));
        total.wall_time = start_wall.elapsed().as_secs_f64();
        total
    }
}

impl SimulationResult {
    /// This run followed by `next`, which started where it ended.
    fn then(self, next: SimulationResult) -> SimulationResult {
        let local_time_stepping = match (self.local_time_stepping, next.local_time_stepping) {
            (Some(a), Some(b)) => Some(MultirateStats {
                steps: a.steps + b.steps,
                element_evaluations: a.element_evaluations + b.element_evaluations,
                global_evaluations: a.global_evaluations + b.global_evaluations,
                finest_level: a.finest_level.max(b.finest_level),
            }),
            (a, b) => a.or(b),
        };
        SimulationResult {
            final_time: next.final_time,
            n_steps: self.n_steps + next.n_steps,
            dt_min: self.dt_min.min(next.dt_min),
            dt_max: self.dt_max.max(next.dt_max),
            wall_time: self.wall_time + next.wall_time,
            success: self.success && next.success,
            error: self.error.or(next.error),
            local_time_stepping,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::boundary::Reflective2D;
    use crate::equations::ShallowWater2D;
    use crate::mesh::Mesh2D;
    use crate::operators::{DGOperators2D, GeometricFactors2D};
    use crate::physics::PhysicsBuilder;
    use crate::solver::SWESolution2D;
    use crate::types::{Depth, ElementIndex};

    fn k(idx: usize) -> ElementIndex {
        ElementIndex::new(idx)
    }
    use crate::time::SSPRK3;

    fn create_test_setup() -> (crate::physics::SWEPhysics2D<Reflective2D>, SWESolution2D) {
        let mesh = Arc::new(Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2));
        let ops = Arc::new(DGOperators2D::new(2));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let equation = ShallowWater2D::with_h_min(9.81, Depth::new(1e-6));
        let bc = Reflective2D::default();

        let physics = PhysicsBuilder::swe_2d(mesh.clone(), ops.clone(), geom, equation, bc).build();

        let mut state = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for ki in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                state.set_state(k(ki), i, crate::solver::SWEState2D::new(1.0, 0.0, 0.0));
            }
        }

        (physics, state)
    }

    #[test]
    fn test_simulation_basic() {
        let (physics, mut state) = create_test_setup();
        let integrator = SSPRK3;

        let sim = Simulation::new(physics, integrator).with_cfl(0.5);

        let result = sim.run(&mut state, 0.0, 0.01);

        assert!(result.success);
        assert!(result.n_steps > 0);
        assert!(result.final_time >= 0.01 - 1e-10);
    }

    #[test]
    fn test_simulation_with_callback() {
        let (physics, mut state) = create_test_setup();
        let integrator = SSPRK3;

        let sim = Simulation::new(physics, integrator).with_cfl(0.5);

        let mut callback_count = 0;
        let result = sim.run_with_callback(&mut state, 0.0, 0.01, |_state, _time| {
            callback_count += 1;
        });

        assert!(result.success);
        assert!(callback_count > 0);
    }

    /// Steps that should land on t_end exactly, up to round-off, take no
    /// extra sliver step: ten steps of 0.005 add up to 0.049999999999999996,
    /// and the run used to take an 11th step of 7e-18 s. (Sums of 3 and 100
    /// steps round up instead, and always landed.)
    #[test]
    fn round_off_leaves_no_sliver_step() {
        let (physics, mut state) = create_test_setup();
        let dt = 0.005; // below the CFL step (0.016)
        let sim = Simulation::new(physics, SSPRK3)
            .with_cfl(0.5)
            .with_dt_max(dt);
        for n in [3, 10, 100, 1000] {
            let result = sim.run(&mut state, 0.0, n as f64 * dt);
            assert_eq!(result.n_steps, n, "{n} steps of {dt}");
            assert_eq!(result.final_time, n as f64 * dt);
            assert!(
                result.dt_min > 0.999 * dt,
                "{n} steps: dt_min {}",
                result.dt_min
            );
        }
    }

    /// An interval that is not a multiple of dt: callbacks land exactly on
    /// t_start + k·interval (they used to fire at the first step past it and
    /// drift by up to one dt per interval), and the run still ends at t_end.
    #[test]
    fn callbacks_land_on_the_interval() {
        let (physics, mut state) = create_test_setup();
        let sim = Simulation::new(physics, SSPRK3)
            .with_cfl(0.5)
            .with_callback_interval(0.0037);
        let (t0, t_end) = (0.5, 0.52);
        let mut times = Vec::new();
        let result = sim.run_with_callback(&mut state, t0, t_end, |_, t| times.push(t));

        assert!(result.success);
        assert_eq!(result.final_time, t_end);
        assert_eq!(times.len(), 6, "{times:?}"); // t0 and 5 intervals
        for (k, &t) in times.iter().enumerate() {
            let expected = t0 + k as f64 * 0.0037;
            assert!(
                (t - expected).abs() < 1e-15,
                "callback {k} at {t}, expected {expected}"
            );
        }
    }

    /// A run ends before the step after its stop flag is set, at the time it had
    /// reached, and says it stopped.
    #[test]
    fn a_run_ends_when_its_stop_flag_is_set() {
        let (physics, mut state) = create_test_setup();
        let stop = Arc::new(AtomicBool::new(false));
        let sim = Simulation::new(physics, SSPRK3)
            .with_cfl(0.5)
            .with_callback_interval(0.002)
            .with_stop(stop.clone());
        let mut times = Vec::new();
        let result = sim.run_with_callback(&mut state, 0.0, 1.0, |_, t| {
            times.push(t);
            if times.len() == 3 {
                stop.store(true, Ordering::Relaxed);
            }
        });
        assert!(!result.success);
        assert_eq!(result.final_time, 0.004);
        assert_eq!(times, [0.0, 0.002, 0.004]);
        assert!(result.error.unwrap().starts_with("Stopped"));
    }

    #[test]
    fn test_simulation_max_steps() {
        let (physics, mut state) = create_test_setup();
        let integrator = SSPRK3;

        let sim = Simulation::new(physics, integrator)
            .with_cfl(0.5)
            .with_max_steps(5);

        let result = sim.run(&mut state, 0.0, 100.0);

        assert!(!result.success);
        assert_eq!(result.n_steps, 5);
    }

    /// dh/dt = −1 everywhere; `post_process` clamps h ≥ 0. Records the smallest
    /// depth any RHS evaluation saw and how often `post_process` ran.
    struct DrainingPhysics {
        mesh: Mesh2D,
        ops: DGOperators2D,
        geom: GeometricFactors2D,
        min_h_seen: std::sync::Mutex<f64>,
        post_process_calls: std::sync::atomic::AtomicUsize,
    }

    impl crate::physics::PhysicsModuleInfo for DrainingPhysics {
        fn name(&self) -> &'static str {
            "draining"
        }
        fn description(&self) -> &str {
            "test: uniform drain with a positivity clamp"
        }
        fn n_variables(&self) -> usize {
            3
        }
        fn variable_names(&self) -> &[&'static str] {
            &["h", "hu", "hv"]
        }
    }

    impl PhysicsModule<SWESolution2D> for DrainingPhysics {
        fn compute_rhs(&self, state: &SWESolution2D, _time: f64) -> SWESolution2D {
            let seen = state.h_data().iter().cloned().fold(f64::INFINITY, f64::min);
            let mut min_h = self.min_h_seen.lock().unwrap();
            *min_h = min_h.min(seen);
            let mut rhs = SWESolution2D::new(state.n_elements, state.n_nodes);
            rhs.h_data_mut().fill(-1.0);
            rhs
        }
        fn compute_dt(&self, _state: &SWESolution2D, _cfl: f64) -> f64 {
            1.0
        }
        fn post_process(&self, state: &mut SWESolution2D) {
            self.post_process_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            for h in state.h_data_mut() {
                *h = h.max(0.0);
            }
        }
        fn mesh(&self) -> &Mesh2D {
            &self.mesh
        }
        fn operators(&self) -> &DGOperators2D {
            &self.ops
        }
        fn geometry(&self) -> &GeometricFactors2D {
            &self.geom
        }
        fn order(&self) -> usize {
            self.ops.order
        }
    }

    #[test]
    fn test_simulation_limits_every_stage() {
        // P0.19 regression: `Simulation` called `post_process` once per step, so
        // the 2nd and 3rd SSP-RK3 stages evaluated the RHS on unlimited states
        // (here h = 0.5 − 1 = −0.5), voiding the Zhang–Shu positivity guarantee.
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 1, 1);
        let ops = DGOperators2D::new(1);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let mut state = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        state.h_data_mut().fill(0.5);
        let physics = DrainingPhysics {
            mesh,
            ops,
            geom,
            min_h_seen: std::sync::Mutex::new(f64::INFINITY),
            post_process_calls: std::sync::atomic::AtomicUsize::new(0),
        };

        let sim = Simulation::new(physics, SSPRK3);
        let result = sim.run(&mut state, 0.0, 2.0);

        assert!(result.success);
        assert_eq!(result.n_steps, 2);
        let physics = sim.physics();
        let calls = physics
            .post_process_calls
            .load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            calls,
            3 * result.n_steps,
            "post_process must run after every stage"
        );
        let min_h = *physics.min_h_seen.lock().unwrap();
        assert!(min_h >= 0.0, "an RHS evaluation saw h = {min_h}");
        assert!(state.h_data().iter().all(|&h| h >= 0.0));
    }

    /// Regression (P2.1): a blown-up 2D state stepped NaN on to `t_end` and
    /// reported success, as `Simulation3D` did before 2026-10-10. It now
    /// stops at the first step that leaves a non-finite value.
    #[test]
    fn a_blown_up_2d_run_fails() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 1, 1);
        let ops = DGOperators2D::new(1);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let mut state = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        state.h_data_mut().fill(5.0);
        state.hu_data_mut()[0] = f64::NAN;
        let physics = DrainingPhysics {
            mesh,
            ops,
            geom,
            min_h_seen: std::sync::Mutex::new(f64::INFINITY),
            post_process_calls: std::sync::atomic::AtomicUsize::new(0),
        };

        let result = Simulation::new(physics, SSPRK3).run(&mut state, 0.0, 4.0);
        assert!(!result.success, "a blown-up run reported success");
        assert_eq!(result.n_steps, 1, "it should stop after the first step");
        assert!(
            result
                .error
                .as_deref()
                .is_some_and(|e| e.contains("blown up")),
            "{:?}",
            result.error
        );
    }

    /// A run in coupling intervals whose exchange leaves the physics alone is
    /// the single run bit for bit: the same states, steps and callback times;
    /// the exchange sees each interval's start and end.
    #[test]
    fn a_run_in_coupling_intervals_is_the_single_run() {
        let wave = || {
            let (physics, mut state) = create_test_setup();
            for (p, h) in state.h_data_mut().iter_mut().enumerate() {
                *h += 0.1 * (0.7 * p as f64).sin();
            }
            (physics, state)
        };
        let (callback, interval, t_end) = (0.0625, 0.125, 0.5);

        let (physics, mut single) = wave();
        let sim = Simulation::new(physics, SSPRK3).with_callback_interval(callback);
        let mut single_times = Vec::new();
        let single_result = sim.run_with_callback(&mut single, 0.0, t_end, |_, t| {
            single_times.push(t);
        });

        let (physics, mut coupled) = wave();
        let mut sim = Simulation::new(physics, SSPRK3).with_callback_interval(callback);
        let (mut times, mut exchanges) = (Vec::new(), Vec::new());
        let result = sim.run_with_exchange(
            &mut coupled,
            0.0,
            t_end,
            interval,
            |_, _, t, t_next| exchanges.push((t, t_next)),
            |_, t| times.push(t),
        );

        assert!(result.success && single_result.success);
        assert_eq!(result.final_time, t_end);
        assert_eq!(result.n_steps, single_result.n_steps);
        assert_eq!(times, single_times);
        assert_eq!(
            exchanges,
            [(0.0, 0.125), (0.125, 0.25), (0.25, 0.375), (0.375, 0.5)]
        );
        assert_eq!(coupled.h_data(), single.h_data());
        assert_eq!(coupled.hu_data(), single.hu_data());
        assert_eq!(coupled.hv_data(), single.hv_data());
    }

    /// A 2D beach (bed −8 → +1.5 m) sloshing up and down its shore under
    /// `WetDry`, with Coriolis and Smagorinsky's viscosity over a background:
    /// the step follows the depth, the dry nodes and the strain. `bed_shift`
    /// moves the bed (m).
    type Beach = crate::physics::SWEPhysics2D<Reflective2D>;
    type BeachCallback<'a> = &'a mut dyn FnMut(&Beach, &SWESolution2D, f64);

    fn beach(bed_shift: f64) -> (Beach, SWESolution2D) {
        use crate::mesh::Bathymetry2D;
        use crate::solver::SWEFormulation2D;
        use crate::source::{CoriolisSource2D, HorizontalViscosity2D};
        let length = 600.0;
        let mesh = Arc::new(Mesh2D::uniform_rectangle(0.0, length, 0.0, 150.0, 8, 2));
        let ops = Arc::new(DGOperators2D::new(2));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
            -8.0 + 9.5 * x / length + 0.4 * y / 150.0 + bed_shift
        }));
        let physics = PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom,
            ShallowWater2D::new(9.81),
            Reflective2D::default(),
        )
        .with_bathymetry(bathymetry.clone())
        .with_formulation(SWEFormulation2D::WetDry)
        .with_wet_dry_correction(true)
        .with_source(CoriolisSource2D::f_plane(1.2e-4))
        .with_viscosity(HorizontalViscosity2D::smagorinsky(0.1).with_background(0.05))
        .build();
        let mut state = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for ki in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let [x, _] = mesh.reference_to_physical(k(ki), ops.nodes_r[i], ops.nodes_s[i]);
                let idx = ki * ops.n_nodes + i;
                let eta = 0.3 * (std::f64::consts::PI * x / length).cos();
                let h = (eta - bathymetry.data[idx]).max(0.0);
                state.set_state(k(ki), i, crate::solver::SWEState2D::new(h, 0.0, 0.0));
            }
        }
        (physics, state)
    }

    /// TODO P5.3 gate: a 2D run restarted halfway, from a file, in a new
    /// simulation, ends bit for bit where the uninterrupted run does, with
    /// global steps and with local time stepping. A restart of another bed is
    /// refused.
    #[test]
    fn a_restarted_2d_run_ends_bit_for_bit_where_the_uninterrupted_run_does() {
        use crate::io::{Restart2D, RestartError};
        use crate::time::MultirateSSPRK3;
        let (interval, t_end) = (20.0, 300.0);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("beach.restart");
        fn run<I: TimeIntegrator<SWESolution2D>>(
            integrator: I,
            state: &mut SWESolution2D,
            t_start: f64,
            t_end: f64,
            callback: BeachCallback,
        ) -> SimulationResult {
            let (physics, _) = beach(0.0);
            let sim = Simulation::new(physics, integrator)
                .with_cfl(0.5)
                .with_callback_interval(20.0);
            sim.run_with_callback(state, t_start, t_end, |s, t| callback(sim.physics(), s, t))
        }
        for local in [false, true] {
            let integrate = |state: &mut SWESolution2D, t0, callback: BeachCallback| match local {
                false => run(SSPRK3, state, t0, t_end, callback),
                true => run(MultirateSSPRK3::new(3), state, t0, t_end, callback),
            };
            let (_, mut uninterrupted) = beach(0.0);
            let mut checkpoint = None;
            let result = integrate(&mut uninterrupted, 0.0, &mut |physics, s, t| {
                if (t - 7.0 * interval).abs() < 1e-9 {
                    checkpoint = Some(physics.restart(s, t));
                }
            });
            assert!(result.success, "{:?}", result.error);
            if local {
                let stats = result.local_time_stepping.expect("local time stepping");
                assert!(stats.finest_level > 0, "every element on one level");
            }
            let checkpoint = checkpoint.expect("a callback at 140 s");
            assert!(
                checkpoint.state.h_data().contains(&0.0),
                "the beach has dry nodes"
            );
            checkpoint.write(&path).unwrap();

            let restart = Restart2D::read(&path).unwrap();
            let (physics, _) = beach(0.0);
            physics.check_restart(&restart).unwrap();
            let mut resumed = restart.state.clone();
            let resumed_result = integrate(&mut resumed, restart.time, &mut |_, _, _| {});
            assert!(resumed_result.success, "{:?}", resumed_result.error);
            assert_eq!(resumed_result.final_time, t_end);
            for (j, (a, b)) in uninterrupted.data.iter().zip(&resumed.data).enumerate() {
                let differ = a.iter().zip(b).filter(|(p, q)| p.to_bits() != q.to_bits());
                assert_eq!(differ.count(), 0, "local {local}: variable {j} differs");
            }

            let (shifted, _) = beach(1e-9);
            assert!(matches!(
                shifted.check_restart(&restart),
                Err(RestartError::Domain(_))
            ));
        }
    }

    #[test]
    #[should_panic(expected = "must divide the coupling interval")]
    fn a_coupling_interval_is_a_multiple_of_the_callback_interval() {
        let (physics, mut state) = create_test_setup();
        let mut sim = Simulation::new(physics, SSPRK3).with_callback_interval(0.05);
        sim.run_with_exchange(&mut state, 0.0, 0.2, 0.12, |_, _, _, _| {}, |_, _| {});
    }

    #[test]
    fn test_simulation_config() {
        let config = SimulationConfig::default();
        assert_eq!(config.cfl, 0.5);
        assert!(config.dt_max.is_none());
        assert!(config.dt_min.is_none());
    }

    #[test]
    fn test_simulation_result() {
        let result = SimulationResult::success(10.0, 100, 0.001, 0.01, 1.5);
        assert!(result.success);
        assert!(result.error.is_none());

        let result = SimulationResult::failure(5.0, 50, "Test error".to_string());
        assert!(!result.success);
        assert_eq!(result.error.as_deref(), Some("Test error"));
    }
}
