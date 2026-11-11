//! 2D bottom friction source terms for shallow water equations.
//!
//! Bottom friction dissipates momentum and is particularly important in
//! shallow coastal areas and narrow straits. The most common formulation
//! is Manning's equation.
//!
//! Manning friction (2D):
//!   S_hu = -g n² |u| u / h^{1/3}
//!   S_hv = -g n² |u| v / h^{1/3}
//!
//! where n is the Manning coefficient and |u| = sqrt(u² + v²).
//!
//! Friction becomes stiff in shallow water: with SSP-RK3 the explicit source
//! flips the momentum sign once Δt·C_f|u|/h > 2.5, i.e. in centimetre-deep
//! water at every wet/dry front. Laws that implement [`BottomFriction2D`] can
//! instead be applied point-implicitly in every RK stage
//! (`SWEPhysics2DBuilder::with_implicit_friction`, via
//! `TimeIntegrator::step_with_relaxation`).

use crate::mesh::Mesh2D;
use crate::operators::DGOperators2D;
use crate::solver::SWEState2D;
use crate::source::{ElementSources, SourceContext2D, SourceTerm2D};
use crate::types::ElementIndex;

/// Cube root of `x`, within 2 ulp of `f64::cbrt`, about twice as fast as the
/// MSVC C runtime's `cbrt`. Manning friction takes one per node and RK
/// stage, which made it a tenth of the Frøya step.
///
/// The exponent divided by 3 in the bit pattern gives a guess within a few
/// percent, and three Halley iterations `y ← y + y (x − y³)/(2y³ + x)`
/// (cubic convergence) take it to round-off. In this correction form the
/// last rounding is that of a small update, so the result stays within
/// 2 ulp (the product form `y (y³ + 2x)/(2y³ + x)` reaches 2.1); the ratio
/// is taken before the product so that `y (x − y³)` cannot underflow for
/// tiny `x`. Zero, subnormal, negative and non-finite inputs go to
/// `f64::cbrt`.
#[inline]
pub(crate) fn cbrt(x: f64) -> f64 {
    if !(x >= f64::MIN_POSITIVE && x.is_finite()) {
        return x.cbrt();
    }
    cbrt_refine(x, cbrt_guess(x))
}

/// The first guess of [`cbrt`] for a normal positive `x`, within a few
/// percent (integer arithmetic on the bit pattern).
#[inline(always)]
fn cbrt_guess(x: f64) -> f64 {
    // 0x2A9F7893782DA1CE: (1023 − 1023/3) in the exponent field, with the
    // mantissa offset that minimises the guess error (Kahan)
    f64::from_bits(x.to_bits() / 3 + 0x2A9F_7893_782D_A1CE)
}

/// The Halley iterations of [`cbrt`] from the guess `y`: floating-point
/// arithmetic without branches, so a loop over nodes vectorises.
#[inline(always)]
fn cbrt_refine(x: f64, mut y: f64) -> f64 {
    for _ in 0..3 {
        let y3 = y * y * y;
        y += y * ((x - y3) / (2.0 * y3 + x));
    }
    y
}

/// [`manning_rate`] of a run of nodes, `out[i]` from `g_n2(i)`, the depth
/// `h[i]` floored at `h_min` and `speed[i]`: the same values bit for bit
/// (zero at zero speed), with the divisions of the nodes side by side
/// instead of one node's chain after another's.
#[inline]
fn manning_rates(
    g_n2: impl Fn(usize) -> f64,
    h_min: f64,
    h: &[f64],
    speed: &[f64],
    out: &mut [f64],
) {
    let n = out.len();
    let (h, speed) = (&h[..n], &speed[..n]);
    // The guess needs normal numbers; anything else one node at a time
    let normal = |h: f64| h >= f64::MIN_POSITIVE && h.is_finite();
    if !h.iter().all(|&h| normal(h.max(h_min))) {
        for i in 0..n {
            out[i] = if speed[i] > 0.0 {
                manning_rate(g_n2(i), h[i].max(h_min), speed[i])
            } else {
                0.0
            };
        }
        return;
    }
    for i in 0..n {
        out[i] = cbrt_guess(h[i].max(h_min));
    }
    for i in 0..n {
        let h = h[i].max(h_min);
        out[i] = g_n2(i) * speed[i] / (cbrt_refine(h, out[i]) * h);
    }
}

/// A bottom-friction law written as a linear damping of the momentum,
///
/// ```text
/// S = −Λ(h, |u|)·(0, hu, hv),
/// ```
///
/// which can be applied point-implicitly: `hu ← hu / (1 + Δt·Λ)` never
/// changes the sign of the momentum, for any Δt. For quadratic drag
/// (Λ ∝ |u|) with Λ taken at the start of the step, this is the exact
/// solution of `d(hu)/dt = −Λ hu` over Δt at fixed depth.
pub trait BottomFriction2D: Send + Sync {
    /// Damping rate Λ ≥ 0 (1/s) at mesh node `node` (element-major,
    /// `k·n_nodes + i`), depth `h > 0` and speed `speed = |u|`. Uniform laws
    /// ignore `node`.
    fn damping_rate(&self, node: usize, h: f64, speed: f64) -> f64;

    /// [`Self::damping_rate`] of the consecutive mesh nodes from `first_node`
    /// with depths `h > 0` and speeds `speed`, into `out` (one value per
    /// node; zero where the speed is zero): the same values bit for bit. A
    /// law overrides it to evaluate an element's nodes in one call, side by
    /// side (Manning's cube roots and divisions vectorise).
    fn damping_rates(&self, first_node: usize, h: &[f64], speed: &[f64], out: &mut [f64]) {
        for (i, out) in out.iter_mut().enumerate() {
            *out = if speed[i] > 0.0 {
                self.damping_rate(first_node + i, h[i], speed[i])
            } else {
                0.0
            };
        }
    }

    /// Number of mesh nodes (`K·n_nodes`) a spatially varying law was built
    /// for, so that the damping can check it against the solution; `None`
    /// for laws that do not depend on the node.
    fn n_total_nodes(&self) -> Option<usize> {
        None
    }
}

/// A shared law, such as [`crate::physics::SWEPhysics2D::friction`], wrapped
/// by another (e.g. [`crate::source::WaveCurrentFriction2D`]).
impl<F: BottomFriction2D + ?Sized> BottomFriction2D for std::sync::Arc<F> {
    #[inline]
    fn damping_rate(&self, node: usize, h: f64, speed: f64) -> f64 {
        (**self).damping_rate(node, h, speed)
    }

    #[inline]
    fn damping_rates(&self, first_node: usize, h: &[f64], speed: &[f64], out: &mut [f64]) {
        (**self).damping_rates(first_node, h, speed, out);
    }

    fn n_total_nodes(&self) -> Option<usize> {
        (**self).n_total_nodes()
    }
}

/// Manning bottom friction source term for 2D shallow water equations.
///
/// S = (0, -g n² |u| u / h^{1/3}, -g n² |u| v / h^{1/3})
///
/// where |u| = sqrt(u² + v²) is the velocity magnitude.
///
/// Manning coefficient n has units of s/m^{1/3} and depends on bed roughness:
/// - Smooth concrete: n ≈ 0.012
/// - Natural channels: n ≈ 0.03-0.05
/// - Floodplains with vegetation: n ≈ 0.1-0.15
/// - Rocky fjord bottoms: n ≈ 0.025-0.035
///
/// # Example
///
/// ```ignore
/// use dg::source::{ManningFriction2D, CombinedSource2D, CoriolisSource2D};
///
/// let friction = ManningFriction2D::standard(0.025);  // Rocky fjord bottom
/// let coriolis = CoriolisSource2D::norwegian_coast();
///
/// let combined = CombinedSource2D::new(vec![&friction, &coriolis]);
/// ```
#[derive(Clone, Debug)]
pub struct ManningFriction2D {
    /// Gravitational acceleration (m/s²)
    pub g: f64,
    /// Manning coefficient (s/m^{1/3})
    pub manning_n: f64,
    /// Minimum depth for friction calculation (m)
    pub h_min: f64,
}

impl ManningFriction2D {
    /// Create a new Manning friction source term.
    ///
    /// # Arguments
    /// * `g` - Gravitational acceleration (m/s²)
    /// * `manning_n` - Manning coefficient (s/m^{1/3})
    pub fn new(g: f64, manning_n: f64) -> Self {
        Self {
            g,
            manning_n,
            h_min: 1e-6,
        }
    }

    /// Create with custom minimum depth.
    pub fn with_h_min(g: f64, manning_n: f64, h_min: f64) -> Self {
        Self {
            g,
            manning_n,
            h_min,
        }
    }

    /// Standard gravity (9.81 m/s²) with given Manning coefficient.
    pub fn standard(manning_n: f64) -> Self {
        Self::new(9.81, manning_n)
    }

    /// Typical values for natural water bodies.
    pub fn natural_channel() -> Self {
        Self::standard(0.035)
    }

    /// Typical values for rocky fjord bottoms.
    pub fn fjord() -> Self {
        Self::standard(0.025)
    }

    /// Compute the friction coefficient C_f.
    ///
    /// τ = ρ C_f |u| u, where C_f = g n² h^{-1/3}
    #[inline]
    pub fn friction_coefficient(&self, h: f64) -> f64 {
        let h_eff = h.max(self.h_min);
        self.g * self.manning_n * self.manning_n / cbrt(h_eff)
    }

    /// Compute friction source term explicitly.
    #[inline]
    pub fn explicit_source(&self, state: &SWEState2D, h_min: f64) -> SWEState2D {
        if state.h < h_min {
            return SWEState2D::zero();
        }

        let h_inv = 1.0 / state.h;
        let speed = (state.hu * state.hu + state.hv * state.hv).sqrt() * h_inv;
        if speed < 1e-14 {
            return SWEState2D::zero();
        }

        // S = (0, -C_f |u| u, -C_f |u| v) = -Λ (0, hu, hv)
        let rate = self.rate(state.h, speed);
        SWEState2D {
            h: 0.0,
            hu: -rate * state.hu,
            hv: -rate * state.hv,
        }
    }

    /// Semi-implicit friction update for 2D.
    ///
    /// For stiff friction (shallow water), explicit treatment may require
    /// very small time steps. Semi-implicit treatment is unconditionally stable.
    ///
    /// Given: du/dt = -C_f |u| u / h (component-wise)
    /// Linearize: du/dt ≈ -C_f |u^n| u^{n+1} / h
    /// Solution: u^{n+1} = u^n / (1 + dt * C_f * |u^n| / h)
    ///
    /// This is the exact solution of the friction ODE over `dt` (|u| obeys
    /// d|u|/dt = −(C_f/h)|u|², solved by |u^n| / (1 + dt C_f |u^n| / h)).
    ///
    /// # Arguments
    /// * `state` - Current state
    /// * `dt` - Time step
    ///
    /// # Returns
    /// Updated state after friction
    pub fn semi_implicit_update(&self, state: &SWEState2D, dt: f64) -> SWEState2D {
        if state.h < self.h_min {
            return *state;
        }

        let speed = (state.hu * state.hu + state.hv * state.hv).sqrt() / state.h;
        if speed < 1e-14 {
            return *state;
        }

        let factor = 1.0 / (1.0 + dt * self.rate(state.h, speed));
        SWEState2D {
            h: state.h,
            hu: factor * state.hu,
            hv: factor * state.hv,
        }
    }

    /// Check if friction will be stiff for given state and time step.
    ///
    /// Friction is considered stiff if dt * C_f * |u| / h > 1
    pub fn is_stiff_for_state(&self, state: &SWEState2D, dt: f64) -> bool {
        if state.h < self.h_min {
            return false;
        }

        let speed = (state.hu * state.hu + state.hv * state.hv).sqrt() / state.h;
        dt * self.rate(state.h, speed) > 1.0
    }

    /// Λ = C_f |u| / h = g n² |u| / h^{4/3}.
    #[inline]
    fn rate(&self, h: f64, speed: f64) -> f64 {
        manning_rate(
            self.g * self.manning_n * self.manning_n,
            h.max(self.h_min),
            speed,
        )
    }
}

/// Manning's damping rate Λ = g n² |u| / h^{4/3} from `g_n2` = g·n² at the
/// (already floored) depth `h`.
#[inline]
fn manning_rate(g_n2: f64, h: f64, speed: f64) -> f64 {
    g_n2 * speed / (cbrt(h) * h)
}

impl BottomFriction2D for ManningFriction2D {
    #[inline]
    fn damping_rate(&self, _node: usize, h: f64, speed: f64) -> f64 {
        self.rate(h, speed)
    }

    fn damping_rates(&self, _first_node: usize, h: &[f64], speed: &[f64], out: &mut [f64]) {
        let g_n2 = self.g * self.manning_n * self.manning_n;
        manning_rates(|_| g_n2, self.h_min, h, speed, out);
    }
}

impl SourceTerm2D for ManningFriction2D {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        self.explicit_source(&ctx.state, ctx.h_min)
    }

    fn name(&self) -> &'static str {
        "manning_friction_2d"
    }

    fn is_stiff(&self) -> bool {
        // Manning friction can be stiff in shallow water
        true
    }
}

/// Chezy friction formulation for 2D.
///
/// Alternative to Manning, with constant friction coefficient:
/// S_hu = -C_D |u| u
/// S_hv = -C_D |u| v
///
/// where C_D is the drag coefficient (dimensionless), so the bed stress is
/// τ_b = ρ C_D |u| u. In terms of the Chezy coefficient C (m^{1/2}/s),
/// C_D = g / C²; Manning's n corresponds to C_D = g n² / h^{1/3}.
#[derive(Clone, Debug)]
pub struct ChezyFriction2D {
    /// Drag coefficient (dimensionless)
    pub c_d: f64,
    /// Minimum depth (m)
    pub h_min: f64,
}

impl ChezyFriction2D {
    /// Create a new Chezy friction source term.
    pub fn new(c_d: f64) -> Self {
        Self { c_d, h_min: 1e-6 }
    }

    /// Compute friction source explicitly.
    pub fn explicit_source(&self, state: &SWEState2D) -> SWEState2D {
        if state.h < self.h_min {
            return SWEState2D::zero();
        }

        let speed = (state.hu * state.hu + state.hv * state.hv).sqrt() / state.h;
        if speed < 1e-14 {
            return SWEState2D::zero();
        }

        // S = (0, -C_D |u| u, -C_D |u| v) = -Λ (0, hu, hv)
        let rate = self.damping_rate(0, state.h, speed);
        SWEState2D {
            h: 0.0,
            hu: -rate * state.hu,
            hv: -rate * state.hv,
        }
    }
}

impl BottomFriction2D for ChezyFriction2D {
    /// Λ = C_D |u| / h.
    #[inline]
    fn damping_rate(&self, _node: usize, h: f64, speed: f64) -> f64 {
        self.c_d * speed / h.max(self.h_min)
    }
}

impl SourceTerm2D for ChezyFriction2D {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        self.explicit_source(&ctx.state)
    }

    fn name(&self) -> &'static str {
        "chezy_friction_2d"
    }

    fn is_stiff(&self) -> bool {
        true
    }
}

/// Manning friction with a coefficient n(x, y) that varies in space (bed
/// types, calibrated roughness maps), held as g·n² at every mesh node.
///
/// The field is evaluated once at construction, so the law can be applied
/// point-implicitly ([`BottomFriction2D`], through
/// `SWEPhysics2DBuilder::with_implicit_friction`) and costs no closure call
/// per node and stage. It belongs to the mesh and order it was built for.
///
/// # Example
///
/// ```
/// use dgcom::mesh::Mesh2D;
/// use dgcom::operators::DGOperators2D;
/// use dgcom::source::SpatiallyVaryingManning2D;
///
/// let mesh = Mesh2D::uniform_rectangle(0.0, 20.0, 0.0, 10.0, 4, 2);
/// let ops = DGOperators2D::new(2);
/// // Rougher bed in the shallows (x < 5 m)
/// let friction = SpatiallyVaryingManning2D::new(9.81, &mesh, &ops, |x, _y| {
///     if x < 5.0 { 0.05 } else { 0.025 }
/// });
/// assert_eq!(friction.n_total_nodes(), 8 * 9);
/// ```
#[derive(Clone, Debug)]
pub struct SpatiallyVaryingManning2D {
    /// Gravitational acceleration (m/s²)
    g: f64,
    /// g·n² per node, element-major (`k·n_nodes + i`)
    g_n2: Vec<f64>,
    /// Nodes per element
    n_nodes: usize,
    /// Minimum depth (m)
    pub h_min: f64,
}

impl SpatiallyVaryingManning2D {
    /// Evaluate `manning_fn(x, y)` (s/m^{1/3}) at the nodes of `mesh` at the
    /// order of `ops`.
    ///
    /// # Panics
    /// If `manning_fn` returns a negative or non-finite coefficient.
    pub fn new(
        g: f64,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        manning_fn: impl Fn(f64, f64) -> f64,
    ) -> Self {
        let n = ElementIndex::iter(mesh.n_elements)
            .flat_map(|k| {
                (0..ops.n_nodes)
                    .map(move |i| mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]))
            })
            .map(|[x, y]| manning_fn(x, y))
            .collect();
        Self::from_nodal(g, ops.n_nodes, n)
    }

    /// Standard gravity (9.81 m/s²) with a coefficient `manning_fn(x, y)`.
    pub fn standard(
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        manning_fn: impl Fn(f64, f64) -> f64,
    ) -> Self {
        Self::new(9.81, mesh, ops, manning_fn)
    }

    /// From Manning coefficients `n` (s/m^{1/3}) at every node, element-major
    /// with `n_nodes` nodes per element (e.g. projected from a roughness map).
    ///
    /// # Panics
    /// If `n.len()` is not a multiple of `n_nodes`, or a coefficient is
    /// negative or not finite.
    pub fn from_nodal(g: f64, n_nodes: usize, n: Vec<f64>) -> Self {
        assert!(
            n_nodes > 0 && n.len().is_multiple_of(n_nodes),
            "{} Manning coefficients for {n_nodes} nodes per element",
            n.len()
        );
        let g_n2 = n
            .into_iter()
            .map(|n| {
                assert!(
                    n.is_finite() && n >= 0.0,
                    "Manning coefficient {n} is negative or not finite"
                );
                g * n * n
            })
            .collect();
        Self {
            g,
            g_n2,
            n_nodes,
            h_min: 1e-6,
        }
    }

    /// Manning coefficient n (s/m^{1/3}) at node `i` of element `k`.
    pub fn manning_n(&self, k: ElementIndex, i: usize) -> f64 {
        (self.g_n2[k.as_usize() * self.n_nodes + i] / self.g).sqrt()
    }

    /// Number of mesh nodes (`K·n_nodes`) this was built for.
    pub fn n_total_nodes(&self) -> usize {
        self.g_n2.len()
    }
}

impl BottomFriction2D for SpatiallyVaryingManning2D {
    /// Λ = g n² |u| / h^{4/3} with the node's n.
    #[inline]
    fn damping_rate(&self, node: usize, h: f64, speed: f64) -> f64 {
        manning_rate(self.g_n2[node], h.max(self.h_min), speed)
    }

    fn damping_rates(&self, first_node: usize, h: &[f64], speed: &[f64], out: &mut [f64]) {
        let g_n2 = &self.g_n2[first_node..first_node + out.len()];
        manning_rates(|i| g_n2[i], self.h_min, h, speed, out);
    }

    fn n_total_nodes(&self) -> Option<usize> {
        Some(self.g_n2.len())
    }
}

impl SourceTerm2D for SpatiallyVaryingManning2D {
    /// The per-node context carries no node index, so the coefficient cannot
    /// be looked up here: the 2D SWE RHS goes through
    /// [`SourceTerm2D::add_element`].
    ///
    /// # Panics
    /// Always.
    fn evaluate(&self, _ctx: &SourceContext2D) -> SWEState2D {
        panic!(
            "SpatiallyVaryingManning2D is a per-node field: evaluate it per element \
             (SourceTerm2D::add_element) or apply it as implicit friction"
        )
    }

    fn add_element(
        &self,
        element: &ElementSources<'_>,
        _h: &mut [f64],
        hu: &mut [f64],
        hv: &mut [f64],
    ) {
        let n = element.n_nodes();
        assert_eq!(
            self.n_nodes, n,
            "SpatiallyVaryingManning2D was built for another order"
        );
        let base = element.element.as_usize() * n;
        let [h_in, hu_in, hv_in] = &element.solution.data;
        // Same arithmetic as `ManningFriction2D::explicit_source`
        for i in 0..n {
            let (h, qx, qy) = (h_in[base + i], hu_in[base + i], hv_in[base + i]);
            if h < element.h_min {
                continue;
            }
            let speed = (qx * qx + qy * qy).sqrt() * (1.0 / h);
            if speed < 1e-14 {
                continue;
            }
            // S = −Λ (0, hu, hv)
            let rate = self.damping_rate(base + i, h, speed);
            hu[i] -= rate * qx;
            hv[i] -= rate * qy;
        }
    }

    fn name(&self) -> &'static str {
        "spatially_varying_manning_2d"
    }

    fn is_stiff(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 10.0;
    const TOL: f64 = 1e-10;

    /// The fast cube root is within 2 ulp of `f64::cbrt` over the whole
    /// normal range, exact on perfect cubes, and defers the edge cases.
    #[test]
    fn cbrt_matches_std() {
        let mut worst: f64 = 0.0;
        let mut x = f64::MIN_POSITIVE;
        while x < 1e300 {
            for m in [1.0, 1.37, 2.0, 2.9, 4.0, 5.5, 7.99] {
                let v = x * m;
                worst = worst.max((cbrt(v) - v.cbrt()).abs() / v.cbrt());
            }
            x *= 1.7;
        }
        assert!(
            worst <= 2.0 * f64::EPSILON,
            "worst relative error {worst:e}"
        );
        // Exact binary cubes (0.001 is not one)
        for c in [1.0, 8.0, 27.0, 3.375, 0.125, 1e9] {
            assert_eq!(cbrt(c), c.cbrt(), "{c}");
        }
        assert_eq!(cbrt(0.0), 0.0);
        assert_eq!(cbrt(-8.0), -2.0);
        assert!(cbrt(f64::NAN).is_nan());
        assert_eq!(cbrt(f64::INFINITY), f64::INFINITY);
        assert_eq!(cbrt(1e-310), 1e-310_f64.cbrt());
    }

    fn make_context(h: f64, hu: f64, hv: f64) -> SourceContext2D {
        SourceContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::new(h, hu, hv),
            0.0,
            (0.0, 0.0),
            G,
            1e-6,
        )
    }

    #[test]
    fn test_manning_zero_velocity() {
        let friction = ManningFriction2D::new(G, 0.03);
        let ctx = make_context(2.0, 0.0, 0.0);
        let s = friction.evaluate(&ctx);

        assert!(s.h.abs() < TOL);
        assert!(s.hu.abs() < TOL);
        assert!(s.hv.abs() < TOL);
    }

    #[test]
    fn test_manning_x_direction() {
        let friction = ManningFriction2D::new(G, 0.03);

        // Positive x-velocity
        let ctx = make_context(2.0, 4.0, 0.0); // h=2, u=2, v=0
        let s = friction.evaluate(&ctx);

        assert!(s.h.abs() < TOL);
        assert!(s.hu < 0.0, "Friction should oppose positive x-velocity");
        assert!(s.hv.abs() < TOL, "No friction in y when v=0");
    }

    #[test]
    fn test_manning_y_direction() {
        let friction = ManningFriction2D::new(G, 0.03);

        // Positive y-velocity
        let ctx = make_context(2.0, 0.0, 4.0); // h=2, u=0, v=2
        let s = friction.evaluate(&ctx);

        assert!(s.h.abs() < TOL);
        assert!(s.hu.abs() < TOL, "No friction in x when u=0");
        assert!(s.hv < 0.0, "Friction should oppose positive y-velocity");
    }

    #[test]
    fn test_manning_diagonal_flow() {
        let friction = ManningFriction2D::new(G, 0.03);

        // Diagonal flow: u = v = 2
        let h = 2.0;
        let ctx = make_context(h, 2.0 * h, 2.0 * h); // h=2, u=2, v=2
        let s = friction.evaluate(&ctx);

        // Both components should be negative (opposing flow)
        assert!(s.hu < 0.0, "hu friction should be negative");
        assert!(s.hv < 0.0, "hv friction should be negative");

        // Components should be equal for equal velocities
        assert!(
            (s.hu - s.hv).abs() < TOL,
            "Equal velocities should give equal friction"
        );
    }

    #[test]
    fn test_manning_shallow_stronger() {
        let friction = ManningFriction2D::new(G, 0.03);

        // Same velocity, different depths
        let u = 2.0;
        let h_deep = 4.0;
        let h_shallow = 1.0;

        let ctx_deep = make_context(h_deep, u * h_deep, 0.0);
        let ctx_shallow = make_context(h_shallow, u * h_shallow, 0.0);

        let s_deep = friction.evaluate(&ctx_deep);
        let s_shallow = friction.evaluate(&ctx_shallow);

        assert!(
            s_shallow.hu.abs() > s_deep.hu.abs(),
            "Friction should be stronger in shallow water: shallow={}, deep={}",
            s_shallow.hu.abs(),
            s_deep.hu.abs()
        );
    }

    #[test]
    fn test_manning_coefficient_scaling() {
        let friction_smooth = ManningFriction2D::new(G, 0.01);
        let friction_rough = ManningFriction2D::new(G, 0.05);

        let ctx = make_context(2.0, 4.0, 0.0);

        let s_smooth = friction_smooth.evaluate(&ctx);
        let s_rough = friction_rough.evaluate(&ctx);

        // Friction scales as n²
        let ratio = s_rough.hu / s_smooth.hu;
        let expected_ratio = (0.05 / 0.01_f64).powi(2);
        assert!(
            (ratio - expected_ratio).abs() < 0.01,
            "Friction should scale as n²: ratio={}, expected={}",
            ratio,
            expected_ratio
        );
    }

    #[test]
    fn test_semi_implicit_update() {
        let friction = ManningFriction2D::new(G, 0.03);
        let state = SWEState2D::new(2.0, 4.0, 2.0); // h=2, u=2, v=1
        let dt = 0.1;

        let state_new = friction.semi_implicit_update(&state, dt);

        // Depth unchanged
        assert!((state_new.h - state.h).abs() < TOL);

        // Velocities should decrease
        let u_old = state.hu / state.h;
        let v_old = state.hv / state.h;
        let u_new = state_new.hu / state_new.h;
        let v_new = state_new.hv / state_new.h;

        assert!(u_new.abs() < u_old.abs(), "u should decrease");
        assert!(v_new.abs() < v_old.abs(), "v should decrease");

        // Signs should be preserved
        assert!(u_new > 0.0, "u should stay positive");
        assert!(v_new > 0.0, "v should stay positive");
    }

    #[test]
    fn test_semi_implicit_stability() {
        let friction = ManningFriction2D::new(G, 0.1); // Strong friction
        let state = SWEState2D::new(0.1, 0.2, 0.3); // Shallow, fast flow
        let dt = 10.0; // Very large time step

        let state_new = friction.semi_implicit_update(&state, dt);

        // Should not blow up
        assert!(state_new.h > 0.0);
        assert!(state_new.hu.is_finite());
        assert!(state_new.hv.is_finite());
        // Should not overshoot
        assert!(state_new.hu >= 0.0);
        assert!(state_new.hv >= 0.0);
    }

    #[test]
    fn test_is_stiff_for_state() {
        let friction = ManningFriction2D::new(G, 0.1);

        // Deep water - not stiff
        let deep = SWEState2D::new(10.0, 10.0, 0.0);
        assert!(!friction.is_stiff_for_state(&deep, 0.01));

        // Shallow water, large dt - stiff
        let shallow = SWEState2D::new(0.01, 0.1, 0.0);
        assert!(friction.is_stiff_for_state(&shallow, 1.0));
    }

    #[test]
    fn test_chezy_friction() {
        let chezy = ChezyFriction2D::new(0.002);
        let ctx = make_context(2.0, 4.0, 2.0);
        let s = chezy.evaluate(&ctx);

        assert!(s.h.abs() < TOL);
        assert!(s.hu < 0.0, "hu friction should oppose flow");
        assert!(s.hv < 0.0, "hv friction should oppose flow");
    }

    /// Regression: quadratic drag on the depth-integrated momentum is
    /// `S_hu = −c_d |u| u` (m²/s²) with dimensionless `c_d`; it previously
    /// carried an extra `1/h`, making it depth-dependent and wrong in magnitude.
    #[test]
    fn test_chezy_friction_magnitude() {
        let c_d = 0.0025;
        let chezy = ChezyFriction2D::new(c_d);
        let (u, v): (f64, f64) = (2.0, 1.0);
        let speed = (u * u + v * v).sqrt();

        for h in [0.5, 2.0, 50.0] {
            let s = chezy.evaluate(&make_context(h, h * u, h * v));
            assert!(
                (s.hu - (-c_d * speed * u)).abs() < TOL,
                "h={h}: hu {}",
                s.hu
            );
            assert!(
                (s.hv - (-c_d * speed * v)).abs() < TOL,
                "h={h}: hv {}",
                s.hv
            );
        }

        // Chezy with c_d = g n² / h^{1/3} is Manning at that depth.
        let (n, h) = (0.03, 7.0);
        let manning = ManningFriction2D::new(G, n);
        let chezy = ChezyFriction2D::new(manning.friction_coefficient(h));
        let ctx = make_context(h, h * u, h * v);
        let (s_c, s_m) = (chezy.evaluate(&ctx), manning.evaluate(&ctx));
        assert!((s_c.hu - s_m.hu).abs() < TOL && (s_c.hv - s_m.hv).abs() < TOL);
    }

    #[test]
    fn test_dry_cell_no_friction() {
        let friction = ManningFriction2D::new(G, 0.03);
        let ctx = make_context(1e-10, 1e-10, 1e-10);
        let s = friction.evaluate(&ctx);

        assert!(s.h.abs() < TOL);
        assert!(s.hu.abs() < TOL);
        assert!(s.hv.abs() < TOL);
    }

    /// Manning coefficient varying inside and across elements: rough near
    /// the left of the domain, smooth to the right.
    fn roughness(x: f64, y: f64) -> f64 {
        0.02 + 0.03 * (-x / 7.0).exp() + 0.002 * y
    }

    /// Mesh, operators and a flow with h, u, v varying from node to node.
    fn varying_flow() -> (Mesh2D, DGOperators2D, crate::solver::SWESolution2D) {
        let mesh = Mesh2D::uniform_rectangle(0.0, 20.0, 0.0, 10.0, 4, 3);
        let ops = DGOperators2D::new(3);
        let mut q = crate::solver::SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let h = 0.5 + 0.1 * x + 0.05 * y;
                let state = SWEState2D::from_primitives(h, (0.3 * y).sin(), 0.2 - 0.02 * x);
                q.set_state(k, i, state);
            }
        }
        (mesh, ops, q)
    }

    /// Each node carries its own coefficient: the point-implicit rate and the
    /// explicit source equal those of a uniform Manning law with that node's
    /// n, bit for bit.
    #[test]
    fn test_spatially_varying_manning_is_manning_per_node() {
        let (mesh, ops, q) = varying_flow();
        let friction = SpatiallyVaryingManning2D::new(G, &mesh, &ops, roughness);
        assert_eq!(
            BottomFriction2D::n_total_nodes(&friction),
            Some(q.h_data().len())
        );

        let mut out = [
            vec![0.0; ops.n_nodes],
            vec![0.0; ops.n_nodes],
            vec![0.0; ops.n_nodes],
        ];
        let mut rough_vs_smooth = (0.0_f64, f64::INFINITY);
        for k in ElementIndex::iter(mesh.n_elements) {
            let element = ElementSources {
                element: k,
                time: 0.0,
                solution: &q,
                mesh: &mesh,
                ops: &ops,
                bathymetry: None,
                g: G,
                h_min: 1e-6,
            };
            for v in &mut out {
                v.fill(0.0);
            }
            let [oh, ohu, ohv] = &mut out;
            friction.add_element(&element, oh, ohu, ohv);
            for i in 0..ops.n_nodes {
                let (x, y) = element.position(i);
                let uniform = ManningFriction2D::new(G, roughness(x, y));
                let node = k.as_usize() * ops.n_nodes + i;
                let state = q.get_state(k, i);
                let speed = (state.hu * state.hu + state.hv * state.hv).sqrt() / state.h;
                assert_eq!(
                    friction.damping_rate(node, state.h, speed),
                    uniform.damping_rate(node, state.h, speed)
                );
                let expected = uniform.evaluate(&element.context(i));
                assert_eq!((oh[i], ohu[i], ohv[i]), (0.0, expected.hu, expected.hv));
                assert!((friction.manning_n(k, i) - roughness(x, y)).abs() < 1e-15);
                let n = friction.manning_n(k, i);
                rough_vs_smooth = (rough_vs_smooth.0.max(n), rough_vs_smooth.1.min(n));
            }
        }
        // The field does vary
        assert!(rough_vs_smooth.0 > 2.0 * rough_vs_smooth.1);
    }

    #[test]
    #[should_panic(expected = "per-node field")]
    fn test_spatially_varying_manning_has_no_pointwise_evaluate() {
        let friction = SpatiallyVaryingManning2D::from_nodal(G, 4, vec![0.03; 8]);
        friction.evaluate(&make_context(2.0, 4.0, 0.0));
    }

    #[test]
    #[should_panic(expected = "nodes per element")]
    fn test_spatially_varying_manning_rejects_partial_elements() {
        SpatiallyVaryingManning2D::from_nodal(G, 4, vec![0.03; 6]);
    }

    #[test]
    #[should_panic(expected = "negative or not finite")]
    fn test_spatially_varying_manning_rejects_negative_n() {
        SpatiallyVaryingManning2D::from_nodal(G, 2, vec![0.03, -0.01]);
    }

    #[test]
    fn test_preset_friction_values() {
        let natural = ManningFriction2D::natural_channel();
        assert!((natural.manning_n - 0.035).abs() < 1e-10);

        let fjord = ManningFriction2D::fjord();
        assert!((fjord.manning_n - 0.025).abs() < 1e-10);
    }

    #[test]
    fn test_is_stiff_trait() {
        let manning = ManningFriction2D::standard(0.03);
        assert!(manning.is_stiff());

        let chezy = ChezyFriction2D::new(0.002);
        assert!(chezy.is_stiff());
    }
}
