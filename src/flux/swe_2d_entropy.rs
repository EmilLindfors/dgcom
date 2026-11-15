//! Entropy-conservative and entropy-stable fluxes for the 2D shallow water
//! equations with bathymetry (Wintermeyer, Winters, Gassner & Kopriva 2017).
//!
//! These are the building blocks of the split-form DGSEM selected with
//! `SWEFormulation2D::EntropyStable` / `EntropyConservative` in the 2D SWE RHS.
//!
//! # Mathematical formulation
//!
//! With bed elevation B, the SWE have the convex entropy (total energy)
//!
//! ```text
//! E = ½h(u² + v²) + ½gh² + ghB
//! w = ∂E/∂q = (g(h + B) − ½(u² + v²), u, v)     entropy variables
//! ψ = ½gh² (u, v)                               entropy flux potential
//! ```
//!
//! The Wintermeyer et al. two-point flux in a direction m = (m_x, m_y) (a unit
//! normal or a contravariant metric vector) is
//!
//! ```text
//! F#(q_L, q_R)·m = ( M,  M{{u}} + ½g h_L h_R m_x,  M{{v}} + ½g h_L h_R m_y )
//! M = {{hu}} m_x + {{hv}} m_y,    {{a}} = ½(a_L + a_R),    [[a]] = a_R − a_L
//! ```
//!
//! It is symmetric, consistent (`F#(q, q) = F(q)`) and entropy conservative
//! together with a nonconservative bed term:
//!
//! ```text
//! [[w]]·(F#·m) = [[ψ·m]] + g {{h u·m}} [[B]]
//! ```
//!
//! The DG operator cancels `g{{h u·m}}[[B]]` with the collocated bed source
//! `−g h_i Σ_j D_ij B_j` in the volume and with the interface term
//! `½g h⁻ (B⁺ − B⁻) n` ([`wintermeyer_bed_interface_term_2d`]) at faces. The
//! same pairing makes the scheme well-balanced for *any* nodal bathymetry: at
//! lake at rest the pressure part of the flux-differencing volume term is
//! `g h_i Σ_j D_ij h_j`, which cancels `g h_i Σ_j D_ij B_j` because
//! `Σ_j D_ij (h_j + B_j) = η Σ_j D_ij = 0`.
//!
//! # Entropy-stable interface dissipation
//!
//! [`entropy_stable_dissipation_2d`] is local Lax–Friedrichs dissipation in
//! entropy variables, `−½λ H̄ [[w]]` with `H̄ = ∂q/∂w` at the arithmetic-mean
//! state. It simplifies to
//!
//! ```text
//! H̄[[w]] = ( [[η]],  {{u}}[[η]] + {{h}}[[u]],  {{v}}[[η]] + {{h}}[[v]] ),   η = h + B
//! [[w]]·H̄[[w]] = g[[η]]² + {{h}}([[u]]² + [[v]]²) ≥ 0
//! ```
//!
//! so it always produces entropy, reduces to the usual jump in `(h, hu, hv)`
//! (with arithmetic-mean products) for continuous B, and vanishes at lake at
//! rest even across bathymetry jumps, where `[[w]] = 0`.
//!
//! # References
//!
//! - Wintermeyer, Winters, Gassner & Kopriva (2017), "An entropy stable nodal
//!   discontinuous Galerkin method for the two dimensional shallow water
//!   equations on unstructured curvilinear meshes with discontinuous
//!   bathymetry", J. Comput. Phys. 340.
//! - Fjordholm, Mishra & Tadmor (2011), "Well-balanced and energy stable
//!   schemes for the shallow water equations with discontinuous topography",
//!   J. Comput. Phys. 230.
//! - Gassner, Winters & Kopriva (2016), "A well balanced and entropy
//!   conservative discontinuous Galerkin spectral element method for the
//!   shallow water equations", Appl. Math. Comput. 272.

use crate::solver::SWEState2D;
use crate::types::Lane;

/// SWE state at one node together with its bed elevation and velocity.
///
/// Precomputing the velocity once per node keeps the O(p³) flux-differencing
/// loops free of divisions. `T` is `f64` except in the pointwise physics shared
/// by the scalar, batched and GPU kernels ([`Lane`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SWENodeState2D<T = f64> {
    /// Water depth h
    pub h: T,
    /// x-momentum hu
    pub hu: T,
    /// y-momentum hv
    pub hv: T,
    /// x-velocity hu/h (zero for dry nodes)
    pub u: T,
    /// y-velocity hv/h (zero for dry nodes)
    pub v: T,
    /// Bed elevation B
    pub b: T,
}

impl SWENodeState2D {
    /// Build from conserved variables and bed elevation.
    ///
    /// The velocity is set to zero for `h ≤ h_min`, as in the Riemann solvers.
    #[inline]
    pub fn new(q: &SWEState2D, b: f64, h_min: f64) -> Self {
        let (u, v) = if q.h > h_min {
            (q.hu / q.h, q.hv / q.h)
        } else {
            (0.0, 0.0)
        };
        Self {
            h: q.h,
            hu: q.hu,
            hv: q.hv,
            u,
            v,
            b,
        }
    }
}

impl<T: Lane> SWENodeState2D<T> {
    /// Free-surface elevation η = h + B.
    #[inline(always)]
    pub fn eta(&self) -> T {
        self.h + self.b
    }
}

/// Wintermeyer et al. (2017) entropy-conservative two-point flux `F#(q_L, q_R)·m`.
///
/// `m` need not be a unit vector: the volume term uses the contravariant
/// metric vectors `(r_x, r_y)` and `(s_x, s_y)`, faces the unit normal.
/// `F#(q, q)·m` is the physical flux `F(q)·m`.
#[inline(always)]
pub fn wintermeyer_flux_2d<T: Lane>(
    q_l: &SWENodeState2D<T>,
    q_r: &SWENodeState2D<T>,
    m: (T, T),
    g: f64,
) -> SWEState2D<T> {
    // Every operation is commutative in the two states, so the flux is bitwise
    // symmetric and the face terms of neighbouring elements cancel exactly.
    let half = q_l.h.constant(0.5);
    let mass = half * ((q_l.hu + q_r.hu) * m.0 + (q_l.hv + q_r.hv) * m.1);
    let pressure = q_l.h.constant(0.5 * g) * (q_l.h * q_r.h);
    let u_avg = half * (q_l.u + q_r.u);
    let v_avg = half * (q_l.v + q_r.v);
    SWEState2D {
        h: mass,
        hu: mass * u_avg + pressure * m.0,
        hv: mass * v_avg + pressure * m.1,
    }
}

/// Interface part of the nonconservative bed term, `½g h⁻ (B⁺ − B⁻)·n`.
///
/// Added to the numerical flux of the side with depth `h_int` and bed `b_int`.
/// Zero for continuous bathymetry. Not conservative by design: across a bed
/// step the two sides feel different pressure forces.
#[inline]
pub fn wintermeyer_bed_interface_term_2d(
    h_int: f64,
    b_int: f64,
    b_ext: f64,
    normal: (f64, f64),
    g: f64,
) -> SWEState2D {
    let force = 0.5 * g * h_int * (b_ext - b_int);
    SWEState2D::new(0.0, force * normal.0, force * normal.1)
}

/// Entropy-stable interface dissipation `−½λ H̄[[w]]` (add to the EC flux).
///
/// `λ = max(|u⁻·n| + √(gh⁻), |u⁺·n| + √(gh⁺))`. Conservative (antisymmetric
/// under swapping sides and flipping `normal`) and zero at lake at rest.
#[inline]
pub fn entropy_stable_dissipation_2d(
    q_int: &SWENodeState2D,
    q_ext: &SWENodeState2D,
    normal: (f64, f64),
    g: f64,
) -> SWEState2D {
    let (nx, ny) = normal;
    let speed_int = (q_int.u * nx + q_int.v * ny).abs() + (g * q_int.h.max(0.0)).sqrt();
    let speed_ext = (q_ext.u * nx + q_ext.v * ny).abs() + (g * q_ext.h.max(0.0)).sqrt();
    let scale = -0.5 * speed_int.max(speed_ext);

    let jump_eta = q_ext.eta() - q_int.eta();
    let h_avg = 0.5 * (q_int.h + q_ext.h);
    let u_avg = 0.5 * (q_int.u + q_ext.u);
    let v_avg = 0.5 * (q_int.v + q_ext.v);

    SWEState2D {
        h: scale * jump_eta,
        hu: scale * (u_avg * jump_eta + h_avg * (q_ext.u - q_int.u)),
        hv: scale * (v_avg * jump_eta + h_avg * (q_ext.v - q_int.v)),
    }
}

/// Mathematical entropy (total energy per unit area) `E = ½h|u|² + ½gh² + ghB`.
#[inline]
pub fn swe_entropy_2d(q: &SWENodeState2D, g: f64) -> f64 {
    0.5 * (q.hu * q.u + q.hv * q.v) + 0.5 * g * q.h * q.h + g * q.h * q.b
}

/// Entropy variables `w = (g(h + B) − ½|u|², u, v)`.
#[inline]
pub fn swe_entropy_variables_2d(q: &SWENodeState2D, g: f64) -> SWEState2D {
    SWEState2D::new(g * q.eta() - 0.5 * (q.u * q.u + q.v * q.v), q.u, q.v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equations::ShallowWater2D;

    const G: f64 = 9.81;
    const H_MIN: f64 = 1e-6;

    /// Deterministic pseudo-random wet states with bed elevations.
    fn sample_states(n: usize) -> Vec<SWENodeState2D> {
        let mut seed: u64 = 0x2545_f491_4f6c_dd1d;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let h = 0.5 + 50.0 * next();
                let u = 4.0 * next() - 2.0;
                let v = 4.0 * next() - 2.0;
                let b = 40.0 * next() - 80.0;
                SWENodeState2D::new(&SWEState2D::from_primitives(h, u, v), b, H_MIN)
            })
            .collect()
    }

    fn close(a: f64, b: f64, scale: f64) -> bool {
        (a - b).abs() <= 1e-12 * scale.max(1.0)
    }

    #[test]
    fn test_consistency_with_physical_flux() {
        let equation = ShallowWater2D::new(G);
        for q in sample_states(20) {
            let conserved = SWEState2D::new(q.h, q.hu, q.hv);
            for normal in [(1.0, 0.0), (0.0, 1.0), (0.6, -0.8)] {
                let f = wintermeyer_flux_2d(&q, &q, normal, G);
                let exact = equation.normal_flux(&conserved, normal);
                let scale = exact.hu.abs() + exact.hv.abs();
                assert!(close(f.h, exact.h, scale), "{f:?} vs {exact:?}");
                assert!(close(f.hu, exact.hu, scale), "{f:?} vs {exact:?}");
                assert!(close(f.hv, exact.hv, scale), "{f:?} vs {exact:?}");
            }
        }
    }

    #[test]
    fn test_symmetry() {
        let states = sample_states(10);
        for pair in states.windows(2) {
            let m = (0.3, -1.7);
            let f_lr = wintermeyer_flux_2d(&pair[0], &pair[1], m, G);
            let f_rl = wintermeyer_flux_2d(&pair[1], &pair[0], m, G);
            assert_eq!((f_lr.h, f_lr.hu, f_lr.hv), (f_rl.h, f_rl.hu, f_rl.hv));
        }
    }

    #[test]
    fn test_entropy_conservation_condition() {
        // [[w]]·(F#·m) = [[ψ·m]] + g{{h u·m}}[[B]] for arbitrary states, beds and m
        let states = sample_states(40);
        for pair in states.windows(2) {
            let (l, r) = (&pair[0], &pair[1]);
            for m in [(1.0, 0.0), (0.0, 1.0), (0.37, -2.1)] {
                let f = wintermeyer_flux_2d(l, r, m, G);
                let jump_w = swe_entropy_variables_2d(r, G) - swe_entropy_variables_2d(l, G);
                let lhs = jump_w.h * f.h + jump_w.hu * f.hu + jump_w.hv * f.hv;

                let psi = |q: &SWENodeState2D| 0.5 * G * q.h * q.h * (q.u * m.0 + q.v * m.1);
                let hum = |q: &SWENodeState2D| q.hu * m.0 + q.hv * m.1;
                let rhs = psi(r) - psi(l) + G * 0.5 * (hum(l) + hum(r)) * (r.b - l.b);

                let scale = lhs.abs() + psi(r).abs() + psi(l).abs();
                assert!(close(lhs, rhs, scale), "EC condition: {lhs} vs {rhs}");
            }
        }
    }

    #[test]
    fn test_lake_at_rest_across_bed_step() {
        // F(q⁻)·n − [F#(q⁻, q⁺)·n + ½g h⁻ [[B]] n + dissipation] = 0 at rest
        let eta = 0.3;
        let (b_int, b_ext) = (-120.0, -35.0);
        let q_int = SWENodeState2D::new(&SWEState2D::new(eta - b_int, 0.0, 0.0), b_int, H_MIN);
        let q_ext = SWENodeState2D::new(&SWEState2D::new(eta - b_ext, 0.0, 0.0), b_ext, H_MIN);
        let normal = (0.8, 0.6);

        let flux_diff = wintermeyer_flux_2d(&q_int, &q_int, normal, G)
            - wintermeyer_flux_2d(&q_int, &q_ext, normal, G)
            - wintermeyer_bed_interface_term_2d(q_int.h, q_int.b, q_ext.b, normal, G)
            - entropy_stable_dissipation_2d(&q_int, &q_ext, normal, G);

        let scale = 0.5 * G * q_int.h * q_int.h;
        assert!(flux_diff.h.abs() < 1e-14 * scale, "{flux_diff:?}");
        assert!(flux_diff.hu.abs() < 1e-14 * scale, "{flux_diff:?}");
        assert!(flux_diff.hv.abs() < 1e-14 * scale, "{flux_diff:?}");
    }

    #[test]
    fn test_dissipation_is_entropy_variable_lax_friedrichs() {
        // −½λ H̄[[w]] with H̄ = ∂q/∂w at the mean state; entropy production ≤ 0
        let states = sample_states(40);
        let normal = (0.6, 0.8);
        for pair in states.windows(2) {
            let (l, r) = (&pair[0], &pair[1]);
            let d = entropy_stable_dissipation_2d(l, r, normal, G);
            let jw = swe_entropy_variables_2d(r, G) - swe_entropy_variables_2d(l, G);

            let (h, u, v) = (0.5 * (l.h + r.h), 0.5 * (l.u + r.u), 0.5 * (l.v + r.v));
            let hbar = [
                [1.0 / G, u / G, v / G],
                [u / G, h + u * u / G, u * v / G],
                [v / G, u * v / G, h + v * v / G],
            ];
            let jw = [jw.h, jw.hu, jw.hv];
            let lambda = ((l.u * normal.0 + l.v * normal.1).abs() + (G * l.h).sqrt())
                .max((r.u * normal.0 + r.v * normal.1).abs() + (G * r.h).sqrt());
            let expected: Vec<f64> = hbar
                .iter()
                .map(|row| -0.5 * lambda * (0..3).map(|c| row[c] * jw[c]).sum::<f64>())
                .collect();

            let got = [d.h, d.hu, d.hv];
            let scale = expected.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
            for c in 0..3 {
                assert!(close(got[c], expected[c], scale), "{got:?} vs {expected:?}");
            }

            let production = jw[0] * d.h + jw[1] * d.hu + jw[2] * d.hv;
            assert!(production <= 0.0, "dissipation must not create entropy");
        }
    }

    #[test]
    fn test_dissipation_is_conservative() {
        let states = sample_states(10);
        let normal = (-0.28, 0.96);
        let flipped = (-normal.0, -normal.1);
        for pair in states.windows(2) {
            let d_lr = entropy_stable_dissipation_2d(&pair[0], &pair[1], normal, G);
            let d_rl = entropy_stable_dissipation_2d(&pair[1], &pair[0], flipped, G);
            assert_eq!((d_lr.h, d_lr.hu, d_lr.hv), (-d_rl.h, -d_rl.hu, -d_rl.hv));
        }
    }

    #[test]
    fn test_entropy_variables_are_entropy_gradient() {
        // w = ∂E/∂q by central differences in (h, hu, hv) at fixed B
        for q in sample_states(10) {
            let w = swe_entropy_variables_2d(&q, G);
            let entropy = |h: f64, hu: f64, hv: f64| {
                swe_entropy_2d(
                    &SWENodeState2D::new(&SWEState2D::new(h, hu, hv), q.b, H_MIN),
                    G,
                )
            };
            let eps = 1e-6;
            let dh =
                (entropy(q.h + eps, q.hu, q.hv) - entropy(q.h - eps, q.hu, q.hv)) / (2.0 * eps);
            let dhu =
                (entropy(q.h, q.hu + eps, q.hv) - entropy(q.h, q.hu - eps, q.hv)) / (2.0 * eps);
            let dhv =
                (entropy(q.h, q.hu, q.hv + eps) - entropy(q.h, q.hu, q.hv - eps)) / (2.0 * eps);
            assert!(
                (w.h - dh).abs() < 1e-5 * w.h.abs().max(1.0),
                "{} vs {dh}",
                w.h
            );
            assert!((w.hu - dhu).abs() < 1e-6, "{} vs {dhu}", w.hu);
            assert!((w.hv - dhv).abs() < 1e-6, "{} vs {dhv}", w.hv);
        }
    }
}
