//! 2D Riemann solvers for shallow water equations.
//!
//! These solvers work by rotating to face-aligned coordinates, applying
//! a 1D Riemann solver in the normal direction, and rotating back.
//!
//! For a face with outward normal n = (nx, ny):
//! - Normal velocity: u_n = u·n = u*nx + v*ny
//! - Tangential velocity: u_t = u×n = -u*ny + v*nx
//!
//! The Riemann problem is solved in the (n, t) coordinate system,
//! then the flux is rotated back to (x, y).
//!
//! Reference: Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics"

use crate::solver::SWEState2D;
use crate::types::Lane;

/// Rotate 2D SWE state to face-aligned coordinates.
///
/// Transforms (hu, hv) to (h*u_n, h*u_t) where:
/// - u_n = u*nx + v*ny (normal component)
/// - u_t = -u*ny + v*nx (tangential component)
#[inline]
fn rotate_to_normal(state: &SWEState2D, nx: f64, ny: f64) -> SWEState2D {
    SWEState2D {
        h: state.h,
        hu: state.hu * nx + state.hv * ny,  // h * u_n
        hv: -state.hu * ny + state.hv * nx, // h * u_t
    }
}

/// Rotate flux from face-aligned back to physical coordinates.
///
/// Transforms (F_n, F_t) to (F_x, F_y) where:
/// - F_x = F_n * nx - F_t * ny
/// - F_y = F_n * ny + F_t * nx
#[inline(always)]
pub(crate) fn rotate_from_normal<T: Lane>(flux: &SWEState2D<T>, nx: T, ny: T) -> SWEState2D<T> {
    SWEState2D {
        h: flux.h,
        hu: flux.hu * nx - flux.hv * ny,
        hv: flux.hu * ny + flux.hv * nx,
    }
}

/// Roe numerical flux for 2D shallow water equations.
///
/// Computes F* · n at an interface with outward normal (nx, ny).
///
/// # Arguments
/// * `q_l` - Left (interior) state
/// * `q_r` - Right (exterior) state
/// * `normal` - Outward unit normal (nx, ny)
/// * `g` - Gravitational acceleration
/// * `h_min` - Minimum depth for wet/dry treatment
///
/// # Returns
/// Numerical flux F* · n as SWEState2D
pub fn roe_flux_swe_2d(
    q_l: &SWEState2D,
    q_r: &SWEState2D,
    normal: (f64, f64),
    g: f64,
    h_min: f64,
) -> SWEState2D {
    let (nx, ny) = normal;

    let h_l = q_l.h;
    let h_r = q_r.h;

    // Handle dry cells
    if h_l <= h_min && h_r <= h_min {
        return SWEState2D::zero();
    }

    // Rotate to face-aligned coordinates
    let q_l_rot = rotate_to_normal(q_l, nx, ny);
    let q_r_rot = rotate_to_normal(q_r, nx, ny);

    // Compute velocities in rotated frame
    let (un_l, ut_l) = if h_l > h_min {
        (q_l_rot.hu / h_l, q_l_rot.hv / h_l)
    } else {
        (0.0, 0.0)
    };

    let (un_r, ut_r) = if h_r > h_min {
        (q_r_rot.hu / h_r, q_r_rot.hv / h_r)
    } else {
        (0.0, 0.0)
    };

    // Wave celerities
    let c_l = (g * h_l.max(0.0)).sqrt();
    let c_r = (g * h_r.max(0.0)).sqrt();

    // Physical fluxes in rotated coordinates (in n-direction)
    // F = [h*un, h*un^2 + gh^2/2, h*un*ut]
    let f_l = SWEState2D {
        h: h_l * un_l,
        hu: h_l * un_l * un_l + 0.5 * g * h_l * h_l,
        hv: h_l * un_l * ut_l,
    };

    let f_r = SWEState2D {
        h: h_r * un_r,
        hu: h_r * un_r * un_r + 0.5 * g * h_r * h_r,
        hv: h_r * un_r * ut_r,
    };

    // Roe averages
    let sqrt_h_l = h_l.max(0.0).sqrt();
    let sqrt_h_r = h_r.max(0.0).sqrt();

    let (_h_roe, un_roe, ut_roe, c_roe) = if sqrt_h_l + sqrt_h_r > 1e-10 {
        let denom = sqrt_h_l + sqrt_h_r;
        let h_roe = 0.5 * (h_l + h_r);
        let un_roe = (sqrt_h_l * un_l + sqrt_h_r * un_r) / denom;
        let ut_roe = (sqrt_h_l * ut_l + sqrt_h_r * ut_r) / denom;
        let c_roe = (g * h_roe).sqrt();
        (h_roe, un_roe, ut_roe, c_roe)
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

    // Eigenvalues in normal direction
    let lambda_1 = un_roe - c_roe; // Left-going wave
    let lambda_2 = un_roe; // Shear wave (tangential velocity)
    let lambda_3 = un_roe + c_roe; // Right-going wave

    // Jumps
    let delta_h = h_r - h_l;
    let delta_hun = q_r_rot.hu - q_l_rot.hu;
    let delta_hut = q_r_rot.hv - q_l_rot.hv;

    // Wave strengths
    let (alpha_1, alpha_2, alpha_3) = if c_roe > 1e-10 {
        let inv_2c = 0.5 / c_roe;
        // α_1 = (Δh*un + c*Δh - Δ(hun)) / (2c)
        // α_2 = Δ(hut) - ut*Δh (shear wave)
        // α_3 = (Δh*un - c*Δh + Δ(hun)) / (2c)
        let alpha_1 = inv_2c * ((un_roe + c_roe) * delta_h - delta_hun);
        let alpha_3 = inv_2c * (-(un_roe - c_roe) * delta_h + delta_hun);
        let alpha_2 = delta_hut - ut_roe * delta_h;
        (alpha_1, alpha_2, alpha_3)
    } else {
        (0.5 * delta_h, delta_hut, 0.5 * delta_h)
    };

    // Right eigenvectors (columns)
    // r1 = [1, un-c, ut]^T
    // r2 = [0, 0, 1]^T (shear wave)
    // r3 = [1, un+c, ut]^T
    let r1 = SWEState2D::new(1.0, un_roe - c_roe, ut_roe);
    let r2 = SWEState2D::new(0.0, 0.0, 1.0);
    let r3 = SWEState2D::new(1.0, un_roe + c_roe, ut_roe);

    // Entropy fix for transonic rarefactions
    let lambda_1_abs = entropy_fix(lambda_1, un_l - c_l, un_r - c_r);
    let lambda_2_abs = lambda_2.abs();
    let lambda_3_abs = entropy_fix(lambda_3, un_l + c_l, un_r + c_r);

    // Roe flux in rotated coordinates
    // F* = 0.5(F_L + F_R) - 0.5 * Σ |λ_i| α_i r_i
    let f_avg = 0.5 * (f_l + f_r);
    let dissipation = 0.5
        * (lambda_1_abs * alpha_1 * r1 + lambda_2_abs * alpha_2 * r2 + lambda_3_abs * alpha_3 * r3);

    let flux_rot = f_avg - dissipation;

    // Rotate back to physical coordinates
    rotate_from_normal(&flux_rot, nx, ny)
}

/// Entropy fix for transonic rarefactions (Harten-Hyman).
fn entropy_fix(lambda_roe: f64, lambda_l: f64, lambda_r: f64) -> f64 {
    if lambda_l < 0.0 && lambda_r > 0.0 {
        let delta = lambda_r - lambda_l;
        if delta.abs() > 1e-10 {
            0.5 * (lambda_roe.abs() + delta)
        } else {
            lambda_roe.abs()
        }
    } else {
        lambda_roe.abs()
    }
}

/// HLL numerical flux for 2D shallow water equations.
///
/// Uses a two-wave approximation. More robust than Roe for strong shocks.
///
/// # Arguments
/// * `q_l` - Left (interior) state
/// * `q_r` - Right (exterior) state
/// * `normal` - Outward unit normal (nx, ny)
/// * `g` - Gravitational acceleration
/// * `h_min` - Minimum depth for wet/dry treatment
pub fn hll_flux_swe_2d(
    q_l: &SWEState2D,
    q_r: &SWEState2D,
    normal: (f64, f64),
    g: f64,
    h_min: f64,
) -> SWEState2D {
    let (nx, ny) = normal;

    // Rotate to face-aligned coordinates
    let q_l_rot = rotate_to_normal(q_l, nx, ny);
    let q_r_rot = rotate_to_normal(q_r, nx, ny);

    // Normal and tangential velocities
    let velocity = |q: &SWEState2D| {
        if q.h > h_min {
            (q.hu / q.h, q.hv / q.h)
        } else {
            (0.0, 0.0)
        }
    };
    let flux_rot = hll_flux_face_aligned(
        HllSide::new(&q_l_rot, velocity(&q_l_rot)),
        HllSide::new(&q_r_rot, velocity(&q_r_rot)),
        g,
        h_min,
    );
    rotate_from_normal(&flux_rot, nx, ny)
}

/// One side of a face for [`hll_flux_face_aligned`], in face-aligned
/// components: depth, normal and tangential momentum, and the velocities the
/// wave speeds and fluxes use (zero on a side at or below `h_min`; the
/// momenta still enter the dissipation term).
#[derive(Clone, Copy, Debug)]
pub(crate) struct HllSide<T = f64> {
    pub h: T,
    pub hun: T,
    pub hut: T,
    pub un: T,
    pub ut: T,
}

impl<T: Lane> HllSide<T> {
    /// The physical flux in face-aligned components, with `half_g` = ½g.
    #[inline(always)]
    fn flux(self, half_g: T) -> SWEState2D<T> {
        SWEState2D {
            h: self.h * self.un,
            hu: self.h * self.un * self.un + half_g * self.h * self.h,
            hv: self.h * self.un * self.ut,
        }
    }
}

impl HllSide {
    #[inline]
    fn new(q_rot: &SWEState2D, (un, ut): (f64, f64)) -> Self {
        Self {
            h: q_rot.h,
            hun: q_rot.hu,
            hut: q_rot.hv,
            un,
            ut,
        }
    }
}

/// The HLL flux of [`hll_flux_swe_2d`] in face-aligned components (mass,
/// normal and tangential momentum), from sides whose velocities are already
/// known, so that callers holding velocities (the split-form kernel) skip
/// the divisions.
///
/// Generic over [`Lane`] (the scalar, batched and GPU kernels share it): its
/// branches are selects. Both sides at or below `h_min` give zero; else the
/// left flux where `s_l ≥ 0`, the right where `s_r ≤ 0`, the star flux
/// between.
#[inline(always)]
pub(crate) fn hll_flux_face_aligned<T: Lane>(
    l: HllSide<T>,
    r: HllSide<T>,
    g: f64,
    h_min: f64,
) -> SWEState2D<T> {
    let (h_l, h_r) = (l.h, r.h);
    let zero = h_l.constant(0.0);
    let h_min_v = h_l.constant(h_min);
    let dry = h_l.at_most(h_min_v) & h_r.at_most(h_min_v);

    let g_v = h_l.constant(g);
    let c_l = (g_v * h_l.max(zero)).sqrt();
    let c_r = (g_v * h_r.max(zero)).sqrt();

    // Wave speed estimates (Einfeldt)
    let (s_l, s_r) = einfeldt_speeds_2d(h_l, h_r, l.un, r.un, c_l, c_r, g, h_min);

    // Physical fluxes in rotated coordinates
    let half_g = h_l.constant(0.5 * g);
    let (f_l, f_r) = (l.flux(half_g), r.flux(half_g));

    // HLL flux in rotated coordinates
    let inv_ds = h_l.constant(1.0) / (s_r - s_l);
    let s_lr = s_l * s_r;
    let star = SWEState2D {
        h: inv_ds * (s_r * f_l.h - s_l * f_r.h + s_lr * (h_r - h_l)),
        hu: inv_ds * (s_r * f_l.hu - s_l * f_r.hu + s_lr * (r.hun - l.hun)),
        hv: inv_ds * (s_r * f_l.hv - s_l * f_r.hv + s_lr * (r.hut - l.hut)),
    };
    let (from_l, from_r) = (s_l.at_least(zero), s_r.at_most(zero));
    let pick = |f_l: T, f_r: T, star: T| {
        T::select(
            dry,
            zero,
            T::select(from_l, f_l, T::select(from_r, f_r, star)),
        )
    };
    SWEState2D {
        h: pick(f_l.h, f_r.h, star.h),
        hu: pick(f_l.hu, f_r.hu, star.hu),
        hv: pick(f_l.hv, f_r.hv, star.hv),
    }
}

/// Einfeldt wave speed estimates for 2D (generic over [`Lane`]: the
/// branches are selects).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn einfeldt_speeds_2d<T: Lane>(
    h_l: T,
    h_r: T,
    un_l: T,
    un_r: T,
    c_l: T,
    c_r: T,
    g: f64,
    h_min: f64,
) -> (T, T) {
    let zero = h_l.constant(0.0);
    let two = h_l.constant(2.0);
    let h_min = h_l.constant(h_min);

    // Roe-averaged quantities (zero where both depths vanish)
    let sqrt_h_l = h_l.max(zero).sqrt();
    let sqrt_h_r = h_r.max(zero).sqrt();
    let roe = (sqrt_h_l + sqrt_h_r).greater_than(h_l.constant(1e-10));
    let h_roe = h_l.constant(0.5) * (h_l + h_r);
    let un_roe = T::select(
        roe,
        (sqrt_h_l * un_l + sqrt_h_r * un_r) / (sqrt_h_l + sqrt_h_r),
        zero,
    );
    let c_roe = T::select(roe, (h_l.constant(g) * h_roe).sqrt(), zero);

    let s_l = T::select(
        h_l.greater_than(h_min),
        (un_l - c_l).min(un_roe - c_roe),
        un_r - two * c_r,
    );
    let s_r = T::select(
        h_r.greater_than(h_min),
        (un_r + c_r).max(un_roe + c_roe),
        un_l + two * c_l,
    );
    (s_l, s_r)
}

/// Rusanov (Local Lax-Friedrichs) flux for 2D SWE.
///
/// Simple and robust, but more diffusive than Roe or HLL.
///
/// F* = 0.5(F_L + F_R) · n - 0.5 * λ_max * (q_R - q_L)
pub fn rusanov_flux_swe_2d(
    q_l: &SWEState2D,
    q_r: &SWEState2D,
    normal: (f64, f64),
    g: f64,
    h_min: f64,
) -> SWEState2D {
    let (nx, ny) = normal;

    let h_l = q_l.h;
    let h_r = q_r.h;

    // Handle dry cells
    if h_l <= h_min && h_r <= h_min {
        return SWEState2D::zero();
    }

    // Velocities
    let (u_l, v_l) = if h_l > h_min {
        (q_l.hu / h_l, q_l.hv / h_l)
    } else {
        (0.0, 0.0)
    };

    let (u_r, v_r) = if h_r > h_min {
        (q_r.hu / h_r, q_r.hv / h_r)
    } else {
        (0.0, 0.0)
    };

    // Normal velocities
    let un_l = u_l * nx + v_l * ny;
    let un_r = u_r * nx + v_r * ny;

    // Wave celerities
    let c_l = (g * h_l.max(0.0)).sqrt();
    let c_r = (g * h_r.max(0.0)).sqrt();

    // Maximum wave speed
    let lambda_max = (un_l.abs() + c_l).max(un_r.abs() + c_r);

    // Physical fluxes F·n = F*nx + G*ny
    // F = [hu, h*u² + gh²/2, h*u*v]
    // G = [hv, h*u*v, h*v² + gh²/2]
    // F·n = [h*un, (h*u² + gh²/2)*nx + h*u*v*ny, h*u*v*nx + (h*v² + gh²/2)*ny]
    let pressure_l = 0.5 * g * h_l * h_l;
    let f_l = SWEState2D {
        h: h_l * un_l,
        hu: (h_l * u_l * u_l + pressure_l) * nx + h_l * u_l * v_l * ny,
        hv: h_l * u_l * v_l * nx + (h_l * v_l * v_l + pressure_l) * ny,
    };

    let pressure_r = 0.5 * g * h_r * h_r;
    let f_r = SWEState2D {
        h: h_r * un_r,
        hu: (h_r * u_r * u_r + pressure_r) * nx + h_r * u_r * v_r * ny,
        hv: h_r * u_r * v_r * nx + (h_r * v_r * v_r + pressure_r) * ny,
    };

    // Rusanov flux
    SWEState2D {
        h: 0.5 * (f_l.h + f_r.h) - 0.5 * lambda_max * (h_r - h_l),
        hu: 0.5 * (f_l.hu + f_r.hu) - 0.5 * lambda_max * (q_r.hu - q_l.hu),
        hv: 0.5 * (f_l.hv + f_r.hv) - 0.5 * lambda_max * (q_r.hv - q_l.hv),
    }
}

/// Enum to select 2D SWE flux type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SWEFluxType2D {
    /// Roe approximate Riemann solver
    Roe,
    /// HLL solver (more robust for strong shocks)
    HLL,
    /// Rusanov/Local Lax-Friedrichs (simple, robust, diffusive)
    Rusanov,
}

/// Compute numerical flux using selected method.
pub fn compute_flux_swe_2d(
    q_l: &SWEState2D,
    q_r: &SWEState2D,
    normal: (f64, f64),
    g: f64,
    h_min: f64,
    flux_type: SWEFluxType2D,
) -> SWEState2D {
    match flux_type {
        SWEFluxType2D::Roe => roe_flux_swe_2d(q_l, q_r, normal, g, h_min),
        SWEFluxType2D::HLL => hll_flux_swe_2d(q_l, q_r, normal, g, h_min),
        SWEFluxType2D::Rusanov => rusanov_flux_swe_2d(q_l, q_r, normal, g, h_min),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 10.0;
    const H_MIN: f64 = 1e-6;
    const TOL: f64 = 1e-10;

    #[test]
    fn test_rotation() {
        // State: h=2, u=3, v=4
        let state = SWEState2D::from_primitives(2.0, 3.0, 4.0);

        // Rotate to x-direction (n = (1, 0))
        let rotated = rotate_to_normal(&state, 1.0, 0.0);
        assert!((rotated.h - 2.0).abs() < TOL);
        assert!((rotated.hu - 6.0).abs() < TOL); // h*u_n = 2*3
        assert!((rotated.hv - 8.0).abs() < TOL); // h*u_t = 2*4

        // Rotate back
        let back = rotate_from_normal(&rotated, 1.0, 0.0);
        assert!((back.h - state.h).abs() < TOL);
        assert!((back.hu - state.hu).abs() < TOL);
        assert!((back.hv - state.hv).abs() < TOL);
    }

    #[test]
    fn test_rotation_45_degrees() {
        // n = (1/√2, 1/√2) - 45 degrees
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        let state = SWEState2D::from_primitives(1.0, 1.0, 0.0);

        let rotated = rotate_to_normal(&state, sqrt2_inv, sqrt2_inv);
        // u_n = u*nx + v*ny = 1 * 1/√2 + 0 * 1/√2 = 1/√2
        assert!((rotated.hu - sqrt2_inv).abs() < TOL);
        // u_t = -u*ny + v*nx = -1 * 1/√2 + 0 * 1/√2 = -1/√2
        assert!((rotated.hv - (-sqrt2_inv)).abs() < TOL);
    }

    #[test]
    fn test_roe_flux_continuous() {
        // For continuous solution, flux should equal physical flux · n
        // State: h=2, u=3, v=1, so hu=6, hv=2
        let state = SWEState2D::from_primitives(2.0, 3.0, 1.0);

        let flux_x = roe_flux_swe_2d(&state, &state, (1.0, 0.0), G, H_MIN);
        let flux_y = roe_flux_swe_2d(&state, &state, (0.0, 1.0), G, H_MIN);

        // F_x = [hu, h*u² + gh²/2, h*u*v] = [6, 18+20, 6] = [6, 38, 6]
        assert!((flux_x.h - 6.0).abs() < TOL);
        assert!((flux_x.hu - 38.0).abs() < TOL);
        assert!((flux_x.hv - 6.0).abs() < TOL);

        // G_y = [hv, h*u*v, h*v² + gh²/2] = [2, 6, 2+20] = [2, 6, 22]
        assert!((flux_y.h - 2.0).abs() < TOL);
        assert!((flux_y.hu - 6.0).abs() < TOL);
        assert!((flux_y.hv - 22.0).abs() < TOL);
    }

    #[test]
    fn test_roe_flux_still_water() {
        let state = SWEState2D::new(2.0, 0.0, 0.0);

        let flux = roe_flux_swe_2d(&state, &state, (1.0, 0.0), G, H_MIN);

        // Still water in x: F = [0, gh^2/2, 0] = [0, 20, 0]
        assert!(flux.h.abs() < TOL);
        assert!((flux.hu - 20.0).abs() < TOL);
        assert!(flux.hv.abs() < TOL);
    }

    #[test]
    fn test_hll_flux_continuous() {
        // State: h=2, u=3, v=1
        let state = SWEState2D::from_primitives(2.0, 3.0, 1.0);

        let flux = hll_flux_swe_2d(&state, &state, (1.0, 0.0), G, H_MIN);

        // Should equal physical flux: [6, 38, 6]
        assert!((flux.h - 6.0).abs() < TOL);
        assert!((flux.hu - 38.0).abs() < TOL);
        assert!((flux.hv - 6.0).abs() < TOL);
    }

    #[test]
    fn test_rusanov_flux_continuous() {
        // State: h=2, u=3, v=1
        let state = SWEState2D::from_primitives(2.0, 3.0, 1.0);

        let flux = rusanov_flux_swe_2d(&state, &state, (1.0, 0.0), G, H_MIN);

        // Should equal physical flux: [6, 38, 6]
        assert!((flux.h - 6.0).abs() < TOL);
        assert!((flux.hu - 38.0).abs() < TOL);
        assert!((flux.hv - 6.0).abs() < TOL);
    }

    #[test]
    fn test_dam_break_x() {
        // Dam break: high water on left, low on right
        let q_l = SWEState2D::new(2.0, 0.0, 0.0);
        let q_r = SWEState2D::new(0.5, 0.0, 0.0);

        let flux_roe = roe_flux_swe_2d(&q_l, &q_r, (1.0, 0.0), G, H_MIN);
        let flux_hll = hll_flux_swe_2d(&q_l, &q_r, (1.0, 0.0), G, H_MIN);

        // Mass should flow to the right (positive flux)
        assert!(flux_roe.h > 0.0);
        assert!(flux_hll.h > 0.0);

        // Momentum flux should be positive
        assert!(flux_roe.hu > 0.0);
        assert!(flux_hll.hu > 0.0);
    }

    #[test]
    fn test_dam_break_diagonal() {
        // Dam break with 45-degree normal
        let q_l = SWEState2D::new(2.0, 0.0, 0.0);
        let q_r = SWEState2D::new(0.5, 0.0, 0.0);

        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        let flux = roe_flux_swe_2d(&q_l, &q_r, (sqrt2_inv, sqrt2_inv), G, H_MIN);

        // Mass should flow in the normal direction
        assert!(flux.h > 0.0);
    }

    #[test]
    fn test_dry_cells() {
        let q_l = SWEState2D::new(0.0, 0.0, 0.0);
        let q_r = SWEState2D::new(0.0, 0.0, 0.0);

        let flux_roe = roe_flux_swe_2d(&q_l, &q_r, (1.0, 0.0), G, H_MIN);
        let flux_hll = hll_flux_swe_2d(&q_l, &q_r, (1.0, 0.0), G, H_MIN);
        let flux_rus = rusanov_flux_swe_2d(&q_l, &q_r, (1.0, 0.0), G, H_MIN);

        for flux in [flux_roe, flux_hll, flux_rus] {
            assert!(flux.h.abs() < TOL);
            assert!(flux.hu.abs() < TOL);
            assert!(flux.hv.abs() < TOL);
        }
    }

    #[test]
    fn test_fluxes_agree_on_uniform_flow() {
        // All fluxes should agree for uniform flow
        let state = SWEState2D::from_primitives(2.0, 1.0, 0.5);
        let normal = (0.6, 0.8); // Not axis-aligned

        let flux_roe = roe_flux_swe_2d(&state, &state, normal, G, H_MIN);
        let flux_hll = hll_flux_swe_2d(&state, &state, normal, G, H_MIN);
        let flux_rus = rusanov_flux_swe_2d(&state, &state, normal, G, H_MIN);

        assert!((flux_roe.h - flux_hll.h).abs() < TOL);
        assert!((flux_roe.hu - flux_hll.hu).abs() < TOL);
        assert!((flux_roe.hv - flux_hll.hv).abs() < TOL);

        assert!((flux_roe.h - flux_rus.h).abs() < TOL);
        assert!((flux_roe.hu - flux_rus.hu).abs() < TOL);
        assert!((flux_roe.hv - flux_rus.hv).abs() < TOL);
    }

    #[test]
    fn test_flux_conservation() {
        // Flux leaving one cell = flux entering neighbor (opposite normals)
        let q_l = SWEState2D::from_primitives(2.0, 1.0, 0.5);
        let q_r = SWEState2D::from_primitives(1.5, 0.8, 0.3);

        let flux_out = roe_flux_swe_2d(&q_l, &q_r, (1.0, 0.0), G, H_MIN);
        let flux_in = roe_flux_swe_2d(&q_r, &q_l, (-1.0, 0.0), G, H_MIN);

        // flux_out + flux_in should be zero (conservation)
        assert!((flux_out.h + flux_in.h).abs() < TOL);
        assert!((flux_out.hu + flux_in.hu).abs() < TOL);
        assert!((flux_out.hv + flux_in.hv).abs() < TOL);
    }

    #[test]
    fn test_compute_flux_dispatch() {
        let q_l = SWEState2D::from_primitives(2.0, 1.0, 0.0);
        let q_r = SWEState2D::from_primitives(1.5, 0.8, 0.0);
        let normal = (1.0, 0.0);

        let roe = compute_flux_swe_2d(&q_l, &q_r, normal, G, H_MIN, SWEFluxType2D::Roe);
        let hll = compute_flux_swe_2d(&q_l, &q_r, normal, G, H_MIN, SWEFluxType2D::HLL);
        let rus = compute_flux_swe_2d(&q_l, &q_r, normal, G, H_MIN, SWEFluxType2D::Rusanov);

        // They should all be finite
        assert!(roe.h.is_finite());
        assert!(hll.h.is_finite());
        assert!(rus.h.is_finite());
    }

    #[test]
    fn test_tangential_velocity_preserved() {
        // Pure tangential flow (perpendicular to normal)
        // n = (1, 0), so tangential is y-direction
        // State: h=1, u=0, v=2 (pure tangential flow)
        let state = SWEState2D::from_primitives(1.0, 0.0, 2.0);

        let flux = roe_flux_swe_2d(&state, &state, (1.0, 0.0), G, H_MIN);

        // With zero normal velocity, mass flux should be zero
        assert!(flux.h.abs() < TOL);
        // x-momentum flux is just pressure: gh^2/2 = 5
        assert!((flux.hu - 5.0).abs() < TOL);
        // y-momentum flux should be zero (no normal velocity to advect it)
        assert!(flux.hv.abs() < TOL);
    }
}
