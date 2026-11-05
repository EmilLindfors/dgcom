//! Equation of State (EOS) for seawater.
//!
//! Computes density from potential temperature, salinity, and pressure (depth).
//!
//! Based on the standard Jackett and McDougall (1995) or UNESCO 1980 EOS.
//!
//! # Equations
//!
//! $\rho(S, T, z)$
//!
//! In Boussinesq models, we often care about $\rho_{pot}$ referenced to surface
//! or the in-situ density perturbation.
//!
//! For this implementation, we use a linearized or simplified nonlinear EOS suitable
//! for coastal applications.

use crate::solver::core::blocks::{Blocks, for_each_block};
use crate::solver::state::Solution3D;

/// Equation of State trait.
pub trait EquationOfState: Send + Sync {
    /// Compute density from T, S, and depth.
    ///
    /// # Arguments
    /// * `t`: Potential temperature [deg C]
    /// * `s`: Salinity [PSU]
    /// * `z`: Depth [m] (negative downwards)
    ///
    /// Returns density [kg/m^3].
    fn compute_density(&self, t: f64, s: f64, z: f64) -> f64;

    /// Update the density field in the 3D solution.
    fn update_density(&self, state: &mut Solution3D) {
        let all = Blocks::All(state.n_elements);
        self.update_density_on(state, all);
    }

    /// Update the density field on `elements`, from
    /// [`Self::compute_density`] at the surface (no pressure dependence).
    fn update_density_on(&self, state: &mut Solution3D, elements: Blocks) {
        let Solution3D {
            temp, salt, rho, ..
        } = state;
        let (temp, salt): (&[f64], &[f64]) = (temp, salt);
        let len = rho.len() / elements.n().max(1);
        for_each_block(
            elements,
            [&mut rho[..]],
            || (),
            |_, k, [rho]| {
                let block = k * len..(k + 1) * len;
                for ((r, &t), &s) in rho.iter_mut().zip(&temp[block.clone()]).zip(&salt[block]) {
                    *r = self.compute_density(t, s, 0.0);
                }
            },
        );
    }
}

/// Linear Equation of State.
///
/// $\rho = \rho_0 (1 - \alpha (T - T_0) + \beta (S - S_0))$
#[derive(Debug, Clone, Copy)]
pub struct LinearEOS {
    pub rho0: f64,
    pub t0: f64,
    pub s0: f64,
    pub alpha: f64, // Thermal expansion [1/C]
    pub beta: f64,  // Haline contraction [1/PSU]
}

impl Default for LinearEOS {
    fn default() -> Self {
        Self {
            rho0: 1025.0,
            t0: 10.0,
            s0: 35.0,
            alpha: 1.7e-4,
            beta: 7.6e-4,
        }
    }
}

impl EquationOfState for LinearEOS {
    fn compute_density(&self, t: f64, s: f64, _z: f64) -> f64 {
        self.rho0 * (1.0 - self.alpha * (t - self.t0) + self.beta * (s - self.s0))
    }
}

/// UNESCO 1980 Equation of State (simplified).
///
/// Ignores pressure effects (potential density at surface pressure).
#[derive(Debug, Clone, Copy)]
pub struct UnescoEOS {
    pub rho0: f64,
}

impl EquationOfState for UnescoEOS {
    fn compute_density(&self, t: f64, s: f64, _z: f64) -> f64 {
        // Simple polynomial fit (Jackett & McDougall 1995 or similar)
        // Values for coefficients...
        // This is a placeholder for the full nonlinear implementation.
        let t2 = t * t;
        let s32 = s.powf(1.5);

        // Approx coeff for sigma-t (kg/m^3 - 1000)
        999.842594 + 6.793952e-2 * t - 9.095290e-3 * t2
            + 1.001685e-4 * t2 * t
            + (0.824493 - 4.0899e-3 * t + 7.6438e-5 * t2) * s
            - 5.72466e-3 * s32
            + 4.8314e-4 * s * s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_eos() {
        let eos = LinearEOS::default();
        // T=T0, S=S0 -> rho=rho0
        let rho = eos.compute_density(10.0, 35.0, 0.0);
        assert!((rho - 1025.0).abs() < 1e-10);

        // Warmer -> lighter
        let rho_warm = eos.compute_density(20.0, 35.0, 0.0);
        assert!(rho_warm < 1025.0);

        // Saltier -> heavier
        let rho_salty = eos.compute_density(10.0, 36.0, 0.0);
        assert!(rho_salty > 1025.0);
    }

    #[test]
    fn test_update_density() {
        let n_elem = 1;
        let n_nodes = 1;
        let n_levels = 2;
        let mut state = Solution3D::new(n_elem, n_nodes, n_levels);

        state.temp.fill(20.0);
        state.salt.fill(35.0);

        let eos = LinearEOS::default();
        eos.update_density(&mut state);

        let expected = eos.compute_density(20.0, 35.0, 0.0);
        assert!((state.rho[0] - expected).abs() < 1e-10);
        assert!((state.rho[1] - expected).abs() < 1e-10);
    }
}
