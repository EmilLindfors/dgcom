//! Equation of State for seawater density.
//!
//! Computes seawater density ρ(T, S, p) from temperature, salinity, and pressure.
//! This module implements the UNESCO EOS-80 formulation which is standard in
//! oceanographic modeling.
//!
//! # References
//!
//! - UNESCO (1981): Tenth report of the joint panel on oceanographic tables and standards.
//! - Millero & Poisson (1981): International one-atmosphere equation of state of seawater.
//!
//! # Units
//!
//! - Temperature: °C (ITS-90)
//! - Salinity: PSU (practical salinity units)
//! - Pressure: dbar (decibars), where 1 dbar ≈ 1 m depth
//! - Density: kg/m³
//!
//! # Norwegian Coast Context
//!
//! Typical values:
//! - Temperature: 4-18°C (seasonal)
//! - Salinity: 25-35 PSU (fresher in fjords due to river input)
//! - Surface density: 1020-1028 kg/m³
//!
//! Density gradients drive baroclinic circulation in fjords where fresh
//! river water meets saline Atlantic water.

/// Reference density for seawater (kg/m³).
///
/// Used for Boussinesq approximation: ρ = ρ₀(1 + ρ'/ρ₀)
pub const RHO_0: f64 = 1025.0;

/// Equation of State calculator for seawater.
///
/// Provides methods to compute density and related quantities from
/// temperature, salinity, and optionally pressure.
#[derive(Clone, Debug)]
pub struct EquationOfState {
    /// Reference density (kg/m³)
    pub rho_0: f64,
    /// Include pressure effects (if false, uses surface pressure only)
    pub include_pressure: bool,
}

impl Default for EquationOfState {
    fn default() -> Self {
        Self::new()
    }
}

impl EquationOfState {
    /// Create a new equation of state calculator with default reference density.
    pub fn new() -> Self {
        Self {
            rho_0: RHO_0,
            include_pressure: false,
        }
    }

    /// Create with custom reference density.
    pub fn with_rho_0(rho_0: f64) -> Self {
        Self {
            rho_0,
            include_pressure: false,
        }
    }

    /// Create with pressure effects included.
    pub fn with_pressure() -> Self {
        Self {
            rho_0: RHO_0,
            include_pressure: true,
        }
    }

    /// Compute seawater density at surface pressure (p = 0).
    ///
    /// Uses the UNESCO EOS-80 one-atmosphere equation of state.
    ///
    /// # Arguments
    /// * `temperature` - Temperature in °C
    /// * `salinity` - Salinity in PSU
    ///
    /// # Returns
    /// Density in kg/m³
    ///
    /// # Example
    /// ```
    /// use dgcom::equations::EquationOfState;
    ///
    /// let eos = EquationOfState::new();
    ///
    /// // Typical Norwegian coastal water
    /// let rho = eos.density_surface(8.0, 34.0);
    /// assert!((rho - 1026.4).abs() < 0.5); // Approximately 1026 kg/m³
    ///
    /// // Fresh water at 4°C (maximum density)
    /// let rho_fresh = eos.density_surface(4.0, 0.0);
    /// assert!((rho_fresh - 1000.0).abs() < 0.1);
    /// ```
    pub fn density_surface(&self, temperature: f64, salinity: f64) -> f64 {
        // UNESCO EOS-80 coefficients for surface density
        let t = temperature;
        let s = salinity;

        // Pure water density (Bigg formula)
        let rho_w = 999.842594 + 6.793952e-2 * t - 9.095290e-3 * t.powi(2)
            + 1.001685e-4 * t.powi(3)
            - 1.120083e-6 * t.powi(4)
            + 6.536336e-9 * t.powi(5);

        // Coefficients for salinity terms
        let a0 = 8.24493e-1;
        let a1 = -4.0899e-3;
        let a2 = 7.6438e-5;
        let a3 = -8.2467e-7;
        let a4 = 5.3875e-9;

        let b0 = -5.72466e-3;
        let b1 = 1.0227e-4;
        let b2 = -1.6546e-6;

        let c0 = 4.8314e-4;

        // Salinity contribution
        let a = a0 + a1 * t + a2 * t.powi(2) + a3 * t.powi(3) + a4 * t.powi(4);
        let b = b0 + b1 * t + b2 * t.powi(2);

        rho_w + a * s + b * s.powf(1.5) + c0 * s.powi(2)
    }

    /// Compute seawater density at given pressure.
    ///
    /// ρ(S, T, p) = ρ(S, T, 0) / (1 − p / K(S, T, p)), with the UNESCO secant
    /// bulk modulus `K` and `p` in bar (converted from the dbar argument).
    ///
    /// # Arguments
    /// * `temperature` - Temperature in °C
    /// * `salinity` - Salinity in PSU
    /// * `pressure` - Pressure in dbar (≈ depth in meters)
    ///
    /// # Returns
    /// Density in kg/m³
    pub fn density(&self, temperature: f64, salinity: f64, pressure: f64) -> f64 {
        if !self.include_pressure || pressure.abs() < 1e-6 {
            return self.density_surface(temperature, salinity);
        }

        let t = temperature;
        let s = salinity;
        // The UNESCO (Millero & Poisson 1981) bulk-modulus fit is in bar.
        let p_bar = pressure / 10.0;

        // Surface density
        let rho_0 = self.density_surface(t, s);

        // Secant bulk modulus K(S, T, p)
        let k = self.secant_bulk_modulus(t, s, p_bar);

        // Density at pressure
        rho_0 / (1.0 - p_bar / k)
    }

    /// Compute density anomaly σ = ρ - 1000 kg/m³.
    ///
    /// The density anomaly is often more convenient to work with since
    /// seawater density is always close to 1000 kg/m³.
    pub fn sigma(&self, temperature: f64, salinity: f64) -> f64 {
        self.density_surface(temperature, salinity) - 1000.0
    }

    /// Compute buoyancy b = -g(ρ - ρ₀)/ρ₀.
    ///
    /// Positive buoyancy indicates water lighter than reference.
    ///
    /// # Arguments
    /// * `temperature` - Temperature in °C
    /// * `salinity` - Salinity in PSU
    /// * `g` - Gravitational acceleration (m/s²)
    pub fn buoyancy(&self, temperature: f64, salinity: f64, g: f64) -> f64 {
        let rho = self.density_surface(temperature, salinity);
        -g * (rho - self.rho_0) / self.rho_0
    }

    /// Compute density ratio ρ'/ρ₀ = (ρ - ρ₀)/ρ₀.
    ///
    /// Used in Boussinesq approximation for pressure gradient.
    pub fn density_ratio(&self, temperature: f64, salinity: f64) -> f64 {
        let rho = self.density_surface(temperature, salinity);
        (rho - self.rho_0) / self.rho_0
    }

    /// Compute hydrostatic pressure from depth.
    ///
    /// Approximates pressure as p ≈ ρ₀ * g * |z| / 10000 (converts Pa to dbar).
    ///
    /// # Arguments
    /// * `z` - Physical depth (negative downwards, in meters)
    /// * `g` - Gravitational acceleration (m/s²)
    ///
    /// # Returns
    /// Pressure in dbar (decibars)
    pub fn hydrostatic_pressure_approx(z: f64, g: f64) -> f64 {
        // p = rho * g * h
        // 1 dbar = 10^4 Pa
        // p[dbar] = 1025 * 9.81 * |z| / 10000 ≈ |z|

        let depth = z.abs();
        RHO_0 * g * depth / 10000.0
    }

    /// Compute density for a vertical column (batched).
    ///
    /// Useful for SIMD acceleration in 3D models.
    ///
    /// # Arguments
    /// * `temp` - Temperature column (°C)
    /// * `salt` - Salinity column (PSU)
    /// * `z` - Vertical coordinate column (m, negative downwards)
    /// * `rho_out` - Output density column (kg/m³)
    /// * `g` - Gravitational acceleration (m/s²)
    pub fn compute_density_column(
        &self,
        temp: &[f64],
        salt: &[f64],
        z: &[f64],
        rho_out: &mut [f64],
        g: f64,
    ) {
        debug_assert_eq!(temp.len(), salt.len());
        debug_assert_eq!(temp.len(), z.len());
        debug_assert_eq!(temp.len(), rho_out.len());

        for i in 0..temp.len() {
            let p = if self.include_pressure {
                Self::hydrostatic_pressure_approx(z[i], g)
            } else {
                0.0
            };
            rho_out[i] = self.density(temp[i], salt[i], p);
        }
    }

    /// Thermal expansion coefficient α = -(1/ρ)(∂ρ/∂T)|_{S,p}.
    ///
    /// Units: 1/°C
    ///
    /// Typical value: ~2×10⁻⁴ /°C for seawater at 10°C, 35 PSU
    pub fn thermal_expansion(&self, temperature: f64, salinity: f64) -> f64 {
        let dt = 0.01; // Small temperature perturbation
        let rho = self.density_surface(temperature, salinity);
        let rho_plus = self.density_surface(temperature + dt, salinity);
        let rho_minus = self.density_surface(temperature - dt, salinity);

        -((rho_plus - rho_minus) / (2.0 * dt)) / rho
    }

    /// Haline contraction coefficient β = (1/ρ)(∂ρ/∂S)|_{T,p}.
    ///
    /// Units: 1/PSU
    ///
    /// Typical value: ~7.5×10⁻⁴ /PSU for seawater at 10°C, 35 PSU
    pub fn haline_contraction(&self, temperature: f64, salinity: f64) -> f64 {
        let ds = 0.01; // Small salinity perturbation
        let rho = self.density_surface(temperature, salinity);
        let rho_plus = self.density_surface(temperature, salinity + ds);
        let rho_minus = self.density_surface(temperature, salinity - ds);

        ((rho_plus - rho_minus) / (2.0 * ds)) / rho
    }

    /// Compute the secant bulk modulus K(S, T, p) in bar for pressure effects.
    ///
    /// Internal helper function for density at depth. `pressure_bar` is in
    /// **bar** (1 bar = 10 dbar), the unit of the UNESCO fit.
    fn secant_bulk_modulus(&self, temperature: f64, salinity: f64, pressure_bar: f64) -> f64 {
        let t = temperature;
        let s = salinity;
        let p = pressure_bar;

        // Pure water secant bulk modulus
        let kw = 19652.21 + 148.4206 * t - 2.327105 * t.powi(2) + 1.360477e-2 * t.powi(3)
            - 5.155288e-5 * t.powi(4);

        // Salinity contribution at p=0
        let k0 = kw
            + s * (54.6746 - 0.603459 * t + 1.09987e-2 * t.powi(2) - 6.1670e-5 * t.powi(3))
            + s.powf(1.5) * (7.944e-2 + 1.6483e-2 * t - 5.3009e-4 * t.powi(2));

        // Pressure contribution
        let aw = 3.239908 + 1.43713e-3 * t + 1.16092e-4 * t.powi(2) - 5.77905e-7 * t.powi(3);
        let a =
            aw + s * (2.2838e-3 - 1.0981e-5 * t - 1.6078e-6 * t.powi(2)) + s.powf(1.5) * 1.91075e-4;

        let bw = 8.50935e-5 - 6.12293e-6 * t + 5.2787e-8 * t.powi(2);
        let b = bw + s * (-9.9348e-7 + 2.0816e-8 * t + 9.1697e-10 * t.powi(2));

        k0 + p * (a + b * p)
    }

    /// Compute the speed of sound in seawater.
    ///
    /// Uses UNESCO formula. Returns speed in m/s.
    ///
    /// # Arguments
    /// * `temperature` - Temperature in °C
    /// * `salinity` - Salinity in PSU
    /// * `pressure` - Pressure in dbar (≈ depth in m)
    pub fn sound_speed(&self, temperature: f64, salinity: f64, pressure: f64) -> f64 {
        let t = temperature;
        let s = salinity;
        let p = pressure / 10.0; // Convert dbar to bar

        // Chen-Millero formula (simplified)
        let c00 = 1402.388;
        let c01 = 5.03830;
        let c02 = -5.81090e-2;
        let c03 = 3.3432e-4;
        let c04 = -1.47797e-6;
        let c05 = 3.1419e-9;

        let c10 = 0.153563;
        let c11 = 6.8999e-4;
        let c12 = -8.1829e-6;
        let c13 = 1.3632e-7;
        let c14 = -6.1260e-10;

        let c20 = 3.1260e-5;
        let c21 = -1.7111e-6;
        let c22 = 2.5986e-8;
        let c23 = -2.5353e-10;
        let c24 = 1.0415e-12;

        let c30 = -9.7729e-9;
        let c31 = 3.8513e-10;
        let c32 = -2.3654e-12;

        let a00 = 1.389;
        let a01 = -1.262e-2;
        let a02 = 7.166e-5;
        let a03 = 2.008e-6;
        let a04 = -3.21e-8;

        let a10 = 9.4742e-5;
        let a11 = -1.2583e-5;
        let a12 = -6.4928e-8;
        let a13 = 1.0515e-8;
        let a14 = -2.0142e-10;

        let a20 = -3.9064e-7;
        let a21 = 9.1061e-9;
        let a22 = -1.6009e-10;
        let a23 = 7.994e-12;

        let a30 = 1.100e-10;
        let a31 = 6.651e-12;
        let a32 = -3.391e-13;

        let b00 = -1.922e-2;
        let b01 = -4.42e-5;
        let b10 = 7.3637e-5;
        let b11 = 1.7950e-7;

        let d00 = 1.727e-3;
        let d10 = -7.9836e-6;

        // Pure water contribution
        let cw = c00
            + c01 * t
            + c02 * t.powi(2)
            + c03 * t.powi(3)
            + c04 * t.powi(4)
            + c05 * t.powi(5)
            + (c10 + c11 * t + c12 * t.powi(2) + c13 * t.powi(3) + c14 * t.powi(4)) * p
            + (c20 + c21 * t + c22 * t.powi(2) + c23 * t.powi(3) + c24 * t.powi(4)) * p.powi(2)
            + (c30 + c31 * t + c32 * t.powi(2)) * p.powi(3);

        // Salinity contribution
        let a = a00
            + a01 * t
            + a02 * t.powi(2)
            + a03 * t.powi(3)
            + a04 * t.powi(4)
            + (a10 + a11 * t + a12 * t.powi(2) + a13 * t.powi(3) + a14 * t.powi(4)) * p
            + (a20 + a21 * t + a22 * t.powi(2) + a23 * t.powi(3)) * p.powi(2)
            + (a30 + a31 * t + a32 * t.powi(2)) * p.powi(3);

        let b = b00 + b01 * t + (b10 + b11 * t) * p;

        let d = d00 + d10 * p;

        cw + a * s + b * s.powf(1.5) + d * s.powi(2)
    }

    /// Compute the freezing point of seawater.
    ///
    /// Returns temperature in °C at which seawater freezes.
    ///
    /// # Arguments
    /// * `salinity` - Salinity in PSU
    /// * `pressure` - Pressure in dbar (≈ depth in m)
    pub fn freezing_point(&self, salinity: f64, pressure: f64) -> f64 {
        let s = salinity;
        let p = pressure;

        // UNESCO formula
        -0.0575 * s + 1.710523e-3 * s.powf(1.5) - 2.154996e-4 * s.powi(2) - 7.53e-4 * p
    }
}

/// Linear equation of state for simplified/faster calculations.
///
/// ρ = ρ₀ * (1 - α(T - T₀) + β(S - S₀))
///
/// This is a good approximation for small T, S variations and is much
/// faster to compute than the full UNESCO formula.
#[derive(Clone, Debug)]
pub struct LinearEquationOfState {
    /// Reference density (kg/m³)
    pub rho_0: f64,
    /// Reference temperature (°C)
    pub t_0: f64,
    /// Reference salinity (PSU)
    pub s_0: f64,
    /// Thermal expansion coefficient (1/°C)
    pub alpha: f64,
    /// Haline contraction coefficient (1/PSU)
    pub beta: f64,
}

impl Default for LinearEquationOfState {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearEquationOfState {
    /// Create a linear EOS with typical Norwegian coast parameters.
    ///
    /// Reference state: T₀ = 8°C, S₀ = 34 PSU
    pub fn new() -> Self {
        Self {
            rho_0: 1026.0,
            t_0: 8.0,
            s_0: 34.0,
            alpha: 1.7e-4, // Typical thermal expansion at 8°C
            beta: 7.6e-4,  // Typical haline contraction
        }
    }

    /// Create with custom reference state and coefficients.
    pub fn with_params(rho_0: f64, t_0: f64, s_0: f64, alpha: f64, beta: f64) -> Self {
        Self {
            rho_0,
            t_0,
            s_0,
            alpha,
            beta,
        }
    }

    /// Compute density using linear approximation.
    pub fn density(&self, temperature: f64, salinity: f64) -> f64 {
        self.rho_0
            * (1.0 - self.alpha * (temperature - self.t_0) + self.beta * (salinity - self.s_0))
    }

    /// Compute density anomaly ρ' = ρ - ρ₀.
    pub fn density_anomaly(&self, temperature: f64, salinity: f64) -> f64 {
        self.rho_0 * (-self.alpha * (temperature - self.t_0) + self.beta * (salinity - self.s_0))
    }

    /// Compute buoyancy b = -g(ρ - ρ₀)/ρ₀ = g*α*(T - T₀) - g*β*(S - S₀).
    pub fn buoyancy(&self, temperature: f64, salinity: f64, g: f64) -> f64 {
        g * (self.alpha * (temperature - self.t_0) - self.beta * (salinity - self.s_0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 0.1; // 0.1 kg/m³ tolerance

    #[test]
    fn test_pure_water_density() {
        let eos = EquationOfState::new();

        // Pure water at 4°C has maximum density ~1000 kg/m³
        let rho_4c = eos.density_surface(4.0, 0.0);
        assert!((rho_4c - 1000.0).abs() < TOL);

        // Pure water at 0°C
        let rho_0c = eos.density_surface(0.0, 0.0);
        assert!((rho_0c - 999.84).abs() < TOL);

        // Pure water at 20°C
        let rho_20c = eos.density_surface(20.0, 0.0);
        assert!((rho_20c - 998.2).abs() < TOL);
    }

    #[test]
    fn test_seawater_density() {
        let eos = EquationOfState::new();

        // Standard seawater: T=10°C, S=35 PSU
        let rho = eos.density_surface(10.0, 35.0);
        assert!((rho - 1026.97).abs() < TOL);

        // Cold Atlantic water: T=0°C, S=35 PSU
        let rho_cold = eos.density_surface(0.0, 35.0);
        assert!((rho_cold - 1028.1).abs() < TOL);

        // Warm tropical: T=25°C, S=35 PSU
        let rho_warm = eos.density_surface(25.0, 35.0);
        assert!((rho_warm - 1023.3).abs() < TOL);
    }

    #[test]
    fn test_norwegian_coastal_density() {
        let eos = EquationOfState::new();

        // Typical Norwegian coastal water
        let rho = eos.density_surface(8.0, 34.0);
        assert!(rho > 1020.0 && rho < 1030.0);

        // Fresh fjord surface water
        let rho_fresh = eos.density_surface(10.0, 20.0);
        assert!(rho_fresh < rho); // Fresher water is lighter
    }

    #[test]
    fn test_density_temperature_dependence() {
        let eos = EquationOfState::new();

        // Warmer water should be lighter (at constant salinity)
        let rho_cold = eos.density_surface(5.0, 35.0);
        let rho_warm = eos.density_surface(15.0, 35.0);
        assert!(rho_cold > rho_warm);
    }

    #[test]
    fn test_density_salinity_dependence() {
        let eos = EquationOfState::new();

        // Saltier water should be heavier (at constant temperature)
        let rho_fresh = eos.density_surface(10.0, 30.0);
        let rho_salty = eos.density_surface(10.0, 35.0);
        assert!(rho_salty > rho_fresh);
    }

    #[test]
    fn test_thermal_expansion() {
        let eos = EquationOfState::new();

        // Typical thermal expansion coefficient
        let alpha = eos.thermal_expansion(10.0, 35.0);
        assert!(alpha > 1e-4 && alpha < 3e-4); // ~2e-4 /°C
    }

    #[test]
    fn test_haline_contraction() {
        let eos = EquationOfState::new();

        // Typical haline contraction coefficient
        let beta = eos.haline_contraction(10.0, 35.0);
        assert!(beta > 5e-4 && beta < 1e-3); // ~7.5e-4 /PSU
    }

    #[test]
    fn test_freezing_point() {
        let eos = EquationOfState::new();

        // Seawater freezes at lower temperature than fresh water
        let tf_fresh = eos.freezing_point(0.0, 0.0);
        let tf_sw = eos.freezing_point(35.0, 0.0);

        assert!((tf_fresh - 0.0).abs() < 0.1);
        assert!(tf_sw < -1.0 && tf_sw > -2.5); // About -1.9°C for S=35
    }

    #[test]
    fn test_sound_speed() {
        let eos = EquationOfState::new();

        // Sound speed in seawater ~1500 m/s
        let c = eos.sound_speed(10.0, 35.0, 0.0);
        assert!(c > 1480.0 && c < 1550.0);

        // Sound speed increases with temperature
        let c_cold = eos.sound_speed(0.0, 35.0, 0.0);
        let c_warm = eos.sound_speed(20.0, 35.0, 0.0);
        assert!(c_warm > c_cold);
    }

    #[test]
    fn test_linear_eos() {
        let eos_full = EquationOfState::new();
        let eos_lin = LinearEquationOfState::new();

        // At reference state, densities should be reasonably close
        // (linear is an approximation, not exact)
        let rho_full = eos_full.density_surface(8.0, 34.0);
        let rho_lin = eos_lin.density(8.0, 34.0);
        assert!((rho_full - rho_lin).abs() < 2.0);

        // For small perturbations, linear should give reasonable values
        let rho_full_warm = eos_full.density_surface(10.0, 34.0);
        let rho_lin_warm = eos_lin.density(10.0, 34.0);
        // Both should be in valid density range
        assert!(rho_full_warm > 1020.0 && rho_full_warm < 1030.0);
        assert!(rho_lin_warm > 1020.0 && rho_lin_warm < 1030.0);

        // Both should show density decrease with warming
        assert!(rho_full_warm < rho_full);
        assert!(rho_lin_warm < rho_lin);
    }

    #[test]
    fn test_buoyancy() {
        let eos = EquationOfState::new();
        let g = 9.81;

        // Fresh water should be positively buoyant relative to seawater
        let b_fresh = eos.buoyancy(10.0, 0.0, g);
        let b_salty = eos.buoyancy(10.0, 35.0, g);

        assert!(b_fresh > b_salty);
    }

    #[test]
    fn test_hydrostatic_pressure() {
        let g = 9.81;
        let z = -100.0; // 100m depth

        let p = EquationOfState::hydrostatic_pressure_approx(z, g);

        // p ≈ 1025 * 9.81 * 100 / 10000 ≈ 100.55 dbar
        assert!((p - 100.55).abs() < 1.0);
    }

    /// Regression: `density` takes pressure in dbar but the UNESCO secant bulk
    /// modulus is fitted in bar. Passing dbar straight through made seawater
    /// 10× too compressible (ρ(5 °C, 35, 1000 dbar) ≈ 1069.5 instead of ≈ 1032.3).
    ///
    /// Check values from UNESCO (1981) / Fofonoff & Millard (1983), tabulated
    /// at p = 0 and 1000 bar = 10 000 dbar.
    #[test]
    fn test_unesco_check_values() {
        let eos = EquationOfState::with_pressure();
        // (T °C, S psu, p dbar, ρ kg/m³)
        let cases = [
            (5.0, 0.0, 0.0, 999.96675),
            (5.0, 0.0, 10_000.0, 1044.12802),
            (25.0, 0.0, 0.0, 997.04796),
            (25.0, 0.0, 10_000.0, 1037.90204),
            (5.0, 35.0, 0.0, 1027.67547),
            (5.0, 35.0, 10_000.0, 1069.48914),
            (25.0, 35.0, 0.0, 1023.34306),
            (25.0, 35.0, 10_000.0, 1062.53817),
        ];
        for (t, s, p, rho_ref) in cases {
            let rho = eos.density(t, s, p);
            assert!(
                (rho - rho_ref).abs() < 1e-5,
                "ρ(T={t}, S={s}, p={p} dbar) = {rho}, UNESCO check value {rho_ref}"
            );
        }

        // An intermediate pressure: 1000 dbar (≈ 1000 m) compresses by ~4.6 kg/m³.
        let rho = eos.density(5.0, 35.0, 1000.0);
        assert!((rho - 1032.3).abs() < 0.1, "ρ(5, 35, 1000 dbar) = {rho}");
    }

    #[test]
    fn test_compute_density_column() {
        let eos = EquationOfState::with_pressure();
        let g = 9.81;
        let n = 5;

        let temp = vec![10.0; n];
        let salt = vec![35.0; n];
        // increasing depth
        let z: Vec<f64> = (0..n).map(|i| -(i as f64) * 100.0).collect();
        let mut rho = vec![0.0; n];

        eos.compute_density_column(&temp, &salt, &z, &mut rho, g);

        // Density should increase with depth due to pressure
        for i in 0..n - 1 {
            assert!(rho[i + 1] > rho[i]);
        }
    }
}
