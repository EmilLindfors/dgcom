/// Vertical column of data for mixing calculation.
///
/// Contains references to 3D fields for a single vertical column.
/// Depths `z` are positive upwards, z=0 at surface (usually).
pub struct Column<'a> {
    /// Depth at cell centers (rho-points), length N
    pub z_r: &'a [f64],
    /// Depth at cell interfaces (w-points), length N+1
    pub z_w: &'a [f64],
    /// Zonal velocity at cell centers, length N
    pub u: &'a [f64],
    /// Meridional velocity at cell centers, length N
    pub v: &'a [f64],
    /// Potential density at cell centers, length N
    pub rho: &'a [f64],
    /// Gravitational acceleration (m/s²) and reference density (kg/m³) of
    /// the model: `N² = −(g/ρ₀) ∂ρ/∂z`.
    pub g: f64,
    pub rho0: f64,
    /// Friction velocities `u* = √(|τ|/ρ₀)` (m/s) of the surface stress and
    /// of the bottom stress (prescribed plus drag) on this column.
    pub surface_friction_velocity: f64,
    pub bottom_friction_velocity: f64,
    /// The surface roughness of this column (m), e.g. `≈ 0.6 H_s` from the
    /// waves; closures with a roughness take the larger of it and their own,
    /// `None` for their own alone.
    pub surface_roughness: Option<f64>,
}

/// Prognostic turbulence of one column at its w-points (bottom to surface,
/// length N+1 each), for closures that carry it
/// ([`VerticalMixing::initial_turbulence`]); empty slices otherwise.
pub struct Turbulence<'a> {
    /// Turbulent kinetic energy `k` (m²/s²).
    pub tke: &'a mut [f64],
    /// The closure's length-scale variable (GLS `ψ`).
    pub gls: &'a mut [f64],
    /// Work space the closure may resize and use; the caller keeps it across
    /// columns, so nothing is allocated after the first.
    pub scratch: &'a mut Vec<f64>,
}

/// Surface and bottom forcing for mixing.
///
/// As a model's configuration ([`crate::physics::Hydrostatic3D::forcing`]) it
/// is the same on every column; a spatially varying surface stress adds to
/// it ([`crate::physics::SurfaceStress3D`]). The closures receive each
/// column's own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Forcing {
    /// Surface wind stress (tx, ty) [N/m^2]
    pub surface_stress: [f64; 2],
    /// Bottom stress (bx, by) [N/m^2]
    pub bottom_stress: [f64; 2],
    /// Net surface heat flux [W/m^2] (positive into the ocean), entering
    /// the temperature as `Q/(ρ₀ c_p)` at the surface; per-column fluxes
    /// add to it ([`crate::physics::SurfaceFluxes3D`])
    pub surface_heat_flux: f64,
}

/// Trait for vertical mixing closure schemes.
pub trait VerticalMixing: Send + Sync {
    /// Compute vertical eddy viscosity and diffusivity profiles.
    ///
    /// Returns a tuple `(Av, Kt)` where:
    /// - `Av`: Vertical eddy viscosity [m^2/s] at w-points (length N+1)
    /// - `Kt`: Vertical eddy diffusivity [m^2/s] at w-points (length N+1)
    ///
    /// Note: Boundary values at top/bottom are usually determined by BCs or specific matching,
    /// but here we return the full profile.
    fn compute_mixing(&self, column: &Column, forcing: &Forcing) -> (Vec<f64>, Vec<f64>) {
        let n_w = column.z_w.len();
        let (mut av, mut kt) = (vec![0.0; n_w], vec![0.0; n_w]);
        self.compute_mixing_into(column, forcing, &mut av, &mut kt);
        (av, kt)
    }

    /// [`Self::compute_mixing`] into `av` and `kt` (each of length N+1),
    /// without allocating: the vertical diffusion calls it for every column
    /// every step.
    fn compute_mixing_into(
        &self,
        column: &Column,
        forcing: &Forcing,
        av: &mut [f64],
        kt: &mut [f64],
    );

    /// The initial `[k, ψ]` of a closure with prognostic turbulence (stored
    /// in `Solution3D::tke`, `Solution3D::gls`), `None` for a diagnostic one.
    fn initial_turbulence(&self) -> Option<[f64; 2]> {
        None
    }

    /// Advance the column's turbulence over `dt`, then write `av` and `kt`
    /// from it. The vertical diffusion calls this once per column and step,
    /// before its implicit solve. Diagnostic closures ignore `dt` and
    /// `turbulence`: [`Self::compute_mixing_into`].
    fn step_mixing_into(
        &self,
        column: &Column,
        forcing: &Forcing,
        _dt: f64,
        _turbulence: Turbulence<'_>,
        av: &mut [f64],
        kt: &mut [f64],
    ) {
        self.compute_mixing_into(column, forcing, av, kt);
    }
}

/// A boxed closure, chosen at run time (`Box<dyn VerticalMixing>`).
impl<M: VerticalMixing + ?Sized> VerticalMixing for Box<M> {
    fn compute_mixing_into(
        &self,
        column: &Column,
        forcing: &Forcing,
        av: &mut [f64],
        kt: &mut [f64],
    ) {
        (**self).compute_mixing_into(column, forcing, av, kt);
    }

    fn initial_turbulence(&self) -> Option<[f64; 2]> {
        (**self).initial_turbulence()
    }

    fn step_mixing_into(
        &self,
        column: &Column,
        forcing: &Forcing,
        dt: f64,
        turbulence: Turbulence<'_>,
        av: &mut [f64],
        kt: &mut [f64],
    ) {
        (**self).step_mixing_into(column, forcing, dt, turbulence, av, kt);
    }
}

/// Constant vertical mixing.
///
/// Simply returns constant values for viscosity and diffusivity.
#[derive(Debug, Clone)]
pub struct ConstantMixing {
    pub eddy_viscosity: f64,
    pub eddy_diffusivity: f64,
}

impl ConstantMixing {
    pub fn new(viscosity: f64, diffusivity: f64) -> Self {
        Self {
            eddy_viscosity: viscosity,
            eddy_diffusivity: diffusivity,
        }
    }
}

impl VerticalMixing for ConstantMixing {
    fn compute_mixing_into(
        &self,
        _column: &Column,
        _forcing: &Forcing,
        av: &mut [f64],
        kt: &mut [f64],
    ) {
        av.fill(self.eddy_viscosity);
        kt.fill(self.eddy_diffusivity);
    }
}

/// Pacanowski-Philander (1981) mixing scheme.
///
/// Computes mixing based on Richardson number:
/// Av = v0 / (1 + 5*Ri)^2 + vb
/// Kt = v0 / (1 + 5*Ri)^3 + kb
#[derive(Debug, Clone)]
pub struct PacanowskiPhilanderMixing {
    pub v0: f64,   // Max viscosity [m^2/s]
    pub vb: f64,   // Background viscosity [m^2/s]
    pub kb: f64,   // Background diffusivity [m^2/s]
    pub g: f64,    // Gravity [m/s^2]
    pub rho0: f64, // Reference density [kg/m^3]
}

impl PacanowskiPhilanderMixing {
    pub fn new(v0: f64, vb: f64, kb: f64, g: f64, rho0: f64) -> Self {
        Self {
            v0,
            vb,
            kb,
            g,
            rho0,
        }
    }
}

impl Default for PacanowskiPhilanderMixing {
    fn default() -> Self {
        Self::new(1.0e-2, 1.0e-4, 1.0e-5, 9.81, 1025.0)
    }
}

impl VerticalMixing for PacanowskiPhilanderMixing {
    fn compute_mixing_into(
        &self,
        column: &Column,
        _forcing: &Forcing,
        av: &mut [f64],
        kt: &mut [f64],
    ) {
        let n = column.z_r.len();

        // Initialize with background values
        av.fill(self.vb);
        kt.fill(self.kb);

        // Loop over internal interfaces (1 to N-1)
        // Interfaces 0 (bottom) and N (surface) are boundaries.
        // We compute Ri at interfaces.
        // Note: z_w has length N+1 (0..N). z_r has length N (0..N-1).
        // Interface k is between cell k-1 and k.
        // So we iterate k from 1 to N-1.

        for k in 1..n {
            // Compute vertical gradients at interface k
            // z_r[k] is above z_r[k-1] (assuming indices increase upwards 0->N-1)
            // Wait, standard ROMS indices: 0 is bottom, N-1 is surface.
            // z_w[0] is bottom, z_w[N] is surface.
            // z_r[k] is center of cell k.
            // Interface k is between z_r[k-1] and z_r[k].

            let dz = column.z_r[k] - column.z_r[k - 1];
            if dz < 1e-10 {
                continue;
            }

            let du = column.u[k] - column.u[k - 1];
            let dv = column.v[k] - column.v[k - 1];
            let drho = column.rho[k] - column.rho[k - 1]; // usually negative for stable stratification (rho decreases upwards)

            let shear2 = (du * du + dv * dv) / (dz * dz);
            // N2 = -g/rho0 * d_rho/dz
            let n2 = -self.g / self.rho0 * drho / dz;

            // Gradient Richardson number Ri = N^2 / S^2
            let ri = if shear2 > 1e-10 {
                n2 / shear2
            } else {
                // If shear is zero, Ri is infinite (if N2 > 0).
                // PP term 1/(1+5Ri)^2 -> 0. Mixing -> background.
                f64::INFINITY
            };

            // Limit Ri to be non-negative for standard PP (shear instability)
            // If unstable (N2 < 0), we want large mixing.
            // Standard PP doesn't handle N2 < 0 explicitly, usually separate logic.
            // We use background for now if unstable, or simple convective adjustment logic?
            // Let's just clamp Ri at 0.
            let ri = if ri < 0.0 { 0.0 } else { ri };

            if ri.is_infinite() {
                av[k] = self.vb;
                kt[k] = self.kb;
            } else {
                let term = 1.0 + 5.0 * ri;
                let term2 = term * term;
                let term3 = term2 * term;

                av[k] = self.v0 / term2 + self.vb;
                kt[k] = self.v0 / term3 + self.kb;
            }
        }

        // Boundary values (bottom and surface)
        // Usually set to background or matched to log layer.
        // We leave them as background (initialized above).
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_mixing() {
        let n = 10;
        let z_r = vec![0.0; n];
        let z_w = vec![0.0; n + 1];
        let u = vec![0.0; n];
        let v = vec![0.0; n];
        let rho = vec![0.0; n];

        let column = Column {
            z_r: &z_r,
            z_w: &z_w,
            u: &u,
            v: &v,
            rho: &rho,
            g: 9.81,
            rho0: 1025.0,
            surface_friction_velocity: 0.0,
            bottom_friction_velocity: 0.0,
            surface_roughness: None,
        };

        let forcing = Forcing {
            surface_stress: [0.0, 0.0],
            bottom_stress: [0.0, 0.0],
            surface_heat_flux: 0.0,
        };

        let mix = ConstantMixing::new(0.1, 0.01);
        let (av, kt) = mix.compute_mixing(&column, &forcing);

        assert_eq!(av.len(), n + 1);
        assert_eq!(kt.len(), n + 1);

        for val in av {
            assert_eq!(val, 0.1);
        }
        for val in kt {
            assert_eq!(val, 0.01);
        }
    }

    #[test]
    fn test_pp_mixing_no_shear() {
        // No shear, constant density -> Ri = 0/0 -> handled as infinite (background)
        // Wait, if shear is zero, Ri is infinite (if N2 > 0).
        // If N2 = 0, Ri = 0/0.
        // My implementation: if shear2 < 1e-10, Ri = Infinity.
        // Then term = 1/(1+5*Inf)^2 = 0.
        // So returns background.

        let n = 5;
        // dz = 1.0
        let z_r = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let z_w = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let u = vec![1.0; n];
        let v = vec![0.0; n];
        let rho = vec![1025.0; n]; // Constant density

        let column = Column {
            z_r: &z_r,
            z_w: &z_w,
            u: &u,
            v: &v,
            rho: &rho,
            g: 9.81,
            rho0: 1025.0,
            surface_friction_velocity: 0.0,
            bottom_friction_velocity: 0.0,
            surface_roughness: None,
        };

        let forcing = Forcing {
            surface_stress: [0.0, 0.0],
            bottom_stress: [0.0, 0.0],
            surface_heat_flux: 0.0,
        };

        let mix = PacanowskiPhilanderMixing::default();
        let (av, kt) = mix.compute_mixing(&column, &forcing);

        // Should be background
        for k in 1..n {
            assert!((av[k] - mix.vb).abs() < 1e-10);
            assert!((kt[k] - mix.kb).abs() < 1e-10);
        }
    }

    #[test]
    fn test_pp_mixing_shear_instability() {
        // High shear, no stratification -> Ri = 0.
        // Mixing should be max (v0 + vb).

        let n = 3;
        // dz = 1.0
        let z_r = vec![0.5, 1.5, 2.5];
        let z_w = vec![0.0, 1.0, 2.0, 3.0];

        // u increases by 1 m/s per meter -> shear = 1.
        let u = vec![0.0, 1.0, 2.0];
        let v = vec![0.0; n];
        let rho = vec![1025.0; n]; // Constant density -> N2 = 0.

        let column = Column {
            z_r: &z_r,
            z_w: &z_w,
            u: &u,
            v: &v,
            rho: &rho,
            g: 9.81,
            rho0: 1025.0,
            surface_friction_velocity: 0.0,
            bottom_friction_velocity: 0.0,
            surface_roughness: None,
        };

        let forcing = Forcing {
            surface_stress: [0.0, 0.0],
            bottom_stress: [0.0, 0.0],
            surface_heat_flux: 0.0,
        };

        let mix = PacanowskiPhilanderMixing::default();
        let (av, kt) = mix.compute_mixing(&column, &forcing);

        // Ri = 0. term = 1.
        // Av = v0/1 + vb
        // Kt = v0/1 + kb

        // Check internal interfaces (k=1, 2)
        // Interface 1 is between cell 0 and 1.
        // Interface 2 is between cell 1 and 2.
        for k in 1..n {
            assert!((av[k] - (mix.v0 + mix.vb)).abs() < 1e-10);
            assert!((kt[k] - (mix.v0 + mix.kb)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_pp_mixing_stratified() {
        // Shear and stratification.
        // shear = 1.0.
        // N2 = -g/rho0 * drho/dz
        // drho/dz = -1.0 (stable).
        // N2 = -9.81/1025 * (-1) = 0.00957
        // Ri = 0.00957 / 1.0 = 0.00957

        // term = 1 + 5 * 0.00957 = 1.04785
        // av = v0 / term^2 + vb

        let n = 3;
        let z_r = vec![0.5, 1.5, 2.5];
        let z_w = vec![0.0, 1.0, 2.0, 3.0];

        let u = vec![0.0, 1.0, 2.0];
        let v = vec![0.0; n];
        let rho = vec![1025.0, 1024.0, 1023.0]; // decreasing rho upwards

        let column = Column {
            z_r: &z_r,
            z_w: &z_w,
            u: &u,
            v: &v,
            rho: &rho,
            g: 9.81,
            rho0: 1025.0,
            surface_friction_velocity: 0.0,
            bottom_friction_velocity: 0.0,
            surface_roughness: None,
        };

        let forcing = Forcing {
            surface_stress: [0.0, 0.0],
            bottom_stress: [0.0, 0.0],
            surface_heat_flux: 0.0,
        };

        let mix = PacanowskiPhilanderMixing::default();
        let (av, _kt) = mix.compute_mixing(&column, &forcing);

        let dz = 1.0;
        let shear2 = 1.0;
        // ρ falls by 1 kg/m³ per dz upwards: N² = −(g/ρ₀) ∂ρ/∂z
        let drho_dz = -1.0 / dz;
        let n2 = -mix.g / mix.rho0 * drho_dz;
        let ri = n2 / shear2;
        let term = 1.0 + 5.0 * ri;
        let expected_av = mix.v0 / (term * term) + mix.vb;

        for k in 1..n {
            assert!((av[k] - expected_av).abs() < 1e-10);
        }
    }
}
