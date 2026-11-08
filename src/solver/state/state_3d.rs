//! 3D state types for baroclinic ocean modeling.
//!
//! # Storage Layout
//!
//! The 3D solution uses a hybrid storage layout:
//!
//! - **Barotropic (2D)**: stored as standard [`DGSolution2D`]
//! - **Baroclinic (3D)**: stored as flat vectors with layout `[n_elements][n_nodes][n_levels]`
//!
//! This layout ensures that vertical columns are contiguous in memory, which is optimal
//! for vertical physics operations (vertical advection, diffusion, mixing) that are
//! typically the bottleneck in mode-split ocean models.
//!
//! Layout detail: `data[k * n_nodes * n_levels + i * n_levels + level]`
//! where:
//! - `k`: element index
//! - `i`: node index
//! - `level`: vertical level index (0 = bottom, N = surface)

use crate::solver::DGSolution2D;
use crate::types::ElementIndex;

/// 3D Ocean Model Solution State.
///
/// Contains both the fast barotropic state (2D) and the slow baroclinic state (3D).
#[derive(Clone)]
pub struct Solution3D {
    /// Number of elements
    pub n_elements: usize,
    /// Number of nodes per element (horizontal)
    pub n_nodes: usize,
    /// Number of vertical levels
    pub n_levels: usize,

    // --- Barotropic State (2D) ---
    /// Free surface elevation η (m)
    pub eta: DGSolution2D,
    /// Depth-averaged u-velocity (m/s)
    pub ubar: DGSolution2D,
    /// Depth-averaged v-velocity (m/s)
    pub vbar: DGSolution2D,

    // --- Baroclinic State (3D) ---
    /// 3D u-velocity (m/s)
    pub u: Vec<f64>,
    /// 3D v-velocity (m/s)
    pub v: Vec<f64>,
    /// Vertical velocity diagnostic (m/s) at the layer centres: the sigma-coordinate
    /// Ω, averaged from its w-point values by `Hydrostatic3D::post_process`
    /// (output only; the RHS uses Ω at the interfaces).
    pub w: Vec<f64>,
    /// Temperature (°C)
    pub temp: Vec<f64>,
    /// Salinity (PSU)
    pub salt: Vec<f64>,
    /// Density (kg/m³)
    pub rho: Vec<f64>,
    /// Eddy viscosity (m²/s) of the momentum at the w-points,
    /// `[n_elements][n_nodes][n_levels + 1]`, bottom to surface (as
    /// [`Self::tke`]); set by the vertical diffusion, zero before.
    pub eddy_viscosity: Vec<f64>,
    /// Eddy diffusivity (m²/s) of the tracers at the w-points, like
    /// [`Self::eddy_viscosity`].
    pub eddy_diffusivity: Vec<f64>,

    // --- Turbulence (prognostic closures only) ---
    /// Turbulent kinetic energy `k` (m²/s²) of a two-equation closure
    /// ([`crate::physics::GlsMixing`]) at the w-points,
    /// `[n_elements][n_nodes][n_levels + 1]`, bottom to surface; empty
    /// without one (the vertical diffusion allocates it on first use).
    /// Advected by the 3D stages (as an inventory over the w-cells, see
    /// `crate::physics::Hydrostatic3D::with_turbulence_advection`), then
    /// stepped by the closure once per baroclinic step.
    pub tke: Vec<f64>,
    /// The closure's length-scale variable `ψ` (GLS), like [`Self::tke`].
    pub gls: Vec<f64>,
}

impl Solution3D {
    /// Set the temperature and salinity of the columns `fill` picks from
    /// `profile(z) → (T, S)` at their layer centres
    /// `z = η + σ_l (η − B)`, over the bed `bed[node]` and the surface `eta`
    /// already in the state.
    pub fn fill_tracers(
        &mut self,
        sigma: &crate::vertical::SigmaGrid,
        bed: &[f64],
        fill: impl Fn(usize) -> bool,
        profile: impl Fn(f64) -> (f64, f64),
    ) {
        let nl = self.n_levels;
        for idx in (0..bed.len()).filter(|&idx| fill(idx)) {
            let eta = self.eta.data[idx];
            for (l, &s) in sigma.sigma_rho().iter().enumerate() {
                let z = eta + s * (eta - bed[idx]);
                (self.temp[idx * nl + l], self.salt[idx * nl + l]) = profile(z);
            }
        }
    }

    /// Keep only the shear of the velocity of the columns `sheared` picks
    /// (each layer minus the column's depth mean on `sigma`) and set the
    /// others at rest: a parent's vertical structure under a depth mean at
    /// rest, for a run whose depth mean ramps up from rest. The number of
    /// sheared columns.
    pub fn keep_shear(
        &mut self,
        sigma: &crate::vertical::SigmaGrid,
        sheared: impl Fn(usize) -> bool,
    ) -> usize {
        let nl = self.n_levels;
        let mut n = 0;
        for idx in 0..self.n_elements * self.n_nodes {
            let keep = sheared(idx);
            let column = idx * nl..(idx + 1) * nl;
            for field in [&mut self.u, &mut self.v] {
                let values = &mut field[column.clone()];
                let mean = if keep {
                    sigma.depth_average(values)
                } else {
                    0.0
                };
                values
                    .iter_mut()
                    .for_each(|x| *x = if keep { *x - mean } else { 0.0 });
            }
            n += usize::from(keep);
        }
        n
    }

    /// Create a new 3D solution initialized to zero.
    pub fn new(n_elements: usize, n_nodes: usize, n_levels: usize) -> Self {
        let n_3d = n_elements * n_nodes * n_levels;
        // Note: Viscosity/Diffusivity often defined at w-points (n_levels + 1)
        // or rho-points depending on scheme.
        // For standard implicit diffusion, we usually need them at interfaces.
        // Let's assume n_levels + 1 for now if we want to be precise, or n_levels if we interpolate.
        // BUT, Solution3D layout is strict n_levels per column.
        // If we need n_levels+1, we might need a separate structure or assume bottom=0/top=0 and store n_levels.
        // ROMS stores Ak, Av at w-points (interfaces).
        // If we store at n_levels, we are missing one interface.
        // However, surface and bottom fluxes are BCs.
        // Let's stick to n_levels (rho points) and interpolate to w-points for flux, or store at w-points.
        // If we change the size, it breaks the "uniform column" stride logic.
        // Let's keep it n_levels for now (cell centers) and assume we interpolate to faces.
        // Or, we can just say this is the "effective" viscosity at the center.

        Self {
            n_elements,
            n_nodes,
            n_levels,

            // 2D fields
            eta: DGSolution2D::new(n_elements, n_nodes),
            ubar: DGSolution2D::new(n_elements, n_nodes),
            vbar: DGSolution2D::new(n_elements, n_nodes),

            // 3D fields
            u: vec![0.0; n_3d],
            v: vec![0.0; n_3d],
            w: vec![0.0; n_3d],
            temp: vec![0.0; n_3d],
            salt: vec![0.0; n_3d],
            rho: vec![0.0; n_3d],
            eddy_viscosity: vec![0.0; n_elements * n_nodes * (n_levels + 1)],
            eddy_diffusivity: vec![0.0; n_elements * n_nodes * (n_levels + 1)],
            tke: Vec::new(),
            gls: Vec::new(),
        }
    }

    /// Get a mutable reference to a vertical column for a 3D variable.
    ///
    /// This is a fast operation (O(1)) because columns are contiguous in memory.
    ///
    /// # Arguments
    /// * `data`: Reference to one of the 3D data vectors (e.g. `&mut self.u`)
    /// * `n_nodes`: Number of nodes per element
    /// * `n_levels`: Number of vertical levels
    /// * `k`: Element index
    /// * `i`: Node index
    #[inline(always)]
    pub fn get_column_mut(
        data: &mut [f64],
        n_nodes: usize,
        n_levels: usize,
        k: ElementIndex,
        i: usize,
    ) -> &mut [f64] {
        let start = (k.as_usize() * n_nodes + i) * n_levels;
        &mut data[start..start + n_levels]
    }

    /// Get a reference to a vertical column for a 3D variable.
    #[inline(always)]
    pub fn get_column(
        data: &[f64],
        n_nodes: usize,
        n_levels: usize,
        k: ElementIndex,
        i: usize,
    ) -> &[f64] {
        let start = (k.as_usize() * n_nodes + i) * n_levels;
        &data[start..start + n_levels]
    }

    /// Get a reference to the u-velocity column at (k, i).
    #[inline(always)]
    pub fn u_column(&self, k: ElementIndex, i: usize) -> &[f64] {
        Self::get_column(&self.u, self.n_nodes, self.n_levels, k, i)
    }

    /// Get a mutable reference to the u-velocity column at (k, i).
    #[inline(always)]
    pub fn u_column_mut(&mut self, k: ElementIndex, i: usize) -> &mut [f64] {
        Self::get_column_mut(&mut self.u, self.n_nodes, self.n_levels, k, i)
    }

    /// Get a reference to the v-velocity column at (k, i).
    #[inline(always)]
    pub fn v_column(&self, k: ElementIndex, i: usize) -> &[f64] {
        Self::get_column(&self.v, self.n_nodes, self.n_levels, k, i)
    }

    /// Get a mutable reference to the v-velocity column at (k, i).
    #[inline(always)]
    pub fn v_column_mut(&mut self, k: ElementIndex, i: usize) -> &mut [f64] {
        Self::get_column_mut(&mut self.v, self.n_nodes, self.n_levels, k, i)
    }

    /// Get a reference to the vertical velocity w column at (k, i).
    #[inline(always)]
    pub fn w_column(&self, k: ElementIndex, i: usize) -> &[f64] {
        Self::get_column(&self.w, self.n_nodes, self.n_levels, k, i)
    }

    /// Get a mutable reference to the vertical velocity w column at (k, i).
    #[inline(always)]
    pub fn w_column_mut(&mut self, k: ElementIndex, i: usize) -> &mut [f64] {
        Self::get_column_mut(&mut self.w, self.n_nodes, self.n_levels, k, i)
    }

    /// Get a reference to the temperature column at (k, i).
    #[inline(always)]
    pub fn temp_column(&self, k: ElementIndex, i: usize) -> &[f64] {
        Self::get_column(&self.temp, self.n_nodes, self.n_levels, k, i)
    }

    /// Get a mutable reference to the temperature column at (k, i).
    #[inline(always)]
    pub fn temp_column_mut(&mut self, k: ElementIndex, i: usize) -> &mut [f64] {
        Self::get_column_mut(&mut self.temp, self.n_nodes, self.n_levels, k, i)
    }

    /// Get a reference to the salinity column at (k, i).
    #[inline(always)]
    pub fn salt_column(&self, k: ElementIndex, i: usize) -> &[f64] {
        Self::get_column(&self.salt, self.n_nodes, self.n_levels, k, i)
    }

    /// Get a mutable reference to the salinity column at (k, i).
    #[inline(always)]
    pub fn salt_column_mut(&mut self, k: ElementIndex, i: usize) -> &mut [f64] {
        Self::get_column_mut(&mut self.salt, self.n_nodes, self.n_levels, k, i)
    }

    /// Get a reference to the density column at (k, i).
    #[inline(always)]
    pub fn rho_column(&self, k: ElementIndex, i: usize) -> &[f64] {
        Self::get_column(&self.rho, self.n_nodes, self.n_levels, k, i)
    }

    /// Get a mutable reference to the density column at (k, i).
    #[inline(always)]
    pub fn rho_column_mut(&mut self, k: ElementIndex, i: usize) -> &mut [f64] {
        Self::get_column_mut(&mut self.rho, self.n_nodes, self.n_levels, k, i)
    }

    /// The eddy viscosity column at (k, i): `n_levels + 1` w-points.
    #[inline(always)]
    pub fn eddy_viscosity_column(&self, k: ElementIndex, i: usize) -> &[f64] {
        Self::get_column(&self.eddy_viscosity, self.n_nodes, self.n_levels + 1, k, i)
    }

    /// Get a mutable reference to the eddy viscosity column at (k, i).
    #[inline(always)]
    pub fn eddy_viscosity_column_mut(&mut self, k: ElementIndex, i: usize) -> &mut [f64] {
        Self::get_column_mut(
            &mut self.eddy_viscosity,
            self.n_nodes,
            self.n_levels + 1,
            k,
            i,
        )
    }

    /// The eddy diffusivity column at (k, i): `n_levels + 1` w-points.
    #[inline(always)]
    pub fn eddy_diffusivity_column(&self, k: ElementIndex, i: usize) -> &[f64] {
        Self::get_column(
            &self.eddy_diffusivity,
            self.n_nodes,
            self.n_levels + 1,
            k,
            i,
        )
    }

    /// Get a mutable reference to the eddy diffusivity column at (k, i).
    #[inline(always)]
    pub fn eddy_diffusivity_column_mut(&mut self, k: ElementIndex, i: usize) -> &mut [f64] {
        Self::get_column_mut(
            &mut self.eddy_diffusivity,
            self.n_nodes,
            self.n_levels + 1,
            k,
            i,
        )
    }

    /// Get the value of a 3D variable at a specific point.
    #[inline(always)]
    pub fn get_value(&self, data: &[f64], k: ElementIndex, i: usize, level: usize) -> f64 {
        let idx = (k.as_usize() * self.n_nodes + i) * self.n_levels + level;
        data[idx]
    }

    /// Set the value of a 3D variable at a specific point.
    #[inline(always)]
    pub fn set_value(&self, data: &mut [f64], k: ElementIndex, i: usize, level: usize, value: f64) {
        let idx = (k.as_usize() * self.n_nodes + i) * self.n_levels + level;
        data[idx] = value;
    }

    /// Fill all state variables with zeros.
    pub fn zero(&mut self) {
        self.eta.fill(0.0);
        self.ubar.fill(0.0);
        self.vbar.fill(0.0);
        self.u.fill(0.0);
        self.v.fill(0.0);
        self.w.fill(0.0);
        self.temp.fill(0.0);
        self.salt.fill(0.0);
        self.rho.fill(0.0);
        self.eddy_viscosity.fill(0.0);
        self.eddy_diffusivity.fill(0.0);
        self.tke.fill(0.0);
        self.gls.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solution_3d_creation() {
        let n_elements = 2;
        let n_nodes = 4;
        let n_levels = 5;

        let sol = Solution3D::new(n_elements, n_nodes, n_levels);

        assert_eq!(sol.u.len(), n_elements * n_nodes * n_levels);
        assert_eq!(sol.eta.data.len(), n_elements * n_nodes);
    }

    #[test]
    fn test_column_access() {
        let n_elements = 1;
        let n_nodes = 1;
        let n_levels = 3;

        let mut sol = Solution3D::new(n_elements, n_nodes, n_levels);
        let k = ElementIndex::new(0);

        // Modify a column
        {
            let col = sol.temp_column_mut(k, 0);
            col[0] = 10.0;
            col[1] = 11.0;
            col[2] = 12.0;
        }

        // Check values
        assert_eq!(sol.temp[0], 10.0);
        assert_eq!(sol.temp[1], 11.0);
        assert_eq!(sol.temp[2], 12.0);

        // Check helper accessor
        assert_eq!(sol.get_value(&sol.temp, k, 0, 1), 11.0);
    }

    #[test]
    fn test_column_contiguous() {
        let n_elements = 2;
        let n_nodes = 2;
        let n_levels = 3;
        let mut sol = Solution3D::new(n_elements, n_nodes, n_levels);

        // Fill data sequentially
        for i in 0..sol.u.len() {
            sol.u[i] = i as f64;
        }

        // Element 0, Node 0 should have values 0, 1, 2
        let col00 = sol.u_column(ElementIndex::new(0), 0);
        assert_eq!(col00, &[0.0, 1.0, 2.0]);

        // Element 0, Node 1 should have values 3, 4, 5
        let col01 = sol.u_column(ElementIndex::new(0), 1);
        assert_eq!(col01, &[3.0, 4.0, 5.0]);
    }
}
