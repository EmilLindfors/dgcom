//! Surface heat and freshwater fluxes on the 3D columns, per mesh node.
//!
//! [`Forcing::surface_heat_flux`](crate::physics::Forcing) is one heat flux
//! for the whole domain. A weather model's heat budget varies along the
//! coast and depends on each column's own sea-surface temperature, so
//! [`Hydrostatic3D::with_surface_fluxes`](crate::physics::Hydrostatic3D::with_surface_fluxes)
//! takes a [`SurfaceFluxes3D`]: given every column's surface temperature and
//! salinity, it writes the column's [`SurfaceBudget`](crate::physics::air_sea::SurfaceBudget)
//! fields. They enter the implicit vertical diffusion, at the middle of the
//! step `tⁿ + Δt/2` and from the surface layer at `tⁿ`:
//!
//! - the non-solar heat flux `Q_ns/(ρ₀ c_p)` as the temperature flux at the
//!   surface;
//! - the net shortwave `Q_sw`, absorbed with depth by a
//!   [`JerlovWater`](crate::physics::air_sea::JerlovWater) profile: layer `k`
//!   gains `Q_sw (I(d_top) − I(d_bot))/(ρ₀ c_p Δz_k)`, and the bottom layer
//!   whatever reaches its top, so the column takes all of it;
//! - E − P as the virtual salt flux `S F/ρ_fw` of the surface salinity.
//!
//! So the heat content `ρ₀ c_p ∫ T dz` of a column changes by exactly
//! `Δt (Q_ns + Q_sw)` over a step, and its salt `∫ S dz` by `Δt S F/ρ_fw`.
//! Columns thinner than the minimum column depth take no fluxes (their
//! tracers are the 3D model's to mask). Implementations:
//! [`crate::source::GriddedSurfaceFluxes`] (the COARE 3.5 bulk formulae on a
//! weather model's fields), and anything with a closed form for tests.

/// Every column's surface layer (`[element][node]`).
#[derive(Clone, Copy, Debug)]
pub struct SeaSurface<'a> {
    /// Temperature of the surface layer (°C)
    pub temperature: &'a [f64],
    /// Salinity of the surface layer
    pub salinity: &'a [f64],
}

/// Every column's surface fluxes (`[element][node]`), positive into the
/// ocean as [`crate::physics::air_sea::SurfaceBudget`].
#[derive(Debug)]
pub struct SurfaceFluxFields<'a> {
    /// Net heat flux at the surface without the shortwave (W/m²)
    pub nonsolar: &'a mut [f64],
    /// Net shortwave entering the water (W/m²)
    pub shortwave: &'a mut [f64],
    /// Evaporation minus precipitation (kg m⁻² s⁻¹)
    pub freshwater: &'a mut [f64],
}

/// Surface heat and freshwater fluxes on the mesh nodes.
pub trait SurfaceFluxes3D: Send + Sync {
    /// Overwrite `fluxes` (one entry per mesh node, `[element][node]`) with
    /// the fluxes at simulation time `t` over a sea whose surface layer is
    /// `surface`.
    fn surface_fluxes_into(&self, t: f64, surface: SeaSurface<'_>, fluxes: SurfaceFluxFields<'_>);
}
