//! Generic length scale (GLS) turbulence closure: two transport equations
//! for the turbulent kinetic energy `k` and a length-scale variable `ψ`
//! (Umlauf & Burchard 2003; ROMS: Warner et al. 2005), per water column.
//!
//! # Equations
//!
//! ```text
//!     ∂k/∂t = ∂/∂z(ν/σ_k ∂k/∂z) + P + B − ε,
//!     ∂ψ/∂t = ∂/∂z(ν/σ_ψ ∂ψ/∂z) + (ψ/k)(c₁P + c₃B − c₂ε),
//!     ψ = (c_μ⁰)^p k^m l^n,    ε = (c_μ⁰)^(3+p/n) k^(3/2+m/n) ψ^(−1/n),
//! ```
//!
//! with shear production `P = ν S²`, buoyancy production `B = −ν′ N²`, the
//! length `l = (c_μ⁰)³ k^(3/2)/ε` and the eddy viscosity and diffusivity
//! `ν = c_μ √k l`, `ν′ = c_μ′ √k l`. The exponents `(p, m, n)` select the
//! model: `(3, 3/2, −1)` is k-ε (`ψ = ε`), `(−1, 1/2, −1)` k-ω, `(2, 1, −2/3)`
//! Umlauf & Burchard's "generic" model ([`GlsParameters`]).
//!
//! The stability functions `c_μ`, `c_μ′` are Canuto et al.'s (2001), version
//! A or B, in their quasi-equilibrium form (Umlauf & Burchard 2005; GOTM's
//! `cmue_d`): functions of `α_N = (k/ε)² N²` alone, with `α_N` bounded below
//! where convection would make them singular. `c_μ⁰` is their neutral value,
//! so that a log layer has `k = u*²/(c_μ⁰)²` exactly.
//!
//! Three constants follow from the others, as in GOTM:
//! - `c₃` in stable stratification (`B < 0`) from the steady-state
//!   Richardson number `Ri_st` (0.25): homogeneous shear turbulence at
//!   `Ri = Ri_st` neither grows nor decays ([`GlsMixing::c3_minus`]);
//! - `c₃` in unstable stratification such that the ε equation the model
//!   implies has k-ε's `c₃ε⁺ = 1` (in the presets, [`GlsParameters`]);
//! - von Kármán's constant `κ = c_μ⁰ √(σ_ψ (c₂ − c₁)/n²)`, the one the ψ
//!   equation keeps in a log layer ([`GlsMixing::kappa`]).
//!
//! Galperin et al.'s (1988) limit `l ≤ c_lim √(2k)/N` (stable water) bounds
//! the length.
//!
//! # Discretisation (Umlauf & Burchard 2005; GOTM `tkeeq`, `genericeq`)
//!
//! `k` and `ψ` live at the w-points. Once per baroclinic step, before the
//! implicit vertical diffusion of momentum and tracers that uses the new `ν`,
//! `ν′`:
//! 1. `S²`, `N²` at the interior w-points from the layer-centre velocity and
//!    density, `P`, `B` from the old `ν`, `ν′`, `ε`;
//! 2. `k`, then `ψ`, by one backward-Euler step with implicit diffusion. The
//!    sinks are linearised in the new value (Patankar 1980: a negative
//!    `B` is a sink when `P + B < 0`), so both stay positive for any `Δt`;
//! 3. `ε` from the new `k`, `ψ`, bounded below by `ε_min` and the Galperin
//!    limit, and `ψ` reset to the bounded `ε`.
//!
//! The boundaries are logarithmic layers with flux conditions at the
//! boundary layers' centres: no flux of `k`, and the flux of `ψ` of a log
//! layer, `−n (c_μ⁰)^(p+1) κ^(n+1)/σ_ψ k^(m+1/2) (z′ + z₀)^n`, at a height
//! `z′` of half the end layer over roughness `z₀` (GOTM's Neumann
//! "logarithmic" conditions). The boundary w-points themselves carry
//! `k = u*²/(c_μ⁰)²`, `l = κ z₀`, and nothing reads them: not the `k` and `ψ`
//! solves (the `ψ` flux takes the `k` of the first interior w-point), not
//! the momentum and tracer solves (which take the surface and bottom fluxes
//! directly). So the friction velocities in [`Column`] are diagnostic: the
//! wind and the bed reach the turbulence through the shear production of the
//! stresses' momentum fluxes. In a layer of constant stress the interior
//! holds the log layer's `k = u*²/(c_μ⁰)²` exactly at equilibrium, so the
//! boundary values agree with it; from rest the top interior w-point needs
//! ≈ 36 h/u* to get there (h the top layer's thickness; ROMS's default
//! Dirichlet surface value, or wave injection, would put `u*` into the
//! interior).
//!
//! The surface roughness is constant, or Charnock's `z₀ₛ = α u*²/g` above
//! it ([`GlsMixing::with_charnock_roughness`]).
//!
//! ## Breaking waves
//!
//! [`GlsMixing::with_wave_breaking`] replaces the surface log layer by
//! Craig & Banner's (1994) injection of `k`, the flux `c_w u*³`. Without
//! shear, diffusion of the injected `k` balances dissipation in a layer
//! `k = K s^(−a)`, `l = L s` (`s` the depth plus `z₀ₛ`), with `a`, `L` fixed
//! by the model's constants ([`GlsMixing::shear_free_layer`]; Umlauf &
//! Burchard 2003) and `K` by the flux. At the top layer's centre `s_c`
//! (GOTM's injection condition, evaluated there as the log layer's is):
//! - the flux of `k` is that layer's, `c_w u*³ (z₀ₛ/s_c)^(3a/2)`;
//! - the flux of `ψ` is that layer's, `−(n − a m) (c_μ⁰)^(p+1) L^(n+1)/σ_ψ
//!   k^(m+½) s_c^n` with `k` from the top interior w-point along the power
//!   law, blended with the log layer's by how deep `s_c` lies in the
//!   wave-affected layer: weight `1 − ln(s_c/z₀ₛ)/ln(s*/z₀ₛ)`, where `s*`
//!   is the depth at which the shear-free `k` falls to the log layer's
//!   `u*²/(c_μ⁰)²` (≈ 2–4 z₀ₛ). With a roughness well below half the top
//!   layer (Charnock's at metre layers) the wave-affected layer is not
//!   resolved and the column is the log layer's; with a wave height's
//!   (Terray et al. 1996: `z₀ₛ ≈ 0.6 H_s`) the top layers hold the
//!   injected `k`. Imposing the shear-free condition at any depth instead
//!   shrank the length in the log layer below and cut the near-surface
//!   mixing by 10–100× at metre layers; a weight from the local `P/ε`
//!   (Burchard 2001's interpolation of `σ_ψ`) fed back on itself.
//! - The surface w-point carries the layer's `k = K z₀ₛ^(−a)`, `l = L z₀ₛ`.
//!
//! k-ε's shear-free layer is thin (`a` ≈ 5, `L` ≈ 0.09 against κ ≈ 0.4):
//! it needs centimetre layers, and at metre layers it lowers the
//! near-surface diffusivity. Use k-ω (`a` 2.5, `L` 0.24) or the generic
//! model (2.0, 0.19, designed for it) with waves.
//!
//! In the 3D model `k` and `ψ` are also advected by the flow, as ROMS does
//! (`Hydrostatic3D::with_turbulence_advection`, on by default): carried
//! through the baroclinic stages over the control volumes of the w-points,
//! then stepped here in each column (operator splitting). The step takes a
//! point that the advection left without positive `k` or `ψ` as having no
//! turbulence (both at their minima). With the advection off the model is
//! local to each column, as in GOTM.
//!
//! # Verification
//!
//! - Wind entrainment into linear stratification (Kato & Phillips 1969):
//!   the mixed layer is within 1.5 % of Price's (1979) `1.05 u* √(t/N₀)`
//!   at 12–30 h with k-ε, at Δt = 10 and 60 s alike (gate in
//!   `physics::vertical_diffusion`).
//! - Open-channel flow: bottom stress `u*²` at steady state, `k` in local
//!   equilibrium with the linear stress to 3 %, the wall length `κz` at the
//!   first w-point (0.96 at 80 levels).
//! - Wind against a log-layer drag: the constant-stress layer holds
//!   `k = u*²/(c_μ⁰)²` and `ν ∂u/∂z = u*²` at every w-point to 1e-10.
//! - Homogeneous shear turbulence is steady at `Ri_st` (growth rate
//!   ≈ 10⁻⁵ S, against ±10⁻² S at Ri = 0.2 and 0.3).
//! - Convection: an unstable column overturns from the minimum turbulence
//!   within hours, with no convective adjustment.
//! - Breaking waves over still water reach the shear-free layer (`k` to
//!   10 / 7 % at 0.1 m layers with k-ω / generic, first order in the layer
//!   thickness; k-ε to 3 % at 5 mm); under wind with `z₀ₛ` = 0.5 m they
//!   raise `k` 4× and the diffusivity 2–3× at the top interior w-point,
//!   leave mid-depth and the Kato–Phillips entrainment as they were, and
//!   with Charnock's roughness change nothing at 0.5 m layers.

use crate::physics::vertical_mixing::{Column, Forcing, Turbulence, VerticalMixing};
use crate::solver::algorithms::tridiagonal::solve_tridiagonal;

/// Exponents and constants of a GLS model (see the module docs). The
/// presets are ROMS's (`ocean.in`), except for `c₃`: `c₃⁻` is computed
/// ([`GlsMixing::c3_minus`]), and `c₃⁺` is GOTM's, `(3/2 − c₃ε⁺)·n + m`
/// for the `c₃ε⁺ = 1` of k-ε's ε equation (ROMS has 1 for every model,
/// which in k-ω is `c₃ε⁺ = 2`: k-ω then cannot start convection from
/// weak turbulence).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlsParameters {
    /// Exponents of `ψ = (c_μ⁰)^p k^m l^n`
    pub p: f64,
    pub m: f64,
    pub n: f64,
    /// Schmidt numbers of `k` and `ψ`
    pub sigma_k: f64,
    pub sigma_psi: f64,
    /// `c₁`, `c₂` of the `ψ` equation, and `c₃` in unstable stratification
    /// (`B > 0`)
    pub c1: f64,
    pub c2: f64,
    pub c3_plus: f64,
}

impl GlsParameters {
    /// k-ε (Rodi 1987).
    pub const K_EPSILON: Self = Self {
        p: 3.0,
        m: 1.5,
        n: -1.0,
        sigma_k: 1.0,
        sigma_psi: 1.3,
        c1: 1.44,
        c2: 1.92,
        c3_plus: 1.0,
    };

    /// k-ω (Wilcox 1988): `ψ = ω = ε/((c_μ⁰)⁴ k)`.
    pub const K_OMEGA: Self = Self {
        p: -1.0,
        m: 0.5,
        n: -1.0,
        sigma_k: 2.0,
        sigma_psi: 2.0,
        c1: 0.555,
        c2: 0.833,
        c3_plus: 0.0,
    };

    /// Umlauf & Burchard's (2003) "generic" model.
    pub const GENERIC: Self = Self {
        p: 2.0,
        m: 1.0,
        n: -0.67,
        sigma_k: 0.8,
        sigma_psi: 1.07,
        c1: 1.0,
        c2: 1.22,
        c3_plus: 0.665,
    };
}

/// Stability functions of [`GlsMixing`]: Canuto et al. (2001), quasi-equilibrium.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum StabilityFunctions {
    /// Version A (GOTM's and ROMS's default).
    #[default]
    CanutoA,
    /// Version B.
    CanutoB,
}

/// The quasi-equilibrium stability functions' polynomial coefficients
/// (Umlauf & Burchard 2005, App. A; GOTM `cmue_d`), `c_μ⁰`, and the lowest
/// `α_N` they are evaluated at.
#[derive(Clone, Copy, Debug)]
struct Stability {
    d: [f64; 6],
    n: [f64; 3],
    nt: [f64; 3],
    cm0: f64,
    an_min: f64,
}

impl Stability {
    fn new(functions: StabilityFunctions) -> Self {
        // Pressure-strain and pressure-scrambling constants (GOTM `init_scnd`)
        let [cc1, cc2, cc3, cc4, cc5, cc6, ct1, ct2, ct3, ct4, ct5, ctt]: [f64; 12] =
            match functions {
                StabilityFunctions::CanutoA => [
                    5.0, 0.8, 1.968, 1.136, 0.0, 0.4, 5.95, 0.6, 1.0, 0.0, 0.3333, 0.72,
                ],
                StabilityFunctions::CanutoB => [
                    5.0, 0.6983, 1.9664, 1.094, 0.0, 0.495, 5.6, 0.6, 1.0, 0.0, 0.3333, 0.477,
                ],
            };
        let a1 = 2.0 / 3.0 - cc2 / 2.0;
        let a2 = 1.0 - cc3 / 2.0;
        let a3 = 1.0 - cc4 / 2.0;
        let _a4 = cc5 / 2.0;
        let a5 = 0.5 - cc6 / 2.0;
        let at1 = 1.0 - ct2;
        let at2 = 1.0 - ct3;
        let at3 = 2.0 * (1.0 - ct4);
        let _at4 = 2.0 * (1.0 - ct5);
        let at5 = 2.0 * ctt * (1.0 - ct5);
        let nn = 0.5 * cc1;
        let nt = ct1;

        let d0 = 36.0 * nn.powi(3) * nt.powi(2);
        let d1 = 84.0 * a5 * at3 * nn.powi(2) * nt + 36.0 * at5 * nn.powi(3) * nt;
        let d2 = 9.0 * (at2.powi(2) - at1.powi(2)) * nn.powi(3)
            - 12.0 * (a2.powi(2) - 3.0 * a3.powi(2)) * nn * nt.powi(2);
        let d3 = 12.0 * a5 * at3 * (a2 * at1 - 3.0 * a3 * at2) * nn
            + 12.0 * a5 * at3 * (a3.powi(2) - a2.powi(2)) * nt
            + 12.0 * at5 * (3.0 * a3.powi(2) - a2.powi(2)) * nn * nt;
        let d4 = 48.0 * a5.powi(2) * at3.powi(2) * nn + 36.0 * a5 * at3 * at5 * nn.powi(2);
        let d5 = 3.0 * (a2.powi(2) - 3.0 * a3.powi(2)) * (at1.powi(2) - at2.powi(2)) * nn;
        let n0 = 36.0 * a1 * nn.powi(2) * nt.powi(2);
        let n1 = -12.0 * a5 * at3 * (at1 + at2) * nn.powi(2)
            + 8.0 * a5 * at3 * (6.0 * a1 - a2 - 3.0 * a3) * nn * nt
            + 36.0 * a1 * at5 * nn.powi(2) * nt;
        let n2 = 9.0 * a1 * (at2.powi(2) - at1.powi(2)) * nn.powi(2);
        let nt0 = 12.0 * at3 * nn.powi(3) * nt;
        let nt1 = 12.0 * a5 * at3.powi(2) * nn.powi(2);
        let nt2 = 9.0 * a1 * at3 * (at1 - at2) * nn.powi(2)
            + (6.0 * a1 * (a2 - 3.0 * a3) - 4.0 * (a2.powi(2) - 3.0 * a3.powi(2))) * at3 * nn * nt;

        // The neutral value (GOTM `compute_cm0`), and the convective limit
        // of α_N where the denominator vanishes
        let cm0 = ((a2.powi(2) - 3.0 * a3.powi(2) + 3.0 * a1 * nn) / (3.0 * nn.powi(2))).powf(0.25);
        let an_min = (-(d1 + nt0) + ((d1 + nt0).powi(2) - 4.0 * d0 * (d4 + nt1)).sqrt())
            / (2.0 * (d4 + nt1));
        Self {
            d: [d0, d1, d2, d3, d4, d5],
            n: [n0, n1, n2],
            nt: [nt0, nt1, nt2],
            cm0,
            an_min,
        }
    }

    /// `(c_μ, c_μ′)` at `α_N = (k/ε)² N²`, with `α_S` from `P + B = ε`.
    #[inline]
    fn eval(&self, an: f64) -> (f64, f64) {
        let [d0, d1, d2, d3, d4, d5] = self.d;
        let [n0, n1, n2] = self.n;
        let [nt0, nt1, nt2] = self.nt;
        let an = an.max(0.5 * self.an_min);
        let tmp0 = -d0 - (d1 + nt0) * an - (d4 + nt1) * an * an;
        let tmp1 = -d2 + n0 + (n1 - d3 - nt2) * an;
        let tmp2 = n2 - d5;
        let a_s = if tmp2.abs() < 1e-10 {
            -tmp0 / tmp1
        } else {
            (-tmp1 + (tmp1 * tmp1 - 4.0 * tmp0 * tmp2).sqrt()) / (2.0 * tmp2)
        };
        let dcm = d0 + d1 * an + d2 * a_s + d3 * an * a_s + d4 * an * an + d5 * a_s * a_s;
        let ncm = n0 + n1 * an + n2 * a_s;
        let ncmp = nt0 + nt1 * an + nt2 * a_s;
        let cm3_inv = 1.0 / self.cm0.powi(3);
        (cm3_inv * ncm / dcm, cm3_inv * ncmp / dcm)
    }

    /// `c₃` for which homogeneous shear turbulence at `Ri = ri` is steady:
    /// `c₂ + (c₁ − c₂) c_μ/(c_μ′ Ri)` at the `α_N` where
    /// `c_μ α_N/Ri − c_μ′ α_N = (c_μ⁰)⁻³` (GOTM `compute_cpsi3`).
    fn c3_for(&self, c1: f64, c2: f64, ri: f64) -> f64 {
        let residual = |an: f64| {
            let (cm, cmp) = self.eval(an);
            cm * an / ri - cmp * an - self.cm0.powi(-3)
        };
        let (mut an, e) = (5.0, 1e-8);
        for _ in 0..200 {
            let f = residual(an);
            let step = -f * e / (residual(an + e) - f);
            assert!(
                step.abs() <= 100.0,
                "steady-state Richardson number {ri} is out of the stability functions' range"
            );
            an += 0.5 * step;
            if step.abs() < 1e-10 {
                break;
            }
        }
        let (cm, cmp) = self.eval(an);
        c2 + (c1 - c2) / ri * cm / cmp
    }
}

/// Injection of TKE by breaking surface waves (Craig & Banner 1994), with
/// the shear-free layer it makes in a GLS model
/// ([`GlsMixing::with_wave_breaking`], [`GlsMixing::shear_free_layer`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaveBreaking {
    /// `c_w` of the surface flux of `k`, `c_w u*³`.
    pub c_w: f64,
    /// Decay rate `a` of `k ∝ (d + z₀ₛ)^(−a)` below the surface.
    pub decay: f64,
    /// Slope `L` of the length `l = L (d + z₀ₛ)`.
    pub slope: f64,
    /// Depth of the wave-affected layer over the roughness, `s*/z₀ₛ`: where
    /// the shear-free layer's `k` falls to the log layer's `u*²/(c_μ⁰)²`.
    pub extent: f64,
}

impl WaveBreaking {
    /// How far a point at `s/z₀ₛ` is inside the wave-affected layer, on a
    /// log scale: 1 at the surface (`s = z₀ₛ`), 0 at its depth `s*` and below.
    #[inline]
    pub fn share_above(&self, s_over_z0: f64) -> f64 {
        if self.extent <= 1.0 {
            return 0.0;
        }
        (1.0 - s_over_z0.ln() / self.extent.ln()).clamp(0.0, 1.0)
    }
}

/// GLS vertical mixing (see the module docs). The turbulence is stored in
/// `Solution3D::tke`, `Solution3D::gls` and stepped by the vertical
/// diffusion ([`crate::physics::apply_vertical_diffusion`]).
#[derive(Clone, Debug)]
pub struct GlsMixing {
    params: GlsParameters,
    stability: Stability,
    c3_minus: f64,
    kappa: f64,
    /// Lower bounds of `k` (m²/s²) and `ε` (m²/s³).
    k_min: f64,
    eps_min: f64,
    /// Galperin's `c_lim`, if the length is limited.
    length_limit: Option<f64>,
    /// Added to the turbulent viscosity and diffusivity (m²/s).
    background_viscosity: f64,
    background_diffusivity: f64,
    /// Roughness lengths of the surface (the minimum under Charnock's) and
    /// of the bed (m).
    z0_surface: f64,
    z0_bottom: f64,
    /// Charnock's constant of the surface roughness `α u*²/g`, if any.
    charnock: Option<f64>,
    /// The surface's TKE injection; a log layer without.
    wave_breaking: Option<WaveBreaking>,
}

impl GlsMixing {
    /// Default lower bound of `k` (GOTM's).
    pub const DEFAULT_K_MIN: f64 = 1e-8;
    /// Default lower bound of `ε` (GOTM's).
    pub const DEFAULT_EPS_MIN: f64 = 1e-12;
    /// Galperin et al.'s (1988) length limit.
    pub const DEFAULT_LENGTH_LIMIT: f64 = 0.53;
    /// Default steady-state Richardson number.
    pub const DEFAULT_RI_ST: f64 = 0.25;
    /// Default background viscosity and diffusivity (ROMS's usual `Akv_bak`,
    /// `Akt_bak`).
    pub const DEFAULT_BACKGROUND_VISCOSITY: f64 = 1e-5;
    pub const DEFAULT_BACKGROUND_DIFFUSIVITY: f64 = 1e-6;
    /// Default roughness of the surface and the bed (ROMS's `Zos`, `Zob`).
    pub const DEFAULT_ROUGHNESS: f64 = 0.02;
    /// Charnock's constant of the water side, `z₀ₛ = α u*²/g` with the
    /// water's friction velocity (Stacey 1999; ROMS's `charnok_alpha`,
    /// GOTM's `charnock_val`).
    pub const DEFAULT_CHARNOCK: f64 = 1400.0;
    /// Craig & Banner's (1994) `c_w` of the wave-breaking flux `c_w u*³`
    /// (ROMS's `crgban_cw`, GOTM's `cw`).
    pub const DEFAULT_WAVE_BREAKING: f64 = 100.0;

    /// A GLS model with `params` and `stability` functions; `c₃⁻` for
    /// [`Self::DEFAULT_RI_ST`] (see [`Self::with_steady_state_richardson`]).
    ///
    /// # Panics
    /// If `σ_ψ(c₂ − c₁)/n²` is not positive (no log layer), or `n ≥ 0`.
    pub fn new(params: GlsParameters, stability: StabilityFunctions) -> Self {
        let stability = Stability::new(stability);
        assert!(
            params.n < 0.0,
            "GLS exponent n must be negative, got {}",
            params.n
        );
        let radicand = params.sigma_psi * (params.c2 - params.c1) / params.n.powi(2);
        assert!(
            radicand > 0.0,
            "GLS constants have no log layer: σ_ψ(c₂ − c₁)/n² = {radicand}"
        );
        Self {
            params,
            stability,
            c3_minus: stability.c3_for(params.c1, params.c2, Self::DEFAULT_RI_ST),
            kappa: stability.cm0 * radicand.sqrt(),
            k_min: Self::DEFAULT_K_MIN,
            eps_min: Self::DEFAULT_EPS_MIN,
            length_limit: Some(Self::DEFAULT_LENGTH_LIMIT),
            background_viscosity: Self::DEFAULT_BACKGROUND_VISCOSITY,
            background_diffusivity: Self::DEFAULT_BACKGROUND_DIFFUSIVITY,
            z0_surface: Self::DEFAULT_ROUGHNESS,
            z0_bottom: Self::DEFAULT_ROUGHNESS,
            charnock: None,
            wave_breaking: None,
        }
    }

    /// k-ε with Canuto A stability functions (GOTM's default model).
    pub fn k_epsilon() -> Self {
        Self::new(GlsParameters::K_EPSILON, StabilityFunctions::CanutoA)
    }

    /// k-ω with Canuto A stability functions.
    pub fn k_omega() -> Self {
        Self::new(GlsParameters::K_OMEGA, StabilityFunctions::CanutoA)
    }

    /// Umlauf & Burchard's generic model with Canuto A stability functions.
    pub fn generic() -> Self {
        Self::new(GlsParameters::GENERIC, StabilityFunctions::CanutoA)
    }

    /// `c₃⁻` from the steady-state Richardson number `ri` instead.
    pub fn with_steady_state_richardson(mut self, ri: f64) -> Self {
        assert!(
            ri > 0.0,
            "steady-state Richardson number must be positive, got {ri}"
        );
        self.c3_minus = self.stability.c3_for(self.params.c1, self.params.c2, ri);
        self
    }

    /// Roughness lengths `z0_surface`, `z0_bottom` (m) of the boundary log
    /// layers. Match the bed's to the drag's ([`crate::physics::BottomDrag3D`]).
    pub fn with_roughness(mut self, z0_surface: f64, z0_bottom: f64) -> Self {
        assert!(
            z0_surface > 0.0 && z0_bottom > 0.0,
            "roughness lengths must be positive, got {z0_surface}, {z0_bottom}"
        );
        self.z0_surface = z0_surface;
        self.z0_bottom = z0_bottom;
        self
    }

    /// Charnock's surface roughness `z₀ₛ = max(α u*²/g, z₀)` from the
    /// surface stress, with `z₀` the surface roughness of
    /// [`Self::with_roughness`] as the minimum (ROMS's `CHARNOK`). `alpha`
    /// is for the water's friction velocity: [`Self::DEFAULT_CHARNOCK`].
    pub fn with_charnock_roughness(mut self, alpha: f64) -> Self {
        assert!(
            alpha > 0.0,
            "Charnock's constant must be positive, got {alpha}"
        );
        self.charnock = Some(alpha);
        self
    }

    /// TKE injected at the surface by breaking waves, the flux `c_w u*³`
    /// (Craig & Banner 1994; [`Self::DEFAULT_WAVE_BREAKING`]), in place of
    /// the surface log layer. Below the surface the model then holds its
    /// shear-free layer ([`Self::shear_free_layer`]); the flux conditions
    /// of `k` and `ψ` are that layer's, at the top layer's centre (see the
    /// module docs).
    ///
    /// # Panics
    /// If `c_w` is negative, or the model has no shear-free layer.
    pub fn with_wave_breaking(mut self, c_w: f64) -> Self {
        assert!(
            c_w >= 0.0,
            "wave-breaking c_w must be non-negative, got {c_w}"
        );
        let (decay, slope) = self.shear_free_layer().unwrap_or_else(|| {
            panic!(
                "GLS model {:?} has no shear-free layer for wave breaking",
                self.params
            )
        });
        // k at the surface over the log layer's, (σ_k c_w/(a c_μ⁰ L))^(2/3) (c_μ⁰)²
        let cm0 = self.stability.cm0;
        let surface_ratio =
            (self.params.sigma_k * c_w / (decay * cm0 * slope)).powf(2.0 / 3.0) * cm0 * cm0;
        self.wave_breaking = Some(WaveBreaking {
            c_w,
            decay,
            slope,
            extent: surface_ratio.powf(1.0 / decay),
        });
        self
    }

    /// The surface's wave breaking, if configured.
    pub fn wave_breaking(&self) -> Option<WaveBreaking> {
        self.wave_breaking
    }

    /// The model's shear-free layer under a source of `k` at a wall,
    /// `(a, L)`: `k = K s^(−a)`, `l = L s` at a distance `s` from the wall
    /// (plus its roughness), where the diffusion of `k` and `ψ` balances
    /// dissipation (no shear, no stratification). With `c_μ = c_μ⁰` (the
    /// quasi-equilibrium stability functions without stratification), the
    /// `k` and `ψ` equations give `(3/2) a² L² = σ_k (c_μ⁰)²` and
    /// `b (b − a/2) L² = c₂ σ_ψ (c_μ⁰)²` with `b = n − a m`, so that
    /// `a` is the positive root of
    /// `(2σ_k m(m + ½) − 3c₂σ_ψ) a² − 2σ_k n(2m + ½) a + 2σ_k n² = 0`
    /// (Umlauf & Burchard 2003, §4; GOTM's `gen_alpha`, `gen_l`). The
    /// presets: k-ε `a` 4.97, `L` 0.087; k-ω 2.53, 0.24; generic 2.00, 0.19
    /// (designed for 2, after Terray et al. 1996). `None` if no root is
    /// positive.
    pub fn shear_free_layer(&self) -> Option<(f64, f64)> {
        let GlsParameters {
            m,
            n,
            sigma_k,
            sigma_psi,
            c2,
            ..
        } = self.params;
        let qa = 2.0 * sigma_k * m * (m + 0.5) - 3.0 * c2 * sigma_psi;
        let qb = -2.0 * sigma_k * n * (2.0 * m + 0.5);
        let qc = 2.0 * sigma_k * n * n;
        // The smallest positive root (`qc > 0`: with `qa < 0` there is one)
        let roots = if qa.abs() < 1e-12 {
            [-qc / qb, f64::NAN]
        } else {
            let disc = (qb * qb - 4.0 * qa * qc).sqrt();
            [(-qb + disc) / (2.0 * qa), (-qb - disc) / (2.0 * qa)]
        };
        let decay = roots
            .into_iter()
            .filter(|&a| a > 0.0)
            .min_by(f64::total_cmp)?;
        let slope = self.stability.cm0 * (2.0 * sigma_k / 3.0).sqrt() / decay;
        Some((decay, slope))
    }

    /// Surface roughness at the surface friction velocity `u_star`, with the
    /// column's own (from the waves), if any: the largest of the constant
    /// roughness, the column's and Charnock's.
    #[inline]
    fn surface_roughness(&self, u_star: f64, g: f64, column: Option<f64>) -> f64 {
        let z0 = column.map_or(self.z0_surface, |z0| z0.max(self.z0_surface));
        match self.charnock {
            Some(alpha) => (alpha * u_star * u_star / g).max(z0),
            None => z0,
        }
    }

    /// Background viscosity and diffusivity (m²/s), added to the turbulent ones.
    pub fn with_background(mut self, viscosity: f64, diffusivity: f64) -> Self {
        assert!(
            viscosity >= 0.0 && diffusivity >= 0.0,
            "background mixing must be non-negative, got {viscosity}, {diffusivity}"
        );
        self.background_viscosity = viscosity;
        self.background_diffusivity = diffusivity;
        self
    }

    /// Galperin's length limit `l ≤ c_lim √(2k)/N` with `c_lim`, or none.
    pub fn with_length_limit(mut self, c_lim: Option<f64>) -> Self {
        self.length_limit = c_lim;
        self
    }

    /// Lower bounds of `k` (m²/s²) and `ε` (m²/s³).
    pub fn with_minimum(mut self, k_min: f64, eps_min: f64) -> Self {
        assert!(
            k_min > 0.0 && eps_min > 0.0,
            "minimum k and ε must be positive, got {k_min}, {eps_min}"
        );
        self.k_min = k_min;
        self.eps_min = eps_min;
        self
    }

    /// The model's exponents and constants.
    pub fn parameters(&self) -> GlsParameters {
        self.params
    }

    /// Neutral stability function `c_μ⁰`.
    pub fn cm0(&self) -> f64 {
        self.stability.cm0
    }

    /// `c₃` in stable stratification.
    pub fn c3_minus(&self) -> f64 {
        self.c3_minus
    }

    /// The von Kármán constant of the model's log layer.
    pub fn kappa(&self) -> f64 {
        self.kappa
    }

    /// Stability functions `(c_μ, c_μ′)` at `α_N = (k/ε)² N²`.
    pub fn stability_functions(&self, an: f64) -> (f64, f64) {
        self.stability.eval(an)
    }

    /// `ε` of `k` and `ψ`.
    #[inline]
    pub fn dissipation(&self, k: f64, psi: f64) -> f64 {
        let GlsParameters { p, m, n, .. } = self.params;
        self.stability.cm0.powf(3.0 + p / n) * k.powf(1.5 + m / n) * psi.powf(-1.0 / n)
    }

    /// `ψ` of `k` and `ε`.
    #[inline]
    pub fn psi(&self, k: f64, eps: f64) -> f64 {
        let GlsParameters { p, m, n, .. } = self.params;
        let cm0 = self.stability.cm0;
        let length = cm0.powi(3) * k.powf(1.5) / eps;
        cm0.powf(p) * k.powf(m) * length.powf(n)
    }

    /// Turbulent viscosity and diffusivity `(c_μ, c_μ′)·√k l` of `k`, `ε`
    /// and `N²`.
    #[inline]
    fn coefficients(&self, k: f64, eps: f64, n2: f64) -> (f64, f64) {
        let tau = k / eps;
        let (cm, cmp) = self.stability.eval(tau * tau * n2);
        let x = self.stability.cm0.powi(3) * k * k / eps;
        (cm * x, cmp * x)
    }

    /// `ε` bounded below by `ε_min` and, in stable water, the length limit.
    #[inline]
    fn bounded_dissipation(&self, k: f64, eps: f64, n2: f64) -> f64 {
        let eps = eps.max(self.eps_min);
        match self.length_limit {
            Some(c_lim) if n2 > 0.0 => {
                let cm0 = self.stability.cm0;
                eps.max(cm0.powi(3) / (std::f64::consts::SQRT_2 * c_lim) * k * n2.sqrt())
            }
            _ => eps,
        }
    }

    /// `ψ` at a wall with friction-velocity `k` and roughness `z0`.
    #[inline]
    fn wall_psi(&self, k: f64, z0: f64) -> f64 {
        let GlsParameters { p, m, n, .. } = self.params;
        self.stability.cm0.powf(p) * k.powf(m) * (self.kappa * z0).powf(n)
    }

    /// Flux of `ψ` into the water of a log layer at `z′ + z₀` from a wall,
    /// with `k` there.
    #[inline]
    fn wall_psi_flux(&self, k: f64, height: f64) -> f64 {
        let GlsParameters {
            p, m, n, sigma_psi, ..
        } = self.params;
        -n * self.stability.cm0.powf(p + 1.0) * self.kappa.powf(n + 1.0) / sigma_psi
            * k.powf(m + 0.5)
            * height.powf(n)
    }

    /// Flux of `ψ` into the water at a distance `s` from the surface (plus
    /// its roughness) in the shear-free layer of `wave`, with `k` there:
    /// `−(n − a m) (c_μ⁰)^(p+1) L^(n+1)/σ_ψ k^(m+½) s^n`, the log layer's
    /// flux for `a = 0`, `L = κ`.
    #[inline]
    fn shear_free_psi_flux(&self, wave: &WaveBreaking, k: f64, s: f64) -> f64 {
        let GlsParameters {
            p, m, n, sigma_psi, ..
        } = self.params;
        -(n - wave.decay * m) * self.stability.cm0.powf(p + 1.0) * wave.slope.powf(n + 1.0)
            / sigma_psi
            * k.powf(m + 0.5)
            * s.powf(n)
    }

    /// `k` and `ψ` at the surface itself of the shear-free layer under the
    /// flux `c_w u*³` at roughness `z0`: `K z₀^(−a) = (σ_k c_w/(a c_μ⁰ L))^(2/3) u*²`,
    /// `l = L z₀`.
    #[inline]
    fn wave_surface(&self, wave: &WaveBreaking, u_star: f64, z0: f64) -> (f64, f64) {
        let GlsParameters {
            p, m, n, sigma_k, ..
        } = self.params;
        let cm0 = self.stability.cm0;
        let k = ((sigma_k * wave.c_w / (wave.decay * cm0 * wave.slope)).powf(2.0 / 3.0)
            * u_star
            * u_star)
            .max(self.k_min);
        (k, cm0.powf(p) * k.powf(m) * (wave.slope * z0).powf(n))
    }
}

/// Per-w-point buffers of one column step, in [`Turbulence::scratch`].
struct Fields<'a> {
    /// Layer thicknesses (the first N)
    h: &'a mut [f64],
    n2: &'a mut [f64],
    s2: &'a mut [f64],
    /// Viscosity, `P`, `B`, `ε` of the old turbulence
    num: &'a mut [f64],
    production: &'a mut [f64],
    buoyancy: &'a mut [f64],
    eps: &'a mut [f64],
    k_old: &'a mut [f64],
    psi_old: &'a mut [f64],
}

/// Tridiagonal system of the interior w-points, in [`Turbulence::scratch`].
struct Solver<'a> {
    a: &'a mut [f64],
    b: &'a mut [f64],
    c: &'a mut [f64],
    d: &'a mut [f64],
    x: &'a mut [f64],
    c_prime: &'a mut [f64],
    d_prime: &'a mut [f64],
}

/// `scratch` split into [`Fields`] and [`Solver`] for `nw` w-points.
fn work(scratch: &mut Vec<f64>, nw: usize) -> (Fields<'_>, Solver<'_>) {
    const N_FIELDS: usize = 9;
    const N_SOLVER: usize = 7;
    let len = (N_FIELDS + N_SOLVER) * nw;
    if scratch.len() < len {
        scratch.resize(len, 0.0);
    }
    let mut chunks = scratch[..len].chunks_exact_mut(nw);
    let mut next = || chunks.next().expect("sized above");
    (
        Fields {
            h: next(),
            n2: next(),
            s2: next(),
            num: next(),
            production: next(),
            buoyancy: next(),
            eps: next(),
            k_old: next(),
            psi_old: next(),
        },
        Solver {
            a: next(),
            b: next(),
            c: next(),
            d: next(),
            x: next(),
            c_prime: next(),
            d_prime: next(),
        },
    )
}

/// One backward-Euler step of `∂Y/∂t = ∂/∂z(D ∂Y/∂z) + Q + L·Y` at the
/// interior w-points `1..nw−1` of `y`: `diff(j)` is `D` at w-point `j`
/// (averaged to the layers between), `source(j) = (Q, L)` with `L ≤ 0`
/// taken at the new time, and `flux_bottom`, `flux_top` enter the water at
/// the boundary layers' centres, replacing the diffusion across those
/// layers (GOTM `diff_face` with Neumann conditions).
fn solve_interior(
    y: &mut [f64],
    h: &[f64],
    diff: impl Fn(usize) -> f64,
    source: impl Fn(usize) -> (f64, f64),
    flux_bottom: f64,
    flux_top: f64,
    dt: f64,
    s: &mut Solver<'_>,
) {
    let nw = y.len();
    let m = nw - 2;
    for j in 1..nw - 1 {
        let row = j - 1;
        let volume = 0.5 * (h[j - 1] + h[j]);
        let lower = if j == 1 {
            0.0
        } else {
            dt * 0.5 * (diff(j) + diff(j - 1)) / h[j - 1] / volume
        };
        let upper = if j == nw - 2 {
            0.0
        } else {
            dt * 0.5 * (diff(j) + diff(j + 1)) / h[j] / volume
        };
        let (q, l) = source(j);
        s.a[row] = -lower;
        s.c[row] = -upper;
        s.b[row] = 1.0 + lower + upper - dt * l;
        s.d[row] = y[j] + dt * q;
        if j == 1 {
            s.d[row] += dt * flux_bottom / volume;
        }
        if j == nw - 2 {
            s.d[row] += dt * flux_top / volume;
        }
    }
    solve_tridiagonal(
        &s.a[..m],
        &s.b[..m],
        &s.c[..m],
        &s.d[..m],
        &mut s.x[..m],
        &mut s.c_prime[..m],
        &mut s.d_prime[..m],
    );
    y[1..nw - 1].copy_from_slice(&s.x[..m]);
}

/// Patankar's split of a source `prod + buoy − diss` of `Y` into `(Q, L)`
/// with `Q ≥ 0` explicit and `L·Y` (`L ≤ 0`) implicit: a negative `buoy` is a
/// sink when the sum of the productions is negative.
#[inline]
fn patankar(prod: f64, buoy: f64, diss: f64, y_old: f64) -> (f64, f64) {
    if prod + buoy > 0.0 {
        (prod + buoy, -diss / y_old)
    } else {
        (prod, -(diss - buoy) / y_old)
    }
}

impl VerticalMixing for GlsMixing {
    /// The mixing of the turbulence at its minimum (there is no state to
    /// step here).
    fn compute_mixing_into(
        &self,
        _column: &Column,
        _forcing: &Forcing,
        av: &mut [f64],
        kt: &mut [f64],
    ) {
        let (num, nuh) = self.coefficients(self.k_min, self.eps_min, 0.0);
        av.fill(num + self.background_viscosity);
        kt.fill(nuh + self.background_diffusivity);
    }

    /// `k = k_min`, `ε = ε_min`.
    fn initial_turbulence(&self) -> Option<[f64; 2]> {
        Some([self.k_min, self.psi(self.k_min, self.eps_min)])
    }

    fn step_mixing_into(
        &self,
        column: &Column,
        _forcing: &Forcing,
        dt: f64,
        turbulence: Turbulence<'_>,
        av: &mut [f64],
        kt: &mut [f64],
    ) {
        let Turbulence {
            tke,
            gls: psi,
            scratch,
        } = turbulence;
        let nl = column.u.len();
        let nw = nl + 1;
        assert!(
            tke.len() == nw && psi.len() == nw,
            "GLS turbulence must be at the {nw} w-points, got {} and {}",
            tke.len(),
            psi.len()
        );
        let (f, mut solver) = work(scratch, nw);
        let GlsParameters {
            sigma_k,
            sigma_psi,
            c1,
            c2,
            c3_plus,
            ..
        } = self.params;
        let cm0 = self.stability.cm0;
        let u_surface = column.surface_friction_velocity;
        let z0_surface = self.surface_roughness(u_surface, column.g, column.surface_roughness);

        for l in 0..nl {
            f.h[l] = column.z_w[l + 1] - column.z_w[l];
        }
        // Shear and stratification at the interior w-points, held to the ends
        let buoyancy_scale = column.g / column.rho0;
        for j in 1..nl {
            let dz = column.z_r[j] - column.z_r[j - 1];
            let (du, dv) = (column.u[j] - column.u[j - 1], column.v[j] - column.v[j - 1]);
            f.s2[j] = (du * du + dv * dv) / (dz * dz);
            f.n2[j] = -buoyancy_scale * (column.rho[j] - column.rho[j - 1]) / dz;
        }
        if nl >= 2 {
            (f.s2[0], f.n2[0]) = (f.s2[1], f.n2[1]);
            (f.s2[nl], f.n2[nl]) = (f.s2[nl - 1], f.n2[nl - 1]);
        } else {
            f.s2.fill(0.0);
            f.n2.fill(0.0);
        }

        if nl >= 2 {
            // P, B, ε of the old turbulence, at least the minima. Advection
            // (`Hydrostatic3D::with_turbulence_advection`) can leave values
            // below them, or not even positive: such a point is taken as
            // having no turbulence (k and ε at their minima), not as `k`
            // with the minimum ε, whose length `k^(3/2)/ε_min` would mix the
            // column at tens of m²/s
            for j in 1..nl {
                let valid = tke[j] > 0.0 && psi[j] > 0.0;
                let (k, unbounded) = if valid {
                    let k = tke[j].max(self.k_min);
                    (k, self.dissipation(k, psi[j]))
                } else {
                    (self.k_min, self.eps_min)
                };
                let eps = unbounded.max(self.eps_min);
                if !valid || k != tke[j] || eps != unbounded {
                    tke[j] = k;
                    psi[j] = self.psi(k, eps);
                }
                let (num, nuh) = self.coefficients(tke[j], eps, f.n2[j]);
                f.eps[j] = eps;
                f.num[j] = num;
                f.production[j] = num * f.s2[j];
                f.buoyancy[j] = -nuh * f.n2[j];
            }
            f.k_old.copy_from_slice(tke);
            f.psi_old.copy_from_slice(psi);
            let Fields {
                h,
                n2,
                num,
                production,
                buoyancy,
                eps,
                k_old,
                psi_old,
                ..
            } = &f;

            // k, with no flux through the log layers, and the shear-free
            // layer's flux at the top layer's centre under breaking waves
            let top_centre = 0.5 * h[nl - 1] + z0_surface;
            let k_flux_top = self.wave_breaking.map_or(0.0, |wave| {
                wave.c_w * u_surface.powi(3) * (z0_surface / top_centre).powf(1.5 * wave.decay)
            });
            solve_interior(
                tke,
                h,
                |j| num[j] / sigma_k,
                |j| patankar(production[j], buoyancy[j], eps[j], k_old[j]),
                0.0,
                k_flux_top,
                dt,
                &mut solver,
            );
            for k in &mut tke[1..nl] {
                *k = k.max(self.k_min);
            }

            // ψ, with the boundary layers' fluxes at the end layers' centres.
            // Under breaking waves the top one is the shear-free layer's
            // where the top interior w-point has no shear production and the
            // log layer's where it is in equilibrium (`P = ε`), linear in
            // `P/ε` between
            let flux_bottom = self.wall_psi_flux(tke[1], 0.5 * h[0] + self.z0_bottom);
            let log_flux_top = self.wall_psi_flux(tke[nl - 1], top_centre);
            let flux_top = match &self.wave_breaking {
                None => log_flux_top,
                Some(wave) => {
                    let w = wave.share_above(top_centre / z0_surface);
                    let k_centre =
                        tke[nl - 1] * ((h[nl - 1] + z0_surface) / top_centre).powf(wave.decay);
                    w * self.shear_free_psi_flux(wave, k_centre, top_centre)
                        + (1.0 - w) * log_flux_top
                }
            };
            solve_interior(
                psi,
                h,
                |j| num[j] / sigma_psi,
                |j| {
                    let ratio = psi_old[j] / k_old[j];
                    let c3 = if buoyancy[j] > 0.0 {
                        c3_plus
                    } else {
                        self.c3_minus
                    };
                    patankar(
                        c1 * ratio * production[j],
                        c3 * ratio * buoyancy[j],
                        c2 * ratio * eps[j],
                        psi_old[j],
                    )
                },
                flux_bottom,
                flux_top,
                dt,
                &mut solver,
            );

            // ε bounded, and ψ consistent with it
            for j in 1..nl {
                let eps = self.bounded_dissipation(tke[j], self.dissipation(tke[j], psi[j]), n2[j]);
                psi[j] = self.psi(tke[j], eps);
            }
        }

        // The boundary layers at the boundary w-points
        tke[0] = (column.bottom_friction_velocity.powi(2) / (cm0 * cm0)).max(self.k_min);
        psi[0] = self.wall_psi(tke[0], self.z0_bottom);
        match &self.wave_breaking {
            None => {
                tke[nl] = (u_surface * u_surface / (cm0 * cm0)).max(self.k_min);
                psi[nl] = self.wall_psi(tke[nl], z0_surface);
            }
            Some(wave) => (tke[nl], psi[nl]) = self.wave_surface(wave, u_surface, z0_surface),
        }

        for j in 0..nw {
            let eps = self.bounded_dissipation(tke[j], self.dissipation(tke[j], psi[j]), f.n2[j]);
            let (num, nuh) = self.coefficients(tke[j], eps, f.n2[j]);
            av[j] = num + self.background_viscosity;
            kt[j] = nuh + self.background_diffusivity;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A column of `nl` uniform layers over `depth` with uniform shear `s`
    /// and `N² = ri·s²`, its w-point turbulence at `k`, `ε`.
    struct Homogeneous {
        z_r: Vec<f64>,
        z_w: Vec<f64>,
        u: Vec<f64>,
        v: Vec<f64>,
        rho: Vec<f64>,
        tke: Vec<f64>,
        gls: Vec<f64>,
        /// Surface friction velocity (m/s)
        u_star: f64,
    }

    const G: f64 = 9.81;
    const RHO0: f64 = 1025.0;

    impl Homogeneous {
        fn new(
            mixing: &GlsMixing,
            nl: usize,
            depth: f64,
            s: f64,
            ri: f64,
            k: f64,
            eps: f64,
        ) -> Self {
            let dz = depth / nl as f64;
            let z_w: Vec<f64> = (0..=nl).map(|j| -depth + j as f64 * dz).collect();
            let z_r: Vec<f64> = (0..nl).map(|l| -depth + (l as f64 + 0.5) * dz).collect();
            let n2 = ri * s * s;
            Self {
                u: z_r.iter().map(|z| s * z).collect(),
                v: vec![0.0; nl],
                rho: z_r.iter().map(|z| RHO0 * (1.0 - n2 / G * z)).collect(),
                tke: vec![k; nl + 1],
                gls: vec![mixing.psi(k, eps); nl + 1],
                u_star: 0.0,
                z_r,
                z_w,
            }
        }

        fn step(&mut self, mixing: &GlsMixing, dt: f64, scratch: &mut Vec<f64>) -> Vec<f64> {
            let nw = self.z_w.len();
            let (mut av, mut kt) = (vec![0.0; nw], vec![0.0; nw]);
            mixing.step_mixing_into(
                &Column {
                    z_r: &self.z_r,
                    z_w: &self.z_w,
                    u: &self.u,
                    v: &self.v,
                    rho: &self.rho,
                    g: G,
                    rho0: RHO0,
                    surface_friction_velocity: self.u_star,
                    bottom_friction_velocity: 0.0,
                    surface_roughness: None,
                },
                &Forcing {
                    surface_stress: [0.0; 2],
                    bottom_stress: [0.0; 2],
                    surface_heat_flux: 0.0,
                },
                dt,
                Turbulence {
                    tke: &mut self.tke,
                    gls: &mut self.gls,
                    scratch,
                },
                &mut av,
                &mut kt,
            );
            av
        }
    }

    /// Canuto et al.'s neutral stability function `c_μ⁰` (Umlauf & Burchard
    /// 2005, Table 3: 0.5270 for A, 0.5540 for B), equal to the stability
    /// function at `α_N = 0` so that a log layer has `k = u*²/(c_μ⁰)²`; and
    /// the derived constants of the presets (ROMS: c₃⁻ = 0.05 for the generic
    /// model; κ near 0.4).
    #[test]
    fn canuto_stability_functions_have_the_published_neutral_values() {
        for (functions, published) in [
            (StabilityFunctions::CanutoA, 0.5270),
            (StabilityFunctions::CanutoB, 0.5540),
        ] {
            let gls = GlsMixing::new(GlsParameters::K_EPSILON, functions);
            let (cm, cmp) = gls.stability_functions(0.0);
            assert!(
                (gls.cm0() - published).abs() < 1e-3,
                "{functions:?}: c_μ⁰ {} against {published}",
                gls.cm0()
            );
            assert!(
                (cm - gls.cm0()).abs() < 1e-12,
                "{functions:?}: c_μ(0) {cm} against c_μ⁰ {}",
                gls.cm0()
            );
            assert!(
                cmp > cm,
                "{functions:?}: neutral Prandtl number {} ≥ 1",
                cm / cmp
            );
            // Stable stratification damps the mixing, momentum less than heat
            let (cm_s, cmp_s) = gls.stability_functions(5.0);
            assert!(cm_s < cm && cmp_s < cmp && cm_s / cmp_s > cm / cmp);
        }
        let generic = GlsMixing::generic();
        assert!(
            (generic.c3_minus() - 0.05).abs() < 0.01,
            "generic c₃⁻ {}",
            generic.c3_minus()
        );
        for gls in [GlsMixing::k_epsilon(), GlsMixing::k_omega(), generic] {
            assert!((0.37..0.43).contains(&gls.kappa()), "κ {}", gls.kappa());
        }
    }

    /// `c₃⁻` makes homogeneous shear turbulence steady at the steady-state
    /// Richardson number: in uniform shear and stratification, `k` grows
    /// below `Ri_st` = 0.25 and decays above it, and at `Ri_st` its growth
    /// rate is three orders of magnitude smaller than either (≈ 10⁻⁵ S).
    #[test]
    fn homogeneous_turbulence_is_steady_at_the_steady_state_richardson_number() {
        let (s, dt) = (0.05, 2.0);
        for gls in [
            GlsMixing::k_epsilon(),
            GlsMixing::k_omega(),
            GlsMixing::generic(),
        ] {
            let gls = gls.with_length_limit(None).with_background(0.0, 0.0);
            let rate = |ri: f64| {
                let mut column = Homogeneous::new(&gls, 200, 200.0, s, ri, 1e-4, 1e-6);
                let mut scratch = Vec::new();
                let (mid, mut k_half) = (100, 0.0);
                let steps = (60.0 / s / dt) as usize;
                for n in 0..steps {
                    if n == steps / 2 {
                        k_half = column.tke[mid];
                    }
                    column.step(&gls, dt, &mut scratch);
                }
                (column.tke[mid] / k_half).ln() / (steps / 2) as f64 / dt / s
            };
            let (below, at, above) = (rate(0.2), rate(GlsMixing::DEFAULT_RI_ST), rate(0.3));
            let name = format!("{:?}", gls.parameters());
            assert!(below > 0.0 && above < 0.0, "{name}: rates {below}, {above}");
            assert!(
                at.abs() < 0.01 * below.min(-above),
                "{name}: growth rate {at} at Ri_st against {below}, {above}"
            );
        }
    }

    /// The Patankar sinks keep `k` and `ψ` positive and finite for any step,
    /// in strong stratification and in convection.
    #[test]
    fn turbulence_stays_positive_for_any_time_step() {
        for gls in [
            GlsMixing::k_epsilon(),
            GlsMixing::k_omega(),
            GlsMixing::generic(),
        ] {
            for ri in [-2.0, 0.0, 10.0] {
                let mut column = Homogeneous::new(&gls, 30, 30.0, 0.1, ri, 1e-3, 1e-5);
                let mut scratch = Vec::new();
                for dt in [1e6, 1.0, 1e4] {
                    let av = column.step(&gls, dt, &mut scratch);
                    assert!(
                        column
                            .tke
                            .iter()
                            .chain(&column.gls)
                            .chain(&av)
                            .all(|x| x.is_finite() && *x > 0.0),
                        "Ri {ri}, dt {dt}: k {:?}",
                        column.tke
                    );
                }
            }
        }
    }

    /// Advection can leave `k` or `ψ` not even positive (an upwind DG
    /// undershoot): the step takes such a point as having no turbulence
    /// (`k`, `ε` at their minima), and the rest of the column is stepped as
    /// if it had been. Before, a negative `ψ` gave a negative or NaN `ε`
    /// there, the implicit solve spread the NaN over the column, and the
    /// final bounds reset the whole column's turbulence to the minima.
    #[test]
    fn turbulence_that_is_not_positive_is_taken_as_none() {
        for gls in [
            GlsMixing::k_epsilon(),
            GlsMixing::k_omega(),
            GlsMixing::generic(),
        ] {
            let [k_min, psi_min] = gls.initial_turbulence().expect("prognostic");
            let run = |bad: bool| {
                let mut column = Homogeneous::new(&gls, 10, 10.0, 0.01, 0.0, 1e-4, 1e-7);
                if bad {
                    column.tke[3] = -1e-6;
                    column.gls[4] = -1e-9;
                    column.gls[5] = 0.0;
                    column.tke[6] = f64::NAN;
                } else {
                    // What they are taken as
                    for j in 3..=6 {
                        column.tke[j] = k_min;
                        column.gls[j] = psi_min;
                    }
                }
                let mut scratch = Vec::new();
                let av = column.step(&gls, 60.0, &mut scratch);
                (column.tke, column.gls, av)
            };
            let (bad, reference) = (run(true), run(false));
            assert!(
                bad.0
                    .iter()
                    .chain(&bad.1)
                    .chain(&bad.2)
                    .all(|x| x.is_finite() && *x > 0.0),
                "{:?}: k {:?}, ψ {:?}",
                gls.parameters(),
                bad.0,
                bad.1
            );
            // Away from the bad points the column is the reference's
            for (name, a, b) in [("k", &bad.0, &reference.0), ("ψ", &bad.1, &reference.1)] {
                for j in [1, 8, 9] {
                    assert!(
                        (a[j] - b[j]).abs() <= 1e-12 * b[j].abs(),
                        "{:?}: {name} at w-point {j}: {:.6e} against {:.6e}",
                        gls.parameters(),
                        a[j],
                        b[j]
                    );
                }
            }
            assert!(bad.0[1] > 10.0 * k_min, "test regime: k {:.3e}", bad.0[1]);
        }
    }

    /// A single layer has no interior w-point: only the boundary log layers.
    #[test]
    fn a_single_layer_has_only_the_boundary_layers() {
        let gls = GlsMixing::k_epsilon();
        let mut column = Homogeneous::new(&gls, 1, 0.5, 0.0, 0.0, 1e-4, 1e-6);
        let av = column.step(&gls, 10.0, &mut Vec::new());
        assert_eq!(column.tke, vec![GlsMixing::DEFAULT_K_MIN; 2]);
        assert!(av.iter().all(|x| x.is_finite() && *x > 0.0));
    }

    /// The shear-free layer's `(a, L)` satisfy both of its balances, the
    /// `k` equation's `(3/2) a² L² = σ_k (c_μ⁰)²` and the `ψ` equation's
    /// `b (b − a/2) L² = c₂ σ_ψ (c_μ⁰)²`, `b = n − a m`; the generic model
    /// has GOTM's `gen_alpha` = −2 (Umlauf & Burchard 2003 designed it for
    /// that) and a slope near its `gen_l` = 0.2.
    #[test]
    fn the_shear_free_layer_balances_both_equations() {
        for gls in [
            GlsMixing::k_epsilon(),
            GlsMixing::k_omega(),
            GlsMixing::generic(),
        ] {
            let GlsParameters {
                m,
                n,
                sigma_k,
                sigma_psi,
                c2,
                ..
            } = gls.parameters();
            let (a, l) = gls.shear_free_layer().expect("a shear-free layer");
            let cm0_2 = gls.cm0().powi(2);
            let b = n - a * m;
            let name = format!("{:?}", gls.parameters());
            assert!(
                (1.5 * a * a * l * l / (sigma_k * cm0_2) - 1.0).abs() < 1e-12,
                "{name}: k balance, a {a}, L {l}"
            );
            assert!(
                (b * (b - 0.5 * a) * l * l / (c2 * sigma_psi * cm0_2) - 1.0).abs() < 1e-12,
                "{name}: ψ balance, a {a}, L {l}"
            );
        }
        let (a, l) = GlsMixing::generic().shear_free_layer().unwrap();
        assert!(
            (a - 2.0).abs() < 0.01 && (l - 0.2).abs() < 0.01,
            "generic: a {a}, L {l}"
        );
    }

    /// Charnock's roughness follows the stress above its minimum.
    #[test]
    fn charnock_roughness_has_the_constant_roughness_as_its_minimum() {
        let gls = GlsMixing::k_epsilon().with_charnock_roughness(GlsMixing::DEFAULT_CHARNOCK);
        let z0 = |u_star: f64| gls.surface_roughness(u_star, G, None);
        assert!((z0(0.02) - 1400.0 * 4e-4 / G).abs() < 1e-15);
        assert_eq!(z0(0.002), GlsMixing::DEFAULT_ROUGHNESS);
        assert_eq!(
            GlsMixing::k_epsilon().surface_roughness(0.02, G, None),
            GlsMixing::DEFAULT_ROUGHNESS
        );
    }

    /// The steady column under breaking waves over still, unstratified
    /// water: `(max |k/k_a − 1|, max |l/l_a − 1|)` over 0.5–2.5 m deep
    /// against the shear-free layer `k_a = K s^(−a)`, `l_a = L s`.
    fn shear_free_errors(
        gls: &GlsMixing,
        nl: usize,
        depth: f64,
        u_star: f64,
        z0: f64,
    ) -> (f64, f64) {
        let (a, slope) = gls.shear_free_layer().unwrap();
        let k_surface = (gls.parameters().sigma_k * gls.wave_breaking().unwrap().c_w
            / (a * gls.cm0() * slope))
            .powf(2.0 / 3.0)
            * u_star
            * u_star;
        let mut column = Homogeneous::new(gls, nl, depth, 0.0, 0.0, 1e-14, 1e-20);
        column.u_star = u_star;
        let (mut scratch, dt) = (Vec::new(), 60.0);
        for _ in 0..(48.0 * 3600.0 / dt) as usize {
            column.step(gls, dt, &mut scratch);
        }
        assert!(
            (column.tke[nl] / k_surface - 1.0).abs() < 1e-12,
            "surface k {} against {k_surface}",
            column.tke[nl]
        );
        let (mut k_err, mut l_err) = (0.0_f64, 0.0_f64);
        for j in 1..nl {
            let s = -column.z_w[j] + z0;
            if !(0.5..=2.5).contains(&(s - z0)) {
                continue;
            }
            let k = column.tke[j];
            let l = gls.cm0().powi(3) * k.powf(1.5) / gls.dissipation(k, column.gls[j]);
            k_err = k_err.max((k / (k_surface * (z0 / s).powf(a)) - 1.0).abs());
            l_err = l_err.max((l / (slope * s) - 1.0).abs());
        }
        (k_err, l_err)
    }

    /// Breaking waves over still, unstratified water (Craig & Banner 1994):
    /// with the flux `c_w u*³` at the surface, the column settles into the
    /// model's shear-free layer, `k = K s^(−a)` with
    /// `K = (σ_k c_w/(a c_μ⁰ L))^(2/3) u*² z₀^a` and `l = L s`, `s` the depth
    /// plus `z₀` (here 0.5 m, a wave height's). The surface condition blends
    /// towards the log layer's by the top layer centre's depth inside the
    /// wave-affected layer, so the error is first order in the layer
    /// thickness. Measured over 0.5–2.5 m deep, at 0.4 / 0.2 / 0.1 m layers:
    /// `k` 39 / 19 / 10 % (k-ω), 18 / 11 / 7.3 % (generic), the length
    /// within 4 %. k-ε's layer (`a` ≈ 5, `l` ≈ 0.09 s) is too thin for
    /// these layers (its turbulence collapses below ≈ 0.7 m even from the
    /// exact profile): at 1 cm and 5 mm layers `k` holds to 4.3 / 2.7 %,
    /// the length to 0.5 %.
    #[test]
    fn breaking_waves_make_the_shear_free_layer() {
        let (u_star, z0) = (0.01, 0.5);
        let waves = |gls: GlsMixing| {
            gls.with_roughness(z0, 0.02)
                .with_minimum(1e-14, 1e-20)
                .with_wave_breaking(GlsMixing::DEFAULT_WAVE_BREAKING)
        };
        for gls in [GlsMixing::k_omega(), GlsMixing::generic()] {
            let gls = waves(gls);
            let name = format!("{:?}", gls.parameters());
            let errors: Vec<(f64, f64)> = [50, 100, 200]
                .into_iter()
                .map(|nl| shear_free_errors(&gls, nl, 20.0, u_star, z0))
                .collect();
            for pair in errors.windows(2) {
                assert!(
                    pair[1].0 < pair[0].0 / 1.4,
                    "{name}: k error {:.3} at half the layers against {:.3}",
                    pair[1].0,
                    pair[0].0
                );
            }
            let finest = errors[2].0;
            assert!(finest < 0.11, "{name}: k error {finest:.3} at 0.1 m layers");
            assert!(
                errors.iter().all(|&(_, l)| l < 0.04),
                "{name}: length errors {errors:?}"
            );
        }
        let gls = waves(GlsMixing::k_epsilon());
        let errors: Vec<(f64, f64)> = [500, 1000]
            .into_iter()
            .map(|nl| shear_free_errors(&gls, nl, 5.0, u_star, z0))
            .collect();
        assert!(
            errors[0].0 < 0.05 && errors[1].0 < 0.03 && errors.iter().all(|&(_, l)| l < 0.01),
            "k-ε: errors {errors:?} at 1 cm and 5 mm layers"
        );
    }
}
