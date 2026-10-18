//! Air–sea fluxes of heat and fresh water: the COARE 3.5 bulk formulae.
//!
//! The ocean gains heat at the surface from the net shortwave and longwave
//! radiation and loses it to the air by the turbulent sensible and latent
//! heat fluxes; it loses fresh water by evaporation and gains it from
//! precipitation. With the near-surface weather (10 m wind, 2 m air
//! temperature and humidity, sea-level pressure, downwelling radiation and
//! precipitation, [`AirState`]) and the sea-surface temperature and
//! salinity, [`Coare35::budget`] gives a column's [`SurfaceBudget`]:
//!
//! ```text
//! Q_ns = ε (LW↓ − σ T_s⁴) − H_s − H_l      (into the ocean, at the surface)
//! Q_sw = (1 − α) SW↓                        (into the ocean, absorbed below)
//! F    = E − P,   E = H_l / L_v
//! ```
//!
//! with ε = 0.97, α = 0.055 and the turbulent fluxes
//! `H_s = −ρ_a c_pa u* θ*`, `H_l = −ρ_a L_v u* q*` (upward positive) of
//! Monin–Obukhov similarity, iterated for the scales `u*`, `θ*`, `q*` as
//! COARE 3.5 does (Fairall et al. 2003; Edson et al. 2013): a Charnock
//! roughness growing with the 10 m neutral wind, the scalar roughness of
//! COARE 3.0, the stability functions of Beljaars & Holtslag (1991) for
//! stable air and Grachev et al. (2000) for convection, and a convective
//! gustiness that keeps the fluxes finite in calm air. It is a port of
//! NOAA's reference implementation (`pycoare.coare_35`) without the cool
//! skin, the warm layer and the wave inputs: the model's top layer is
//! metres thick and its temperature stands for the bulk SST. The tests
//! match that implementation to 1e-9 over stable, unstable, calm and storm
//! cases.
//!
//! The fluxes enter a 3D model through [`crate::physics::SurfaceFluxes3D`]:
//! `Q_ns/(ρ₀ c_p)` as the temperature flux at the surface, `Q_sw` absorbed
//! with depth by a [`JerlovWater`] profile, and `F` as the virtual salt
//! flux `S F/ρ_fw` (the volume of the fresh water is not added to the
//! column, as in ROMS without its `SALINITY` volume flux).

/// Specific heat capacity of sea water (J kg⁻¹ K⁻¹), ROMS's `Cp`.
pub const SEAWATER_HEAT_CAPACITY: f64 = 3985.0;

/// Density of fresh water (kg/m³), converting E − P to a volume flux.
pub const FRESH_WATER_DENSITY: f64 = 1000.0;

/// von Kármán's constant of COARE.
const VON: f64 = 0.4;
/// Kelvin at 0 °C, as COARE.
const TDK: f64 = 273.16;
/// Gas constant of dry air (J kg⁻¹ K⁻¹)
const RGAS: f64 = 287.1;
/// Specific heat of air (J kg⁻¹ K⁻¹)
const CPA: f64 = 1004.67;
/// Convective gustiness coefficient
const BETA: f64 = 1.2;
/// Charnock's parameter as a function of the 10 m neutral wind, `A1 U + A2`,
/// constant above `UMAX`
const A1: f64 = 0.0017;
const A2: f64 = -0.0050;
const UMAX: f64 = 19.0;
/// Emissivity of the sea surface
const EMISSIVITY: f64 = 0.97;
/// Stefan–Boltzmann constant (W m⁻² K⁻⁴)
const STEFAN_BOLTZMANN: f64 = 5.67e-8;
/// Shortwave albedo of the sea surface
const ALBEDO: f64 = 0.055;

/// The near-surface atmosphere over one column.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AirState {
    /// Wind speed at the wind height relative to the surface (m/s)
    pub wind_speed: f64,
    /// Air temperature at the temperature height (°C)
    pub air_temperature: f64,
    /// Relative humidity at the humidity height (fraction, 0–1)
    pub relative_humidity: f64,
    /// Air pressure at sea level (Pa)
    pub pressure: f64,
    /// Downwelling shortwave radiation at the surface (W/m²)
    pub shortwave: f64,
    /// Downwelling longwave radiation at the surface (W/m²)
    pub longwave: f64,
    /// Precipitation (kg m⁻² s⁻¹)
    pub precipitation: f64,
}

/// The turbulent fluxes of [`Coare35::turbulent`], COARE's signs: heat
/// fluxes positive upward, out of the ocean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TurbulentFluxes {
    /// Wind stress magnitude (N/m²)
    pub stress: f64,
    /// Sensible heat flux (W/m², upward)
    pub sensible: f64,
    /// Latent heat flux (W/m², upward)
    pub latent: f64,
    /// Evaporation (kg m⁻² s⁻¹)
    pub evaporation: f64,
    /// Friction velocity of the air `u*` (m/s)
    pub friction_velocity: f64,
}

/// A column's surface budget, positive into the ocean.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SurfaceBudget {
    /// Net heat flux at the surface without the shortwave: net longwave
    /// minus the sensible and latent fluxes (W/m², into the ocean)
    pub nonsolar: f64,
    /// Net shortwave entering the water (W/m²), absorbed with depth
    pub shortwave: f64,
    /// Evaporation minus precipitation (kg m⁻² s⁻¹): the ocean's loss of
    /// fresh water
    pub freshwater: f64,
}

/// The COARE 3.5 bulk algorithm; see the module docs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coare35 {
    /// Height of the wind (m)
    pub wind_height: f64,
    /// Height of the air temperature (m)
    pub temperature_height: f64,
    /// Height of the humidity (m)
    pub humidity_height: f64,
    /// Height of the atmospheric boundary layer for the convective
    /// gustiness (m)
    pub boundary_layer_height: f64,
    /// Gravity (m/s²)
    pub g: f64,
    /// Iterations of the similarity scales
    pub iterations: usize,
}

impl Default for Coare35 {
    fn default() -> Self {
        Self::new()
    }
}

impl Coare35 {
    /// The 10 m wind and the 2 m air temperature and humidity of a weather
    /// model (MET Nordic, MEPS, ERA5), g = 9.81 m/s², 10 iterations.
    pub fn new() -> Self {
        Self {
            wind_height: 10.0,
            temperature_height: 2.0,
            humidity_height: 2.0,
            boundary_layer_height: 600.0,
            g: 9.81,
            iterations: 10,
        }
    }

    /// With the normal gravity at latitude `lat` (degrees), as COARE.
    pub fn at_latitude(lat: f64) -> Self {
        Self {
            g: normal_gravity(lat),
            ..Self::new()
        }
    }

    /// The heights of the wind and of the air temperature and humidity (m).
    pub fn with_heights(self, wind: f64, temperature: f64, humidity: f64) -> Self {
        Self {
            wind_height: wind,
            temperature_height: temperature,
            humidity_height: humidity,
            ..self
        }
    }

    /// The surface budget of a column with sea-surface temperature `sst`
    /// (°C) and salinity `sss` under `air`.
    pub fn budget(&self, air: &AirState, sst: f64, sss: f64) -> SurfaceBudget {
        let turbulent = self.turbulent(air, sst, sss);
        SurfaceBudget {
            nonsolar: net_longwave(sst, air.longwave) - turbulent.sensible - turbulent.latent,
            shortwave: (1.0 - ALBEDO) * air.shortwave.max(0.0),
            freshwater: turbulent.evaporation - air.precipitation,
        }
    }

    /// The turbulent fluxes over a sea at `sst` (°C) and salinity `sss`.
    pub fn turbulent(&self, air: &AirState, sst: f64, sss: f64) -> TurbulentFluxes {
        let (zu, zt, zq) = (
            self.wind_height,
            self.temperature_height,
            self.humidity_height,
        );
        let (g, zi) = (self.g, self.boundary_layer_height);
        let t = air.air_temperature;
        let p = air.pressure / 100.0; // mb
        let qs = sea_humidity(sst, p, sss);
        let q = air_humidity(t, p, air.relative_humidity);
        let lhvap = latent_heat(sst);
        let rhoa = p * 100.0 / (RGAS * (t + TDK) * (1.0 + 0.61 * q));
        let visa = 1.326e-5 * (1.0 + 6.542e-3 * t + 8.301e-6 * t * t - 4.84e-9 * t * t * t);

        // First guess
        let du = air.wind_speed;
        let dt = sst - t - 0.0098 * zt;
        let dq = qs - q;
        let ta = t + TDK;
        let mut ut = du.hypot(0.5);
        let u10 = ut * (10.0 / 1e-4_f64).ln() / (zu / 1e-4).ln();
        let mut usr = 0.035 * u10;
        let zo10 = 0.011 * usr * usr / g + 0.11 * visa / usr;
        let cd10 = (VON / (10.0 / zo10).ln()).powi(2);
        let ct10 = 0.00115 / cd10.sqrt();
        let zot10 = 10.0 / (VON / ct10).exp();
        let cd = (VON / (zu / zo10).ln()).powi(2);
        let ct = VON / (zt / zot10).ln();
        let cc = VON * ct / cd;
        let ribcu = -zu / zi / 0.004 / BETA.powi(3);
        let ribu = -g * zu / ta * (dt + 0.61 * ta * dq) / (ut * ut);
        // Stable with a thin Obukhov length: keep the first iteration's
        let thin = cc * ribu * (1.0 + 27.0 / 9.0 * ribu / cc) > 50.0;
        let zetu = if ribu < 0.0 {
            cc * ribu / (1.0 + ribu / ribcu)
        } else {
            cc * ribu * (1.0 + 27.0 / 9.0 * ribu / cc)
        };
        let l10 = zu / zetu;
        usr = ut * VON / ((zu / zo10).ln() - psi_u_40(zu / l10));
        let mut tsr = -dt * VON / ((zt / zot10).ln() - psi_t(zt / l10));
        let mut qsr = -dq * VON / ((zq / zot10).ln() - psi_t(zq / l10));
        let mut charn = charnock(u10);
        let mut gf = f64::INFINITY;
        let mut first = (usr, tsr, qsr);

        for i in 0..self.iterations {
            let zet = VON * g * zu / ta * (tsr + 0.61 * ta * qsr) / (usr * usr);
            let obukhov = zu / zet;
            let zo = charn * usr * usr / g + 0.11 * visa / usr;
            let rr = zo * usr / visa;
            let zoq = (5.8e-5 / rr.powf(0.72)).min(1.6e-4);
            let zot = zoq;
            usr = ut * VON / ((zu / zo).ln() - psi_u(zu / obukhov));
            qsr = -dq * VON / ((zq / zoq).ln() - psi_t(zq / obukhov));
            tsr = -dt * VON / ((zt / zot).ln() - psi_t(zt / obukhov));
            let tvsr = tsr + 0.61 * ta * qsr;
            let buoyancy = -g / ta * usr * tvsr;
            let ug = if buoyancy > 0.0 {
                BETA * (buoyancy * zi).cbrt()
            } else {
                0.2
            };
            ut = du.hypot(ug);
            gf = if du != 0.0 { ut / du } else { f64::INFINITY };
            if i == 0 {
                first = (usr, tsr, qsr);
            }
            charn = charnock(usr / VON / gf * (10.0 / zo).ln());
        }
        if thin {
            (usr, tsr, qsr) = first;
        }
        let latent = -rhoa * lhvap * usr * qsr;
        TurbulentFluxes {
            stress: rhoa * usr * usr / gf,
            sensible: -rhoa * CPA * usr * tsr,
            latent,
            evaporation: latent / lhvap,
            friction_velocity: usr,
        }
    }
}

/// Net longwave into the ocean (W/m²) at sea-surface temperature `sst`
/// (°C) under downwelling longwave `longwave` (W/m²).
pub fn net_longwave(sst: f64, longwave: f64) -> f64 {
    EMISSIVITY * (longwave - STEFAN_BOLTZMANN * (sst + TDK).powi(4))
}

/// Latent heat of vaporisation (J/kg) at `sst` (°C).
pub fn latent_heat(sst: f64) -> f64 {
    (2.501 - 0.00237 * sst) * 1e6
}

/// Normal gravity (m/s²) at latitude `lat` (degrees), Somigliana's formula
/// on WGS 84.
pub fn normal_gravity(lat: f64) -> f64 {
    let s2 = lat.to_radians().sin().powi(2);
    let e2: f64 = 8.1819190842622e-2_f64.powi(2);
    let (a, b) = (6_378_137.0, 6_356_752.314);
    let (gamma_p, gamma_e) = (9.832_184_937_9, 9.780_325_335_9);
    let k = b * gamma_p / (a * gamma_e) - 1.0;
    gamma_e * (1.0 + k * s2) / (1.0 - e2 * s2).sqrt()
}

/// Saturation vapour pressure (mb) at `t` (°C) and pressure `p` (mb),
/// Buck (1981) with the enhancement factor.
fn saturation_vapour_pressure(t: f64, p: f64) -> f64 {
    6.1121 * (17.502 * t / (240.97 + t)).exp() * (1.0007 + 3.46e-6 * p)
}

/// Saturation specific humidity (kg/kg) over sea water at `t` (°C),
/// salinity `s` and pressure `p` (mb).
fn sea_humidity(t: f64, p: f64, s: f64) -> f64 {
    let es = (1.0 - 0.02 * s / 35.0) * saturation_vapour_pressure(t, p);
    0.622 * es / (p - 0.378 * es)
}

/// Specific humidity (kg/kg) of air at `t` (°C), pressure `p` (mb) and
/// relative humidity `rh` (fraction).
fn air_humidity(t: f64, p: f64, rh: f64) -> f64 {
    let em = rh * saturation_vapour_pressure(t, p);
    0.62197 * em / (p - 0.378 * em)
}

/// Charnock's parameter at the 10 m neutral wind `u` (m/s).
fn charnock(u: f64) -> f64 {
    A1 * u.min(UMAX) + A2
}

/// Free-convective profile function of Grachev et al. (2000).
fn psi_convective(x: f64) -> f64 {
    let sqrt3 = 3f64.sqrt();
    1.5 * ((x * x + x + 1.0) / 3.0).ln() - sqrt3 * ((2.0 * x + 1.0) / sqrt3).atan()
        + std::f64::consts::PI / sqrt3
}

/// The velocity profile function at `zet = z/L`, with the Kansas constants
/// `(γ_K, γ_C)` and the stable slope `a`.
fn psi_velocity(zet: f64, [gamma_k, gamma_c]: [f64; 2], a: f64) -> f64 {
    if zet >= 0.0 {
        let (b, c, d) = (0.75, 5.0, 0.35);
        -(a * zet + b * (zet - c / d) * (-(d * zet).min(50.0)).exp() + b * c / d)
    } else {
        let x = (1.0 - gamma_k * zet).powf(0.25);
        let kansas = 2.0 * ((1.0 + x) / 2.0).ln() + ((1.0 + x * x) / 2.0).ln() - 2.0 * x.atan()
            + std::f64::consts::FRAC_PI_2;
        let convective = psi_convective((1.0 - gamma_c * zet).cbrt());
        let f = zet * zet / (1.0 + zet * zet);
        (1.0 - f) * kansas + f * convective
    }
}

/// COARE 3.5's velocity profile function (`psiu_26`).
fn psi_u(zet: f64) -> f64 {
    psi_velocity(zet, [15.0, 10.15], 0.7)
}

/// The first guess's velocity profile function (`psiu_40`).
fn psi_u_40(zet: f64) -> f64 {
    psi_velocity(zet, [18.0, 10.0], 1.0)
}

/// COARE 3.5's scalar profile function (`psit_26`).
fn psi_t(zet: f64) -> f64 {
    if zet >= 0.0 {
        let (b, c, d) = (0.6667, 5.0, 0.35);
        -((1.0 + 2.0 / 3.0 * zet).powf(1.5)
            + b * (zet - c / d) * (-(d * zet).min(50.0)).exp()
            + b * c / d
            - 1.0)
    } else {
        let x = (1.0 - 15.0 * zet).sqrt();
        let kansas = 2.0 * ((1.0 + x) / 2.0).ln();
        let convective = psi_convective((1.0 - 34.15 * zet).cbrt());
        let f = zet * zet / (1.0 + zet * zet);
        (1.0 - f) * kansas + f * convective
    }
}

/// The optical water type of the shortwave absorption: Jerlov's classes as
/// two exponentials (Paulson & Simpson 1977),
///
/// ```text
/// I(d)/I(0) = R e^(−d/ζ₁) + (1 − R) e^(−d/ζ₂)
/// ```
///
/// the red part absorbed in the top metre or two, the blue-green part over
/// `ζ₂`. Open ocean is type I; coastal water with plankton and dissolved
/// matter (CDOM from the rivers along the Norwegian coast) absorbs higher,
/// type II–III.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum JerlovWater {
    /// Clear open ocean (R 0.58, ζ₁ 0.35 m, ζ₂ 23 m)
    I,
    /// R 0.62, ζ₁ 0.6 m, ζ₂ 20 m
    IA,
    /// R 0.67, ζ₁ 1.0 m, ζ₂ 17 m
    IB,
    /// Coastal shelf water (R 0.77, ζ₁ 1.5 m, ζ₂ 14 m)
    #[default]
    II,
    /// Turbid coastal water (R 0.78, ζ₁ 1.4 m, ζ₂ 7.9 m)
    III,
}

impl JerlovWater {
    /// `(R, ζ₁, ζ₂)`.
    pub fn parameters(self) -> (f64, f64, f64) {
        match self {
            Self::I => (0.58, 0.35, 23.0),
            Self::IA => (0.62, 0.6, 20.0),
            Self::IB => (0.67, 1.0, 17.0),
            Self::II => (0.77, 1.5, 14.0),
            Self::III => (0.78, 1.4, 7.9),
        }
    }

    /// The fraction of the net shortwave that reaches `depth` (m) below the
    /// surface.
    pub fn transmitted(self, depth: f64) -> f64 {
        let (r, zeta_1, zeta_2) = self.parameters();
        let depth = depth.max(0.0);
        r * (-depth / zeta_1).exp() + (1.0 - r) * (-depth / zeta_2).exp()
    }
}

impl std::str::FromStr for JerlovWater {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_uppercase().as_str() {
            "I" | "1" => Ok(Self::I),
            "IA" => Ok(Self::IA),
            "IB" => Ok(Self::IB),
            "II" | "2" => Ok(Self::II),
            "III" | "3" => Ok(Self::III),
            _ => Err(format!(
                "unknown Jerlov water type {s:?} (I, IA, IB, II, III)"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(wind, air T, RH, SST, p, SW↓, LW↓, [τ, H_s, H_l, upward LW, u*])` from
    /// NOAA's `pycoare.coare_35` 0.x (zu 10 m, zt = zq 2 m, lat 63.7°,
    /// zi 600 m, jcool 0, 10 iterations, S 35).
    #[allow(clippy::type_complexity)]
    const REFERENCE: [(f64, f64, f64, f64, f64, f64, f64, [f64; 5]); 7] = [
        // Moderate wind, the sea warmer than the air
        (
            8.0,
            10.0,
            0.85,
            12.0,
            101000.0,
            400.0,
            330.0,
            [
                9.8621320987e-02,
                2.6361951703e+01,
                6.7586872482e+01,
                4.3571457692e+01,
                2.8332401355e-01,
            ],
        ),
        // Warm air over a cold sea: stable
        (
            3.0,
            16.0,
            0.7,
            11.0,
            101500.0,
            600.0,
            340.0,
            [
                1.9741091519e-03,
                -5.7366197460e+00,
                2.4256339345e-01,
                2.8796931956e+01,
                4.0323590332e-02,
            ],
        ),
        // Strong wind
        (
            15.0,
            8.0,
            0.8,
            10.0,
            100000.0,
            50.0,
            320.0,
            [
                5.2256260312e-01,
                4.8865928521e+01,
                1.3214641602e+02,
                4.3175698318e+01,
                6.5156447559e-01,
            ],
        ),
        // Calm: free convection, the gustiness carries the fluxes
        (
            0.5,
            5.0,
            0.6,
            12.0,
            102000.0,
            0.0,
            300.0,
            [
                1.3547512522e-03,
                2.8123819391e+01,
                5.2015472316e+01,
                7.2671457692e+01,
                4.8118500218e-02,
            ],
        ),
        // Storm: Charnock's parameter capped
        (
            22.0,
            10.0,
            0.9,
            10.5,
            99000.0,
            100.0,
            350.0,
            [
                1.4743375729e+00,
                1.7858951114e+01,
                8.1153281360e+01,
                1.6579677057e+01,
                1.1030989557e+00,
            ],
        ),
        // Very stable: the first iteration's scales (z/L > 50)
        (
            1.0,
            20.0,
            0.9,
            8.0,
            101300.0,
            700.0,
            360.0,
            [
                8.7628176097e-07,
                -5.2446344811e-03,
                -7.0780681020e-03,
                -5.5083890380e+00,
                8.6510579451e-04,
            ],
        ),
        // Neutral and saturated
        (
            6.0,
            12.0,
            1.0,
            12.0,
            101300.0,
            200.0,
            350.0,
            [
                4.2797524264e-02,
                -1.8647326360e-01,
                -4.0809326084e+00,
                2.4171457692e+01,
                1.8652374323e-01,
            ],
        ),
    ];

    fn air(case: &(f64, f64, f64, f64, f64, f64, f64, [f64; 5])) -> AirState {
        let &(wind_speed, air_temperature, relative_humidity, _, pressure, sw, lw, _) = case;
        AirState {
            wind_speed,
            air_temperature,
            relative_humidity,
            pressure,
            shortwave: sw,
            longwave: lw,
            precipitation: 0.0,
        }
    }

    /// The port gives NOAA's reference implementation's fluxes to 1e-9
    /// (relative; 1e-12 W/m² absolute near zero).
    #[test]
    fn coare_matches_the_reference_implementation() {
        let coare = Coare35::at_latitude(63.7);
        for case in &REFERENCE {
            let sst = case.3;
            let [tau, hs, hl, rnl, usr] = case.7;
            let f = coare.turbulent(&air(case), sst, 35.0);
            for (name, got, want) in [
                ("τ", f.stress, tau),
                ("H_s", f.sensible, hs),
                ("H_l", f.latent, hl),
                ("u*", f.friction_velocity, usr),
                ("LW↑", -net_longwave(sst, case.6), rnl),
            ] {
                assert!(
                    (got - want).abs() <= 1e-9 * want.abs() + 1e-12,
                    "{name} = {got:e}, reference {want:e} in case {case:?}"
                );
            }
        }
    }

    /// The budget: shortwave less the albedo, the non-solar flux the net
    /// longwave less the turbulent fluxes, E − P from the latent heat.
    #[test]
    fn the_budget_adds_the_fluxes_into_the_ocean() {
        let coare = Coare35::at_latitude(63.7);
        let mut state = air(&REFERENCE[0]);
        state.precipitation = 2.0 / 3600.0;
        let f = coare.turbulent(&state, 12.0, 33.0);
        let b = coare.budget(&state, 12.0, 33.0);
        assert!((b.shortwave - 0.945 * 400.0).abs() < 1e-12);
        let nonsolar = net_longwave(12.0, 330.0) - f.sensible - f.latent;
        assert!((b.nonsolar - nonsolar).abs() < 1e-12);
        assert!(
            b.nonsolar < -100.0,
            "a warm sea in wind loses heat: {}",
            b.nonsolar
        );
        let evaporation = f.latent / latent_heat(12.0);
        assert!((b.freshwater - (evaporation - state.precipitation)).abs() < 1e-18);
        // Night: no shortwave, and none from a slightly negative analysis
        state.shortwave = -0.01;
        assert_eq!(coare.budget(&state, 12.0, 33.0).shortwave, 0.0);
    }

    /// Without differences in temperature and humidity there is no sensible
    /// or latent flux; a fresher sea evaporates more (higher saturation
    /// humidity), a warmer one too.
    #[test]
    fn fluxes_follow_the_air_sea_differences() {
        let coare = Coare35::new().with_heights(10.0, 10.0, 10.0);
        // The air at the sea's temperature less the lapse over 10 m
        let sst = 12.0;
        let t = sst - 0.0098 * 10.0;
        let p = 1013.0;
        let q = sea_humidity(sst, p, 35.0);
        let rh = q * p / (0.62197 + 0.378 * q) / saturation_vapour_pressure(t, p);
        let state = AirState {
            wind_speed: 8.0,
            air_temperature: t,
            relative_humidity: rh,
            pressure: p * 100.0,
            shortwave: 0.0,
            longwave: 300.0,
            precipitation: 0.0,
        };
        let f = coare.turbulent(&state, sst, 35.0);
        assert!(f.sensible.abs() < 1e-12 && f.latent.abs() < 1e-9, "{f:?}");
        // 5 psu fresher: 0.3 % more vapour pressure at the surface, 0.7 W/m²
        let fresh = coare.turbulent(&state, sst, 30.0);
        let warm = coare.turbulent(&state, sst + 1.0, 35.0);
        assert!(
            fresh.latent > 0.5 && warm.latent > fresh.latent,
            "{fresh:?} {warm:?}"
        );
        assert!(warm.sensible > 1.0);
    }

    /// Normal gravity at the equator, the pole and Frøya.
    #[test]
    fn normal_gravity_on_wgs84() {
        assert!((normal_gravity(0.0) - 9.780_325_335_9).abs() < 1e-10);
        assert!((normal_gravity(90.0) - 9.832_184_937_9).abs() < 1e-9);
        assert!((normal_gravity(63.7) - 9.8215).abs() < 1e-3);
    }

    /// Each Jerlov profile starts at 1, decays monotonically, keeps more
    /// light at depth for clearer water, and parses from its name.
    #[test]
    fn jerlov_profiles() {
        let types = [
            JerlovWater::I,
            JerlovWater::IA,
            JerlovWater::IB,
            JerlovWater::II,
            JerlovWater::III,
        ];
        for water in types {
            assert!((water.transmitted(0.0) - 1.0).abs() < 1e-15);
            assert_eq!(water.transmitted(-1.0), 1.0);
            let mut last = 1.0;
            for d in 1..100 {
                let t = water.transmitted(d as f64);
                assert!(t < last && t > 0.0);
                last = t;
            }
        }
        for pair in types.windows(2) {
            assert!(
                pair[0].transmitted(5.0) > pair[1].transmitted(5.0),
                "{pair:?}"
            );
        }
        // Type I: 0.42 e^(−10/23) of the light at 10 m
        let expected = 0.58 * (-10.0 / 0.35_f64).exp() + 0.42 * (-10.0 / 23.0_f64).exp();
        assert!((JerlovWater::I.transmitted(10.0) - expected).abs() < 1e-15);
        assert_eq!("ii".parse::<JerlovWater>(), Ok(JerlovWater::II));
        assert!("IV".parse::<JerlovWater>().is_err());
    }
}
