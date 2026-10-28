//! Tidal astronomy: Doodson/Schureman equilibrium arguments and nodal corrections.
//!
//! Harmonic constituents obtained from analysis or a catalogue (amplitude `H`,
//! Greenwich phase lag `G`) predict elevation as
//!
//! ```text
//! η(t) = Σᵢ Hᵢ cos(ωᵢ t − Gᵢ)
//! ```
//!
//! but this ignores two astronomical effects that a real prediction needs:
//!
//! 1. **Equilibrium argument `V₀`** — the astronomical phase of the constituent
//!    at the prediction epoch. Without it the tide is at the wrong phase (a fixed
//!    per-constituent offset), because `ω t` alone measures phase from an
//!    arbitrary origin rather than from the true lunar/solar configuration.
//! 2. **Nodal corrections `f`, `u`** — the 18.61-year regression of the lunar
//!    node modulates each constituent's amplitude by a slowly varying factor `f`
//!    and its phase by `u`. For the diurnal constituents this modulation is large:
//!    `f(K1)` ranges over ≈0.88–1.11 and `f(O1)` over ≈0.81–1.18, so omitting it
//!    biases K1/O1 amplitudes by 11–19 %.
//!
//! With both corrections the prediction becomes the standard Schureman form
//!
//! ```text
//! η(t) = Σᵢ fᵢ Hᵢ cos(ωᵢ t + (V₀ + u)ᵢ − Gᵢ)
//! ```
//!
//! where `t` is elapsed time (seconds) from the epoch at which `V₀`, `f`, `u`
//! are evaluated.
//!
//! # Formulation
//!
//! The astronomical mean longitudes are Meeus' fundamental arguments referred to
//! J2000.0 (Meeus, *Astronomical Algorithms* 2nd ed., ch. 22 & 47):
//!
//! - `s`  — mean longitude of the Moon (`L′`)
//! - `h`  — mean longitude of the Sun (`L₀`)
//! - `p`  — mean longitude of lunar perigee (`s − M′`)
//! - `N`  — longitude of the Moon's ascending node (`Ω`)
//! - `p₁` — mean longitude of solar perigee/perihelion (`h − M`)
//!
//! and the mean lunar time `τ = 15° · H_UT + h − s` (`H_UT` = UT hours of day).
//! Each constituent's equilibrium argument is the Doodson combination
//! `V₀ = c_τ τ + c_s s + c_h h + c_p p + c_p₁ p₁ + offset`, and its nodal factors
//! `f`, `u` are the Schureman (1958) closed-form approximations in the node
//! longitude `N`. The constituent's angular speed is the time derivative of the
//! same combination, `ω = c_τ τ′ + c_s s′ + …` ([`angular_speed`],
//! [`constituent_period`]), so `ω t` and `V₀` advance consistently.
//!
//! # References
//!
//! - Doodson (1921), *The harmonic development of the tide-generating potential*.
//! - Schureman (1958), *Manual of Harmonic Analysis and Prediction of Tides*,
//!   US C&GS Special Publication 98.
//! - Meeus (1998), *Astronomical Algorithms*, 2nd ed.
//! - Pugh & Woodworth (2014), *Sea-Level Science*, ch. 3.

use std::f64::consts::PI;

const DEG_TO_RAD: f64 = PI / 180.0;

// Secular (linear) rates of Meeus' fundamental arguments, degrees per Julian
// century. Shared by `AstronomicalArguments::at_julian_date` and the constituent
// speeds so that ω t advances at the same rate as V₀.
/// Moon mean longitude `L′` (Meeus 47.1).
const S_PER_CENTURY: f64 = 481_267.881_234_21;
/// Sun mean longitude `L₀` (Meeus 25.2).
const H_PER_CENTURY: f64 = 36_000.769_827_79;
/// Moon mean anomaly `M′` (Meeus 47.4).
const MP_PER_CENTURY: f64 = 477_198.867_505_5;
/// Sun mean anomaly `M` (Meeus 47.3).
const M_PER_CENTURY: f64 = 35_999.050_290_9;

/// Mean solar hours per Julian century.
const HOURS_PER_CENTURY: f64 = 36_525.0 * 24.0;

/// Wrap an angle in degrees to `[0, 360)`.
#[inline]
fn wrap_deg(x: f64) -> f64 {
    let r = x % 360.0;
    if r < 0.0 { r + 360.0 } else { r }
}

/// Astronomical mean longitudes (degrees, wrapped to `[0, 360)`) and the mean
/// lunar time at a given instant.
///
/// Construct with [`AstronomicalArguments::at_julian_date`] or
/// [`AstronomicalArguments::at_datetime`]. All members are in degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AstronomicalArguments {
    /// Mean lunar time `τ = 15° · H_UT + h − s` (degrees).
    pub tau: f64,
    /// Mean longitude of the Moon, `s` (degrees).
    pub s: f64,
    /// Mean longitude of the Sun, `h` (degrees).
    pub h: f64,
    /// Mean longitude of lunar perigee, `p` (degrees).
    pub p: f64,
    /// Longitude of the Moon's ascending node, `N` (degrees).
    pub n: f64,
    /// Mean longitude of solar perigee, `p₁` (degrees).
    pub p1: f64,
}

impl AstronomicalArguments {
    /// Compute the mean longitudes at a Julian Date (UTC).
    ///
    /// UTC is used directly in place of Terrestrial Time; the resulting ΔT error
    /// (tens of seconds) is negligible for the slowly varying nodal/equilibrium
    /// terms this module supplies.
    pub fn at_julian_date(jd_utc: f64) -> Self {
        // Julian centuries from J2000.0 (JD 2451545.0 = 2000-01-01 12:00).
        let t = (jd_utc - 2_451_545.0) / 36_525.0;
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;

        // Meeus (47.1): Moon mean longitude L′.
        let s = 218.316_447_7 + S_PER_CENTURY * t - 0.001_578_6 * t2 + t3 / 538_841.0
            - t4 / 65_194_000.0;
        // Meeus (25.2): Sun mean longitude L₀.
        let h = 280.466_456_7 + H_PER_CENTURY * t + 0.000_303_2 * t2;
        // Meeus (47.4): Moon mean anomaly M′.
        let mp = 134.963_396_4 + MP_PER_CENTURY * t + 0.008_741_4 * t2 + t3 / 69_699.0
            - t4 / 14_712_000.0;
        // Meeus (47.3): Sun mean anomaly M.
        let m = 357.529_109_2 + M_PER_CENTURY * t - 0.000_153_6 * t2 + t3 / 24_490_000.0;
        // Meeus (47.7): longitude of the ascending node Ω.
        let node = 125.044_547_9 - 1_934.136_289_1 * t + 0.002_075_4 * t2 + t3 / 467_441.0
            - t4 / 60_616_000.0;

        // Lunar perigee p = s − M′; solar perigee p₁ = h − M.
        let p = s - mp;
        let p1 = h - m;

        // Mean lunar time τ = 15°·H_UT + h − s, with H_UT the UT hours of day.
        // JD days start at noon, so (jd + 0.5) mod 1 is the fraction past midnight.
        let day_fraction = (jd_utc + 0.5).rem_euclid(1.0);
        let h_ut = day_fraction * 24.0;
        let tau = 15.0 * h_ut + h - s;

        Self {
            tau: wrap_deg(tau),
            s: wrap_deg(s),
            h: wrap_deg(h),
            p: wrap_deg(p),
            n: wrap_deg(node),
            p1: wrap_deg(p1),
        }
    }

    /// Compute the mean longitudes at a Gregorian UTC calendar instant.
    ///
    /// `month` is 1–12, `day` is 1–31, `hour` 0–23, `minute` 0–59; `second` may
    /// be fractional.
    pub fn at_datetime(
        year: i32,
        month: u32,
        day: u32,
        hour: u32,
        minute: u32,
        second: f64,
    ) -> Self {
        Self::at_julian_date(julian_date(year, month, day, hour, minute, second))
    }
}

/// Julian Date (UTC) of a Gregorian calendar instant.
///
/// Meeus (7.1), valid for the Gregorian calendar. `month` is 1–12; `hour`,
/// `minute`, and `second` give the time of day (UTC).
pub fn julian_date(year: i32, month: u32, day: u32, hour: u32, minute: u32, second: f64) -> f64 {
    let (y, m) = if month <= 2 {
        (year - 1, month as i32 + 12)
    } else {
        (year, month as i32)
    };

    let a = (y as f64 / 100.0).floor();
    let b = 2.0 - a + (a / 4.0).floor();

    let day_fraction = (hour as f64 + minute as f64 / 60.0 + second / 3600.0) / 24.0;

    (365.25 * (y as f64 + 4716.0)).floor() + (30.6001 * (m as f64 + 1.0)).floor() + day as f64 + b
        - 1524.5
        + day_fraction
}

/// Nodal / equilibrium correction for a single constituent at a fixed epoch.
///
/// Apply as `η = f · H · cos(ω t + phase_offset − G)` where
/// `phase_offset = (V₀ + u)` and `t` is elapsed time from the epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodalCorrection {
    /// Nodal amplitude factor `f` (dimensionless, ≈1).
    pub f: f64,
    /// Nodal phase correction `u` (degrees).
    pub u_deg: f64,
    /// Equilibrium argument `V₀` at the epoch (degrees).
    pub v0_deg: f64,
}

impl NodalCorrection {
    /// The identity correction (`f = 1`, `u = V₀ = 0`).
    pub const IDENTITY: Self = Self {
        f: 1.0,
        u_deg: 0.0,
        v0_deg: 0.0,
    };

    /// Total phase to add to `ω t`, i.e. `(V₀ + u)` in radians.
    #[inline]
    pub fn phase_offset_rad(&self) -> f64 {
        (self.v0_deg + self.u_deg) * DEG_TO_RAD
    }
}

/// Nodal correction of a constituent through a run: `V₀` at the epoch
/// (`t = 0`) and the nodal terms linear in the time `t` (s) since it,
///
/// ```text
/// f(t) = f₀ + ḟ t,    u(t) = u₀ + u̇ t,
/// ```
///
/// the first-order expansion of the 18.61-year modulation about the epoch.
/// Its error is at most ½ max|f̈|, ½ max|ü| times t²: for M2 ≈ 0.001° in `u`
/// after a month and 0.12° after a year, for K1 0.004° and 0.5° (`f`: 0.2 %
/// and 0.7 % after a year). Unlike `f`, `u` held at the middle of a run
/// (off by `u̇ T/2` at its ends, 1.5° for K1 over a year), the correction at
/// `t` does not depend on where the run ends, so a run resumed or branched to
/// another end is forced as the run it came from. Built by
/// [`ModelClock::nodal_drift`](crate::time::ModelClock::nodal_drift).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodalDrift {
    /// `f₀`, `u₀` and `V₀` at the epoch.
    pub at_epoch: NodalCorrection,
    /// `ḟ` (per second).
    pub f_rate: f64,
    /// `u̇` (degrees per second).
    pub u_rate_deg: f64,
}

impl NodalDrift {
    /// No correction at any time.
    pub const IDENTITY: Self = Self {
        at_epoch: NodalCorrection::IDENTITY,
        f_rate: 0.0,
        u_rate_deg: 0.0,
    };

    /// `f`, `u` at `t` and `V₀` at the epoch.
    #[inline]
    pub fn at(&self, t: f64) -> NodalCorrection {
        NodalCorrection {
            f: self.at_epoch.f + self.f_rate * t,
            u_deg: self.at_epoch.u_deg + self.u_rate_deg * t,
            v0_deg: self.at_epoch.v0_deg,
        }
    }

    /// `u̇` in radians per second: the drift of the nodal phase, which adds
    /// to the constituent's angular speed `ω`.
    #[inline]
    pub fn u_rate_rad(&self) -> f64 {
        self.u_rate_deg * DEG_TO_RAD
    }

    /// `ḟ / f₀` (per second): the relative drift of the amplitude.
    #[inline]
    pub fn relative_f_rate(&self) -> f64 {
        self.f_rate / self.at_epoch.f
    }
}

/// Doodson coefficients of a constituent's equilibrium argument
/// `V₀ = c_τ τ + c_s s + c_h h + c_p p + c_p₁ p₁ + offset` (degrees).
///
/// The lunar-node coefficient is zero for every constituent handled here: the
/// node's effect is carried entirely by the nodal factors `f`, `u`.
#[derive(Clone, Copy)]
struct Doodson {
    c_tau: f64,
    c_s: f64,
    c_h: f64,
    c_p: f64,
    c_p1: f64,
    offset: f64,
}

impl Doodson {
    #[inline]
    fn v0(&self, a: &AstronomicalArguments) -> f64 {
        wrap_deg(
            self.c_tau * a.tau
                + self.c_s * a.s
                + self.c_h * a.h
                + self.c_p * a.p
                + self.c_p1 * a.p1
                + self.offset,
        )
    }

    /// Angular speed `dV₀/dt` in degrees per mean solar hour, from the secular
    /// rates of the mean longitudes (`τ = 15°·H_UT + h − s`, so `τ′ = 15 + h′ − s′`).
    #[inline]
    fn speed_deg_per_hour(&self) -> f64 {
        let s = S_PER_CENTURY / HOURS_PER_CENTURY;
        let h = H_PER_CENTURY / HOURS_PER_CENTURY;
        let p = (S_PER_CENTURY - MP_PER_CENTURY) / HOURS_PER_CENTURY;
        let p1 = (H_PER_CENTURY - M_PER_CENTURY) / HOURS_PER_CENTURY;
        let tau = 15.0 + h - s;
        self.c_tau * tau + self.c_s * s + self.c_h * h + self.c_p * p + self.c_p1 * p1
    }
}

/// Which family of Schureman nodal `f`/`u` formulas a constituent uses.
#[derive(Clone, Copy)]
enum NodalKind {
    /// No node dependence (solar constituents, long-period Ssa): `f = 1`, `u = 0`.
    None,
    /// M2-type semidiurnal lunar (also N2). Base factor for compounds.
    M2,
    K1,
    O1,
    K2,
    Mf,
    Mm,
    /// Product of `n` M2 base factors: `f = f(M2)ⁿ`, `u = n · u(M2)` (M4, M6, MN4).
    M2Power(i32),
}

/// Static description of a constituent: its Doodson argument and nodal family.
struct ConstituentDef {
    doodson: Doodson,
    nodal: NodalKind,
}

/// Look up the definition of a constituent by name (case-insensitive).
///
/// Returns `None` for names not in the supported set (see [`nodal_correction`]).
fn constituent_def(name: &str) -> Option<ConstituentDef> {
    // Convenience for building a Doodson argument.
    const fn d(c_tau: f64, c_s: f64, c_h: f64, c_p: f64, c_p1: f64, offset: f64) -> Doodson {
        Doodson {
            c_tau,
            c_s,
            c_h,
            c_p,
            c_p1,
            offset,
        }
    }

    let def = match name.to_uppercase().as_str() {
        // --- Semidiurnal ---
        "M2" => ConstituentDef {
            doodson: d(2.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            nodal: NodalKind::M2,
        },
        "S2" => ConstituentDef {
            doodson: d(2.0, 2.0, -2.0, 0.0, 0.0, 0.0),
            nodal: NodalKind::None,
        },
        "N2" => ConstituentDef {
            doodson: d(2.0, -1.0, 0.0, 1.0, 0.0, 0.0),
            nodal: NodalKind::M2,
        },
        "K2" => ConstituentDef {
            doodson: d(2.0, 2.0, 0.0, 0.0, 0.0, 0.0),
            nodal: NodalKind::K2,
        },
        // --- Diurnal (±90° offsets are Schureman's convention) ---
        "K1" => ConstituentDef {
            doodson: d(1.0, 1.0, 0.0, 0.0, 0.0, 90.0),
            nodal: NodalKind::K1,
        },
        "O1" => ConstituentDef {
            doodson: d(1.0, -1.0, 0.0, 0.0, 0.0, -90.0),
            nodal: NodalKind::O1,
        },
        "P1" => ConstituentDef {
            doodson: d(1.0, 1.0, -2.0, 0.0, 0.0, -90.0),
            nodal: NodalKind::None,
        },
        "Q1" => ConstituentDef {
            doodson: d(1.0, -2.0, 0.0, 1.0, 0.0, -90.0),
            nodal: NodalKind::O1,
        },
        // --- Shallow-water / overtides ---
        "M4" => ConstituentDef {
            doodson: d(4.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            nodal: NodalKind::M2Power(2),
        },
        "MS4" => ConstituentDef {
            doodson: d(4.0, 2.0, -2.0, 0.0, 0.0, 0.0),
            nodal: NodalKind::M2, // f(M2)·f(S2) = f(M2)
        },
        "MN4" => ConstituentDef {
            doodson: d(4.0, -1.0, 0.0, 1.0, 0.0, 0.0),
            nodal: NodalKind::M2Power(2),
        },
        "M6" => ConstituentDef {
            doodson: d(6.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            nodal: NodalKind::M2Power(3),
        },
        // --- Long period ---
        "MF" => ConstituentDef {
            doodson: d(0.0, 2.0, 0.0, 0.0, 0.0, 0.0),
            nodal: NodalKind::Mf,
        },
        "MM" => ConstituentDef {
            doodson: d(0.0, 1.0, 0.0, -1.0, 0.0, 0.0),
            nodal: NodalKind::Mm,
        },
        "SSA" => ConstituentDef {
            doodson: d(0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
            nodal: NodalKind::None,
        },
        _ => return None,
    };
    Some(def)
}

/// Names of the supported constituents, in their conventional spelling.
pub const CONSTITUENT_NAMES: [&str; 15] = [
    "M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "M4", "MS4", "MN4", "M6", "Mf", "Mm", "Ssa",
];

/// The conventional (`'static`) spelling of a supported constituent name,
/// matched case-insensitively (`"mf"` → `"Mf"`); `None` if unsupported.
pub fn canonical_name(name: &str) -> Option<&'static str> {
    CONSTITUENT_NAMES
        .iter()
        .find(|c| c.eq_ignore_ascii_case(name.trim()))
        .copied()
}

/// Schureman `f`/`u` for the M2 species (also N2), given node longitude `N` (rad).
fn nodal_m2(n_rad: f64) -> (f64, f64) {
    let f = 1.0004 - 0.0373 * n_rad.cos() + 0.0002 * (2.0 * n_rad).cos();
    let u = -2.14 * n_rad.sin();
    (f, u)
}

/// Compute the nodal factor `f` and phase `u` (degrees) for a nodal family.
fn nodal_factors(kind: NodalKind, n_rad: f64) -> (f64, f64) {
    let (c, c2, c3) = (n_rad.cos(), (2.0 * n_rad).cos(), (3.0 * n_rad).cos());
    let (s, s2, s3) = (n_rad.sin(), (2.0 * n_rad).sin(), (3.0 * n_rad).sin());
    match kind {
        NodalKind::None => (1.0, 0.0),
        NodalKind::M2 => nodal_m2(n_rad),
        NodalKind::K1 => {
            let f = 1.0060 + 0.1150 * c - 0.0088 * c2 + 0.0006 * c3;
            let u = -8.86 * s + 0.68 * s2 - 0.07 * s3;
            (f, u)
        }
        NodalKind::O1 => {
            let f = 1.0089 + 0.1871 * c - 0.0147 * c2 + 0.0014 * c3;
            let u = 10.80 * s - 1.34 * s2 + 0.19 * s3;
            (f, u)
        }
        NodalKind::K2 => {
            let f = 1.0241 + 0.2863 * c + 0.0083 * c2 - 0.0015 * c3;
            let u = -17.74 * s + 0.68 * s2 - 0.04 * s3;
            (f, u)
        }
        NodalKind::Mf => {
            let f = 1.0429 + 0.4135 * c - 0.004 * c2;
            let u = -23.7 * s + 2.7 * s2 - 0.4 * s3;
            (f, u)
        }
        NodalKind::Mm => {
            let f = 1.0 - 0.1300 * c + 0.0013 * c2;
            (f, 0.0)
        }
        NodalKind::M2Power(k) => {
            let (fm, um) = nodal_m2(n_rad);
            (fm.powi(k), k as f64 * um)
        }
    }
}

/// Compute the nodal correction `(f, u, V₀)` for a named constituent at the
/// given astronomical arguments.
///
/// Names are case-insensitive. Supported constituents:
/// M2, S2, N2, K2, K1, O1, P1, Q1, M4, MS4, MN4, M6, Mf, Mm, Ssa.
/// Returns `None` for any other name.
///
/// # Example
///
/// ```
/// use dg_rs::tides::{AstronomicalArguments, nodal_correction};
///
/// let astro = AstronomicalArguments::at_datetime(2024, 1, 1, 0, 0, 0.0);
/// let m2 = nodal_correction("M2", &astro).unwrap();
/// // f is within a few percent of unity for M2.
/// assert!((m2.f - 1.0).abs() < 0.05);
/// ```
pub fn nodal_correction(name: &str, astro: &AstronomicalArguments) -> Option<NodalCorrection> {
    let def = constituent_def(name)?;
    let (f, u_deg) = nodal_factors(def.nodal, astro.n * DEG_TO_RAD);
    Some(NodalCorrection {
        f,
        u_deg,
        v0_deg: def.doodson.v0(astro),
    })
}

/// Angular speed of a named constituent in degrees per mean solar hour.
///
/// Derived from the constituent's Doodson argument and the secular rates of the
/// mean longitudes, i.e. the rate at which its `V₀` advances; agrees with the
/// tabulated speeds of Schureman (1958) to < 10⁻⁷ °/h (M2 28.9841042°/h,
/// K1 15.0410686°/h, …). Names and support as for [`nodal_correction`].
pub fn angular_speed(name: &str) -> Option<f64> {
    constituent_def(name).map(|def| def.doodson.speed_deg_per_hour())
}

/// Period of a named constituent in seconds, `360° / speed`.
///
/// See [`angular_speed`]. Use this — never a rounded period — for anything
/// that accumulates phase over a record: a 12.42 h M2 drifts ≈ 6° per year.
pub fn constituent_period(name: &str) -> Option<f64> {
    angular_speed(name).map(|speed| 360.0 / speed * 3600.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY_SECONDS: f64 = 86_400.0;

    #[test]
    fn canonical_names_cover_the_catalogue() {
        for name in CONSTITUENT_NAMES {
            assert!(constituent_def(name).is_some(), "{name}");
            assert_eq!(canonical_name(&name.to_lowercase()), Some(name));
        }
        assert_eq!(canonical_name("XX9"), None);
    }

    /// Mean longitudes at J2000.0 are the constant terms of Meeus' arguments.
    /// These are textbook values (Meeus ch. 22, 47) and validate the polynomials.
    #[test]
    fn mean_longitudes_at_j2000() {
        // J2000.0 = 2000-01-01 12:00 UTC = JD 2451545.0.
        let a = AstronomicalArguments::at_julian_date(2_451_545.0);
        assert!((a.s - 218.316_447_7).abs() < 1e-4, "s = {}", a.s);
        assert!((a.h - 280.466_456_7).abs() < 1e-4, "h = {}", a.h);
        // p = s − M′ = 218.3164 − 134.9634 = 83.3531 (mod 360).
        assert!((a.p - 83.353_051_3).abs() < 1e-3, "p = {}", a.p);
        // N = Ω = 125.0445.
        assert!((a.n - 125.044_547_9).abs() < 1e-4, "N = {}", a.n);
        // p₁ = h − M = 280.4665 − 357.5291 = −77.0627 → 282.9373 (mod 360).
        assert!((a.p1 - 282.937_347_5).abs() < 1e-3, "p1 = {}", a.p1);
    }

    /// At J2000.0 (noon UT) the mean lunar time is τ = 15·12 + h − s.
    #[test]
    fn mean_lunar_time_at_j2000_noon() {
        let a = AstronomicalArguments::at_julian_date(2_451_545.0);
        let expected = wrap_deg(15.0 * 12.0 + 280.466_456_7 - 218.316_447_7);
        assert!((a.tau - expected).abs() < 1e-3, "tau = {}", a.tau);
    }

    /// The Julian-date conversion matches Meeus' worked examples.
    #[test]
    fn julian_date_reference_values() {
        // Meeus example 7.a: 1957-10-04.81 UT = JD 2436116.31.
        let jd = julian_date(1957, 10, 4, 19, 26, 24.0);
        assert!((jd - 2_436_116.31).abs() < 1e-2, "jd = {jd}");
        // J2000.0.
        let jd2000 = julian_date(2000, 1, 1, 12, 0, 0.0);
        assert!((jd2000 - 2_451_545.0).abs() < 1e-6, "jd2000 = {jd2000}");
    }

    /// The time-derivative of each constituent's equilibrium argument must equal
    /// its tabulated angular speed ω = 360°/period. This simultaneously validates
    /// the Doodson coefficients and the mean-longitude rates. Finite-difference
    /// V₀ across one day and recover the period.
    #[test]
    fn equilibrium_argument_rate_matches_period() {
        // (name, period in hours)
        let cases = [
            ("M2", 12.420_601_2),
            ("S2", 12.0),
            ("N2", 12.658_348_2),
            ("K2", 11.967_234_8),
            ("K1", 23.934_469_7),
            ("O1", 25.819_341_7),
            ("P1", 24.065_890_2),
            ("Q1", 26.868_356_7),
            ("M4", 6.210_300_6),
            ("M6", 4.140_200_4),
        ];
        let jd0 = 2_451_545.0;
        let dt_days = 1.0;
        for (name, period_hours) in cases {
            let a0 = AstronomicalArguments::at_julian_date(jd0);
            let a1 = AstronomicalArguments::at_julian_date(jd0 + dt_days);
            let v0 = nodal_correction(name, &a0).unwrap().v0_deg;
            let v1 = nodal_correction(name, &a1).unwrap().v0_deg;
            // Unwrap the phase advance over one day (many full turns): use the
            // known rate to pick the right multiple of 360°.
            let expected_rate_deg_per_day = 360.0 / period_hours * 24.0;
            let raw = v1 - v0;
            let turns = ((expected_rate_deg_per_day - raw) / 360.0).round();
            let rate = raw + 360.0 * turns;
            let recovered_period = 360.0 / rate * 24.0;
            assert!(
                (recovered_period - period_hours).abs() < 1e-4,
                "{name}: recovered period {recovered_period} h vs {period_hours} h",
            );
        }
    }

    /// Constituent speeds match the tabulated Doodson/Schureman (1958) speeds
    /// (as published by NOAA CO-OPS), and periods are `360°/speed`.
    #[test]
    fn angular_speeds_match_tabulated_values() {
        // (name, tabulated speed in °/h)
        let cases = [
            ("M2", 28.984_104_2),
            ("S2", 30.0),
            ("N2", 28.439_729_5),
            ("K2", 30.082_137_3),
            ("K1", 15.041_068_6),
            ("O1", 13.943_035_6),
            ("P1", 14.958_931_4),
            ("Q1", 13.398_660_9),
            ("M4", 57.968_208_4),
            ("MS4", 58.984_104_2),
            ("MN4", 57.423_833_7),
            ("M6", 86.952_312_7),
            ("Mf", 1.098_033_1),
            ("Mm", 0.544_374_7),
            ("Ssa", 0.082_137_3),
        ];
        for (name, tabulated) in cases {
            let speed = angular_speed(name).unwrap();
            assert!(
                (speed - tabulated).abs() < 1e-6,
                "{name}: {speed} °/h vs tabulated {tabulated} °/h"
            );
            let period = constituent_period(name).unwrap();
            assert!((period * speed - 360.0 * 3600.0).abs() < 1e-6);
        }
        // S2 is exactly two solar cycles per day.
        assert!((constituent_period("S2").unwrap() - 12.0 * 3600.0).abs() < 1e-9);
        assert!(angular_speed("XYZ").is_none());
    }

    /// S2 is pure solar semidiurnal: V₀ = 30°·H_UT, so it vanishes at UT midnight.
    #[test]
    fn s2_equilibrium_argument_is_solar_time() {
        // 2020-06-15 00:00 UT.
        let a = AstronomicalArguments::at_datetime(2020, 6, 15, 0, 0, 0.0);
        let v0 = nodal_correction("S2", &a).unwrap().v0_deg;
        assert!(v0 < 1e-6 || (v0 - 360.0).abs() < 1e-6, "S2 V0 = {v0}");

        // At 06:00 UT, V₀ = 30·6 = 180°.
        let a6 = AstronomicalArguments::at_datetime(2020, 6, 15, 6, 0, 0.0);
        let v6 = nodal_correction("S2", &a6).unwrap().v0_deg;
        assert!((v6 - 180.0).abs() < 1e-6, "S2 V0 at 06:00 = {v6}");
    }

    /// Solar constituents have no lunar-node modulation.
    #[test]
    fn solar_constituents_have_unit_nodal_factor() {
        let a = AstronomicalArguments::at_datetime(2015, 3, 21, 0, 0, 0.0);
        for name in ["S2", "P1", "Ssa"] {
            let c = nodal_correction(name, &a).unwrap();
            assert!((c.f - 1.0).abs() < 1e-12, "{name} f = {}", c.f);
            assert!(c.u_deg.abs() < 1e-12, "{name} u = {}", c.u_deg);
        }
    }

    /// The M2 nodal factor reaches its known extremes at the node passages
    /// N = 0° (f ≈ 0.963) and N = 180° (f ≈ 1.038), with u = 0 at both.
    #[test]
    fn m2_nodal_factor_extremes() {
        let (f0, u0) = nodal_m2(0.0);
        assert!((f0 - 0.9633).abs() < 1e-3, "f(M2, N=0) = {f0}");
        assert!(u0.abs() < 1e-12, "u(M2, N=0) = {u0}");

        let (f180, u180) = nodal_m2(PI);
        assert!((f180 - 1.0379).abs() < 1e-3, "f(M2, N=180) = {f180}");
        assert!(u180.abs() < 1e-12, "u(M2, N=180) = {u180}");
    }

    /// K1 and O1 carry the large diurnal nodal modulation that, if omitted,
    /// biases their amplitudes by 11–19 %. Check the known ranges.
    #[test]
    fn diurnal_nodal_factor_ranges() {
        // K1: ≈0.882 at N=180°, ≈1.113 at N=0°.
        let (f_k1_max, _) = nodal_factors(NodalKind::K1, 0.0);
        let (f_k1_min, _) = nodal_factors(NodalKind::K1, PI);
        assert!((f_k1_max - 1.1128).abs() < 1e-3, "f(K1) max = {f_k1_max}");
        assert!((f_k1_min - 0.8816).abs() < 1e-3, "f(K1) min = {f_k1_min}");
        // The peak-to-peak modulation exceeds 11 %.
        assert!(f_k1_max - f_k1_min > 0.11);

        // O1: ≈0.806 at N=180°, ≈1.183 at N=0°.
        let (f_o1_max, _) = nodal_factors(NodalKind::O1, 0.0);
        let (f_o1_min, _) = nodal_factors(NodalKind::O1, PI);
        assert!((f_o1_max - 1.1827).abs() < 1e-3, "f(O1) max = {f_o1_max}");
        assert!((f_o1_min - 0.8057).abs() < 1e-3, "f(O1) min = {f_o1_min}");
        assert!(f_o1_max - f_o1_min > 0.18);
    }

    /// Compound constituents' nodal factors are products of their parents'.
    #[test]
    fn compound_nodal_factors_are_products() {
        let a = AstronomicalArguments::at_datetime(2010, 9, 1, 0, 0, 0.0);
        let m2 = nodal_correction("M2", &a).unwrap();
        let m4 = nodal_correction("M4", &a).unwrap();
        let m6 = nodal_correction("M6", &a).unwrap();
        assert!((m4.f - m2.f.powi(2)).abs() < 1e-12);
        assert!((m6.f - m2.f.powi(3)).abs() < 1e-12);
        // Phase corrections scale linearly.
        assert!((m4.u_deg - 2.0 * m2.u_deg).abs() < 1e-12);
        assert!((m6.u_deg - 3.0 * m2.u_deg).abs() < 1e-12);
    }

    /// The nodal factor is (nearly) periodic over the 18.61-year node cycle.
    #[test]
    fn nodal_factor_is_periodic_over_node_cycle() {
        let jd0 = julian_date(2000, 1, 1, 0, 0, 0.0);
        // Node regresses 360° in ≈6798.38 days.
        let jd1 = jd0 + 6798.38;
        let a0 = AstronomicalArguments::at_julian_date(jd0);
        let a1 = AstronomicalArguments::at_julian_date(jd1);
        let f0 = nodal_correction("O1", &a0).unwrap().f;
        let f1 = nodal_correction("O1", &a1).unwrap().f;
        assert!(
            (f0 - f1).abs() < 2e-3,
            "f(O1): {f0} vs {f1} one node cycle later"
        );
    }

    #[test]
    fn unknown_constituent_returns_none() {
        let a = AstronomicalArguments::at_julian_date(2_451_545.0);
        assert!(nodal_correction("XYZ", &a).is_none());
    }

    #[test]
    fn phase_offset_combines_v0_and_u() {
        let c = NodalCorrection {
            f: 1.0,
            u_deg: 10.0,
            v0_deg: 50.0,
        };
        assert!((c.phase_offset_rad() - 60.0 * DEG_TO_RAD).abs() < 1e-12);
    }

    #[test]
    fn identity_correction_is_neutral() {
        assert_eq!(NodalCorrection::IDENTITY.f, 1.0);
        assert!(NodalCorrection::IDENTITY.phase_offset_rad().abs() < 1e-15);
    }

    /// Sanity: elapsed-time predictions with the correction stay bounded and the
    /// factor multiplies amplitude while the offset shifts phase.
    #[test]
    fn correction_applies_as_amplitude_and_phase() {
        let a = AstronomicalArguments::at_datetime(2022, 1, 1, 0, 0, 0.0);
        let c = nodal_correction("K1", &a).unwrap();
        let omega = 2.0 * PI / (23.934_469_7 * 3600.0); // K1 angular speed (rad/s)
        let amp = 0.1;
        let g = 45.0_f64.to_radians();
        // η at t = 0 with correction.
        let eta0 = c.f * amp * (c.phase_offset_rad() - g).cos();
        // One full K1 period later the phase advances by exactly 2π.
        let t = DAY_SECONDS * (23.934_469_7 / 24.0);
        let eta_t = c.f * amp * (omega * t + c.phase_offset_rad() - g).cos();
        assert!(
            (eta0 - eta_t).abs() < 1e-9,
            "K1 not periodic: {eta0} vs {eta_t}"
        );
        assert!(c.f > 0.8 && c.f < 1.2);
    }
}
