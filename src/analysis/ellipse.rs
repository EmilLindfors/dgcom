//! Tidal current ellipses: harmonic analysis of a velocity series.
//!
//! Each constituent of a current `(u, v)` traces an ellipse. With the
//! reference constants of the components (amplitude `H`, Greenwich lag `G`,
//! as fitted by [`HarmonicAnalysis::fit`]) and the astronomical argument
//! `χ = ω(t − t₀) + V₀ + u_nodal`,
//!
//! ```text
//! u = f Hᵤ cos(χ − Gᵤ),   v = f Hᵥ cos(χ − Gᵥ),
//! ```
//!
//! the complex velocity `w = u + iv` splits into two counter-rotating
//! circles (Foreman 1978; Xu 2002, *Ellipse parameters conversion and
//! velocity profiles for tidal currents in Matlab*):
//!
//! ```text
//! w = W₊ e^{iχ} + W₋ e^{−iχ},   W₊ = ½(Ũ + iṼ),  W₋ = ½(Ũ* + iṼ*),
//! Ũ = Hᵤ e^{−iGᵤ},  Ṽ = Hᵥ e^{−iGᵥ}.
//! ```
//!
//! The ellipse has semi-major axis `|W₊| + |W₋|` and signed semi-minor axis
//! `|W₊| − |W₋|` (positive: the current turns anticlockwise), inclination
//! `θ = ½(arg W₊ + arg W₋)` of the major axis anticlockwise from the x axis
//! (east for east/north components), and Greenwich lag `g = ½(arg W₋ − arg W₊)`
//! of the maximum current along `θ`:
//!
//! ```text
//! w e^{−iθ} = major cos(χ − g) + i minor sin(χ − g).
//! ```
//!
//! `θ` is taken in `[0°, 180°)`; reversing the axis shifts `g` by 180°.
//!
//! # Skill
//!
//! [`TidalEllipse::complex_difference`] is the vector analogue of the
//! elevation's complex difference `|Z_m − Z_o|`
//! ([`ConstituentComparison::complex_difference`](super::ConstituentComparison::complex_difference)):
//! `√(|ΔŨ|² + |ΔṼ|²)`, the RMS over a period of the vector difference between
//! the two tidal currents times √2. It weighs errors in axes, inclination and
//! phase by their effect on the velocity.

use super::{AnalysisError, HarmonicAnalysis, Inference, ReferenceFit};

/// A complex number `(re, im)`.
type Complex = (f64, f64);

fn polar(amplitude: f64, angle: f64) -> Complex {
    (amplitude * angle.cos(), amplitude * angle.sin())
}

fn abs((re, im): Complex) -> f64 {
    re.hypot(im)
}

fn arg((re, im): Complex) -> f64 {
    im.atan2(re)
}

/// Inclination into `[0°, 180°)` and lag into `[0°, 360°)` (degrees):
/// reversing the axis moves the maximum by half a period, and round-off must
/// not wrap one without the other. An inclination within round-off below
/// 180° is taken as 0° (an east–west current reads 0°, not 179.999…°).
fn normalize(inclination: f64, lag: f64) -> (f64, f64) {
    let turns = inclination.div_euclid(180.0);
    let (mut inclination, mut lag) = (inclination - 180.0 * turns, lag - 180.0 * turns);
    if inclination >= 180.0 - 1e-9 {
        inclination = (inclination - 180.0).max(0.0);
        lag -= 180.0;
    }
    let lag = lag.rem_euclid(360.0);
    (inclination, if lag >= 360.0 { 0.0 } else { lag })
}

/// The tidal ellipse of one constituent (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TidalEllipse {
    /// Constituent name.
    pub name: &'static str,
    /// Semi-major axis (m/s), ≥ 0.
    pub major: f64,
    /// Semi-minor axis (m/s), signed: positive when the current turns
    /// anticlockwise, negative when clockwise.
    pub minor: f64,
    /// Inclination of the major axis anticlockwise from the x axis (east),
    /// in `[0°, 180°)`.
    pub inclination_deg: f64,
    /// Greenwich phase lag of the maximum current along the inclination
    /// (degrees, `[0, 360)`).
    pub lag_deg: f64,
    /// Whether the constants were inferred rather than fitted.
    pub inferred: bool,
}

impl TidalEllipse {
    /// The ellipse of the components `u = Hᵤ cos(χ − Gᵤ)`,
    /// `v = Hᵥ cos(χ − Gᵥ)` (amplitude and lag in degrees).
    pub fn from_components(name: &'static str, u: (f64, f64), v: (f64, f64)) -> Self {
        let u_c = polar(u.0, -u.1.to_radians());
        let v_c = polar(v.0, -v.1.to_radians());
        // W₊ = ½(Ũ + iṼ), W₋ = ½(Ũ* + iṼ*)
        let w_plus = (0.5 * (u_c.0 - v_c.1), 0.5 * (u_c.1 + v_c.0));
        let w_minus = (0.5 * (u_c.0 + v_c.1), 0.5 * (-u_c.1 + v_c.0));
        let (a_plus, a_minus) = (abs(w_plus), abs(w_minus));
        let (e_plus, e_minus) = (arg(w_plus), arg(w_minus));
        let (inclination_deg, lag_deg) = normalize(
            (0.5 * (e_plus + e_minus)).to_degrees(),
            (0.5 * (e_minus - e_plus)).to_degrees(),
        );
        Self {
            name,
            major: a_plus + a_minus,
            minor: a_plus - a_minus,
            inclination_deg,
            lag_deg,
            inferred: false,
        }
    }

    /// Amplitude and lag (degrees) of the u and v components: the inverse
    /// of [`from_components`](Self::from_components).
    pub fn components(&self) -> ((f64, f64), (f64, f64)) {
        let (u, v) = self.complex_components();
        let lag = |c: Complex| (-arg(c)).to_degrees().rem_euclid(360.0);
        ((abs(u), lag(u)), (abs(v), lag(v)))
    }

    /// `(Ũ, Ṽ)`: Ũ = W₊ + W₋*, Ṽ = −i(W₊ − W₋*), with
    /// W± = ½(major ± minor) e^{i(θ ∓ g)}.
    fn complex_components(&self) -> (Complex, Complex) {
        let (theta, g) = (self.inclination_deg.to_radians(), self.lag_deg.to_radians());
        let w_plus = polar(0.5 * (self.major + self.minor), theta - g);
        let w_minus = polar(0.5 * (self.major - self.minor), theta + g);
        let u = (w_plus.0 + w_minus.0, w_plus.1 - w_minus.1);
        let d = (w_plus.0 - w_minus.0, w_plus.1 + w_minus.1);
        (u, (d.1, -d.0))
    }

    /// `√(|ΔŨ|² + |ΔṼ|²)` against `other` (see the module docs).
    pub fn complex_difference(&self, other: &Self) -> f64 {
        let (u1, v1) = self.complex_components();
        let (u2, v2) = other.complex_components();
        ((u1.0 - u2.0).powi(2)
            + (u1.1 - u2.1).powi(2)
            + (v1.0 - v2.0).powi(2)
            + (v1.1 - v2.1).powi(2))
        .sqrt()
    }

    /// The ellipse in axes rotated anticlockwise by `angle_deg` (e.g. from
    /// east/north into mesh axes at the local grid convergence).
    pub fn rotated(&self, angle_deg: f64) -> Self {
        let (inclination_deg, lag_deg) = normalize(self.inclination_deg - angle_deg, self.lag_deg);
        Self {
            inclination_deg,
            lag_deg,
            ..*self
        }
    }

    /// Eccentricity-like ratio `minor / major` in `[−1, 1]` (0 rectilinear,
    /// ±1 circular); 0 for a vanishing ellipse.
    pub fn flattening_ratio(&self) -> f64 {
        if self.major > 0.0 {
            self.minor / self.major
        } else {
            0.0
        }
    }
}

/// Result of [`HarmonicAnalysis::fit_ellipses`].
#[derive(Clone, Debug, PartialEq)]
pub struct EllipseFit {
    /// Mean current `(ū, v̄)` over the record (residual flow).
    pub mean: (f64, f64),
    /// Ellipses in the order of the component fits (fitted, then inferred).
    pub ellipses: Vec<TidalEllipse>,
    /// Reference-constant fit of the u component.
    pub u: ReferenceFit,
    /// Reference-constant fit of the v component.
    pub v: ReferenceFit,
}

impl EllipseFit {
    /// The ellipse of constituent `name`.
    pub fn get(&self, name: &str) -> Option<&TidalEllipse> {
        self.ellipses.iter().find(|e| e.name == name)
    }

    /// The tidal prediction of `(u, v)` at `times_unix` (see
    /// [`ReferenceFit::predict`]).
    pub fn predict(&self, times_unix: &[f64]) -> (Vec<f64>, Vec<f64>) {
        (self.u.predict(times_unix), self.v.predict(times_unix))
    }

    /// Fraction of the current variance (both components) explained.
    pub fn r_squared(&self, u: &[f64], v: &[f64]) -> f64 {
        let variance = |x: &[f64]| {
            let mean = x.iter().sum::<f64>() / x.len() as f64;
            x.iter().map(|a| (a - mean).powi(2)).sum::<f64>()
        };
        let (su, sv) = (variance(u), variance(v));
        if su + sv > 0.0 {
            1.0 - ((1.0 - self.u.r_squared) * su + (1.0 - self.v.r_squared) * sv) / (su + sv)
        } else {
            1.0
        }
    }
}

impl HarmonicAnalysis {
    /// Fit tidal ellipses to the current `(u[i], v[i])` at `times_unix[i]`:
    /// the reference constants of each component (inferences applied to
    /// both with the same ratio and lag offset), converted to ellipses.
    /// Errors as [`Self::fit`].
    pub fn fit_ellipses(
        &self,
        times_unix: &[f64],
        u: &[f64],
        v: &[f64],
    ) -> Result<EllipseFit, AnalysisError> {
        let fit_u = self.fit(times_unix, u)?;
        let fit_v = self.fit(times_unix, v)?;
        Ok(EllipseFit::from_components(fit_u, fit_v))
    }
}

impl Inference {
    /// This inference from its first source alone, with the major-axis
    /// ratio and lag difference of `reference`'s ellipses (a longer current
    /// record at the same place), where it has both constituents; else
    /// unchanged. Applied to both
    /// components alike by [`HarmonicAnalysis::fit_ellipses`].
    pub fn or_from_ellipses(self, reference: Option<&EllipseFit>) -> Self {
        reference
            .and_then(|r| {
                let (a, b) = (r.get(self.name)?, r.get(self.from.name)?);
                (b.major > 0.0).then(|| {
                    Self::single(
                        self.name,
                        self.from.name,
                        a.major / b.major,
                        a.lag_deg - b.lag_deg,
                    )
                })
            })
            .unwrap_or(self)
    }
}

impl EllipseFit {
    /// The ellipses of two component fits of the same analysis.
    fn from_components(fit_u: ReferenceFit, fit_v: ReferenceFit) -> Self {
        let ellipses = fit_u
            .constants
            .iter()
            .zip(&fit_v.constants)
            .map(|(cu, cv)| TidalEllipse {
                inferred: cu.inferred,
                ..TidalEllipse::from_components(
                    cu.name,
                    (cu.amplitude, cu.lag_deg),
                    (cv.amplitude, cv.lag_deg),
                )
            })
            .collect();
        Self {
            mean: (fit_u.mean, fit_v.mean),
            ellipses,
            u: fit_u,
            v: fit_v,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::Inference;
    use crate::tides::constituent_period;
    use crate::time::ModelClock;

    fn angle_error(a: f64, b: f64, period: f64) -> f64 {
        ((a - b + 0.5 * period).rem_euclid(period) - 0.5 * period).abs()
    }

    fn assert_ellipse(e: &TidalEllipse, major: f64, minor: f64, inclination: f64, lag: f64) {
        assert!(
            (e.major - major).abs() < 1e-12,
            "major {} vs {major}",
            e.major
        );
        assert!(
            (e.minor - minor).abs() < 1e-12,
            "minor {} vs {minor}",
            e.minor
        );
        assert!(
            angle_error(e.inclination_deg, inclination, 180.0) < 1e-9,
            "inclination {} vs {inclination}",
            e.inclination_deg
        );
        assert!(
            angle_error(e.lag_deg, lag, 360.0) < 1e-9,
            "lag {} vs {lag}",
            e.lag_deg
        );
    }

    /// Rectilinear and rotary cases with known ellipses.
    #[test]
    fn ellipses_of_simple_currents() {
        // u = cos χ: east–west, maximum eastward at χ = 0
        assert_ellipse(
            &TidalEllipse::from_components("M2", (1.0, 0.0), (0.0, 0.0)),
            1.0,
            0.0,
            0.0,
            0.0,
        );
        // u = −cos χ: the same line, maximum along +east half a period later
        assert_ellipse(
            &TidalEllipse::from_components("M2", (1.0, 180.0), (0.0, 0.0)),
            1.0,
            0.0,
            0.0,
            180.0,
        );
        // u = v = 0.5 cos(χ − 30°): along 45°
        let s = 0.5;
        assert_ellipse(
            &TidalEllipse::from_components("M2", (s, 30.0), (s, 30.0)),
            s * 2f64.sqrt(),
            0.0,
            45.0,
            30.0,
        );
        // u = cos(χ − 10°), v = −cos(χ − 10°): along 135°, towards north-west
        // (u < 0, v > 0) at χ = 190°
        assert_ellipse(
            &TidalEllipse::from_components("M2", (1.0, 10.0), (1.0, 190.0)),
            2f64.sqrt(),
            0.0,
            135.0,
            190.0,
        );
        // Round-off at the 0°/180° wrap: inclination and lag stay consistent
        let e = TidalEllipse::from_components("M2", (0.5, 10.0), (0.1, 280.0));
        assert!(e.inclination_deg < 1e-9 || e.inclination_deg > 180.0 - 1e-9);
        let ((hu, gu), _) = e.components();
        assert!((hu - 0.5).abs() < 1e-12 && angle_error(gu, 10.0, 360.0) < 1e-9);
        // u = 0.3 cos χ, v = 0.3 sin χ = 0.3 cos(χ − 90°): anticlockwise circle
        let e = TidalEllipse::from_components("K1", (0.3, 0.0), (0.3, 90.0));
        assert!((e.major - 0.3).abs() < 1e-12 && (e.minor - 0.3).abs() < 1e-12);
        // … and clockwise
        let e = TidalEllipse::from_components("K1", (0.3, 0.0), (0.3, 270.0));
        assert!((e.major - 0.3).abs() < 1e-12 && (e.minor + 0.3).abs() < 1e-12);
        // Northward rectilinear: 90°
        assert_ellipse(
            &TidalEllipse::from_components("M2", (0.0, 0.0), (0.4, 250.0)),
            0.4,
            0.0,
            90.0,
            250.0,
        );
    }

    /// The ellipse reproduces the current it came from at every phase, and
    /// `components` inverts `from_components`.
    #[test]
    fn ellipse_traces_the_current() {
        for &(u, v) in &[
            ((0.21, 239.15), (0.129, 70.37)),
            ((0.06, 327.9), (0.042, 166.5)),
            ((0.5, 10.0), (0.1, 280.0)),
            ((0.0, 0.0), (0.2, 45.0)),
        ] {
            let e = TidalEllipse::from_components("M2", u, v);
            assert!(e.major >= e.minor.abs() && (0.0..180.0).contains(&e.inclination_deg));
            let ((hu, gu), (hv, gv)) = e.components();
            let (theta, g) = (e.inclination_deg.to_radians(), e.lag_deg.to_radians());
            for step in 0..24 {
                let chi = step as f64 * std::f64::consts::TAU / 24.0;
                let (uu, vv) = (
                    u.0 * (chi - u.1.to_radians()).cos(),
                    v.0 * (chi - v.1.to_radians()).cos(),
                );
                // w e^{−iθ} = major cos(χ − g) + i minor sin(χ − g)
                let (a, b) = (e.major * (chi - g).cos(), e.minor * (chi - g).sin());
                let (ue, ve) = (
                    a * theta.cos() - b * theta.sin(),
                    a * theta.sin() + b * theta.cos(),
                );
                assert!(
                    (ue - uu).abs() < 1e-12 && (ve - vv).abs() < 1e-12,
                    "{u:?} {v:?} at {chi}"
                );
                let (ui, vi) = (
                    hu * (chi - gu.to_radians()).cos(),
                    hv * (chi - gv.to_radians()).cos(),
                );
                assert!((ui - uu).abs() < 1e-12 && (vi - vv).abs() < 1e-12);
            }
        }
    }

    /// The complex difference is √2 × the RMS vector difference over a
    /// period, and invariant under a common rotation.
    #[test]
    fn complex_difference_is_the_rms_vector_difference() {
        let a = TidalEllipse::from_components("M2", (0.21, 239.0), (0.13, 70.0));
        let b = TidalEllipse::from_components("M2", (0.18, 251.0), (0.15, 62.0));
        let n = 360;
        let mean_square = (0..n)
            .map(|i| {
                let chi = i as f64 * std::f64::consts::TAU / n as f64;
                let at = |e: &TidalEllipse| {
                    let ((hu, gu), (hv, gv)) = e.components();
                    (
                        hu * (chi - gu.to_radians()).cos(),
                        hv * (chi - gv.to_radians()).cos(),
                    )
                };
                let ((u1, v1), (u2, v2)) = (at(&a), at(&b));
                (u1 - u2).powi(2) + (v1 - v2).powi(2)
            })
            .sum::<f64>()
            / n as f64;
        let d = a.complex_difference(&b);
        assert!((d - (2.0 * mean_square).sqrt()).abs() < 1e-12, "{d}");
        assert_eq!(a.complex_difference(&a), 0.0);
        let d_rotated = a.rotated(33.0).complex_difference(&b.rotated(33.0));
        assert!((d_rotated - d).abs() < 1e-12);
        // Rotating by the inclination puts the major axis on x
        let r = a.rotated(a.inclination_deg);
        assert!(r.inclination_deg.abs() < 1e-12 && (r.lag_deg - a.lag_deg).abs() < 1e-12);
    }

    /// A month of synthetic current with nodal modulation: the fit returns
    /// the ellipses the series was built from, inferred ones included.
    #[test]
    fn fit_recovers_ellipses() {
        let t0 = ModelClock::at_datetime(2025, 6, 1, 0, 0).epoch_unix;
        let truth = [
            TidalEllipse {
                name: "M2",
                major: 0.25,
                minor: -0.04,
                inclination_deg: 32.0,
                lag_deg: 250.0,
                inferred: false,
            },
            TidalEllipse {
                name: "S2",
                major: 0.09,
                minor: 0.01,
                inclination_deg: 28.0,
                lag_deg: 290.0,
                inferred: false,
            },
            TidalEllipse {
                name: "K1",
                major: 0.05,
                minor: 0.03,
                inclination_deg: 120.0,
                lag_deg: 10.0,
                inferred: false,
            },
            TidalEllipse {
                name: "O1",
                major: 0.04,
                minor: 0.0,
                inclination_deg: 100.0,
                lag_deg: 200.0,
                inferred: false,
            },
        ];
        let p1 = TidalEllipse {
            name: "P1",
            major: 0.331 * 0.05,
            minor: 0.331 * 0.03,
            inferred: true,
            ..truth[2]
        };
        let hours = 30 * 24;
        let times: Vec<f64> = (0..hours).map(|i| t0 + i as f64 * 3600.0).collect();
        let clock = ModelClock::new(t0);
        let t_mid = 0.5 * (times[hours - 1] - t0);
        let mut u = vec![0.07; hours];
        let mut v = vec![-0.02; hours];
        for e in truth.iter().chain([&p1]) {
            let ((hu, gu), (hv, gv)) = e.components();
            let n = clock.nodal_correction(e.name, t_mid).unwrap();
            let omega = std::f64::consts::TAU / constituent_period(e.name).unwrap();
            for (i, &t) in times.iter().enumerate() {
                let chi = omega * (t - t0) + n.phase_offset_rad();
                u[i] += n.f * hu * (chi - gu.to_radians()).cos();
                v[i] += n.f * hv * (chi - gv.to_radians()).cos();
            }
        }
        let fit = HarmonicAnalysis::new(&["M2", "S2", "K1", "O1"])
            .with_inferences(&[Inference::P1_FROM_K1])
            .fit_ellipses(&times, &u, &v)
            .unwrap();
        assert!((fit.mean.0 - 0.07).abs() < 1e-10 && (fit.mean.1 + 0.02).abs() < 1e-10);
        assert!(fit.r_squared(&u, &v) > 1.0 - 1e-12);
        for e in truth.iter().chain([&p1]) {
            let f = fit.get(e.name).unwrap();
            assert!(f.complex_difference(e) < 1e-9, "{}: {f:?} vs {e:?}", e.name);
            assert!((f.minor - e.minor).abs() < 1e-9 && f.inferred == e.inferred);
        }
        let (pu, pv) = fit.predict(&times);
        for i in 0..hours {
            assert!((pu[i] - u[i]).abs() < 1e-9 && (pv[i] - v[i]).abs() < 1e-9);
        }
    }
}
