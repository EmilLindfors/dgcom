//! Minor constituents inferred from major ones by linear admittance.
//!
//! The ocean's response to the tidal potential varies smoothly with
//! frequency within a species, so a minor constituent `k` follows from two
//! neighbours `a` and `b` of the same species (Munk & Cartwright 1966): the
//! admittance, the complex amplitude `Z = H e^{−iG}` over the equilibrium
//! amplitude `E`, is taken linear in frequency between them,
//!
//! ```text
//! Z_k / E_k = (1 − x) Z_a / E_a + x Z_b / E_b,    x = (ω_k − ω_a) / (ω_b − ω_a),
//! ```
//!
//! which is `Z_k = w_a Z_a + w_b Z_b` with real weights. `x` outside
//! `[0, 1]` extrapolates (2N2 and μ2 lie below N2). The weights of
//! [`AdmittanceInference::MINOR_SEMIDIURNALS`] are those of Ray's `perth`
//! tide-prediction code and OTPS's minor-constituent inference: e.g.
//! `Z(ν2) = 0.165 Z(N2) + 0.00487 Z(M2)`, `Z(L2) = 0.0131 Z(M2) + 0.0326 Z(S2)`.
//!
//! The constants are reference constants (nodal modulation removed), so the
//! inferred constituent takes its own `f`, `u` in a prediction.
//!
//! # References
//!
//! - Munk & Cartwright (1966), *Tidal spectroscopy and prediction*, Phil.
//!   Trans. R. Soc. A 259.
//! - Cartwright & Tayler (1971), Cartwright & Edden (1973): the equilibrium
//!   amplitudes.
//! - Le Provost et al. (1991), *Spectroscopy of the world ocean tides from a
//!   finite element hydrodynamic model*: admittance inference of minor waves.

use super::angular_speed;

/// Equilibrium amplitude (m) of a constituent of the tide-generating
/// potential, from the Cartwright–Tayler–Edden expansion (signs absorbed
/// into the 180° offsets of the equilibrium arguments, as in Foreman's
/// tables); `None` for the compound and long-period ones not listed.
///
/// Their ratios are the equilibrium inferences: K2/S2 = 0.272, P1/K1 =
/// 0.331, N2/M2 = Q1/O1 = 0.191.
pub fn equilibrium_amplitude(name: &str) -> Option<f64> {
    Some(match name.trim().to_uppercase().as_str() {
        "M2" => 0.242_334,
        "S2" => 0.112_841,
        "N2" => 0.046_398,
        "K2" => 0.030_704,
        "2N2" => 0.006_141,
        "MU2" => 0.007_408,
        "NU2" => 0.008_811,
        "LDA2" => 0.001_790,
        "L2" => 0.006_863,
        "K1" => 0.141_565,
        "O1" => 0.100_514,
        "P1" => 0.046_843,
        "Q1" => 0.019_256,
        _ => return None,
    })
}

/// Constituent `name` inferred from `lower` and `upper` by linear
/// admittance in frequency; see the module docs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdmittanceInference {
    /// Inferred constituent.
    pub name: &'static str,
    /// Lower reference in frequency.
    pub lower: &'static str,
    /// Upper reference in frequency.
    pub upper: &'static str,
}

impl AdmittanceInference {
    /// The minor semidiurnals 2N2, μ2, ν2 from N2 and M2, and λ2, L2 from M2
    /// and S2.
    pub const MINOR_SEMIDIURNALS: [Self; 5] = [
        Self::new("2N2", "N2", "M2"),
        Self::new("MU2", "N2", "M2"),
        Self::new("NU2", "N2", "M2"),
        Self::new("LDA2", "M2", "S2"),
        Self::new("L2", "M2", "S2"),
    ];

    /// `name` from `lower` and `upper`.
    pub const fn new(name: &'static str, lower: &'static str, upper: &'static str) -> Self {
        Self { name, lower, upper }
    }

    /// The weights `(w_lower, w_upper)` of `Z = w_lower Z_lower + w_upper
    /// Z_upper`; `None` if a constituent has no equilibrium amplitude or
    /// speed, or the references share a speed.
    pub fn weights(&self) -> Option<(f64, f64)> {
        let (omega, omega_a, omega_b) = (
            angular_speed(self.name)?,
            angular_speed(self.lower)?,
            angular_speed(self.upper)?,
        );
        if omega_a == omega_b {
            return None;
        }
        let (e, e_a, e_b) = (
            equilibrium_amplitude(self.name)?,
            equilibrium_amplitude(self.lower)?,
            equilibrium_amplitude(self.upper)?,
        );
        let x = (omega - omega_a) / (omega_b - omega_a);
        Some(((1.0 - x) * e / e_a, x * e / e_b))
    }

    /// The inferred `(H, G)` (G in degrees, `[0, 360)`) from the references'
    /// `(H, G)`; `None` as for [`weights`](Self::weights).
    pub fn infer(&self, lower: (f64, f64), upper: (f64, f64)) -> Option<(f64, f64)> {
        let (wa, wb) = self.weights()?;
        let complex = |(h, g): (f64, f64)| {
            let g = g.to_radians();
            (h * g.cos(), -h * g.sin())
        };
        let (a, b) = (complex(lower), complex(upper));
        let (re, im) = (wa * a.0 + wb * b.0, wa * a.1 + wb * b.1);
        Some((
            re.hypot(im),
            (-im).atan2(re).to_degrees().rem_euclid(360.0),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The weights are Ray's (`perth`, OTPS): 2N2 0.264 N2 − 0.0253 M2,
    /// μ2 0.298 N2 − 0.0264 M2, ν2 0.165 N2 + 0.00487 M2, λ2 0.0040 M2 +
    /// 0.0074 S2, L2 0.0131 M2 + 0.0326 S2.
    #[test]
    fn weights_match_rays_coefficients() {
        let expected = [
            (0.264, -0.0253),
            (0.298, -0.0264),
            (0.165, 0.00487),
            (0.0040, 0.0074),
            (0.0131, 0.0326),
        ];
        for (inference, (a, b)) in AdmittanceInference::MINOR_SEMIDIURNALS.iter().zip(expected) {
            let (wa, wb) = inference.weights().unwrap();
            // Ray's coefficients carry two or three digits
            assert!((wa - a).abs() < 0.006 * a.abs().max(0.1), "{}: {wa} vs {a}", inference.name);
            assert!((wb - b).abs() < 0.03 * b.abs(), "{}: {wb} vs {b}", inference.name);
        }
    }

    /// The equilibrium ratios of the crate's single-source inferences.
    #[test]
    fn equilibrium_ratios() {
        let ratio = |a, b| equilibrium_amplitude(a).unwrap() / equilibrium_amplitude(b).unwrap();
        assert!((ratio("K2", "S2") - 0.272).abs() < 5e-4);
        assert!((ratio("P1", "K1") - 0.331).abs() < 5e-4);
        assert!((ratio("N2", "M2") - 0.191).abs() < 1e-3);
        assert!((ratio("Q1", "O1") - 0.191).abs() < 1e-3);
    }

    /// An ocean whose admittance is the same at every frequency (one gain
    /// and one lag) gives every minor constituent that gain and lag, and a
    /// reference inferred from itself is itself.
    #[test]
    fn constant_admittance_is_reproduced() {
        let (gain, lag) = (3.7, 123.0);
        let z = |name| (gain * equilibrium_amplitude(name).unwrap(), lag);
        for inference in AdmittanceInference::MINOR_SEMIDIURNALS {
            let (h, g) = inference
                .infer(z(inference.lower), z(inference.upper))
                .unwrap();
            let (h0, g0) = z(inference.name);
            assert!((h - h0).abs() < 1e-12 && (g - g0).abs() < 1e-9, "{}", inference.name);
        }
        let itself = AdmittanceInference::new("N2", "N2", "M2");
        assert_eq!(itself.weights().map(|w| w.1), Some(0.0));
        let (h, g) = itself.infer((0.17, 300.0), (0.9, 330.0)).unwrap();
        assert!((h - 0.17).abs() < 1e-12 && (g - 300.0).abs() < 1e-9);
    }

    #[test]
    fn unknown_constituents_infer_nothing() {
        assert!(AdmittanceInference::new("M4", "N2", "M2").weights().is_none());
        assert!(AdmittanceInference::new("NU2", "M2", "M2").weights().is_none());
    }
}
