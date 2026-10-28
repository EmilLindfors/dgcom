//! Tidal astronomy and nodal corrections.
//!
//! This module supplies the astronomical corrections a real tidal prediction
//! needs on top of catalogued amplitude/phase pairs: the equilibrium argument
//! `V₀` (astronomical phase at the prediction epoch) and the 18.61-year nodal
//! modulation `f`, `u`. See [`astronomy`] for the mathematical formulation.
//!
//! # Applying corrections
//!
//! A corrected constituent predicts elevation as
//!
//! ```text
//! η(t) = f · H · cos(ω t + (V₀ + u) − G)
//! ```
//!
//! where `t` is elapsed time (seconds) from the epoch. Given the crate's
//! internal phase convention `φ = −G` (elevation `H·cos(ω t + φ)`), a correction
//! multiplies the amplitude by `f` and adds `(V₀ + u)` to `φ`:
//!
//! ```
//! use dg_rs::tides::{AstronomicalArguments, correct_amplitude_phase, nodal_correction};
//!
//! // Raw catalogue constituent: amplitude H and internal phase φ = −G.
//! let (h, phi) = (0.45_f64, -125.3_f64.to_radians());
//! let astro = AstronomicalArguments::at_datetime(2024, 6, 1, 0, 0, 0.0);
//! let corr = nodal_correction("M2", &astro).unwrap();
//! let (h_corr, phi_corr) = correct_amplitude_phase(h, phi, &corr);
//! assert!((h_corr - corr.f * h).abs() < 1e-12);
//! ```

pub mod astronomy;

pub use astronomy::{
    AstronomicalArguments, CONSTITUENT_NAMES, NodalCorrection, NodalDrift, angular_speed,
    canonical_name, constituent_period, julian_date, nodal_correction,
};

/// Apply a nodal correction to a raw `(amplitude, internal_phase)` pair.
///
/// `internal_phase` is the crate's convention `φ = −G` (radians), so the result
/// evaluates as `f·H·cos(ω t + φ + V₀ + u)`. Returns
/// `(f · amplitude, internal_phase + (V₀ + u))`.
#[inline]
pub fn correct_amplitude_phase(
    amplitude: f64,
    internal_phase: f64,
    correction: &NodalCorrection,
) -> (f64, f64) {
    (
        correction.f * amplitude,
        internal_phase + correction.phase_offset_rad(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correct_amplitude_phase_scales_and_shifts() {
        let corr = NodalCorrection {
            f: 1.05,
            u_deg: 3.0,
            v0_deg: 40.0,
        };
        let (amp, phase) = correct_amplitude_phase(0.5, 0.1, &corr);
        assert!((amp - 0.525).abs() < 1e-12);
        assert!((phase - (0.1 + 43.0_f64.to_radians())).abs() < 1e-12);
    }

    #[test]
    fn identity_correction_leaves_pair_unchanged() {
        let (amp, phase) = correct_amplitude_phase(0.5, 0.1, &NodalCorrection::IDENTITY);
        assert!((amp - 0.5).abs() < 1e-12);
        assert!((phase - 0.1).abs() < 1e-12);
    }
}
