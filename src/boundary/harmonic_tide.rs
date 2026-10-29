//! A uniform harmonic tide `η(t)`, the elevation signal of the harmonic
//! boundary conditions.
//!
//! ```text
//! η(t) = η₀ + R(t) Σᵢ fᵢ Aᵢ cos(ωᵢ t + φᵢ + (V₀ + u)ᵢ)
//! ```
//!
//! with reference constants `Aᵢ` and `φᵢ = −Gᵢ` (Greenwich phase lag), the
//! nodal corrections `fᵢ(t)`, `(V₀ + u(t))ᵢ` of a [`ModelClock`] (`f`, `u`
//! drifting with `t`, [`NodalDrift`]; identity until
//! [`HarmonicTide::with_nodal_corrections`] sets them) and a smooth start-up
//! ramp `R(t)`. The corrections are stored apart from the constants, so setting
//! them again replaces them instead of applying them twice.

use super::BCContext2D;
use super::TidalConstituent;
use super::characteristic::{ExternalState, ExternalStateProvider};
use crate::io::ConstituentData;
use crate::tides::{NodalDrift, canonical_name};
use crate::time::ModelClock;

/// Smooth ramp `R(t)` from 0 at `t ≤ 0` to 1 at `t ≥ duration`
/// (`3τ² − 2τ³`, τ = t / duration); 1 without a (positive) duration.
pub fn tidal_ramp(t: f64, duration: Option<f64>) -> f64 {
    match duration {
        Some(d) if d > 0.0 => {
            let tau = (t / d).clamp(0.0, 1.0);
            tau * tau * (3.0 - 2.0 * tau)
        }
        _ => 1.0,
    }
}

/// A spatially uniform harmonic tide; see the module docs.
#[derive(Clone, Debug)]
pub struct HarmonicTide {
    /// Mean surface elevation η₀ (m); not ramped.
    pub mean_elevation: f64,
    /// Ramp-up duration (s), `None` for none.
    pub ramp_duration: Option<f64>,
    constituents: Vec<TidalConstituent>,
    corrections: Vec<NodalDrift>,
}

impl HarmonicTide {
    /// A tide from reference constants (internal phase `φ = −G`).
    pub fn new(constituents: Vec<TidalConstituent>) -> Self {
        let corrections = vec![NodalDrift::IDENTITY; constituents.len()];
        Self {
            mean_elevation: 0.0,
            ramp_duration: None,
            constituents,
            corrections,
        }
    }

    /// M2 only, amplitude (m) and internal phase (rad).
    pub fn m2(amplitude: f64, phase: f64) -> Self {
        Self::new(vec![TidalConstituent::m2(amplitude, phase)])
    }

    /// The constituents of a constituent file (amplitude, Greenwich lag `G` in
    /// degrees) with its reference level as mean elevation. Errors on a name
    /// outside [`crate::tides::CONSTITUENT_NAMES`].
    pub fn from_constituent_data(data: &ConstituentData) -> Result<Self, String> {
        let constituents = data
            .constituents
            .iter()
            .map(|c| {
                let name = canonical_name(&c.name)
                    .ok_or_else(|| format!("unsupported constituent {}", c.name))?;
                Ok(TidalConstituent::new(
                    name,
                    c.amplitude,
                    c.period,
                    -c.phase_radians(),
                ))
            })
            .collect::<Result<_, String>>()?;
        Ok(Self::new(constituents).with_mean_elevation(data.reference_level))
    }

    /// Set the mean elevation η₀.
    pub fn with_mean_elevation(mut self, mean: f64) -> Self {
        self.mean_elevation = mean;
        self
    }

    /// Ramp the tide up over `duration` seconds from t = 0.
    pub fn with_ramp_up(mut self, duration: f64) -> Self {
        self.ramp_duration = Some(duration);
        self
    }

    /// Nodal corrections for a run on `clock`: `V₀` at t = 0, `f` and `u` at
    /// each time ([`ModelClock::nodal_drift`]). Replaces any earlier
    /// corrections; constituents without tabulated corrections keep the
    /// identity.
    pub fn with_nodal_corrections(mut self, clock: &ModelClock) -> Self {
        self.corrections = self
            .constituents
            .iter()
            .map(|c| clock.nodal_drift(c.name).unwrap_or(NodalDrift::IDENTITY))
            .collect();
        self
    }

    /// Reference constants (uncorrected).
    pub fn constituents(&self) -> &[TidalConstituent] {
        &self.constituents
    }

    /// Nodal corrections, one per constituent.
    pub fn corrections(&self) -> &[NodalDrift] {
        &self.corrections
    }

    /// Ramp factor `R(t)`.
    pub fn ramp_factor(&self, t: f64) -> f64 {
        tidal_ramp(t, self.ramp_duration)
    }

    /// Surface elevation η(t).
    pub fn elevation(&self, t: f64) -> f64 {
        let tide: f64 = self
            .constituents
            .iter()
            .zip(&self.corrections)
            .map(|(c, drift)| {
                let n = drift.at(t);
                n.f * c.amplitude
                    * (c.angular_frequency() * t + c.phase + n.phase_offset_rad()).cos()
            })
            .sum();
        self.mean_elevation + self.ramp_factor(t) * tide
    }
}

impl ExternalStateProvider for HarmonicTide {
    /// Elevation only; the characteristic OBC chooses the velocity.
    fn external_state(&self, ctx: &BCContext2D) -> ExternalState {
        ExternalState::elevation(self.elevation(ctx.time))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn ramp_is_smooth_and_saturates() {
        assert_eq!(tidal_ramp(-1.0, Some(10.0)), 0.0);
        assert_eq!(tidal_ramp(5.0, Some(10.0)), 0.5);
        assert_eq!(tidal_ramp(20.0, Some(10.0)), 1.0);
        assert_eq!(tidal_ramp(3.0, None), 1.0);
        assert_eq!(tidal_ramp(3.0, Some(0.0)), 1.0);
    }

    #[test]
    fn elevation_uses_greenwich_lag() {
        // φ = −G: with G = 90° the peak comes a quarter period after t = 0
        let tide = HarmonicTide::m2(0.5, -PI / 2.0).with_mean_elevation(0.1);
        let period = tide.constituents()[0].period;
        assert!((tide.elevation(0.0) - 0.1).abs() < 1e-12);
        assert!((tide.elevation(0.25 * period) - 0.6).abs() < 1e-12);
    }

    /// Setting the corrections twice must not apply them twice.
    #[test]
    fn nodal_corrections_are_not_applied_twice() {
        let clock = ModelClock::at_datetime(2025, 6, 1, 0, 0);
        let once = HarmonicTide::m2(0.5, 0.3).with_nodal_corrections(&clock);
        let twice = once.clone().with_nodal_corrections(&clock);
        for t in [0.0, 1234.5, 40_000.0] {
            assert_eq!(once.elevation(t), twice.elevation(t));
        }
        let n = clock.nodal_correction("M2", 0.0).unwrap();
        let expected = n.f * 0.5 * (0.3 + n.phase_offset_rad()).cos();
        assert!((once.elevation(0.0) - expected).abs() < 1e-12);
    }

    /// The nodal terms are those of the time the tide is evaluated at, not
    /// of a run's middle (TODO P1.5: a run resumed to another end was forced
    /// differently): ten days in, K1 is `f`, `u` at ten days.
    #[test]
    fn nodal_corrections_follow_the_time() {
        let clock = ModelClock::at_datetime(2025, 6, 15, 0, 0);
        let k1 = TidalConstituent::new(
            "K1",
            0.1,
            crate::tides::constituent_period("K1").unwrap(),
            0.0,
        );
        let tide = HarmonicTide::new(vec![k1.clone()]).with_nodal_corrections(&clock);
        for t in [0.0, 10.0 * 86_400.0 + 1234.5] {
            let n = clock.nodal_correction("K1", t).unwrap();
            let expected = n.f * 0.1 * (k1.angular_frequency() * t + n.phase_offset_rad()).cos();
            // The linear drift is 6e-7 m off; held at the epoch, 1.4e-4 m
            assert!((tide.elevation(t) - expected).abs() < 1e-5, "{t}");
        }
    }

    #[test]
    fn constituent_file_phases_are_lags() {
        let data = crate::io::parse_constituents("# reference_level: 0.2\nm2 0.5 90\n").unwrap();
        let tide = HarmonicTide::from_constituent_data(&data).unwrap();
        assert_eq!(tide.constituents()[0].name, "M2");
        assert!((tide.constituents()[0].phase + PI / 2.0).abs() < 1e-12);
        assert_eq!(tide.mean_elevation, 0.2);

        let bad = crate::io::parse_constituents("XX9 0.5 90\n");
        if let Ok(data) = bad {
            assert!(HarmonicTide::from_constituent_data(&data).is_err());
        }
    }
}
