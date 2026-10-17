//! Least-squares harmonic analysis: tidal *reference* constants `(H, G)` of a
//! series on an absolute (Unix) time axis, with inference of the
//! constituents the record cannot resolve.
//!
//! The basis functions carry the astronomy, so the unknowns are the
//! reference constants directly:
//!
//! ```text
//! η(t) = Z₀ + Σⱼ fⱼ Hⱼ cos(ωⱼ (t − t₀) + V₀ⱼ + uⱼ − Gⱼ)
//! ```
//!
//! with `V₀` at the first sample `t₀` and the nodal `f`, `u` at the record
//! midpoint ([`ModelClock::nodal_correction`]); `G` is the Greenwich phase
//! lag. Two fits of the same constituents are therefore comparable whatever
//! their records, and comparable with a catalogue or an atlas.
//!
//! # Inference
//!
//! Two constituents are separable only if the record is longer than their
//! synodic period `2π/|ω₁ − ω₂|` (Rayleigh criterion). P1 and K1 need 182
//! days, as do K2 and S2, so a one- or two-month record cannot fit them
//! independently: their energy leaks into each other and the fit becomes
//! ill-conditioned. An [`Inference`] ties such a constituent to a resolved one
//! with a fixed amplitude ratio and phase-lag offset (the equilibrium-tide
//! values by default, as in Foreman's `t_tide`), so both enter the basis but
//! share one pair of unknowns. A fitted pair that is not separable is an
//! error ([`AnalysisError::NotSeparable`]); [`HarmonicAnalysis::resolvable`]
//! chooses what a record can fit.
//!
//! # Solution
//!
//! Each sample's row of the design matrix is rotated into the triangular
//! factor of `[A | y]` as it is formed (a Givens QR), so a fit needs O(m²)
//! memory for m unknowns whatever the record's length, and keeps the
//! conditioning of `A` rather than the square of it that the normal
//! equations `AᵀA x = Aᵀy` have.

use super::{AnalysisError, TimeSeries};
use crate::tides::constituent_period;
use crate::time::ModelClock;

/// A constituent inferred from a resolved one: `H = ratio · H_from`,
/// `G = G_from + lag_offset`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Inference {
    /// Inferred constituent.
    pub name: &'static str,
    /// Resolved constituent it follows.
    pub from: &'static str,
    /// Amplitude ratio `H / H_from`.
    pub amplitude_ratio: f64,
    /// Phase-lag offset `G − G_from` (degrees).
    pub lag_offset_deg: f64,
}

impl Inference {
    /// P1 from K1 with the equilibrium amplitude ratio 0.331 (Cartwright &
    /// Tayler 1971) and equal lag.
    pub const P1_FROM_K1: Self = Self {
        name: "P1",
        from: "K1",
        amplitude_ratio: 0.331,
        lag_offset_deg: 0.0,
    };

    /// K2 from S2 with the equilibrium amplitude ratio 0.272 and equal lag.
    pub const K2_FROM_S2: Self = Self {
        name: "K2",
        from: "S2",
        amplitude_ratio: 0.272,
        lag_offset_deg: 0.0,
    };

    /// N2 from M2 with the equilibrium amplitude ratio 0.191 and equal lag
    /// (separable only from 27.6 days).
    pub const N2_FROM_M2: Self = Self {
        name: "N2",
        from: "M2",
        amplitude_ratio: 0.191,
        lag_offset_deg: 0.0,
    };

    /// Q1 from O1 with the equilibrium amplitude ratio 0.191 and equal lag
    /// (separable only from 27.6 days).
    pub const Q1_FROM_O1: Self = Self {
        name: "Q1",
        from: "O1",
        amplitude_ratio: 0.191,
        lag_offset_deg: 0.0,
    };

    /// The equilibrium inferences above.
    pub const EQUILIBRIUM: [Self; 4] = [
        Self::P1_FROM_K1,
        Self::K2_FROM_S2,
        Self::N2_FROM_M2,
        Self::Q1_FROM_O1,
    ];

    /// `name` from `from` with the ratio and lag offset of a reference fit
    /// (e.g. a long record at a nearby gauge), which beats the equilibrium
    /// values wherever the shelf response differs between the two; `None`
    /// if the reference lacks either constituent.
    pub fn from_reference(
        name: &'static str,
        from: &'static str,
        reference: &ReferenceFit,
    ) -> Option<Self> {
        let (a, b) = (reference.get(name)?, reference.get(from)?);
        Some(Self {
            name,
            from,
            amplitude_ratio: a.amplitude / b.amplitude,
            lag_offset_deg: a.lag_deg - b.lag_deg,
        })
    }

    /// This inference with the ratio and lag offset of `reference` where it
    /// has both constituents, else unchanged.
    pub fn or_from(self, reference: Option<&ReferenceFit>) -> Self {
        reference
            .and_then(|r| Self::from_reference(self.name, self.from, r))
            .unwrap_or(self)
    }
}

/// A fitted or inferred reference constant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReferenceConstant {
    /// Constituent name.
    pub name: &'static str,
    /// Amplitude `H` (units of the series).
    pub amplitude: f64,
    /// Greenwich phase lag `G` (degrees, `[0, 360)`).
    pub lag_deg: f64,
    /// Whether the constant was inferred rather than fitted.
    pub inferred: bool,
}

/// Result of [`HarmonicAnalysis::fit`].
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceFit {
    /// Mean `Z₀` over the record.
    pub mean: f64,
    /// Fitted constituents in the analysis's order, then the inferred ones.
    pub constants: Vec<ReferenceConstant>,
    /// Fraction of the variance explained.
    pub r_squared: f64,
    /// RMS of the residual.
    pub residual_rms: f64,
}

impl ReferenceFit {
    /// The constant of constituent `name`.
    pub fn get(&self, name: &str) -> Option<&ReferenceConstant> {
        self.constants.iter().find(|c| c.name == name)
    }

    /// The tidal prediction `Z₀ + Σⱼ fⱼ Hⱼ cos(ωⱼ (t − t₀) + V₀ⱼ + uⱼ − Gⱼ)`
    /// at `times_unix` (fitted and inferred constants), with `V₀` at the
    /// first time and `f`, `u` at the middle of the span: the fit's own
    /// convention, so any record is predicted as it would be fitted.
    pub fn predict(&self, times_unix: &[f64]) -> Vec<f64> {
        let (Some(&t0), Some(&t1)) = (times_unix.first(), times_unix.last()) else {
            return Vec::new();
        };
        let clock = ModelClock::new(t0);
        let terms: Vec<(f64, f64, f64)> = self
            .constants
            .iter()
            .filter_map(|c| {
                let omega = 2.0 * std::f64::consts::PI / constituent_period(c.name)?;
                let n = clock.nodal_correction(c.name, 0.5 * (t1 - t0))?;
                Some((
                    omega,
                    n.f * c.amplitude,
                    n.phase_offset_rad() - c.lag_deg.to_radians(),
                ))
            })
            .collect();
        times_unix
            .iter()
            .map(|&t| {
                self.mean
                    + terms
                        .iter()
                        .map(|&(omega, amp, phase)| amp * (omega * (t - t0) + phase).cos())
                        .sum::<f64>()
            })
            .collect()
    }
}

/// A harmonic analysis: the constituents to fit and those to infer from
/// them. See the module docs.
#[derive(Clone, Debug, PartialEq)]
pub struct HarmonicAnalysis {
    names: Vec<&'static str>,
    inferences: Vec<Inference>,
}

impl HarmonicAnalysis {
    /// Fit `names`, inferring nothing.
    pub fn new(names: &[&'static str]) -> Self {
        Self {
            names: names.to_vec(),
            inferences: Vec::new(),
        }
    }

    /// M2, S2, K1 and O1, the four largest constituents (15 days separate
    /// them).
    pub fn standard() -> Self {
        Self::new(&["M2", "S2", "K1", "O1"])
    }

    /// The constituents that matter along the Norwegian coast from a month's
    /// record: M2, S2, N2, K1 and O1 fitted, P1 inferred from K1.
    pub fn norwegian_coast() -> Self {
        Self::new(&["M2", "S2", "N2", "K1", "O1"]).with_inferences(&[Inference::P1_FROM_K1])
    }

    /// For a record of `length` seconds: the longest prefix-greedy subset of
    /// `candidates` it separates (each candidate, in priority order, is kept
    /// if the record is at least `rayleigh` times its synodic period with
    /// every kept one; `rayleigh` = 1 is the classical criterion; unknown
    /// names are skipped), and of `inferences` those whose constituent is
    /// left out and whose source is kept.
    pub fn resolvable(
        candidates: &[&'static str],
        length: f64,
        rayleigh: f64,
        inferences: &[Inference],
    ) -> Self {
        let mut kept: Vec<(&'static str, f64)> = Vec::new();
        for &name in candidates {
            let Some(period) = constituent_period(name) else {
                continue;
            };
            let omega = 2.0 * std::f64::consts::PI / period;
            let separable = kept.iter().all(|&(_, other)| {
                let synodic = 2.0 * std::f64::consts::PI / (omega - other).abs();
                length >= rayleigh * synodic
            });
            if separable {
                kept.push((name, omega));
            }
        }
        let names: Vec<&'static str> = kept.into_iter().map(|(name, _)| name).collect();
        let inferences = inferences
            .iter()
            .filter(|i| !names.contains(&i.name) && names.contains(&i.from))
            .copied()
            .collect();
        Self { names, inferences }
    }

    /// Also infer `inferences` from the fitted constituents.
    pub fn with_inferences(mut self, inferences: &[Inference]) -> Self {
        self.inferences.extend_from_slice(inferences);
        self
    }

    /// The fitted constituents.
    pub fn names(&self) -> &[&'static str] {
        &self.names
    }

    /// The inferred constituents.
    pub fn inferences(&self) -> &[Inference] {
        &self.inferences
    }

    /// The record length the fitted constituents need to separate pairwise
    /// (Rayleigh criterion: the longest synodic period of a pair, s); 0 for
    /// fewer than two, or unknown names.
    pub fn minimum_record_length(&self) -> f64 {
        let omegas: Vec<f64> = self
            .names
            .iter()
            .filter_map(|&n| constituent_period(n))
            .map(|p| 2.0 * std::f64::consts::PI / p)
            .collect();
        let mut longest = 0.0_f64;
        for (i, &a) in omegas.iter().enumerate() {
            for &b in &omegas[i + 1..] {
                if a != b {
                    longest = longest.max(2.0 * std::f64::consts::PI / (a - b).abs());
                }
            }
        }
        longest
    }

    /// Fit the reference constants to `series` (Unix times).
    pub fn fit_series(&self, series: &TimeSeries) -> Result<ReferenceFit, AnalysisError> {
        self.fit(series.times(), series.values())
    }

    /// Fit the reference constants to the samples `(times_unix[i],
    /// values[i])`; see the module docs.
    ///
    /// Errors on samples and times of different lengths, an unknown
    /// constituent, an inference whose source is not fitted, fewer samples
    /// than unknowns, or a pair of fitted constituents that the record
    /// cannot separate.
    pub fn fit(&self, times_unix: &[f64], values: &[f64]) -> Result<ReferenceFit, AnalysisError> {
        let n = times_unix.len();
        if n != values.len() {
            return Err(AnalysisError::LengthMismatch {
                times: n,
                values: values.len(),
            });
        }
        let names = &self.names;
        let n_unknowns = 1 + 2 * names.len();
        if n < n_unknowns {
            return Err(AnalysisError::TooFewSamples {
                samples: n,
                unknowns: n_unknowns,
            });
        }
        let (t0, t1) = (times_unix[0], times_unix[n - 1]);
        let length = t1 - t0;
        let clock = ModelClock::new(t0);
        let t_mid = 0.5 * length;

        // (ω, f, V₀ + u) of a constituent on this record
        let astronomy = |name: &str| -> Result<(f64, f64, f64), AnalysisError> {
            let period = constituent_period(name)
                .ok_or_else(|| AnalysisError::UnknownConstituent(name.to_string()))?;
            let n = clock
                .nodal_correction(name, t_mid)
                .ok_or_else(|| AnalysisError::NoNodalCorrection(name.to_string()))?;
            Ok((
                2.0 * std::f64::consts::PI / period,
                n.f,
                n.phase_offset_rad(),
            ))
        };
        let fitted: Vec<(f64, f64, f64)> = names
            .iter()
            .map(|n| astronomy(n))
            .collect::<Result<_, _>>()?;

        // Rayleigh criterion between the fitted constituents
        for (i, a) in fitted.iter().enumerate() {
            for (j, b) in fitted.iter().enumerate().skip(i + 1) {
                let synodic = 2.0 * std::f64::consts::PI / (a.0 - b.0).abs();
                if length < synodic {
                    return Err(AnalysisError::NotSeparable {
                        a: names[i],
                        b: names[j],
                        needed_days: synodic / 86_400.0,
                        have_days: length / 86_400.0,
                    });
                }
            }
        }

        // Inferred constituents, attached to the index of their source
        let attached: Vec<(usize, Inference, (f64, f64, f64))> = self
            .inferences
            .iter()
            .map(|inf| {
                let source = names.iter().position(|&n| n == inf.from).ok_or(
                    AnalysisError::InferenceSourceNotFitted {
                        name: inf.name,
                        from: inf.from,
                    },
                )?;
                Ok((source, *inf, astronomy(inf.name)?))
            })
            .collect::<Result<_, AnalysisError>>()?;

        // A row of the design matrix and the sample, [1, (f cos(arg), f sin(arg))
        // per fitted constituent, y], is rotated into the upper-triangular R of
        // [A | y] at once (Givens; Golub & Van Loan 2013, §5.2.5, §6.5.3): no
        // n × m matrix, the conditioning of A rather than of AᵀA, and R's last
        // diagonal is the residual norm.
        let p = n_unknowns + 1;
        let mut r = vec![0.0f64; p * p];
        let mut w = vec![0.0f64; p];
        for (&t_unix, &value) in times_unix.iter().zip(values) {
            let t = t_unix - t0;
            w[0] = 1.0;
            for (j, &(omega, f, phase)) in fitted.iter().enumerate() {
                let arg = omega * t + phase;
                w[1 + 2 * j] = f * arg.cos();
                w[2 + 2 * j] = f * arg.sin();
            }
            for &(j, inf, (omega, f, phase)) in &attached {
                let arg = omega * t + phase - inf.lag_offset_deg.to_radians();
                w[1 + 2 * j] += inf.amplitude_ratio * f * arg.cos();
                w[2 + 2 * j] += inf.amplitude_ratio * f * arg.sin();
            }
            w[n_unknowns] = value;
            for j in 0..p {
                if w[j] == 0.0 {
                    continue;
                }
                let row = &mut r[j * p..(j + 1) * p];
                let rho = row[j].hypot(w[j]);
                let (c, s) = (row[j] / rho, w[j] / rho);
                row[j] = rho;
                for (rk, wk) in row[j + 1..].iter_mut().zip(&mut w[j + 1..]) {
                    (*rk, *wk) = (c * *rk + s * *wk, c * *wk - s * *rk);
                }
            }
        }
        // R x = Qᵀy
        let mut x = vec![0.0f64; n_unknowns];
        for j in (0..n_unknowns).rev() {
            let row = &r[j * p..(j + 1) * p];
            let known: f64 = (j + 1..n_unknowns).map(|k| row[k] * x[k]).sum();
            x[j] = (row[n_unknowns] - known) / row[j];
        }
        let ss_res = r[p * p - 1].powi(2);

        // H cos(arg − G) = H cos G cos(arg) + H sin G sin(arg)
        let constant = |name, j: usize, ratio: f64, offset: f64, inferred| {
            let (c, s) = (x[1 + 2 * j], x[2 + 2 * j]);
            ReferenceConstant {
                name,
                amplitude: ratio * c.hypot(s),
                lag_deg: (s.atan2(c).to_degrees() + offset).rem_euclid(360.0),
                inferred,
            }
        };
        let mut constants: Vec<ReferenceConstant> = names
            .iter()
            .enumerate()
            .map(|(j, &name)| constant(name, j, 1.0, 0.0, false))
            .collect();
        constants.extend(attached.iter().map(|&(j, inf, _)| {
            constant(inf.name, j, inf.amplitude_ratio, inf.lag_offset_deg, true)
        }));

        let mean_value = values.iter().sum::<f64>() / n as f64;
        let ss_tot: f64 = values.iter().map(|v| (v - mean_value).powi(2)).sum();
        Ok(ReferenceFit {
            mean: x[0],
            constants,
            r_squared: if ss_tot > 0.0 {
                1.0 - ss_res / ss_tot
            } else {
                1.0
            },
            residual_rms: (ss_res / n as f64).sqrt(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthesize a series from reference constants on `clock` with the
    /// corrections the fit uses, `hours` hourly samples from `t0`.
    fn synthesize(t0: f64, hours: usize, mean: f64, constants: &[(&str, f64, f64)]) -> Vec<f64> {
        let clock = ModelClock::new(t0);
        let t_mid = 0.5 * (hours - 1) as f64 * 3600.0;
        (0..hours)
            .map(|i| {
                let t = i as f64 * 3600.0;
                mean + constants
                    .iter()
                    .map(|&(name, h, g)| {
                        let n = clock.nodal_correction(name, t_mid).unwrap();
                        let omega = 2.0 * std::f64::consts::PI / constituent_period(name).unwrap();
                        n.f * h * (omega * t + n.phase_offset_rad() - g.to_radians()).cos()
                    })
                    .sum::<f64>()
            })
            .collect()
    }

    fn lag_error(a: f64, b: f64) -> f64 {
        ((a - b + 180.0).rem_euclid(360.0) - 180.0).abs()
    }

    /// A 30-day record with P1 and K2 at their equilibrium ratios: the fit
    /// with inference recovers every constant, inferred ones included.
    #[test]
    fn recovers_reference_constants_with_inference() {
        let t0 = ModelClock::at_datetime(2025, 6, 1, 0, 0).epoch_unix;
        let truth = [
            ("M2", 0.80, 283.0),
            ("S2", 0.28, 318.0),
            ("N2", 0.17, 262.0),
            ("K1", 0.07, 190.0),
            ("O1", 0.05, 45.0),
            ("M4", 0.02, 110.0),
            ("P1", 0.331 * 0.07, 190.0),
            ("K2", 0.272 * 0.28, 318.0),
        ];
        let hours = 30 * 24;
        let values = synthesize(t0, hours, 0.12, &truth);
        let times: Vec<f64> = (0..hours).map(|i| t0 + i as f64 * 3600.0).collect();
        let fit = HarmonicAnalysis::new(&["M2", "S2", "N2", "K1", "O1", "M4"])
            .with_inferences(&[Inference::P1_FROM_K1, Inference::K2_FROM_S2])
            .fit(&times, &values)
            .unwrap();

        assert!((fit.mean - 0.12).abs() < 1e-9);
        assert!(fit.r_squared > 1.0 - 1e-12);
        for (name, h, g) in truth {
            let c = fit.get(name).unwrap();
            assert!(
                (c.amplitude - h).abs() < 1e-8,
                "{name}: H {} vs {h}",
                c.amplitude
            );
            assert!(
                lag_error(c.lag_deg, g) < 1e-6,
                "{name}: G {} vs {g}",
                c.lag_deg
            );
        }
        assert!(fit.get("P1").unwrap().inferred);

        // The prediction reproduces the record, inferred constituents included
        let predicted = fit.predict(&times);
        for (p, v) in predicted.iter().zip(&values) {
            assert!((p - v).abs() < 1e-8, "{p} vs {v}");
        }
    }

    /// A 60-day record of 10-minute samples with gaps and a 3 m datum, plus
    /// a deterministic off-tide wobble: the fit recovers the constants, and
    /// its residual (from the QR's last diagonal) equals the one computed
    /// from the prediction.
    #[test]
    fn residual_from_the_factor_matches_the_prediction() {
        let t0 = ModelClock::at_datetime(2025, 1, 1, 0, 0).epoch_unix;
        let truth: [(&str, f64, f64); 5] = [
            ("M2", 0.80, 283.0),
            ("S2", 0.28, 318.0),
            ("N2", 0.17, 262.0),
            ("K1", 0.07, 190.0),
            ("O1", 0.05, 45.0),
        ];
        let clock = ModelClock::new(t0);
        // Every sample but those in a 2-day gap every 9 days
        let times: Vec<f64> = (0..60 * 144)
            .map(|i| t0 + 600.0 * i as f64)
            .filter(|t| ((t - t0) / 86_400.0) % 9.0 >= 2.0)
            .collect();
        let t_mid = 0.5 * (times[times.len() - 1] - t0);
        let wobble = |t: f64| 0.02 * (2.0 * std::f64::consts::PI * t / (3.7 * 86_400.0)).sin();
        let values: Vec<f64> = times
            .iter()
            .map(|&tu| {
                let t = tu - t0;
                3.0 + wobble(t)
                    + truth
                        .iter()
                        .map(|&(name, h, g)| {
                            let c = clock.nodal_correction(name, t_mid).unwrap();
                            let omega =
                                2.0 * std::f64::consts::PI / constituent_period(name).unwrap();
                            c.f * h * (omega * t + c.phase_offset_rad() - g.to_radians()).cos()
                        })
                        .sum::<f64>()
            })
            .collect();
        let fit = HarmonicAnalysis::new(&["M2", "S2", "N2", "K1", "O1"])
            .fit(&times, &values)
            .unwrap();
        for (name, h, g) in truth {
            let c = fit.get(name).unwrap();
            // The wobble (3.7-day period) leaks a little into the tidal bands
            assert!((c.amplitude - h).abs() < 2e-3, "{name}: H {}", c.amplitude);
            assert!(lag_error(c.lag_deg, g) < 1.0, "{name}: G {}", c.lag_deg);
        }
        assert!((fit.mean - 3.0).abs() < 2e-3, "Z0 {}", fit.mean);
        let predicted = fit.predict(&times);
        let rms = (predicted
            .iter()
            .zip(&values)
            .map(|(p, v)| (p - v).powi(2))
            .sum::<f64>()
            / times.len() as f64)
            .sqrt();
        assert!(rms > 0.01, "the wobble is left in the residual: {rms}");
        assert!(
            (fit.residual_rms - rms).abs() < 1e-12,
            "{} vs {rms}",
            fit.residual_rms
        );
    }

    /// The same constants from two records of one tide that start at
    /// different times and are sampled differently: reference constants do
    /// not depend on the record (the old record-time fit's phases did).
    #[test]
    fn reference_constants_do_not_depend_on_the_record() {
        let truth = [
            ("M2", 0.75, 300.0),
            ("S2", 0.25, 340.0),
            ("K1", 0.06, 165.0),
        ];
        let fit_from = |(month, day, hour): (u32, u32, u32), hours: usize| {
            let t0 = ModelClock::at_datetime(2025, month, day, hour, 0).epoch_unix;
            let values = synthesize(t0, hours, 0.0, &truth);
            let times: Vec<f64> = (0..hours).map(|i| t0 + i as f64 * 3600.0).collect();
            HarmonicAnalysis::new(&["M2", "S2", "K1"])
                .fit(&times, &values)
                .unwrap()
        };
        let (a, b) = (fit_from((6, 1, 0), 16 * 24), fit_from((9, 17, 7), 20 * 24));
        for (name, h, g) in truth {
            for fit in [&a, &b] {
                let c = fit.get(name).unwrap();
                assert!((c.amplitude - h).abs() < 1e-8, "{name}: H {}", c.amplitude);
                assert!(lag_error(c.lag_deg, g) < 1e-6, "{name}: G {}", c.lag_deg);
            }
        }
    }

    /// Two weeks resolve M2/S2 and K1/O1 but not N2 (27.6 d) or P1 (182 d);
    /// a month adds N2 and Q1; a year everything. The inferences kept are
    /// those of left-out constituents with a kept source.
    #[test]
    fn resolvable_constituents_follow_the_rayleigh_criterion() {
        let names = [
            "M2", "S2", "K1", "O1", "N2", "Q1", "P1", "K2", "M4", "MS4", "MN4",
        ];
        let day = 86_400.0;
        let resolvable = |days: f64, rayleigh| {
            HarmonicAnalysis::resolvable(&names, days * day, rayleigh, &Inference::EQUILIBRIUM)
        };
        let two_weeks = resolvable(15.0, 1.0);
        assert_eq!(two_weeks.names(), ["M2", "S2", "K1", "O1", "M4", "MS4"]);
        let inferred: Vec<_> = two_weeks.inferences().iter().map(|i| i.name).collect();
        assert_eq!(inferred, ["P1", "K2", "N2", "Q1"]);
        let month = resolvable(30.0, 1.0);
        assert_eq!(
            month.names(),
            ["M2", "S2", "K1", "O1", "N2", "Q1", "M4", "MS4", "MN4"]
        );
        let inferred: Vec<_> = month.inferences().iter().map(|i| i.name).collect();
        assert_eq!(inferred, ["P1", "K2"]);
        let year = resolvable(366.0, 1.0);
        assert_eq!(year.names(), names);
        assert!(year.inferences().is_empty());
        // Stricter factor drops the pairs that barely separate
        assert_eq!(
            resolvable(30.0, 1.5).names(),
            ["M2", "S2", "K1", "O1", "M4", "MS4"]
        );
        assert!(resolvable(30.0, 1.0).minimum_record_length() <= 30.0 * day);
        assert!(resolvable(366.0, 1.0).minimum_record_length() > 182.0 * day);
    }

    /// An inference from a reference fit carries the reference's ratio and
    /// lag difference.
    #[test]
    fn inference_from_reference_fit() {
        let constant = |name, amplitude, lag_deg| ReferenceConstant {
            name,
            amplitude,
            lag_deg,
            inferred: false,
        };
        let reference = ReferenceFit {
            mean: 0.0,
            constants: vec![constant("M2", 0.8, 300.0), constant("N2", 0.16, 280.0)],
            r_squared: 1.0,
            residual_rms: 0.0,
        };
        let inf = Inference::from_reference("N2", "M2", &reference).unwrap();
        assert!((inf.amplitude_ratio - 0.2).abs() < 1e-12);
        assert!((inf.lag_offset_deg + 20.0).abs() < 1e-12);
        assert!(Inference::from_reference("Q1", "O1", &reference).is_none());
        assert_eq!(Inference::N2_FROM_M2.or_from(Some(&reference)), inf);
        assert_eq!(
            Inference::Q1_FROM_O1.or_from(Some(&reference)),
            Inference::Q1_FROM_O1
        );
        assert_eq!(Inference::N2_FROM_M2.or_from(None), Inference::N2_FROM_M2);
    }

    /// What cannot be fitted is an error, not a panic or a garbage fit.
    #[test]
    fn unfittable_requests_are_errors() {
        let t0 = 1.7e9;
        let times: Vec<f64> = (0..720).map(|i| t0 + i as f64 * 3600.0).collect();
        let values = vec![0.0; 720];
        let fit = |names: &[&'static str], inferences: &[Inference]| {
            HarmonicAnalysis::new(names)
                .with_inferences(inferences)
                .fit(&times, &values)
                .unwrap_err()
        };
        // K1 and P1 cannot be fitted independently from a month
        assert!(matches!(
            fit(&["K1", "P1"], &[]),
            AnalysisError::NotSeparable {
                a: "K1",
                b: "P1",
                ..
            }
        ));
        assert_eq!(
            fit(&["M2"], &[Inference::P1_FROM_K1]),
            AnalysisError::InferenceSourceNotFitted {
                name: "P1",
                from: "K1"
            }
        );
        assert_eq!(
            fit(&["XX9"], &[]),
            AnalysisError::UnknownConstituent("XX9".into())
        );
        let few = HarmonicAnalysis::standard().fit(&times[..8], &values[..8]);
        assert_eq!(
            few.unwrap_err(),
            AnalysisError::TooFewSamples {
                samples: 8,
                unknowns: 9
            }
        );
        let mismatch = HarmonicAnalysis::standard().fit(&times, &values[..10]);
        assert_eq!(
            mismatch.unwrap_err(),
            AnalysisError::LengthMismatch {
                times: 720,
                values: 10
            }
        );
    }
}
