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
//! chooses what a record can fit. The mean is the constituent of speed
//! zero: a record shorter than a constituent's period cannot tell the two
//! apart, so `resolvable` leaves out Ssa under half a year and M2 under
//! 12.4 hours.
//!
//! A constituent left out is not gone from the water: its energy goes to
//! the fitted neighbour it cannot be told from, with a phase that turns over
//! their synodic period. ν2 is 0.19 of N2 and 206 days from it, so a month's
//! N2 is ±20 % off with the month; T2 does 6 % to S2 over a year. The minor
//! semidiurnals are therefore inferred too, each from two neighbours with
//! the weights of a linear admittance ([`Inference::by_admittance`],
//! [`Inference::theory`]), and a source may itself be inferred (a
//! fortnight's ν2 rests on an N2 inferred from M2). Where a longer record
//! at the same place exists, its own ratios replace theory's
//! ([`HarmonicAnalysis::fit_with_reference`]).
//!
//! An inference states that the constituent is in the signal. A model run
//! forced without ν2 has none, and inferring one gives its N2 the beat the
//! inference takes out of a gauge's: the candidates passed for a record are
//! the constituents it carries ([`HarmonicAnalysis::GAUGE_RECORD`] for a
//! gauge, [`HarmonicAnalysis::station_candidates`] of its forcing's for a
//! model).
//!
//! # Solution
//!
//! Each sample's row of the design matrix is rotated into the triangular
//! factor of `[A | y]` as it is formed (a Givens QR), so a fit needs O(m²)
//! memory for m unknowns whatever the record's length, and keeps the
//! conditioning of `A` rather than the square of it that the normal
//! equations `AᵀA x = Aᵀy` have.

use super::{AnalysisError, TimeSeries};
use crate::tides::{AdmittanceInference, constituent_period};
use crate::time::ModelClock;

/// One source of an [`Inference`]: its share of the inferred constant is
/// `amplitude_ratio · H` at the lag `G + lag_offset_deg` of `name`'s `(H, G)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InferenceSource {
    /// The constituent followed: a fitted one, or one that is itself
    /// inferred (ν2 from an N2 that a fortnight infers from M2).
    pub name: &'static str,
    /// Amplitude ratio to the source.
    pub amplitude_ratio: f64,
    /// Phase-lag offset from the source (degrees).
    pub lag_offset_deg: f64,
}

/// A constituent inferred from one other, `H = ratio · H_from`,
/// `G = G_from + lag_offset`, or from two: the sum of both shares as complex
/// constants `H e^{−iG}`, which is an admittance interpolated between two
/// references ([`AdmittanceInference`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Inference {
    /// Inferred constituent.
    pub name: &'static str,
    /// The constituent it follows.
    pub from: InferenceSource,
    /// A second one, whose share is added.
    pub and: Option<InferenceSource>,
}

impl Inference {
    /// P1 from K1 with the equilibrium amplitude ratio 0.331 (Cartwright &
    /// Tayler 1971) and equal lag.
    pub const P1_FROM_K1: Self = Self::single("P1", "K1", 0.331, 0.0);

    /// K2 from S2 with the equilibrium amplitude ratio 0.272 and equal lag.
    pub const K2_FROM_S2: Self = Self::single("K2", "S2", 0.272, 0.0);

    /// N2 from M2 with the equilibrium amplitude ratio 0.191 and equal lag
    /// (separable only from 27.6 days).
    pub const N2_FROM_M2: Self = Self::single("N2", "M2", 0.191, 0.0);

    /// Q1 from O1 with the equilibrium amplitude ratio 0.191 and equal lag
    /// (separable only from 27.6 days).
    pub const Q1_FROM_O1: Self = Self::single("Q1", "O1", 0.191, 0.0);

    /// The equilibrium inferences above.
    pub const EQUILIBRIUM: [Self; 4] = [
        Self::P1_FROM_K1,
        Self::K2_FROM_S2,
        Self::N2_FROM_M2,
        Self::Q1_FROM_O1,
    ];

    /// What theory gives the constituents a record does not resolve, with
    /// no longer record to take their ratios from:
    ///
    /// - [`Self::EQUILIBRIUM`]: P1, K2, N2 and Q1 (with N2's lag from the
    ///   age of the tide in [`HarmonicAnalysis::fit_with_reference`]);
    /// - 2N2, μ2, ν2, λ2, L2 and T2 by linear admittance between N2, M2 and
    ///   S2 ([`AdmittanceInference::MINORS`]), as tide models' predictions
    ///   add them (Ray's `perth`, OTPS).
    ///
    /// The admittance holds for the astronomical ones (ν2 is 0.99 of it at
    /// Bodø and Mausund). μ2 and L2 share their speeds with the compound
    /// tides 2MS2 and 2MN2, which shallow water makes: at Puerto Montt μ2
    /// lies 55° from its admittance. Inferring them still beats leaving
    /// them out, on open and shallow coasts alike, and a record fits them
    /// first of the six ([`HarmonicAnalysis::GAUGE_RECORD`]).
    pub fn theory() -> Vec<Self> {
        let mut rules = Self::EQUILIBRIUM.to_vec();
        rules.extend(
            AdmittanceInference::MINORS
                .iter()
                .filter_map(Self::by_admittance),
        );
        rules
    }

    /// `name` from `from` alone, with an amplitude ratio and a lag offset
    /// (degrees).
    pub const fn single(
        name: &'static str,
        from: &'static str,
        amplitude_ratio: f64,
        lag_offset_deg: f64,
    ) -> Self {
        Self {
            name,
            from: InferenceSource {
                name: from,
                amplitude_ratio,
                lag_offset_deg,
            },
            and: None,
        }
    }

    /// `admittance`'s constituent from its two references with the weights
    /// of the linear admittance (a negative weight, where the constituent
    /// lies outside its references, is a ratio at 180°); `None` as
    /// [`AdmittanceInference::weights`].
    pub fn by_admittance(admittance: &AdmittanceInference) -> Option<Self> {
        let (lower, upper) = admittance.weights()?;
        let source = |name, weight: f64| InferenceSource {
            name,
            amplitude_ratio: weight.abs(),
            lag_offset_deg: if weight < 0.0 { 180.0 } else { 0.0 },
        };
        Some(Self {
            name: admittance.name,
            from: source(admittance.lower, lower),
            and: (upper != 0.0).then(|| source(admittance.upper, upper)),
        })
    }

    /// The sources, one or two.
    pub fn sources(&self) -> impl Iterator<Item = &InferenceSource> {
        std::iter::once(&self.from).chain(&self.and)
    }

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
        Some(Self::single(
            name,
            from,
            a.amplitude / b.amplitude,
            a.lag_deg - b.lag_deg,
        ))
    }

    /// This inference from its first source alone, with the ratio and lag
    /// offset of `reference`, where it has both constituents; else
    /// unchanged.
    pub fn or_from(self, reference: Option<&ReferenceFit>) -> Self {
        reference
            .and_then(|r| Self::from_reference(self.name, self.from.name, r))
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
    /// Constituents of a tide gauge's record, in priority order: the major
    /// ones and the overtides, the minor semidiurnals, and the long-period
    /// ones, so that the seasonal and fortnightly signal does not leak into
    /// the tides. Of the minor ones μ2 comes before 2N2 and L2 before λ2:
    /// each pair is 206 days apart, and the first is the one theory cannot
    /// give in shallow water ([`Inference::theory`]), so it is fitted from
    /// 32 days and the other inferred.
    pub const GAUGE_RECORD: [&'static str; 21] = [
        "M2", "S2", "K1", "O1", "N2", "Q1", "K2", "P1", "M4", "MS4", "MN4", "M6", "NU2", "MU2",
        "2N2", "L2", "LDA2", "T2", "Mf", "Mm", "Ssa",
    ];

    /// The reference constants of a tide gauge's record (Unix times): the
    /// [`Self::GAUGE_RECORD`] constituents the record resolves (Rayleigh
    /// criterion 1), the others inferred from them by theory
    /// ([`Self::fit_with_reference`] with no reference).
    ///
    /// A gauge's water carries the minor semidiurnals whether the record
    /// separates them or not. Left out of a month's fit, ν2 moves N2 by
    /// ±20 % with the month and T2 moves S2 by 6 %; over a year ν2 still
    /// leaves 2 % in N2. Scored against UTide's fit of the year at 17
    /// Norwegian gauges, a 30-day fit's N2 is off by 30 mm without them and
    /// 7 mm with, its S2 by 15 and 7.5 mm, and the tide it predicts for the
    /// rest of the year by 4.8 and 2.9 cm; at six gauges on other coasts by
    /// 11.8 and 8.5 cm (`notes/validation.md`, P3.1 "Inference of the minor
    /// constituents"; `examples/tide_inference_skill.rs`).
    pub fn fit_gauge_record(times: &[f64], values: &[f64]) -> Result<ReferenceFit, AnalysisError> {
        Self::fit_with_reference(&Self::GAUGE_RECORD, times, values, None)
    }

    /// The eight major constituents, which an ocean model's tide is taken
    /// to carry where nothing says what it was forced with (NorKyst-800:
    /// these eight from TPXO).
    pub const MAJOR: [&'static str; 8] = ["M2", "S2", "K1", "O1", "N2", "Q1", "K2", "P1"];

    /// The overtides and compound tides of [`Self::GAUGE_RECORD`], which
    /// shallow water makes of the tide that enters: a model has them
    /// whatever its boundary carries.
    pub const OVERTIDES: [&'static str; 4] = ["M4", "MS4", "MN4", "M6"];

    /// The candidates of a model's record at a station, for
    /// [`Self::fit_with_reference`]: of [`Self::GAUGE_RECORD`], in its
    /// priority order, the constituents its forcing carries (`carried`,
    /// matched without case) and [`Self::OVERTIDES`].
    ///
    /// An inference states that the constituent is in the record, so the
    /// candidates are what the run was forced with, no more and no fewer:
    ///
    /// - a run without ν2 (a nest in NorKyst as it is) has none to infer,
    ///   and an inferred one gives its N2 the ±20 % beat that the inference
    ///   takes out of a gauge's;
    /// - a run with the minor semidiurnals (an atlas corrected from theory)
    ///   has that beat in a month's N2 unless they are inferred, and under
    ///   27.6 days M2 takes a share;
    /// - a run forced with M2 alone has no N2: inferred, it puts a tenth of
    ///   M2 into a fortnight's M2 (|ΔZ|, at any phase of the month).
    pub fn station_candidates(carried: &[&str]) -> Vec<&'static str> {
        Self::GAUGE_RECORD
            .into_iter()
            .filter(|name| {
                Self::OVERTIDES.contains(name)
                    || carried.iter().any(|c| c.eq_ignore_ascii_case(name))
            })
            .collect()
    }

    /// The reference constants of a record that carries `candidates` (Unix
    /// times): those the record resolves (Rayleigh criterion 1) fitted, the
    /// others inferred, where a rule of [`Inference::theory`] gives them,
    /// with the ratio and lag offset of `reference` (a longer record at the
    /// same place, e.g. the gauge's whole record for a model run) where it
    /// has both constituents, else theory's. A short record resolves few
    /// candidates, and their constants carry the unresolved constituents of
    /// their band: a day's M2 holds S2 (`notes/validation.md`, the
    /// 2026-06-12 Mausund outcome), which only a fortnight separates.
    ///
    /// Under a month the reference is what makes the fit: N2 is 22° from
    /// M2's lag on the Norwegian coast, not at equal lag, so a fortnight's
    /// M2 is off by 40 mm with N2 at its equilibrium ratio and by 11 mm
    /// with the gauge's own ratio from another year. Without a reference
    /// that has N2 and M2, a record that separates M2 from S2 is fitted
    /// twice ([`Self::with_the_age_of_the_tide`]): 15 mm.
    pub fn fit_with_reference(
        candidates: &[&'static str],
        times: &[f64],
        values: &[f64],
        reference: Option<&ReferenceFit>,
    ) -> Result<ReferenceFit, AnalysisError> {
        let length = times
            .first()
            .zip(times.last())
            .map_or(0.0, |(t0, t1)| t1 - t0);
        let inferences: Vec<Inference> = Inference::theory()
            .into_iter()
            .filter(|i| candidates.contains(&i.name))
            .map(|i| i.or_from(reference))
            .collect();
        let analysis = Self::resolvable(candidates, length, 1.0, &inferences);
        let fit = analysis.fit(times, values)?;
        match analysis.with_the_age_of_the_tide(&fit) {
            Some(aged) => aged.fit(times, values),
            None => Ok(fit),
        }
    }

    /// This analysis with an N2 inferred from M2 at equal lag
    /// ([`Inference::N2_FROM_M2`]) given the lag that `fit`, this
    /// analysis's own, implies: the lag of the semidiurnal tide taken
    /// linear in frequency through M2 and S2 (one age of the tide, Garrett
    /// & Munk 1971), `G(N2) = G(M2) − 0.536 (G(S2) − G(M2))`. `None`
    /// unless N2 is so inferred and M2 and S2 are fitted.
    ///
    /// The equilibrium ratio gives N2's amplitude within 10 %, but its lag
    /// is 15–35° before M2's on most coasts, which a fortnight's fit then
    /// puts into M2. The lag's slope is the same on both sides of M2 to a
    /// few degrees where the tide is not near an amphidrome: at 17
    /// Norwegian gauges the tide a 15-day fit predicts for the rest of the
    /// year is off by 6.6 cm at equal lag and by 3.8 cm so (worse by 0.3 cm
    /// at Tregde, whose M2 is 10 cm, and by 0.05 cm at Stavanger); at six
    /// gauges on other coasts by 14.4 and 10.7 cm (`notes/validation.md`,
    /// P3.1 "Inference of the minor constituents").
    pub fn with_the_age_of_the_tide(&self, fit: &ReferenceFit) -> Option<Self> {
        let n2 = self
            .inferences
            .iter()
            .position(|i| *i == Inference::N2_FROM_M2)?;
        let fitted = |name: &str| fit.get(name).filter(|c| !c.inferred);
        let (m2, s2) = (fitted("M2")?, fitted("S2")?);
        let speed = |name| constituent_period(name).map(|p| 1.0 / p);
        let (to_n2, to_s2) = (speed("N2")? - speed("M2")?, speed("S2")? - speed("M2")?);
        let m2_to_s2 = (s2.lag_deg - m2.lag_deg + 180.0).rem_euclid(360.0) - 180.0;
        let mut aged = self.clone();
        aged.inferences[n2].from.lag_offset_deg = to_n2 / to_s2 * m2_to_s2;
        Some(aged)
    }

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

    /// What a month's record resolves of the largest constituents: M2, S2,
    /// N2, K1 and O1 fitted, P1 inferred from K1.
    pub fn monthly() -> Self {
        Self::new(&["M2", "S2", "N2", "K1", "O1"]).with_inferences(&[Inference::P1_FROM_K1])
    }

    /// For a record of `length` seconds: the longest prefix-greedy subset of
    /// `candidates` it separates, and of `inferences` those whose
    /// constituent is left out and whose sources are fitted or inferred.
    ///
    /// Each candidate, in priority order, is kept if the record is at least
    /// `rayleigh` times its synodic period with every earlier candidate,
    /// kept or left out (`rayleigh` = 1 is the classical criterion; unknown
    /// names are skipped). One left out is still in the water, in the fit
    /// through its inference or not at all, and a later one fitted beside
    /// it takes what is missing under its own name:
    ///
    /// - a fortnight infers N2 from M2, so it does not fit 2N2 (13.8 days
    ///   from M2, 27.6 from N2), which then held N2's error and made the
    ///   predicted tide worse than with no minor constituent at all;
    /// - 14 days do not separate S2 from M2, so they do not fit K2 either
    ///   (13.7 days from M2, half a year from S2), nor 12 days Q1 without
    ///   O1.
    ///
    /// The mean is in every fit, and it is the constituent of speed zero:
    /// a candidate is separable from it once the record is `rayleigh` times
    /// its own period. Ssa (183 days) on a record of two months, or Mf on
    /// one of four days, is a trend beside the mean: the two cancel inside
    /// the record and the prediction leaves it at once.
    pub fn resolvable(
        candidates: &[&'static str],
        length: f64,
        rayleigh: f64,
        inferences: &[Inference],
    ) -> Self {
        let mut names: Vec<&'static str> = Vec::new();
        // The speeds of the mean and of the earlier candidates
        let mut earlier: Vec<f64> = vec![0.0];
        for &name in candidates {
            let Some(period) = constituent_period(name) else {
                continue;
            };
            let omega = 2.0 * std::f64::consts::PI / period;
            let separable = |other: &f64| {
                let synodic = 2.0 * std::f64::consts::PI / (omega - other).abs();
                length >= rayleigh * synodic
            };
            if earlier.iter().all(separable) {
                names.push(name);
            }
            earlier.push(omega);
        }
        // An inference is kept once each source is fitted or itself inferred
        // (the first of a name wins), so the passes repeat until none is added
        let mut inferred: Vec<Inference> = Vec::new();
        loop {
            let before = inferred.len();
            for i in inferences {
                let known =
                    |name: &str| names.contains(&name) || inferred.iter().any(|k| k.name == name);
                if !known(i.name) && i.sources().all(|s| known(s.name)) {
                    inferred.push(*i);
                }
            }
            if inferred.len() == before {
                break;
            }
        }
        Self {
            names,
            inferences: inferred,
        }
    }

    /// The fitted constituents that `inference` rests on, each with the
    /// amplitude ratio and lag offset (degrees) of its share: its sources,
    /// and the sources of those that are themselves inferred.
    fn shares(&self, inference: &Inference) -> Result<Vec<(usize, f64, f64)>, AnalysisError> {
        let mut shares = Vec::new();
        self.follow(inference, (1.0, 0.0), 0, inference.name, &mut shares)?;
        Ok(shares)
    }

    /// Add `via`'s sources to `shares`, scaled by `(ratio, lag offset)`;
    /// `depth` inferences deep in the chain from `name` (a chain longer
    /// than the inferences is a cycle).
    fn follow(
        &self,
        via: &Inference,
        (ratio, offset): (f64, f64),
        depth: usize,
        name: &'static str,
        shares: &mut Vec<(usize, f64, f64)>,
    ) -> Result<(), AnalysisError> {
        for source in via.sources() {
            let share = (
                ratio * source.amplitude_ratio,
                offset + source.lag_offset_deg,
            );
            if let Some(column) = self.names.iter().position(|&n| n == source.name) {
                shares.push((column, share.0, share.1));
                continue;
            }
            let next = self
                .inferences
                .iter()
                .find(|i| i.name == source.name)
                .filter(|_| depth < self.inferences.len())
                .ok_or(AnalysisError::InferenceSourceNotFitted {
                    name,
                    from: source.name,
                })?;
            self.follow(next, share, depth + 1, name, shares)?;
        }
        Ok(())
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

        // Inferred constituents: their astronomy, and their shares of the
        // fitted ones (column, amplitude ratio, lag offset in degrees and in
        // radians)
        type Attached = (&'static str, (f64, f64, f64), Vec<(usize, f64, f64, f64)>);
        let attached: Vec<Attached> = self
            .inferences
            .iter()
            .map(|inf| {
                let shares = self
                    .shares(inf)?
                    .into_iter()
                    .map(|(j, ratio, offset)| (j, ratio, offset, offset.to_radians()))
                    .collect();
                Ok((inf.name, astronomy(inf.name)?, shares))
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
            for (_, (omega, f, phase), shares) in &attached {
                for &(j, ratio, _, offset) in shares {
                    let arg = omega * t + phase - offset;
                    w[1 + 2 * j] += ratio * f * arg.cos();
                    w[2 + 2 * j] += ratio * f * arg.sin();
                }
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
        constants.extend(attached.iter().map(|(name, _, shares)| match shares[..] {
            [(j, ratio, offset_deg, _)] => constant(*name, j, ratio, offset_deg, true),
            // The shares' sum as complex constants H e^{−iG}
            _ => {
                let (re, im) =
                    shares
                        .iter()
                        .fold((0.0, 0.0), |(re, im), &(j, ratio, _, offset)| {
                            let h = ratio * x[1 + 2 * j].hypot(x[2 + 2 * j]);
                            let g = x[2 + 2 * j].atan2(x[1 + 2 * j]) + offset;
                            (re + h * g.cos(), im + h * g.sin())
                        });
                ReferenceConstant {
                    name,
                    amplitude: re.hypot(im),
                    lag_deg: im.atan2(re).to_degrees().rem_euclid(360.0),
                    inferred: true,
                }
            }
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
        assert!((inf.from.amplitude_ratio - 0.2).abs() < 1e-12);
        assert!((inf.from.lag_offset_deg + 20.0).abs() < 1e-12);
        assert!(inf.and.is_none());
        assert!(Inference::from_reference("Q1", "O1", &reference).is_none());
        assert_eq!(Inference::N2_FROM_M2.or_from(Some(&reference)), inf);
        assert_eq!(
            Inference::Q1_FROM_O1.or_from(Some(&reference)),
            Inference::Q1_FROM_O1
        );
        assert_eq!(Inference::N2_FROM_M2.or_from(None), Inference::N2_FROM_M2);
    }

    /// The semidiurnal constants of a tide whose minor constituents follow
    /// the linear admittance of N2, M2 and S2.
    fn tide_with_minors() -> Vec<(&'static str, f64, f64)> {
        let mut truth = vec![
            ("M2", 0.74, 301.0),
            ("S2", 0.25, 339.0),
            ("N2", 0.15, 279.0),
            ("K1", 0.06, 165.0),
            ("O1", 0.05, 38.0),
        ];
        let of = |truth: &[(&str, f64, f64)], name| {
            let c = truth.iter().find(|c| c.0 == name).unwrap();
            (c.1, c.2)
        };
        for admittance in AdmittanceInference::MINORS {
            let (h, g) = admittance
                .infer(of(&truth, admittance.lower), of(&truth, admittance.upper))
                .unwrap();
            truth.push((admittance.name, h, g));
        }
        truth
    }

    /// A month cannot separate ν2 from N2 (206 days), so a fit without ν2
    /// folds it into N2: N2 swings by ν2's share of it, ±19 %, with the
    /// month (the Norwegian gauges' monthly N2, `notes/validation.md` P3.1).
    /// With the minor constituents inferred by admittance from N2, M2 and S2
    /// every month returns the same constants, the inferred ones included.
    #[test]
    fn minor_semidiurnals_inferred_by_admittance_stop_the_monthly_beat_of_n2() {
        let truth = tide_with_minors();
        let minors: Vec<Inference> = AdmittanceInference::MINORS
            .iter()
            .filter_map(Inference::by_admittance)
            .collect();
        assert_eq!(minors.len(), 6);
        let names = ["M2", "S2", "N2", "K1", "O1"];
        let (mut lowest, mut highest) = (f64::MAX, f64::MIN);
        for month in [1, 3, 5, 7, 9, 11] {
            let t0 = ModelClock::at_datetime(2025, month, 1, 0, 0).epoch_unix;
            let hours = 30 * 24;
            let values = synthesize(t0, hours, 0.0, &truth);
            let times: Vec<f64> = (0..hours).map(|i| t0 + i as f64 * 3600.0).collect();

            let without = HarmonicAnalysis::new(&names).fit(&times, &values).unwrap();
            let n2 = without.get("N2").unwrap().amplitude / 0.15;
            (lowest, highest) = (lowest.min(n2), highest.max(n2));

            let fit = HarmonicAnalysis::new(&names)
                .with_inferences(&minors)
                .fit(&times, &values)
                .unwrap();
            for &(name, h, g) in &truth {
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
                assert_eq!(c.inferred, !names.contains(&name));
            }
            assert!(fit.residual_rms < 1e-8);
        }
        assert!(
            lowest < 0.87 && highest > 1.13,
            "N2 without ν2: {lowest:.3} to {highest:.3} of the truth"
        );
    }

    /// A fortnight infers N2 from M2, and ν2 from that N2 and M2: an
    /// inference may rest on an inferred constituent, and its constant is
    /// the admittance of the constants the fit returns.
    #[test]
    fn an_inference_follows_an_inferred_source_to_the_fitted_ones() {
        let (ratio, offset) = (0.2, -22.0);
        let mut truth = vec![
            ("M2", 0.74, 301.0),
            ("S2", 0.25, 339.0),
            ("N2", ratio * 0.74, 301.0 + offset),
            ("K1", 0.06, 165.0),
            ("O1", 0.05, 38.0),
        ];
        let nu2 = AdmittanceInference::new("NU2", "N2", "M2");
        let (h, g) = nu2.infer((truth[2].1, truth[2].2), (0.74, 301.0)).unwrap();
        truth.push(("NU2", h, g));

        let t0 = ModelClock::at_datetime(2025, 6, 1, 0, 0).epoch_unix;
        let hours = 15 * 24;
        let values = synthesize(t0, hours, 0.0, &truth);
        let times: Vec<f64> = (0..hours).map(|i| t0 + i as f64 * 3600.0).collect();
        // ν2 is listed before the N2 it rests on
        let inferences = [
            Inference::by_admittance(&nu2).unwrap(),
            Inference::single("N2", "M2", ratio, offset),
        ];
        let analysis = HarmonicAnalysis::resolvable(
            &["M2", "S2", "K1", "O1", "N2", "NU2"],
            times[hours - 1] - t0,
            1.0,
            &inferences,
        );
        assert_eq!(analysis.names(), ["M2", "S2", "K1", "O1"]);
        let inferred: Vec<_> = analysis.inferences().iter().map(|i| i.name).collect();
        assert_eq!(inferred, ["N2", "NU2"]);
        let fit = analysis.fit(&times, &values).unwrap();
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
    }

    /// What a gauge's record fits and infers of [`HarmonicAnalysis::GAUGE_RECORD`]
    /// with theory's rules: a fortnight fits no minor semidiurnal (2N2 is
    /// 13.8 days from M2, but 27.6 from the N2 it infers); a month none
    /// either; 35 days fit μ2 and L2 and infer their partners; a year and a
    /// day fits them all.
    #[test]
    fn a_gauge_record_fits_the_minor_semidiurnals_it_separates() {
        const MINOR: [&str; 6] = ["NU2", "MU2", "2N2", "L2", "LDA2", "T2"];
        let of = |days: f64| {
            let analysis = HarmonicAnalysis::resolvable(
                &HarmonicAnalysis::GAUGE_RECORD,
                days * 86_400.0,
                1.0,
                &Inference::theory(),
            );
            let minor = |names: Vec<&'static str>| -> Vec<&'static str> {
                names.into_iter().filter(|n| MINOR.contains(n)).collect()
            };
            (
                minor(analysis.names().to_vec()),
                minor(analysis.inferences().iter().map(|i| i.name).collect()),
                analysis.names().contains(&"N2"),
            )
        };
        let all_inferred = ["2N2", "MU2", "NU2", "LDA2", "L2", "T2"];
        let (fitted, inferred, n2) = of(15.0);
        assert!(fitted.is_empty() && !n2, "{fitted:?}");
        assert_eq!(inferred, all_inferred);
        let (fitted, inferred, n2) = of(30.0);
        assert!(fitted.is_empty() && n2, "{fitted:?}");
        assert_eq!(inferred, all_inferred);
        let (fitted, inferred, _) = of(35.0);
        assert_eq!(fitted, ["MU2", "L2"]);
        assert_eq!(inferred, ["2N2", "NU2", "LDA2", "T2"]);
        let (fitted, inferred, _) = of(366.0);
        assert_eq!(fitted, MINOR);
        assert!(inferred.is_empty());
    }

    /// A long-period constituent is fitted once the record is as long as its
    /// period (the mean is the constituent of speed zero): Mf from 13.7
    /// days, Mm from 27.6, Ssa from 183. On 40 days whose level sags 2 cm
    /// in the middle (weather), a fitted Ssa is that curve, over three times
    /// as high, and the mean goes with it.
    #[test]
    fn a_constituent_longer_than_the_record_is_left_out() {
        let long = |days: f64| -> Vec<&'static str> {
            HarmonicAnalysis::resolvable(
                &HarmonicAnalysis::GAUGE_RECORD,
                days * 86_400.0,
                1.0,
                &Inference::theory(),
            )
            .names()
            .iter()
            .copied()
            .filter(|n| ["Mf", "Mm", "Ssa"].contains(n))
            .collect()
        };
        assert!(long(4.0).is_empty());
        assert_eq!(long(15.0), ["Mf"]);
        assert_eq!(long(30.0), ["Mf", "Mm"]);
        assert_eq!(long(90.0), ["Mf", "Mm"]);
        assert_eq!(long(183.0), ["Mf", "Mm", "Ssa"]);
        // Twelve hours fit no tide at all (S2's period, but M2 is in the
        // water beside it), a day M2 alone
        let short = |hours: f64| {
            HarmonicAnalysis::resolvable(&HarmonicAnalysis::MAJOR, hours * 3600.0, 1.0, &[])
                .names()
                .to_vec()
        };
        assert!(short(12.0).is_empty());
        assert_eq!(short(24.0), ["M2"]);
        // A candidate left out keeps out the later ones the record cannot
        // tell from it: Q1 without O1 (12 days), K2 without S2 (14 days)
        assert_eq!(short(12.0 * 24.0), ["M2", "K1"]);
        assert_eq!(short(14.0 * 24.0), ["M2", "K1", "O1"]);
        assert_eq!(short(15.0 * 24.0), ["M2", "S2", "K1", "O1"]);

        let t0 = ModelClock::at_datetime(2025, 3, 1, 0, 0).epoch_unix;
        let hours = 40 * 24;
        let half = 0.5 * (hours - 1) as f64;
        let truth = [("M2", 0.74, 301.0), ("S2", 0.25, 339.0)];
        let values: Vec<f64> = synthesize(t0, hours, 0.3, &truth)
            .into_iter()
            .enumerate()
            .map(|(i, v)| v + 0.02 * ((i as f64 - half) / half).powi(2))
            .collect();
        let times: Vec<f64> = (0..hours).map(|i| t0 + i as f64 * 3600.0).collect();
        // The mean of the level without the tide: the sag's is a third of it
        let level = 0.3 + 0.02 / 3.0;

        let with_ssa = HarmonicAnalysis::new(&["M2", "S2", "Ssa"])
            .fit(&times, &values)
            .unwrap();
        assert!(
            with_ssa.get("Ssa").unwrap().amplitude > 0.06 && (with_ssa.mean - level).abs() > 0.06,
            "Ssa {}, mean {} vs {level}",
            with_ssa.get("Ssa").unwrap().amplitude,
            with_ssa.mean
        );
        let fit = HarmonicAnalysis::fit_with_reference(&["M2", "S2", "Ssa"], &times, &values, None)
            .unwrap();
        assert!(fit.get("Ssa").is_none());
        assert!(
            (fit.mean - level).abs() < 1e-3,
            "mean {} vs {level}",
            fit.mean
        );
    }

    /// A model's record is fitted with what its forcing carries
    /// (`station_candidates`). Inferring the minor semidiurnals in a run
    /// without them, or leaving them out of a run that has them, puts ν2's
    /// 19 % of N2 into a month's N2; and a run forced with M2 alone has no
    /// N2 to infer (inferred, a fortnight's M2 is a tenth off).
    #[test]
    fn a_models_record_is_fitted_with_what_its_forcing_carries() {
        assert_eq!(
            HarmonicAnalysis::station_candidates(&HarmonicAnalysis::MAJOR),
            [
                "M2", "S2", "K1", "O1", "N2", "Q1", "K2", "P1", "M4", "MS4", "MN4", "M6"
            ]
        );
        assert_eq!(
            HarmonicAnalysis::station_candidates(&["m2", "S2", "nu2", "Z0", "Mf"]),
            ["M2", "S2", "M4", "MS4", "MN4", "M6", "NU2", "Mf"]
        );

        // The majors with K2 and P1 at their equilibrium ratios, and the same
        // with the minor semidiurnals
        let mut with_minors = tide_with_minors();
        with_minors.extend([
            ("K2", 0.272 * 0.25, 339.0),
            ("P1", 0.331 * 0.06, 165.0),
            ("Q1", 0.01, 350.0),
        ]);
        let minor = |name: &str| AdmittanceInference::MINORS.iter().any(|a| a.name == name);
        let majors: Vec<_> = with_minors
            .iter()
            .copied()
            .filter(|c| !minor(c.0))
            .collect();
        let carried = |truth: &[(&'static str, f64, f64)]| -> Vec<&'static str> {
            truth.iter().map(|c| c.0).collect()
        };
        let hours = 30 * 24;
        // The largest error of N2's amplitude over six months of the year,
        // as a share of it
        let n2_error = |truth: &[(&'static str, f64, f64)], candidates: &[&'static str]| {
            [1, 3, 5, 7, 9, 11]
                .into_iter()
                .map(|month| {
                    let t0 = ModelClock::at_datetime(2025, month, 1, 0, 0).epoch_unix;
                    let values = synthesize(t0, hours, 0.0, truth);
                    let times: Vec<f64> = (0..hours).map(|i| t0 + i as f64 * 3600.0).collect();
                    let fit =
                        HarmonicAnalysis::fit_with_reference(candidates, &times, &values, None)
                            .unwrap();
                    (fit.get("N2").unwrap().amplitude / 0.15 - 1.0).abs()
                })
                .fold(0.0, f64::max)
        };
        let of = |truth: &[(&'static str, f64, f64)]| {
            HarmonicAnalysis::station_candidates(&carried(truth))
        };
        for truth in [&majors, &with_minors] {
            let own = n2_error(truth, &of(truth));
            assert!(own < 1e-7, "N2 with the forcing's constituents: {own}");
        }
        let (inferred, left_out) = (
            n2_error(&majors, &of(&with_minors)),
            n2_error(&with_minors, &of(&majors)),
        );
        assert!(
            inferred > 0.1 && left_out > 0.1,
            "N2 with minors inferred that the run lacks {inferred}, left out of one that has \
             them {left_out}"
        );

        // M2 alone over a fortnight, from six days of the month: the largest
        // |ΔZ| of M2 as a share of it. An N2 inferred beside it takes a tenth
        // (9.5 to 11.7 % over the six)
        let hours = 15 * 24;
        let m2 = |candidates: &[&'static str]| {
            [1, 6, 11, 16, 21, 26]
                .into_iter()
                .map(|day| {
                    let t0 = ModelClock::at_datetime(2025, 6, day, 0, 0).epoch_unix;
                    let values = synthesize(t0, hours, 0.0, &[("M2", 0.74, 301.0)]);
                    let times: Vec<f64> = (0..hours).map(|i| t0 + i as f64 * 3600.0).collect();
                    let fit =
                        HarmonicAnalysis::fit_with_reference(candidates, &times, &values, None)
                            .unwrap();
                    let c = fit.get("M2").unwrap();
                    let (a, b) = (c.lag_deg.to_radians(), 301.0_f64.to_radians());
                    let dz = (c.amplitude * a.cos() - 0.74 * b.cos())
                        .hypot(c.amplitude * a.sin() - 0.74 * b.sin());
                    dz / 0.74
                })
                .fold(0.0, f64::max)
        };
        assert!(m2(&HarmonicAnalysis::station_candidates(&["M2"])) < 1e-9);
        let with_n2 = m2(&HarmonicAnalysis::station_candidates(
            &HarmonicAnalysis::MAJOR,
        ));
        assert!(
            with_n2 > 0.09,
            "M2 with an N2 inferred beside it: {with_n2}"
        );
    }

    /// A fortnight cannot fit N2. With N2 at M2's lag (the equilibrium
    /// inference) the 21° that N2 really leads M2 by go into M2 and S2;
    /// with the lag the M2-to-S2 slope implies (`fit_gauge_record`'s second
    /// pass), a tide whose lag is linear in frequency is recovered.
    #[test]
    fn a_fortnights_n2_takes_its_lag_from_the_age_of_the_tide() {
        let (m2, s2) = ((0.74, 301.0), (0.25, 339.0));
        let n2_lag = m2.1 - 0.536 * (s2.1 - m2.1);
        let truth = [
            ("M2", m2.0, m2.1),
            ("S2", s2.0, s2.1),
            ("N2", 0.191 * m2.0, n2_lag),
            ("K1", 0.06, 165.0),
            ("O1", 0.05, 38.0),
        ];
        let t0 = ModelClock::at_datetime(2025, 6, 1, 0, 0).epoch_unix;
        let hours = 15 * 24;
        let values = synthesize(t0, hours, 0.0, &truth);
        let times: Vec<f64> = (0..hours).map(|i| t0 + i as f64 * 3600.0).collect();
        let error = |fit: &ReferenceFit, name: &str| {
            let (c, t) = (
                fit.get(name).unwrap(),
                truth.iter().find(|t| t.0 == name).unwrap(),
            );
            let (a, b) = (c.lag_deg.to_radians(), t.2.to_radians());
            (c.amplitude * a.cos() - t.1 * b.cos()).hypot(c.amplitude * a.sin() - t.1 * b.sin())
        };
        let candidates = ["M2", "S2", "K1", "O1", "N2"];
        let equal_lag = HarmonicAnalysis::resolvable(
            &candidates,
            times[hours - 1] - t0,
            1.0,
            &Inference::EQUILIBRIUM,
        );
        let first = equal_lag.fit(&times, &values).unwrap();
        assert!(
            error(&first, "N2") > 0.04 && error(&first, "M2") > 0.01,
            "at equal lag: N2 {}, M2 {}",
            error(&first, "N2"),
            error(&first, "M2")
        );

        let aged =
            HarmonicAnalysis::fit_with_reference(&candidates, &times, &values, None).unwrap();
        assert!(aged.get("N2").unwrap().inferred);
        assert!(
            lag_error(aged.get("N2").unwrap().lag_deg, n2_lag) < 3.0,
            "N2 lag {} vs {n2_lag}",
            aged.get("N2").unwrap().lag_deg
        );
        for name in ["M2", "S2", "N2"] {
            assert!(
                error(&aged, name) < 0.2 * error(&first, name),
                "{name}: {} vs {} at equal lag",
                error(&aged, name),
                error(&first, name)
            );
        }
        // A month fits N2: nothing to age
        let month = HarmonicAnalysis::resolvable(
            &candidates,
            30.0 * 86_400.0,
            1.0,
            &Inference::EQUILIBRIUM,
        );
        assert!(month.with_the_age_of_the_tide(&first).is_none());
        // Nor is a ratio from a reference touched
        let mut referenced = equal_lag.clone();
        let n2 = referenced
            .inferences
            .iter_mut()
            .find(|i| i.name == "N2")
            .unwrap();
        *n2 = Inference::single("N2", "M2", 0.2, -20.0);
        assert!(referenced.with_the_age_of_the_tide(&first).is_none());
    }

    /// Inferences that rest on each other, or on nothing fitted, are an
    /// error; `resolvable` leaves them out.
    #[test]
    fn inferences_without_a_fitted_source_are_refused() {
        let times: Vec<f64> = (0..720).map(|i| 1.7e9 + i as f64 * 3600.0).collect();
        let values = vec![0.0; 720];
        let cycle = [
            Inference::single("N2", "NU2", 5.0, 0.0),
            Inference::single("NU2", "N2", 0.2, 0.0),
        ];
        let err = HarmonicAnalysis::new(&["M2"])
            .with_inferences(&cycle)
            .fit(&times, &values)
            .unwrap_err();
        assert!(matches!(
            err,
            AnalysisError::InferenceSourceNotFitted { name: "N2", .. }
        ));
        let nu2 = Inference::by_admittance(&AdmittanceInference::new("NU2", "N2", "M2")).unwrap();
        assert_eq!(
            HarmonicAnalysis::new(&["M2"])
                .with_inferences(&[nu2])
                .fit(&times, &values)
                .unwrap_err(),
            AnalysisError::InferenceSourceNotFitted {
                name: "NU2",
                from: "N2"
            }
        );
        let kept = HarmonicAnalysis::resolvable(&["M2"], 30.0 * 86_400.0, 1.0, &cycle);
        assert!(kept.inferences().is_empty());
        let kept = HarmonicAnalysis::resolvable(&["M2"], 30.0 * 86_400.0, 1.0, &[nu2]);
        assert!(kept.inferences().is_empty());
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
