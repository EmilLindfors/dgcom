//! Errors of the harmonic analysis and the station comparisons.

use thiserror::Error;

/// Why a harmonic fit or a station comparison could not be made.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum AnalysisError {
    /// A constituent the tidal tables do not know.
    #[error("unknown constituent {0}")]
    UnknownConstituent(String),
    /// A constituent without a nodal correction.
    #[error("no nodal correction for {0}")]
    NoNodalCorrection(String),
    /// Times and values of different lengths.
    #[error("{times} times for {values} values")]
    LengthMismatch {
        /// Number of times
        times: usize,
        /// Number of values
        values: usize,
    },
    /// Fewer samples than unknowns of the fit.
    #[error("{samples} samples for {unknowns} unknowns")]
    TooFewSamples {
        /// Number of samples
        samples: usize,
        /// Number of unknowns (mean and two per fitted constituent)
        unknowns: usize,
    },
    /// Two fitted constituents the record is too short to separate
    /// (Rayleigh criterion), or one that is longer than the record: the
    /// mean is the constituent of speed zero, `Z0`, and the two need a
    /// record of the constituent's period.
    #[error(
        "{a} and {b} need a {needed_days:.1}-day record to separate (have {have_days:.1} days); \
         infer {b} or leave it out"
    )]
    NotSeparable {
        /// The earlier constituent, or `Z0` (the mean)
        a: &'static str,
        /// The later constituent: the one to infer or leave out
        b: &'static str,
        /// Their synodic period (days)
        needed_days: f64,
        /// The record's length (days)
        have_days: f64,
    },
    /// A record that resolves no candidate, from the mean or from the
    /// earlier candidates of its band: a fit would hold the mean alone.
    #[error(
        "a {have_hours:.1}-hour record resolves no constituent: the first candidate, \
         {first}, needs {needed_hours:.1} h"
    )]
    NothingResolved {
        /// The record's length (hours)
        have_hours: f64,
        /// The first candidate, which the record would fit first
        first: &'static str,
        /// Its period (hours): the record it needs
        needed_hours: f64,
    },
    /// An inference that rests on a constituent neither fitted nor inferred
    /// from fitted ones (or on itself, through a cycle of inferences).
    #[error("{name} is inferred from {from}, which is neither fitted nor inferred from what is")]
    InferenceSourceNotFitted {
        /// Inferred constituent
        name: &'static str,
        /// Its source
        from: &'static str,
    },
    /// Fewer than two observations inside the model's time span.
    #[error("{station}: fewer than two observations inside the model's time span")]
    TooFewPairedSamples {
        /// Station name
        station: String,
    },
    /// No model series at a station.
    #[error("{0}: no model series at the station")]
    NoModelSeries(String),
    /// No observations at a station.
    #[error("{0}: no observations")]
    NoObservations(String),
}
