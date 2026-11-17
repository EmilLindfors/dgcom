//! Tidal astronomy and nodal corrections.
//!
//! This module supplies the astronomical corrections a real tidal prediction
//! needs on top of catalogued amplitude/phase pairs: the equilibrium argument
//! `V₀` (astronomical phase at the prediction epoch) and the 18.61-year nodal
//! modulation `f`, `u`. See [`astronomy`] for the mathematical formulation,
//! and [`admittance`] for minor constituents inferred from major ones.
//!
//! # Applying corrections
//!
//! A corrected constituent predicts elevation as
//!
//! ```text
//! η(t) = f(t) · H · cos(ω t + V₀ + u(t) − G)
//! ```
//!
//! where `t` is elapsed time (seconds) from the epoch, `V₀` the equilibrium
//! argument there and `f`, `u` the nodal terms drifting with `t`
//! ([`NodalDrift`], from [`crate::time::ModelClock::nodal_drift`]). The
//! forcing paths evaluate them at each time (`HarmonicTide`,
//! `BoundaryTides`); none bakes a correction held constant into a
//! constituent's amplitude and phase, which over a long run freezes `u`.
//!
//! ```
//! use dgcom::tides::{AstronomicalArguments, nodal_correction};
//!
//! let astro = AstronomicalArguments::at_datetime(2024, 6, 1, 0, 0, 0.0);
//! let m2 = nodal_correction("M2", &astro).unwrap();
//! assert!((m2.f - 1.0).abs() < 0.04); // M2's nodal factor stays within ±3.7 %
//! ```

pub mod admittance;
pub mod astronomy;

pub use admittance::{AdmittanceInference, equilibrium_amplitude};
pub use astronomy::{
    AstronomicalArguments, CONSTITUENT_NAMES, NodalCorrection, NodalDrift, angular_speed,
    canonical_name, constituent_period, julian_date, nodal_correction,
};
