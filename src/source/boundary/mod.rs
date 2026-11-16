//! Boundary treatment source terms.
//!
//! Sponge layers for absorbing boundaries.

mod sponge;

pub use sponge::{
    RectangularBoundary, SpongeDistanceFn, SpongeLayer2D, SpongeProfile, rectangular_sponge_fn,
};
