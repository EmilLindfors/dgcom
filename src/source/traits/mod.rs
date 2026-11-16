//! Abstract source term traits.
//!
//! - [`SourceTerm`]: 1D source term interface
//! - [`SourceTerm2D`]: 2D source term interface with context
//! - [`CombinedSource`], [`CombinedSource2D`]: Composition helpers

mod source_1d;
mod source_2d;

pub use source_1d::{CombinedSource, SourceTerm};
pub use source_2d::{
    CombinedSource2D, DeviceSource2D, ElementSources, SourceContext2D, SourceTerm2D, SourceTerms2D,
};
