//! Shared public contracts for the AionFM ecosystem.

pub mod error;
pub mod forecast;
pub mod types;
pub mod validation;

pub use error::{AionError, AionResult};
pub use forecast::*;
pub use types::*;
