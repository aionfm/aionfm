//! Data layer skeleton for loading, normalizing, patching, and streaming series.

pub mod loader;
pub mod metadata;
pub mod normalization;
pub mod patch;
pub mod stream;

pub use loader::*;
pub use metadata::*;
pub use normalization::*;
pub use patch::*;
pub use stream::*;
