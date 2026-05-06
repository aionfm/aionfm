//! Data layer for loading, normalizing, patching, and streaming series.

pub mod loader;
pub mod metadata;
pub mod missing;
pub mod normalization;
pub mod patch;
pub mod stream;

pub use loader::*;
pub use metadata::*;
pub use missing::*;
pub use normalization::*;
pub use patch::*;
pub use stream::*;
