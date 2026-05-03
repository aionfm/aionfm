//! Adaptation layer skeleton for domain adapters, calibration, and workflows.

pub mod adapter;
pub mod calibration;
pub mod registry;
pub mod workflow;

pub use adapter::*;
pub use calibration::*;
pub use registry::*;
pub use workflow::*;
