use serde::{Deserialize, Serialize};

/// Optimizer configuration independent of a concrete autodiff backend.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub gradient_clip_norm: Option<f32>,
    pub warmup_steps: usize,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            weight_decay: 0.01,
            gradient_clip_norm: Some(1.0),
            warmup_steps: 1_000,
        }
    }
}

/// Optimizer progress tracked for checkpointing.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OptimizerState {
    pub step: usize,
    pub epoch: usize,
}
