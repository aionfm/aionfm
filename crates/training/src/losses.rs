use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Multi-objective loss weights from the training specification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LossWeights {
    pub next_patch: f32,
    pub masked_reconstruction: f32,
    pub token_likelihood: f32,
    pub quantile: f32,
    pub contrastive: f32,
    pub calibration: f32,
    pub rollout: f32,
}

impl Default for LossWeights {
    fn default() -> Self {
        Self {
            next_patch: 1.0,
            masked_reconstruction: 0.5,
            token_likelihood: 0.5,
            quantile: 1.0,
            contrastive: 0.1,
            calibration: 0.1,
            rollout: 0.1,
        }
    }
}

/// Named loss values for logging and scheduling.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LossBreakdown {
    pub values: BTreeMap<String, f32>,
}

impl LossBreakdown {
    pub fn total(&self, weights: &LossWeights) -> f32 {
        self.values.get("next_patch").copied().unwrap_or_default() * weights.next_patch
            + self
                .values
                .get("masked_reconstruction")
                .copied()
                .unwrap_or_default()
                * weights.masked_reconstruction
            + self
                .values
                .get("token_likelihood")
                .copied()
                .unwrap_or_default()
                * weights.token_likelihood
            + self.values.get("quantile").copied().unwrap_or_default() * weights.quantile
            + self.values.get("contrastive").copied().unwrap_or_default() * weights.contrastive
            + self.values.get("calibration").copied().unwrap_or_default() * weights.calibration
            + self.values.get("rollout").copied().unwrap_or_default() * weights.rollout
    }
}

/// Placeholder loss aggregator to be wired to a tensor backend.
#[derive(Clone, Debug, Default)]
pub struct LossAggregator {
    pub weights: LossWeights,
}

impl LossAggregator {
    pub fn aggregate(&self, breakdown: &LossBreakdown) -> f32 {
        breakdown.total(&self.weights)
    }
}
