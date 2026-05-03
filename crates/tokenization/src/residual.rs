use aionfm_utils::{AionError, AionResult, ValuePatch};
use serde::{Deserialize, Serialize};

/// Baseline and residual representation for a value patch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResidualPatch {
    pub entity_id: String,
    pub start_index: usize,
    pub baseline: f32,
    pub scale: f32,
    pub residuals: Vec<f32>,
}

/// Computes simple patch residuals. Future implementations can replace this with learned baselines.
#[derive(Clone, Debug, Default)]
pub struct ResidualCalculator;

impl ResidualCalculator {
    pub fn compute(&self, patch: &ValuePatch) -> AionResult<ResidualPatch> {
        if patch.values.is_empty() {
            return Err(AionError::Validation(
                "cannot tokenize an empty value patch".into(),
            ));
        }
        let observed: Vec<f32> = patch
            .values
            .iter()
            .zip(patch.mask.observed.iter().chain(std::iter::repeat(&true)))
            .filter_map(|(value, observed)| observed.then_some(*value))
            .filter(|value| value.is_finite())
            .collect();
        let baseline = if observed.is_empty() {
            0.0
        } else {
            observed.iter().sum::<f32>() / observed.len() as f32
        };
        let variance = if observed.is_empty() {
            0.0
        } else {
            observed
                .iter()
                .map(|value| {
                    let centered = value - baseline;
                    centered * centered
                })
                .sum::<f32>()
                / observed.len() as f32
        };
        let scale = variance.sqrt().max(1e-6);
        Ok(ResidualPatch {
            entity_id: patch.entity_id.clone(),
            start_index: patch.start_index,
            baseline,
            scale,
            residuals: patch
                .values
                .iter()
                .map(|value| (value - baseline) / scale)
                .collect(),
        })
    }
}
