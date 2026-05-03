use aionfm_utils::{AionError, AionResult};
use serde::{Deserialize, Serialize};

/// Reversible normalization strategy.
pub trait Normalizer: Send + Sync {
    fn fit(values: &[f32]) -> AionResult<Self>
    where
        Self: Sized;
    fn transform(&self, values: &[f32]) -> Vec<f32>;
    fn inverse_transform(&self, values: &[f32]) -> Vec<f32>;
}

/// Normalization statistics stored for forecast inversion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NormalizationStats {
    pub mean: f32,
    pub scale: f32,
    pub observed_count: usize,
}

/// Standard score normalizer with a small-scale guard.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StandardNormalizer {
    pub stats: NormalizationStats,
}

impl Normalizer for StandardNormalizer {
    fn fit(values: &[f32]) -> AionResult<Self> {
        if values.is_empty() {
            return Err(AionError::Validation(
                "cannot fit normalizer on an empty series".into(),
            ));
        }
        let observed: Vec<f32> = values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        if observed.is_empty() {
            return Err(AionError::Validation(
                "cannot fit normalizer without finite values".into(),
            ));
        }
        let mean = observed.iter().sum::<f32>() / observed.len() as f32;
        let variance = observed
            .iter()
            .map(|value| {
                let centered = value - mean;
                centered * centered
            })
            .sum::<f32>()
            / observed.len() as f32;
        let scale = variance.sqrt().max(1e-6);
        Ok(Self {
            stats: NormalizationStats {
                mean,
                scale,
                observed_count: observed.len(),
            },
        })
    }

    fn transform(&self, values: &[f32]) -> Vec<f32> {
        values
            .iter()
            .map(|value| (value - self.stats.mean) / self.stats.scale)
            .collect()
    }

    fn inverse_transform(&self, values: &[f32]) -> Vec<f32> {
        values
            .iter()
            .map(|value| value * self.stats.scale + self.stats.mean)
            .collect()
    }
}

/// No-op normalizer for already-scaled inputs.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IdentityNormalizer;

impl Normalizer for IdentityNormalizer {
    fn fit(_values: &[f32]) -> AionResult<Self> {
        Ok(Self)
    }

    fn transform(&self, values: &[f32]) -> Vec<f32> {
        values.to_vec()
    }

    fn inverse_transform(&self, values: &[f32]) -> Vec<f32> {
        values.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_normalization_round_trips() {
        let values = vec![2.0, 4.0, 6.0];
        let normalizer = StandardNormalizer::fit(&values).unwrap();
        let restored = normalizer.inverse_transform(&normalizer.transform(&values));
        for (left, right) in values.iter().zip(restored) {
            assert!((left - right).abs() < 1e-5);
        }
    }
}
