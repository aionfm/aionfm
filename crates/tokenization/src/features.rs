use crate::ResidualPatch;
use serde::{Deserialize, Serialize};

/// Behavioral descriptors used by the regime quantizer.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ResidualDescriptor {
    pub slope: f32,
    pub volatility: f32,
    pub curvature: f32,
    pub last_residual: f32,
    pub missing_ratio: f32,
}

impl ResidualDescriptor {
    pub fn as_vector(&self) -> [f32; 5] {
        [
            self.slope,
            self.volatility,
            self.curvature,
            self.last_residual,
            self.missing_ratio,
        ]
    }
}

/// Extracts slope, volatility, curvature, and missingness descriptors.
#[derive(Clone, Debug, Default)]
pub struct FeatureExtractor;

impl FeatureExtractor {
    pub fn extract(&self, patch: &ResidualPatch, observed: &[bool]) -> ResidualDescriptor {
        let values = &patch.residuals;
        if values.is_empty() {
            return ResidualDescriptor::default();
        }
        let first = values.first().copied().unwrap_or_default();
        let last = values.last().copied().unwrap_or_default();
        let slope = if values.len() <= 1 {
            0.0
        } else {
            (last - first) / (values.len() - 1) as f32
        };
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let volatility = (values
            .iter()
            .map(|value| {
                let centered = value - mean;
                centered * centered
            })
            .sum::<f32>()
            / values.len() as f32)
            .sqrt();
        let curvature = if values.len() < 3 {
            0.0
        } else {
            values
                .windows(3)
                .map(|window| window[2] - 2.0 * window[1] + window[0])
                .sum::<f32>()
                / (values.len() - 2) as f32
        };
        let missing_ratio = if observed.is_empty() {
            0.0
        } else {
            observed.iter().filter(|observed| !**observed).count() as f32 / observed.len() as f32
        };
        ResidualDescriptor {
            slope,
            volatility,
            curvature,
            last_residual: last,
            missing_ratio,
        }
    }
}
