use crate::{RegimeLevel, RegimeToken, RegimeVocabulary, ResidualDescriptor};
use aionfm_utils::{AionError, AionResult};
use serde::{Deserialize, Serialize};

/// Assigned token plus distance metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenAssignment {
    pub token_id: u32,
    pub label: String,
    pub distance: f32,
}

/// Quantizer abstraction for learned or fixed codebooks.
pub trait Quantizer: Send + Sync {
    fn assign(&self, descriptor: &ResidualDescriptor) -> AionResult<TokenAssignment>;
    fn vocabulary(&self) -> &RegimeVocabulary;
}

/// Simple nearest-centroid quantizer used as a deterministic skeleton.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CodebookQuantizer {
    vocabulary: RegimeVocabulary,
    centroids: Vec<(u32, [f32; 5])>,
}

impl CodebookQuantizer {
    pub fn new(vocabulary: RegimeVocabulary, centroids: Vec<(u32, [f32; 5])>) -> Self {
        Self {
            vocabulary,
            centroids,
        }
    }

    pub fn default_behavioral_codebook() -> Self {
        let mut vocabulary = RegimeVocabulary::default();
        let tokens = [
            (1, "stable", [0.0, 0.2, 0.0, 0.0, 0.0]),
            (2, "growth", [0.5, 0.3, 0.0, 0.6, 0.0]),
            (3, "decline", [-0.5, 0.3, 0.0, -0.6, 0.0]),
            (4, "volatile", [0.0, 1.0, 0.0, 0.0, 0.0]),
            (5, "missing_or_irregular", [0.0, 0.2, 0.0, 0.0, 0.6]),
        ];
        let centroids = tokens
            .iter()
            .map(|(id, label, centroid)| {
                vocabulary.insert(RegimeToken {
                    id: *id,
                    level: RegimeLevel::Meso,
                    label: (*label).into(),
                    description: None,
                });
                (*id, *centroid)
            })
            .collect();
        Self {
            vocabulary,
            centroids,
        }
    }
}

impl Default for CodebookQuantizer {
    fn default() -> Self {
        Self::default_behavioral_codebook()
    }
}

impl Quantizer for CodebookQuantizer {
    fn assign(&self, descriptor: &ResidualDescriptor) -> AionResult<TokenAssignment> {
        let vector = descriptor.as_vector();
        let (token_id, distance) = self
            .centroids
            .iter()
            .map(|(token_id, centroid)| {
                let distance = vector
                    .iter()
                    .zip(centroid)
                    .map(|(left, right)| {
                        let diff = left - right;
                        diff * diff
                    })
                    .sum::<f32>()
                    .sqrt();
                (*token_id, distance)
            })
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .ok_or_else(|| AionError::Validation("quantizer has no centroids".into()))?;
        let label = self
            .vocabulary
            .token(token_id)
            .map(|token| token.label.clone())
            .unwrap_or_else(|| format!("token_{token_id}"));
        Ok(TokenAssignment {
            token_id,
            label,
            distance,
        })
    }

    fn vocabulary(&self) -> &RegimeVocabulary {
        &self.vocabulary
    }
}
