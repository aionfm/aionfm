use crate::{FeatureExtractor, Quantizer, ResidualCalculator, ResidualDescriptor, TokenAssignment};
use aionfm_utils::{AionResult, ValuePatch};
use serde::{Deserialize, Serialize};

/// Full tokenization output for one value patch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenizedPatch {
    pub entity_id: String,
    pub start_index: usize,
    pub descriptor: ResidualDescriptor,
    pub assignment: TokenAssignment,
}

/// Converts continuous value patches into discrete regime tokens.
#[derive(Clone, Debug)]
pub struct RegimeTokenizer<Q> {
    residuals: ResidualCalculator,
    features: FeatureExtractor,
    quantizer: Q,
}

impl<Q> RegimeTokenizer<Q>
where
    Q: Quantizer,
{
    pub fn new(quantizer: Q) -> Self {
        Self {
            residuals: ResidualCalculator,
            features: FeatureExtractor,
            quantizer,
        }
    }

    pub fn tokenize(&self, patch: &ValuePatch) -> AionResult<TokenizedPatch> {
        let residual = self.residuals.compute(patch)?;
        let descriptor = self.features.extract(&residual, &patch.mask.observed);
        let assignment = self.quantizer.assign(&descriptor)?;
        Ok(TokenizedPatch {
            entity_id: patch.entity_id.clone(),
            start_index: patch.start_index,
            descriptor,
            assignment,
        })
    }

    pub fn quantizer(&self) -> &Q {
        &self.quantizer
    }
}
