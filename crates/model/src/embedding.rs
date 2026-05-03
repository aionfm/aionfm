use aionfm_tokenization::TokenizedPatch;
use aionfm_utils::{CovariatePatch, ValuePatch};

/// Projects continuous value patches into a hidden representation.
#[derive(Clone, Debug)]
pub struct ValuePatchEncoder {
    hidden_dim: usize,
}

impl ValuePatchEncoder {
    pub fn new(hidden_dim: usize) -> Self {
        Self { hidden_dim }
    }

    pub fn encode(&self, patch: &ValuePatch) -> Vec<f32> {
        let mut embedding = vec![0.0; self.hidden_dim];
        if patch.values.is_empty() {
            return embedding;
        }
        for (index, value) in patch.values.iter().enumerate() {
            embedding[index % self.hidden_dim] += *value / patch.values.len() as f32;
        }
        embedding
    }
}

/// Embeds regime token assignments.
#[derive(Clone, Debug)]
pub struct RegimeEmbedding {
    hidden_dim: usize,
}

impl RegimeEmbedding {
    pub fn new(hidden_dim: usize) -> Self {
        Self { hidden_dim }
    }

    pub fn encode(&self, tokenized: &TokenizedPatch) -> Vec<f32> {
        let mut embedding = vec![0.0; self.hidden_dim];
        embedding[tokenized.assignment.token_id as usize % self.hidden_dim] = 1.0;
        embedding
    }
}

/// Encodes aligned covariate patches.
#[derive(Clone, Debug)]
pub struct CovariateEmbedding {
    hidden_dim: usize,
}

impl CovariateEmbedding {
    pub fn new(hidden_dim: usize) -> Self {
        Self { hidden_dim }
    }

    pub fn encode(&self, patch: &CovariatePatch) -> Vec<f32> {
        let mut embedding = vec![0.0; self.hidden_dim];
        for (index, values) in patch.covariates.values().enumerate() {
            let mean = if values.is_empty() {
                0.0
            } else {
                values.iter().sum::<f32>() / values.len() as f32
            };
            embedding[index % self.hidden_dim] += mean;
        }
        embedding
    }
}
