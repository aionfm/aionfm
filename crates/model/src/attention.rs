/// Causal attention mask over a sequence of patches.
#[derive(Clone, Debug)]
pub struct CausalMask {
    pub len: usize,
}

impl CausalMask {
    pub fn allows(&self, query_index: usize, key_index: usize) -> bool {
        query_index < self.len && key_index <= query_index
    }
}

/// Placeholder causal transformer block stack.
#[derive(Clone, Debug)]
pub struct CausalTransformer {
    pub layers: usize,
    pub hidden_dim: usize,
}

impl CausalTransformer {
    pub fn forward(&self, embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
        embeddings.to_vec()
    }
}
