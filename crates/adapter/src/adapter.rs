use aionfm_utils::AionResult;
use serde::{Deserialize, Serialize};

/// Adapter configuration for bottleneck modules inserted into model layers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdapterConfig {
    pub name: String,
    pub domain: String,
    pub bottleneck_dim: usize,
    pub target_layers: Vec<usize>,
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            name: "default".into(),
            domain: "general".into(),
            bottleneck_dim: 64,
            target_layers: vec![],
        }
    }
}

/// Adapter abstraction independent of a concrete tensor backend.
pub trait DomainAdapter: Send + Sync {
    fn config(&self) -> &AdapterConfig;
    fn transform_hidden(&self, hidden: &[f32]) -> AionResult<Vec<f32>>;
}

/// No-op bottleneck adapter skeleton.
#[derive(Clone, Debug)]
pub struct BottleneckAdapter {
    config: AdapterConfig,
}

impl BottleneckAdapter {
    pub fn new(config: AdapterConfig) -> Self {
        Self { config }
    }
}

impl DomainAdapter for BottleneckAdapter {
    fn config(&self) -> &AdapterConfig {
        &self.config
    }

    fn transform_hidden(&self, hidden: &[f32]) -> AionResult<Vec<f32>> {
        Ok(hidden.to_vec())
    }
}
