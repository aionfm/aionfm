use serde::{Deserialize, Serialize};

/// Named model size presets from the specification.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSizePreset {
    Tiny,
    Small,
    Base,
    Large,
    Experimental,
}

/// Core architectural configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub preset: ModelSizePreset,
    pub patch_len: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub memory_slots: usize,
    pub regime_vocab_size: usize,
    pub covariate_dim: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::small()
    }
}

impl ModelConfig {
    pub fn small() -> Self {
        Self {
            preset: ModelSizePreset::Small,
            patch_len: 32,
            hidden_dim: 256,
            num_layers: 8,
            num_heads: 8,
            memory_slots: 128,
            regime_vocab_size: 4096,
            covariate_dim: 64,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.patch_len == 0 {
            return Err("patch_len must be greater than zero".into());
        }
        if self.hidden_dim == 0 || self.num_layers == 0 || self.num_heads == 0 {
            return Err("hidden_dim, num_layers, and num_heads must be greater than zero".into());
        }
        if self.hidden_dim % self.num_heads != 0 {
            return Err("hidden_dim must be divisible by num_heads".into());
        }
        Ok(())
    }
}
