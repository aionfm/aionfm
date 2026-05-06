use crate::MissingPolicy;
use aionfm_utils::{AionError, AionResult, ObservationMask, ValuePatch};
use serde::{Deserialize, Serialize};

/// Patch segmentation configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PatchConfig {
    pub patch_len: usize,
    pub stride: usize,
    #[serde(default)]
    pub drop_incomplete: bool,
    #[serde(default)]
    pub missing_policy: MissingPolicy,
}

impl Default for PatchConfig {
    fn default() -> Self {
        Self {
            patch_len: 32,
            stride: 16,
            drop_incomplete: true,
            missing_policy: MissingPolicy::Preserve,
        }
    }
}

/// Generates value patches for the continuous stream.
#[derive(Clone, Debug)]
pub struct PatchGenerator {
    config: PatchConfig,
}

impl PatchGenerator {
    pub fn new(config: PatchConfig) -> AionResult<Self> {
        if config.patch_len == 0 {
            return Err(AionError::Validation(
                "patch_len must be greater than zero".into(),
            ));
        }
        if config.stride == 0 {
            return Err(AionError::Validation(
                "stride must be greater than zero".into(),
            ));
        }
        Ok(Self { config })
    }

    pub fn generate(&self, entity_id: impl Into<String>, values: &[f32]) -> Vec<ValuePatch> {
        let entity_id = entity_id.into();
        let mut patches = Vec::new();
        let mut start = 0;
        while start < values.len() {
            let end = (start + self.config.patch_len).min(values.len());
            if self.config.drop_incomplete && end - start < self.config.patch_len {
                break;
            }
            let treatment = self.config.missing_policy.apply(&values[start..end]);
            let mut patch_values = treatment.values;
            let mut observed = treatment.observed;
            if !self.config.drop_incomplete && patch_values.len() < self.config.patch_len {
                let missing = self.config.patch_len - patch_values.len();
                patch_values.extend(std::iter::repeat(0.0).take(missing));
                observed.extend(std::iter::repeat(false).take(missing));
            }
            patches.push(ValuePatch {
                entity_id: entity_id.clone(),
                start_index: start,
                values: patch_values,
                mask: ObservationMask { observed },
            });
            start += self.config.stride;
        }
        patches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_overlapping_patches() {
        let generator = PatchGenerator::new(PatchConfig {
            patch_len: 4,
            stride: 2,
            drop_incomplete: true,
            missing_policy: MissingPolicy::Preserve,
        })
        .unwrap();
        let patches = generator.generate("entity", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(patches.len(), 2);
        assert_eq!(patches[1].start_index, 2);
    }
}
