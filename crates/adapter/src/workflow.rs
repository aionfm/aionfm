use crate::{AdapterConfig, AdapterRegistry};
use aionfm_utils::{AdaptationRequest, AdaptationStatus, AionResult};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Supported adaptation strategies.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdaptationMode {
    ZeroShot,
    FewShotCalibration,
    AdapterTuning,
    FullFineTuning,
}

/// Coordinates adapter preparation and calibration.
#[derive(Clone, Debug, Default)]
pub struct AdaptationWorkflow {
    registry: AdapterRegistry,
}

impl AdaptationWorkflow {
    pub fn new(registry: AdapterRegistry) -> Self {
        Self { registry }
    }

    pub fn start(
        &mut self,
        request: AdaptationRequest,
        mode: AdaptationMode,
    ) -> AionResult<AdaptationStatus> {
        if matches!(mode, AdaptationMode::AdapterTuning) {
            let config = AdapterConfig {
                name: request
                    .adapter_name
                    .clone()
                    .unwrap_or_else(|| format!("{}_adapter", request.domain)),
                domain: request.domain.clone(),
                ..AdapterConfig::default()
            };
            self.registry.register(config);
        }
        Ok(AdaptationStatus {
            adaptation_id: Uuid::new_v4(),
            status: "accepted".into(),
            message: Some(format!("adaptation workflow queued in {mode:?} mode")),
        })
    }

    pub fn registry(&self) -> &AdapterRegistry {
        &self.registry
    }
}
