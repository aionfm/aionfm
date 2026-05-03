use aionfm_utils::{EntityMetadata, EntitySeries, Frequency};
use serde::{Deserialize, Serialize};

/// Synthetic trajectory generator configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyntheticConfig {
    pub entity_count: usize,
    pub length: usize,
    pub seasonal_period: usize,
    pub noise_scale: f32,
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            entity_count: 1,
            length: 128,
            seasonal_period: 24,
            noise_scale: 0.05,
        }
    }
}

/// Deterministic synthetic generator for examples and smoke tests.
#[derive(Clone, Debug)]
pub struct SyntheticSeriesGenerator {
    config: SyntheticConfig,
}

impl SyntheticSeriesGenerator {
    pub fn new(config: SyntheticConfig) -> Self {
        Self { config }
    }

    pub fn generate(&self) -> Vec<EntitySeries> {
        (0..self.config.entity_count)
            .map(|entity| {
                let values = (0..self.config.length)
                    .map(|step| {
                        let season = (step as f32 / self.config.seasonal_period.max(1) as f32
                            * std::f32::consts::TAU)
                            .sin();
                        entity as f32 + step as f32 * 0.01 + season
                    })
                    .collect();
                EntitySeries {
                    entity_id: format!("synthetic_{entity}"),
                    target: "value".into(),
                    values,
                    timestamps: vec![],
                    covariates: vec![],
                    metadata: EntityMetadata {
                        entity_id: format!("synthetic_{entity}"),
                        domain: Some("synthetic".into()),
                        frequency: Frequency::Hourly,
                        attributes: Default::default(),
                    },
                }
            })
            .collect()
    }
}
