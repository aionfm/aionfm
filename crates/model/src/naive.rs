use crate::{ForecastHeads, ForecastModel, ModelConfig};
use aionfm_utils::{
    validation::validate_forecast_options, AionResult, ForecastRequest, ForecastResponse,
};

/// Deterministic statistical baseline used until learned AionFM kernels are implemented.
#[derive(Clone, Debug)]
pub struct NaiveForecastModel {
    config: ModelConfig,
    heads: ForecastHeads,
}

impl Default for NaiveForecastModel {
    fn default() -> Self {
        Self {
            config: ModelConfig::default(),
            heads: ForecastHeads::default(),
        }
    }
}

impl NaiveForecastModel {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            heads: ForecastHeads::default(),
        }
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
}

impl ForecastModel for NaiveForecastModel {
    fn model_name(&self) -> &str {
        &self.heads.model_name
    }

    fn model_version(&self) -> &str {
        &self.heads.model_version
    }

    fn forecast(&self, request: &ForecastRequest) -> AionResult<ForecastResponse> {
        validate_forecast_options(&request.options)?;
        let result = self
            .heads
            .baseline_forecast(&request.entity, &request.options);
        Ok(ForecastResponse::new(
            self.model_name(),
            self.model_version(),
            vec![result],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aionfm_utils::{ForecastEntity, ForecastOptions, ForecastRequest};

    #[test]
    fn baseline_returns_requested_horizon() {
        let model = NaiveForecastModel::default();
        let request = ForecastRequest::new(
            ForecastEntity {
                entity_id: "store_42".into(),
                target: "demand".into(),
                historical_values: vec![1.0, 2.0, 3.0],
                frequency: Default::default(),
                covariates: vec![],
                metadata: Default::default(),
            },
            ForecastOptions {
                horizon: 4,
                ..Default::default()
            },
        );
        let response = model.forecast(&request).unwrap();
        assert_eq!(response.results[0].point_forecast.len(), 4);
    }

    #[test]
    fn baseline_extrapolates_simple_trend() {
        let model = NaiveForecastModel::default();
        let request = ForecastRequest::new(
            ForecastEntity {
                entity_id: "store_42".into(),
                target: "demand".into(),
                historical_values: vec![1.0, 2.0, 3.0, 4.0],
                frequency: Default::default(),
                covariates: vec![],
                metadata: Default::default(),
            },
            ForecastOptions {
                horizon: 2,
                ..Default::default()
            },
        );
        let response = model.forecast(&request).unwrap();
        assert!(response.results[0].point_forecast[1] > response.results[0].point_forecast[0]);
    }
}
