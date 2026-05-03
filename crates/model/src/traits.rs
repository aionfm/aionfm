use aionfm_utils::{AionResult, EntityForecast, ForecastRequest, ForecastResponse};

/// Stable public model interface consumed by serving, API, and SDK mock backends.
pub trait ForecastModel: Send + Sync {
    fn model_name(&self) -> &str;
    fn model_version(&self) -> &str;
    fn forecast(&self, request: &ForecastRequest) -> AionResult<ForecastResponse>;

    fn predict_point(&self, request: &ForecastRequest) -> AionResult<Vec<f32>> {
        Ok(self
            .forecast(request)?
            .results
            .into_iter()
            .next()
            .map(|entity| entity.point_forecast)
            .unwrap_or_default())
    }

    fn predict_quantiles(&self, request: &ForecastRequest) -> AionResult<EntityForecast> {
        self.forecast(request)?
            .results
            .into_iter()
            .next()
            .ok_or_else(|| {
                aionfm_utils::AionError::Backend("forecast returned no entity result".into())
            })
    }
}
