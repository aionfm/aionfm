use crate::types::{CovariateSeries, EntityMetadata, Frequency, Metadata};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use uuid::Uuid;

/// Quantile level requested by clients. Values must be in `(0, 1)`.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct QuantileLevel(pub f32);

impl QuantileLevel {
    pub const Q10: Self = Self(0.10);
    pub const Q50: Self = Self(0.50);
    pub const Q90: Self = Self(0.90);

    pub fn key(self) -> String {
        format!("{:.2}", self.0)
    }
}

/// Forecast options shared by API and SDK callers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ForecastOptions {
    pub horizon: usize,
    #[serde(default = "ForecastOptions::default_quantiles")]
    pub quantiles: Vec<QuantileLevel>,
    #[serde(default)]
    pub scenario_count: Option<usize>,
    #[serde(default)]
    pub return_regimes: bool,
    #[serde(default)]
    pub return_scenarios: bool,
    #[serde(default)]
    pub enforce_constraints: bool,
    #[serde(default)]
    pub use_retrieval: bool,
}

impl Default for ForecastOptions {
    fn default() -> Self {
        Self {
            horizon: 1,
            quantiles: Self::default_quantiles(),
            scenario_count: None,
            return_regimes: false,
            return_scenarios: false,
            enforce_constraints: false,
            use_retrieval: false,
        }
    }
}

impl ForecastOptions {
    fn default_quantiles() -> Vec<QuantileLevel> {
        vec![QuantileLevel::Q10, QuantileLevel::Q50, QuantileLevel::Q90]
    }
}

/// One entity in a batch forecast request.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ForecastEntity {
    pub entity_id: String,
    pub target: String,
    pub historical_values: Vec<f32>,
    #[serde(default)]
    pub frequency: Frequency,
    #[serde(default)]
    pub covariates: Vec<CovariateSeries>,
    #[serde(default)]
    pub metadata: EntityMetadata,
}

/// Single-entity request used by model and serving traits.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ForecastRequest {
    pub request_id: Uuid,
    pub entity: ForecastEntity,
    pub options: ForecastOptions,
}

impl ForecastRequest {
    pub fn new(entity: ForecastEntity, options: ForecastOptions) -> Self {
        Self {
            request_id: Uuid::new_v4(),
            entity,
            options,
        }
    }
}

/// Public API payload for one or more entities.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchForecastRequest {
    #[serde(default = "Uuid::new_v4")]
    pub request_id: Uuid,
    pub entities: Vec<ForecastEntity>,
    pub horizon: usize,
    #[serde(default = "ForecastOptions::default_quantiles")]
    pub quantiles: Vec<QuantileLevel>,
    #[serde(default)]
    pub scenario_count: Option<usize>,
    #[serde(default)]
    pub options: RequestOptions,
}

impl BatchForecastRequest {
    pub fn forecast_options(&self) -> ForecastOptions {
        ForecastOptions {
            horizon: self.horizon,
            quantiles: self.quantiles.clone(),
            scenario_count: self.scenario_count,
            return_regimes: self.options.return_regimes,
            return_scenarios: self.options.return_scenarios,
            enforce_constraints: self.options.enforce_constraints,
            use_retrieval: self.options.use_retrieval,
        }
    }
}

/// Optional forecast behavior switches.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RequestOptions {
    #[serde(default)]
    pub return_regimes: bool,
    #[serde(default)]
    pub return_scenarios: bool,
    #[serde(default)]
    pub enforce_constraints: bool,
    #[serde(default)]
    pub use_retrieval: bool,
}

/// Prediction interval bounds for a coverage level.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictionInterval {
    pub lower: Vec<f32>,
    pub upper: Vec<f32>,
}

/// Human-readable interpretability summary.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Explanation {
    #[serde(default)]
    pub current_regime: Option<String>,
    #[serde(default)]
    pub uncertainty_driver: Option<String>,
    #[serde(default)]
    pub change_point_probability: Option<f32>,
    #[serde(default)]
    pub notes: Vec<String>,
}

/// Forecast output for a single entity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntityForecast {
    pub entity_id: String,
    pub forecast_horizon: usize,
    pub frequency: String,
    pub target: String,
    pub point_forecast: Vec<f32>,
    pub quantiles: BTreeMap<String, Vec<f32>>,
    #[serde(default)]
    pub prediction_intervals: BTreeMap<String, PredictionInterval>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scenario_paths: Option<Vec<Vec<f32>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regime_probabilities: Option<BTreeMap<String, f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explanation: Option<Explanation>,
    #[serde(default)]
    pub metadata: Metadata,
}

/// Top-level forecast response returned by serving, API, and SDK layers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ForecastResponse {
    pub response_id: Uuid,
    pub model: String,
    pub model_version: String,
    pub generated_at: DateTime<Utc>,
    pub results: Vec<EntityForecast>,
}

impl ForecastResponse {
    pub fn new(
        model: impl Into<String>,
        version: impl Into<String>,
        results: Vec<EntityForecast>,
    ) -> Self {
        Self {
            response_id: Uuid::new_v4(),
            model: model.into(),
            model_version: version.into(),
            generated_at: Utc::now(),
            results,
        }
    }
}

/// Request for scenario-first generation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScenarioRequest {
    pub forecast: BatchForecastRequest,
    #[serde(default)]
    pub scenario_type: Option<String>,
    #[serde(default)]
    pub forced_regimes: Vec<String>,
}

/// Request for interpretability outputs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InterpretationRequest {
    pub forecast: BatchForecastRequest,
    #[serde(default)]
    pub include_change_points: bool,
    #[serde(default)]
    pub include_attention_summary: bool,
}

/// Descriptor returned by model-list endpoints.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelDescriptor {
    pub name: String,
    pub version: String,
    pub status: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub supported_frequencies: Vec<Frequency>,
}

/// Model adaptation request for adapter workflows.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptationRequest {
    pub domain: String,
    #[serde(default)]
    pub entities: Vec<ForecastEntity>,
    #[serde(default)]
    pub adapter_name: Option<String>,
    #[serde(default)]
    pub calibration: bool,
}

/// Status returned after starting or simulating adaptation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptationStatus {
    pub adaptation_id: Uuid,
    pub status: String,
    #[serde(default)]
    pub message: Option<String>,
}

/// Health status returned by serving and API status endpoints.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServiceStatus {
    pub status: String,
    pub version: String,
    pub model_loaded: bool,
    #[serde(default)]
    pub queue_depth: usize,
    #[serde(default)]
    pub p50_latency_ms: Option<f32>,
    #[serde(default)]
    pub p95_latency_ms: Option<f32>,
}
