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
    pub constraints: ForecastConstraints,
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
            constraints: ForecastConstraints::default(),
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
            constraints: self.options.constraints.clone(),
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
    pub constraints: ForecastConstraints,
    #[serde(default)]
    pub use_retrieval: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hierarchy: Option<HierarchySpec>,
}

/// Constraint hints used by constraint-aware decoding and post-processing.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ForecastConstraints {
    #[serde(default)]
    pub non_negative: bool,
    #[serde(default)]
    pub min_value: Option<f32>,
    #[serde(default)]
    pub max_value: Option<f32>,
    #[serde(default)]
    pub monotonic_non_decreasing: bool,
    #[serde(default)]
    pub integer: bool,
    #[serde(default)]
    pub closed_horizon_indices: Vec<usize>,
    #[serde(default)]
    pub closed_value: Option<f32>,
}

impl ForecastConstraints {
    pub fn has_any(&self) -> bool {
        self.non_negative
            || self.min_value.is_some()
            || self.max_value.is_some()
            || self.monotonic_non_decreasing
            || self.integer
            || !self.closed_horizon_indices.is_empty()
    }
}

/// Prediction interval bounds for a coverage level.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictionInterval {
    pub lower: Vec<f32>,
    pub upper: Vec<f32>,
}

/// Interpretable additive decomposition of a forecast trajectory.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ForecastDecomposition {
    pub baseline: Vec<f32>,
    pub seasonal: Vec<f32>,
    pub event: Vec<f32>,
    pub residual: Vec<f32>,
}

/// Parametric distribution summary for each forecast horizon.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DistributionForecast {
    pub family: String,
    pub location: Vec<f32>,
    pub scale: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub degrees_of_freedom: Option<f32>,
}

/// Future regime trajectory summary for interpretability and scenario exploration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegimeStep {
    pub horizon_index: usize,
    pub label: String,
    pub probability: f32,
    pub change_point_probability: f32,
    pub volatility_state: String,
}

/// Summary of constraint projection work applied to a response.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ConstraintReport {
    pub applied: bool,
    #[serde(default)]
    pub adjusted_points: usize,
    #[serde(default)]
    pub notes: Vec<String>,
}

/// Hierarchical reconciliation strategy.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconciliationMethod {
    BottomUp,
    TopDown,
    MiddleOut,
}

impl Default for ReconciliationMethod {
    fn default() -> Self {
        Self::BottomUp
    }
}

/// Parent-child relationship for hierarchical forecasting.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HierarchyRelation {
    pub parent_entity_id: String,
    pub child_entity_ids: Vec<String>,
}

/// Batch-level hierarchy definition used by serving post-processing.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HierarchySpec {
    #[serde(default)]
    pub method: ReconciliationMethod,
    #[serde(default)]
    pub relations: Vec<HierarchyRelation>,
}

/// Reconciliation summary for aggregate coherence.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ReconciliationReport {
    pub method: ReconciliationMethod,
    #[serde(default)]
    pub adjusted_entities: Vec<String>,
    #[serde(default)]
    pub notes: Vec<String>,
}

/// Retrieved historical analog used for interpretation and retrieval-augmented forecasts.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievalMatch {
    pub source_entity_id: String,
    pub start_index: usize,
    pub window_len: usize,
    pub similarity: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regime_label: Option<String>,
    #[serde(default)]
    pub outcome_preview: Vec<f32>,
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
    pub decomposition: Option<ForecastDecomposition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distribution: Option<DistributionForecast>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imputed_history: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scenario_paths: Option<Vec<Vec<f32>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regime_probabilities: Option<BTreeMap<String, f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regime_timeline: Option<Vec<RegimeStep>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraint_report: Option<ConstraintReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieval_matches: Option<Vec<RetrievalMatch>>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reconciliation_report: Option<ReconciliationReport>,
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
            reconciliation_report: None,
        }
    }
}

/// Severity assigned to monitoring alerts.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

impl Default for AlertSeverity {
    fn default() -> Self {
        Self::Info
    }
}

/// Alert emitted by evaluation or serving monitoring.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MonitoringAlert {
    pub severity: AlertSeverity,
    pub metric: String,
    pub message: String,
    pub value: f32,
    pub threshold: f32,
}

/// Observed values paired with an entity forecast for post-deployment evaluation.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EvaluationObservation {
    pub entity_id: String,
    pub observed: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segment: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_mae: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub missing_rate: Option<f32>,
}

/// Request used by monitoring jobs to score a forecast after observations arrive.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationRequest {
    pub forecast: ForecastResponse,
    #[serde(default)]
    pub observations: Vec<EvaluationObservation>,
}

/// Per-entity forecast quality metrics.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EntityEvaluation {
    pub entity_id: String,
    pub target: String,
    pub horizon: usize,
    pub compared_points: usize,
    pub mae: f32,
    pub rmse: f32,
    pub smape: f32,
    pub wape: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mase: Option<f32>,
    #[serde(default)]
    pub quantile_pinball_loss: BTreeMap<String, f32>,
    #[serde(default)]
    pub quantile_coverage: BTreeMap<String, f32>,
    #[serde(default)]
    pub interval_coverage: BTreeMap<String, f32>,
    #[serde(default)]
    pub interval_width: BTreeMap<String, f32>,
}

/// Evaluation summary for accuracy, calibration, intervals, and drift indicators.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub report_id: Uuid,
    pub generated_at: DateTime<Utc>,
    pub model: String,
    pub model_version: String,
    pub entity_count: usize,
    pub observation_count: usize,
    #[serde(default)]
    pub entities: Vec<EntityEvaluation>,
    #[serde(default)]
    pub metrics: BTreeMap<String, f32>,
    #[serde(default)]
    pub alerts: Vec<MonitoringAlert>,
}

impl EvaluationReport {
    pub fn new(
        model: impl Into<String>,
        version: impl Into<String>,
        entities: Vec<EntityEvaluation>,
        metrics: BTreeMap<String, f32>,
        alerts: Vec<MonitoringAlert>,
    ) -> Self {
        let observation_count = entities.iter().map(|entity| entity.compared_points).sum();
        Self {
            report_id: Uuid::new_v4(),
            generated_at: Utc::now(),
            model: model.into(),
            model_version: version.into(),
            entity_count: entities.len(),
            observation_count,
            entities,
            metrics,
            alerts,
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
    #[serde(default)]
    pub metrics: BTreeMap<String, f32>,
    #[serde(default)]
    pub alerts: Vec<MonitoringAlert>,
}
