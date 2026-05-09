use aionfm_utils::{
    AlertSeverity, EntityEvaluation, EvaluationObservation, EvaluationReport, EvaluationRequest,
    ForecastResponse, MonitoringAlert,
};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Thresholds used to convert monitoring metrics into alerts.
#[derive(Clone, Debug)]
pub struct EvaluationThresholds {
    pub max_smape: f32,
    pub max_wape: f32,
    pub max_calibration_error: f32,
    pub max_missing_rate: f32,
}

impl Default for EvaluationThresholds {
    fn default() -> Self {
        Self {
            max_smape: 0.25,
            max_wape: 0.25,
            max_calibration_error: 0.10,
            max_missing_rate: 0.20,
        }
    }
}

/// Computes post-deployment evaluation reports once observations are available.
#[derive(Clone, Debug, Default)]
pub struct ForecastEvaluator {
    thresholds: EvaluationThresholds,
}

impl ForecastEvaluator {
    pub fn new(thresholds: EvaluationThresholds) -> Self {
        Self { thresholds }
    }

    pub fn evaluate(&self, request: &EvaluationRequest) -> EvaluationReport {
        let observations = request
            .observations
            .iter()
            .map(|observation| (observation.entity_id.as_str(), observation))
            .collect::<BTreeMap<_, _>>();
        let mut entities = Vec::new();
        for forecast in &request.forecast.results {
            let Some(observation) = observations.get(forecast.entity_id.as_str()) else {
                continue;
            };
            let compared_points = forecast
                .point_forecast
                .len()
                .min(observation.observed.len());
            if compared_points == 0 {
                continue;
            }
            let observed = &observation.observed[..compared_points];
            let point = &forecast.point_forecast[..compared_points];
            let mae = mean_absolute_error(point, observed);
            let rmse = root_mean_squared_error(point, observed);
            let smape = symmetric_mape(point, observed);
            let wape = weighted_absolute_percentage_error(point, observed);
            let mase = observation
                .baseline_mae
                .filter(|baseline| baseline.is_finite() && *baseline > 0.0)
                .map(|baseline| mae / baseline.max(1e-6));
            let (quantile_pinball_loss, quantile_coverage) =
                quantile_metrics(&forecast.quantiles, observed);
            let (interval_coverage, interval_width) =
                interval_metrics(&forecast.prediction_intervals, observed);
            entities.push(EntityEvaluation {
                entity_id: forecast.entity_id.clone(),
                target: forecast.target.clone(),
                horizon: forecast.forecast_horizon,
                compared_points,
                mae,
                rmse,
                smape,
                wape,
                mase,
                quantile_pinball_loss,
                quantile_coverage,
                interval_coverage,
                interval_width,
            });
        }
        let metrics = aggregate_metrics(&entities, &request.observations);
        let alerts = self.alerts(&metrics);
        let mut report = EvaluationReport::new(
            request.forecast.model.clone(),
            request.forecast.model_version.clone(),
            entities,
            metrics,
            alerts,
        );
        report.context = request
            .context
            .clone()
            .or_else(|| request.forecast.context.clone());
        report
    }

    fn alerts(&self, metrics: &BTreeMap<String, f32>) -> Vec<MonitoringAlert> {
        let mut alerts = Vec::new();
        push_threshold_alert(
            &mut alerts,
            "overall_smape",
            "sMAPE exceeds monitoring threshold",
            metrics.get("overall_smape").copied().unwrap_or_default(),
            self.thresholds.max_smape,
        );
        push_threshold_alert(
            &mut alerts,
            "overall_wape",
            "WAPE exceeds monitoring threshold",
            metrics.get("overall_wape").copied().unwrap_or_default(),
            self.thresholds.max_wape,
        );
        push_threshold_alert(
            &mut alerts,
            "average_quantile_calibration_error",
            "quantile calibration drift exceeds threshold",
            metrics
                .get("average_quantile_calibration_error")
                .copied()
                .unwrap_or_default(),
            self.thresholds.max_calibration_error,
        );
        push_threshold_alert(
            &mut alerts,
            "max_missing_rate",
            "observed missingness exceeds monitoring threshold",
            metrics.get("max_missing_rate").copied().unwrap_or_default(),
            self.thresholds.max_missing_rate,
        );
        alerts
    }
}

/// Lightweight inference metrics hook.
#[derive(Clone, Debug, Default)]
pub struct InferenceMetrics {
    inner: Arc<Mutex<MetricsInner>>,
}

#[derive(Clone, Debug, Default)]
struct MetricsInner {
    request_count: usize,
    evaluation_count: usize,
    last_latency_ms: Option<f32>,
    last_mae: Option<f32>,
    last_smape: Option<f32>,
    last_alert_count: usize,
    evaluated_observation_count: usize,
    interval_width_sum: f32,
    interval_width_count: usize,
    quantile_crossing_count: usize,
    quantile_check_count: usize,
    scenario_path_count: usize,
    regime_counts: BTreeMap<String, usize>,
}

impl InferenceMetrics {
    pub fn record_latency(&self, duration: Duration) {
        let mut inner = self.inner.lock().expect("metrics mutex poisoned");
        inner.request_count += 1;
        inner.last_latency_ms = Some(duration.as_secs_f32() * 1_000.0);
    }

    pub fn record_response(&self, response: &ForecastResponse) {
        let mut inner = self.inner.lock().expect("metrics mutex poisoned");
        for result in &response.results {
            for interval in result.prediction_intervals.values() {
                for (lower, upper) in interval.lower.iter().zip(&interval.upper) {
                    inner.interval_width_sum += (upper - lower).abs();
                    inner.interval_width_count += 1;
                }
            }
            inner.scenario_path_count += result
                .scenario_paths
                .as_ref()
                .map(Vec::len)
                .unwrap_or_default();
            if let Some(regimes) = &result.regime_probabilities {
                for (label, probability) in regimes {
                    let bucketed = (*probability * 100.0).round() as usize;
                    *inner.regime_counts.entry(label.clone()).or_default() += bucketed.max(1);
                }
            }
            let mut keys = result.quantiles.keys().cloned().collect::<Vec<_>>();
            keys.sort_by(|left, right| {
                left.parse::<f32>()
                    .unwrap_or(0.5)
                    .partial_cmp(&right.parse::<f32>().unwrap_or(0.5))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for pair in keys.windows(2) {
                let Some(left) = result.quantiles.get(&pair[0]) else {
                    continue;
                };
                let Some(right) = result.quantiles.get(&pair[1]) else {
                    continue;
                };
                for (left, right) in left.iter().zip(right) {
                    inner.quantile_check_count += 1;
                    if left > right {
                        inner.quantile_crossing_count += 1;
                    }
                }
            }
        }
    }

    pub fn record_evaluation(&self, report: &EvaluationReport) {
        let mut inner = self.inner.lock().expect("metrics mutex poisoned");
        inner.evaluation_count += 1;
        inner.evaluated_observation_count += report.observation_count;
        inner.last_mae = report.metrics.get("overall_mae").copied();
        inner.last_smape = report.metrics.get("overall_smape").copied();
        inner.last_alert_count = report.alerts.len();
    }

    pub fn request_count(&self) -> usize {
        self.inner
            .lock()
            .expect("metrics mutex poisoned")
            .request_count
    }

    pub fn last_latency_ms(&self) -> Option<f32> {
        self.inner
            .lock()
            .expect("metrics mutex poisoned")
            .last_latency_ms
    }

    pub fn summary(&self) -> BTreeMap<String, f32> {
        let inner = self.inner.lock().expect("metrics mutex poisoned");
        let mut metrics = BTreeMap::from([
            ("request_count".into(), inner.request_count as f32),
            ("evaluation_count".into(), inner.evaluation_count as f32),
            (
                "evaluated_observation_count".into(),
                inner.evaluated_observation_count as f32,
            ),
            ("last_alert_count".into(), inner.last_alert_count as f32),
            (
                "scenario_path_count".into(),
                inner.scenario_path_count as f32,
            ),
            (
                "average_interval_width".into(),
                inner.interval_width_sum / inner.interval_width_count.max(1) as f32,
            ),
            (
                "quantile_crossing_rate".into(),
                inner.quantile_crossing_count as f32 / inner.quantile_check_count.max(1) as f32,
            ),
        ]);
        if let Some(value) = inner.last_mae {
            metrics.insert("last_mae".into(), value);
        }
        if let Some(value) = inner.last_smape {
            metrics.insert("last_smape".into(), value);
        }
        for (label, count) in &inner.regime_counts {
            metrics.insert(format!("regime_{label}_weight"), *count as f32);
        }
        metrics
    }

    pub fn alerts(&self) -> Vec<MonitoringAlert> {
        let summary = self.summary();
        let crossing_rate = summary
            .get("quantile_crossing_rate")
            .copied()
            .unwrap_or_default();
        let mut alerts = Vec::new();
        push_threshold_alert(
            &mut alerts,
            "quantile_crossing_rate",
            "forecast quantiles crossed in recent responses",
            crossing_rate,
            0.0,
        );
        alerts
    }
}

fn mean_absolute_error(point: &[f32], observed: &[f32]) -> f32 {
    mean(
        point
            .iter()
            .zip(observed)
            .map(|(point, observed)| (point - observed).abs()),
    )
}

fn root_mean_squared_error(point: &[f32], observed: &[f32]) -> f32 {
    mean(point.iter().zip(observed).map(|(point, observed)| {
        let error = point - observed;
        error * error
    }))
    .sqrt()
}

fn symmetric_mape(point: &[f32], observed: &[f32]) -> f32 {
    mean(point.iter().zip(observed).map(|(point, observed)| {
        let denominator = point.abs() + observed.abs();
        if denominator <= 1e-6 {
            0.0
        } else {
            2.0 * (point - observed).abs() / denominator
        }
    }))
}

fn weighted_absolute_percentage_error(point: &[f32], observed: &[f32]) -> f32 {
    let numerator = point
        .iter()
        .zip(observed)
        .map(|(point, observed)| (point - observed).abs())
        .sum::<f32>();
    let denominator = observed.iter().map(|value| value.abs()).sum::<f32>();
    numerator / denominator.max(1e-6)
}

fn quantile_metrics(
    quantiles: &BTreeMap<String, Vec<f32>>,
    observed: &[f32],
) -> (BTreeMap<String, f32>, BTreeMap<String, f32>) {
    let mut pinball = BTreeMap::new();
    let mut coverage = BTreeMap::new();
    for (key, forecast) in quantiles {
        let level = key.parse::<f32>().unwrap_or(0.5).clamp(0.0, 1.0);
        let len = forecast.len().min(observed.len());
        if len == 0 {
            continue;
        }
        let mut loss_sum = 0.0;
        let mut covered = 0;
        for (forecast, observed) in forecast.iter().zip(observed).take(len) {
            let error = observed - forecast;
            loss_sum += if error >= 0.0 {
                level * error
            } else {
                (level - 1.0) * error
            };
            if observed <= forecast {
                covered += 1;
            }
        }
        pinball.insert(key.clone(), loss_sum / len as f32);
        coverage.insert(key.clone(), covered as f32 / len as f32);
    }
    (pinball, coverage)
}

fn interval_metrics(
    intervals: &BTreeMap<String, aionfm_utils::PredictionInterval>,
    observed: &[f32],
) -> (BTreeMap<String, f32>, BTreeMap<String, f32>) {
    let mut coverage = BTreeMap::new();
    let mut width = BTreeMap::new();
    for (key, interval) in intervals {
        let len = interval
            .lower
            .len()
            .min(interval.upper.len())
            .min(observed.len());
        if len == 0 {
            continue;
        }
        let mut covered = 0;
        let mut width_sum = 0.0;
        for ((lower, upper), observed) in interval
            .lower
            .iter()
            .zip(&interval.upper)
            .zip(observed)
            .take(len)
        {
            if observed >= lower && observed <= upper {
                covered += 1;
            }
            width_sum += (upper - lower).abs();
        }
        coverage.insert(key.clone(), covered as f32 / len as f32);
        width.insert(key.clone(), width_sum / len as f32);
    }
    (coverage, width)
}

fn aggregate_metrics(
    entities: &[EntityEvaluation],
    observations: &[EvaluationObservation],
) -> BTreeMap<String, f32> {
    let compared_points = entities
        .iter()
        .map(|entity| entity.compared_points)
        .sum::<usize>();
    let point_count = compared_points.max(1) as f32;
    let weighted = |metric: fn(&EntityEvaluation) -> f32| {
        entities
            .iter()
            .map(|entity| metric(entity) * entity.compared_points as f32)
            .sum::<f32>()
            / point_count
    };
    let mut metrics = BTreeMap::from([
        ("entity_count".into(), entities.len() as f32),
        ("observation_count".into(), compared_points as f32),
        ("overall_mae".into(), weighted(|entity| entity.mae)),
        ("overall_rmse".into(), weighted(|entity| entity.rmse)),
        ("overall_smape".into(), weighted(|entity| entity.smape)),
        ("overall_wape".into(), weighted(|entity| entity.wape)),
        (
            "max_missing_rate".into(),
            observations
                .iter()
                .filter_map(|observation| observation.missing_rate)
                .fold(0.0_f32, f32::max),
        ),
    ]);
    let mut calibration_error_sum = 0.0;
    let mut calibration_error_count = 0;
    let mut pinball_sum = 0.0;
    let mut pinball_count = 0;
    for entity in entities {
        for (key, coverage) in &entity.quantile_coverage {
            let level = key.parse::<f32>().unwrap_or(0.5).clamp(0.0, 1.0);
            calibration_error_sum += (coverage - level).abs();
            calibration_error_count += 1;
        }
        for value in entity.quantile_pinball_loss.values() {
            pinball_sum += *value;
            pinball_count += 1;
        }
    }
    metrics.insert(
        "average_quantile_calibration_error".into(),
        calibration_error_sum / calibration_error_count.max(1) as f32,
    );
    metrics.insert(
        "average_pinball_loss".into(),
        pinball_sum / pinball_count.max(1) as f32,
    );
    metrics
}

fn mean(values: impl IntoIterator<Item = f32>) -> f32 {
    let mut sum = 0.0;
    let mut count = 0;
    for value in values {
        if value.is_finite() {
            sum += value;
            count += 1;
        }
    }
    sum / count.max(1) as f32
}

fn push_threshold_alert(
    alerts: &mut Vec<MonitoringAlert>,
    metric: &str,
    message: &str,
    value: f32,
    threshold: f32,
) {
    if value > threshold {
        alerts.push(MonitoringAlert {
            severity: AlertSeverity::Warning,
            metric: metric.into(),
            message: message.into(),
            value,
            threshold,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aionfm_utils::{EntityForecast, EvaluationObservation, ForecastResponse};

    #[test]
    fn evaluator_computes_point_metrics() {
        let forecast = ForecastResponse::new("AionFM", "test", vec![entity()]);
        let report = ForecastEvaluator::default().evaluate(&EvaluationRequest {
            context: None,
            forecast,
            observations: vec![EvaluationObservation {
                entity_id: "entity".into(),
                observed: vec![2.0, 4.0],
                ..Default::default()
            }],
        });
        assert_eq!(report.entity_count, 1);
        assert_eq!(report.metrics.get("overall_mae").copied(), Some(1.0));
    }

    fn entity() -> EntityForecast {
        EntityForecast {
            entity_id: "entity".into(),
            forecast_horizon: 2,
            frequency: "D".into(),
            target: "value".into(),
            point_forecast: vec![1.0, 3.0],
            quantiles: BTreeMap::from([("0.50".into(), vec![1.0, 3.0])]),
            prediction_intervals: BTreeMap::new(),
            decomposition: None,
            distribution: None,
            imputed_history: None,
            scenario_paths: None,
            regime_probabilities: None,
            regime_timeline: None,
            constraint_report: None,
            retrieval_matches: None,
            explanation: None,
            metadata: BTreeMap::new(),
        }
    }
}
