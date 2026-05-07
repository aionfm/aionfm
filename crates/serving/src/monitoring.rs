use aionfm_utils::ForecastResponse;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Lightweight inference metrics hook.
#[derive(Clone, Debug, Default)]
pub struct InferenceMetrics {
    inner: Arc<Mutex<MetricsInner>>,
}

#[derive(Clone, Debug, Default)]
struct MetricsInner {
    request_count: usize,
    last_latency_ms: Option<f32>,
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
        for (label, count) in &inner.regime_counts {
            metrics.insert(format!("regime_{label}_weight"), *count as f32);
        }
        metrics
    }
}
