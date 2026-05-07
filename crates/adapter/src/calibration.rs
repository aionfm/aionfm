use aionfm_utils::EntityForecast;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Post-hoc quantile calibration offsets.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct QuantileCalibration {
    pub offsets: std::collections::BTreeMap<String, f32>,
    #[serde(default)]
    pub empirical_coverage: std::collections::BTreeMap<String, f32>,
    #[serde(default)]
    pub sample_count: usize,
}

/// Forecast/observation pair used to fit post-hoc calibration offsets.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalibrationSample {
    pub forecast: EntityForecast,
    pub observed: Vec<f32>,
}

/// Applies learned offsets to quantile forecasts.
#[derive(Clone, Debug, Default)]
pub struct QuantileCalibrator {
    calibration: QuantileCalibration,
}

impl QuantileCalibrator {
    pub fn new(calibration: QuantileCalibration) -> Self {
        Self { calibration }
    }

    pub fn apply(&self, forecast: &mut EntityForecast) {
        for (key, values) in &mut forecast.quantiles {
            if let Some(offset) = self.calibration.offsets.get(key) {
                for value in values {
                    *value += offset;
                }
            }
        }
        for interval in forecast.prediction_intervals.values_mut() {
            for (index, lower) in interval.lower.iter_mut().enumerate() {
                if let Some(offset) = self.calibration.offsets.get("0.10") {
                    *lower += offset;
                }
                if let (Some(upper), Some(offset)) = (
                    interval.upper.get_mut(index),
                    self.calibration.offsets.get("0.90"),
                ) {
                    *upper += offset;
                }
            }
        }
    }

    pub fn fit(samples: &[CalibrationSample]) -> QuantileCalibration {
        let mut residuals_by_quantile: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        let mut coverage_hits: BTreeMap<String, usize> = BTreeMap::new();
        let mut coverage_total: BTreeMap<String, usize> = BTreeMap::new();
        for sample in samples {
            for (key, forecast_values) in &sample.forecast.quantiles {
                for (forecast, observed) in forecast_values.iter().zip(&sample.observed) {
                    if forecast.is_finite() && observed.is_finite() {
                        residuals_by_quantile
                            .entry(key.clone())
                            .or_default()
                            .push(observed - forecast);
                        *coverage_total.entry(key.clone()).or_default() += 1;
                        if observed <= forecast {
                            *coverage_hits.entry(key.clone()).or_default() += 1;
                        }
                    }
                }
            }
        }
        let offsets = residuals_by_quantile
            .into_iter()
            .map(|(key, mut residuals)| {
                residuals.sort_by(f32::total_cmp);
                (key, median_sorted(&residuals))
            })
            .collect();
        let empirical_coverage = coverage_total
            .into_iter()
            .map(|(key, total)| {
                let hits = coverage_hits.get(&key).copied().unwrap_or_default();
                (key, hits as f32 / total.max(1) as f32)
            })
            .collect();
        QuantileCalibration {
            offsets,
            empirical_coverage,
            sample_count: samples.len(),
        }
    }
}

fn median_sorted(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aionfm_utils::EntityForecast;

    #[test]
    fn fits_quantile_offsets_from_residuals() {
        let forecast = EntityForecast {
            entity_id: "e".into(),
            forecast_horizon: 2,
            frequency: "D".into(),
            target: "value".into(),
            point_forecast: vec![1.0, 2.0],
            quantiles: BTreeMap::from([("0.50".into(), vec![1.0, 2.0])]),
            prediction_intervals: BTreeMap::new(),
            decomposition: None,
            distribution: None,
            imputed_history: None,
            scenario_paths: None,
            regime_probabilities: None,
            regime_timeline: None,
            constraint_report: None,
            explanation: None,
            metadata: BTreeMap::new(),
        };
        let calibration = QuantileCalibrator::fit(&[CalibrationSample {
            forecast,
            observed: vec![2.0, 3.0],
        }]);
        assert_eq!(calibration.offsets.get("0.50").copied(), Some(1.0));
    }
}
