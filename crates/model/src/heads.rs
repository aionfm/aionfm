use crate::{enforce_quantile_monotonicity, merge_reports, ConstraintProjector};
use aionfm_utils::{
    DistributionForecast, EntityForecast, Explanation, ForecastConstraints, ForecastDecomposition,
    ForecastEntity, ForecastOptions, Frequency, PredictionInterval, RegimeStep,
};
use std::collections::BTreeMap;

/// Output heads for point, quantile, interval, scenario, and regime forecasts.
#[derive(Clone, Debug)]
pub struct ForecastHeads {
    pub model_name: String,
    pub model_version: String,
}

impl Default for ForecastHeads {
    fn default() -> Self {
        Self {
            model_name: "AionFM".into(),
            model_version: "0.1-statistical-baseline".into(),
        }
    }
}

impl ForecastHeads {
    pub fn baseline_forecast(
        &self,
        entity: &ForecastEntity,
        options: &ForecastOptions,
    ) -> EntityForecast {
        let imputation = impute_history(&entity.historical_values);
        let values = finite_values(&imputation.values);
        let profile = SeriesProfile::from_values(&values, &entity.frequency);
        let decomposition = profile.decomposition(options.horizon);
        let mut point_forecast = sum_decomposition(&decomposition);
        let projection = build_projection(options, &values);
        let projector = projection
            .as_ref()
            .map(|constraints| ConstraintProjector::new(constraints.clone()));
        let mut reports = Vec::new();
        if let Some(projector) = &projector {
            reports.push(projector.project(&mut point_forecast, "point_forecast"));
        }
        let mut quantiles = BTreeMap::new();
        for quantile in &options.quantiles {
            let score = normal_score(quantile.0);
            let mut forecast = point_forecast
                .iter()
                .enumerate()
                .map(|(index, point)| {
                    let horizon_scale = ((index + 1) as f32).sqrt();
                    point + score * profile.residual_scale * horizon_scale
                })
                .collect::<Vec<_>>();
            if let Some(projector) = &projector {
                reports.push(
                    projector.project(&mut forecast, &format!("quantile_{}", quantile.key())),
                );
            }
            quantiles.insert(quantile.key(), forecast);
        }
        enforce_quantile_monotonicity(&mut quantiles);
        let mut prediction_intervals = profile.prediction_intervals(&point_forecast);
        if let Some(projector) = &projector {
            for (coverage, interval) in &mut prediction_intervals {
                reports.push(
                    projector.project(&mut interval.lower, &format!("interval_{coverage}_lower")),
                );
                reports.push(
                    projector.project(&mut interval.upper, &format!("interval_{coverage}_upper")),
                );
            }
        }
        let scenario_paths = options.return_scenarios.then(|| {
            let mut scenarios =
                profile.scenarios(&point_forecast, options.scenario_count.unwrap_or(8));
            if let Some(projector) = &projector {
                for (index, scenario) in scenarios.iter_mut().enumerate() {
                    reports.push(projector.project(scenario, &format!("scenario_{index}")));
                }
            }
            scenarios
        });
        let regime_probabilities = options
            .return_regimes
            .then(|| profile.regime_probabilities());
        let regime_timeline = options
            .return_regimes
            .then(|| profile.regime_timeline(options.horizon));
        let current_regime = profile.primary_regime();
        let constraint_report = merge_reports(reports);
        EntityForecast {
            entity_id: entity.entity_id.clone(),
            forecast_horizon: options.horizon,
            frequency: entity.frequency.code().into(),
            target: entity.target.clone(),
            point_forecast,
            quantiles,
            prediction_intervals,
            decomposition: Some(decomposition),
            distribution: Some(profile.distribution(options.horizon)),
            imputed_history: imputation.has_missing.then_some(imputation.values),
            scenario_paths,
            regime_probabilities,
            regime_timeline,
            constraint_report,
            explanation: Some(Explanation {
                current_regime: Some(current_regime),
                uncertainty_driver: Some(profile.uncertainty_driver()),
                change_point_probability: Some(profile.change_point_probability()),
                notes: vec![
                    "Statistical baseline: trend, seasonality, residual scale, and regime heuristics.".into(),
                    "Replace with the dual-stream AionFM backend for learned forecasts.".into(),
                ],
            }),
            metadata: entity.metadata.attributes.clone(),
        }
    }
}

#[derive(Clone, Debug)]
struct SeriesProfile {
    values: Vec<f32>,
    last_value: f32,
    slope: f32,
    residual_scale: f32,
    seasonal_offsets: Vec<f32>,
    volatility_ratio: f32,
    event_impulse: f32,
}

impl SeriesProfile {
    fn from_values(values: &[f32], frequency: &Frequency) -> Self {
        let values = if values.is_empty() {
            vec![0.0]
        } else {
            values.to_vec()
        };
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let last_value = values.last().copied().unwrap_or_default();
        let slope = estimate_slope(&values);
        let residual_scale = estimate_residual_scale(&values, slope)
            .max(last_value.abs() * 0.01)
            .max(1e-3);
        let seasonal_offsets = estimate_seasonal_offsets(&values, seasonal_period(frequency));
        let volatility_ratio = residual_scale / mean.abs().max(1.0);
        let event_impulse = estimate_event_impulse(&values, slope, residual_scale);
        Self {
            values,
            last_value,
            slope,
            residual_scale,
            seasonal_offsets,
            volatility_ratio,
            event_impulse,
        }
    }

    fn decomposition(&self, horizon: usize) -> ForecastDecomposition {
        let mut baseline = Vec::with_capacity(horizon);
        let mut seasonal = Vec::with_capacity(horizon);
        let mut event = Vec::with_capacity(horizon);
        let mut residual = Vec::with_capacity(horizon);
        for step in 0..horizon {
            baseline.push(self.last_value + self.slope * (step + 1) as f32);
            seasonal.push(self.seasonal_at(step));
            event.push(self.event_impulse * (-(step as f32) / 3.0).exp());
            residual.push(0.0);
        }
        ForecastDecomposition {
            baseline,
            seasonal,
            event,
            residual,
        }
    }

    fn prediction_intervals(&self, point: &[f32]) -> BTreeMap<String, PredictionInterval> {
        let mut intervals = BTreeMap::new();
        for (coverage, score) in [("80", 1.2816_f32), ("95", 1.96_f32)] {
            let (lower, upper): (Vec<f32>, Vec<f32>) = point
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let spread = score * self.residual_scale * ((index + 1) as f32).sqrt();
                    (value - spread, value + spread)
                })
                .unzip();
            intervals.insert(coverage.into(), PredictionInterval { lower, upper });
        }
        intervals
    }

    fn scenarios(&self, point: &[f32], count: usize) -> Vec<Vec<f32>> {
        let count = count.max(1);
        (0..count)
            .map(|scenario| {
                let centered = scenario as f32 - (count.saturating_sub(1)) as f32 / 2.0;
                let phase = (scenario + 1) as f32 * 0.73;
                point
                    .iter()
                    .enumerate()
                    .map(|(step, point)| {
                        let drift =
                            centered * self.residual_scale * 0.35 * ((step + 1) as f32).sqrt();
                        let wave = ((step + 1) as f32 * phase).sin() * self.residual_scale * 0.25;
                        point + drift + wave
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn distribution(&self, horizon: usize) -> DistributionForecast {
        DistributionForecast {
            family: "student_t".into(),
            location: sum_decomposition(&self.decomposition(horizon)),
            scale: (0..horizon)
                .map(|index| self.residual_scale * ((index + 1) as f32).sqrt())
                .collect(),
            degrees_of_freedom: Some(7.0),
        }
    }

    fn regime_timeline(&self, horizon: usize) -> Vec<RegimeStep> {
        let primary = self.primary_regime();
        let change_point_probability = self.change_point_probability();
        let volatility_state = if self.volatility_ratio > 0.25 {
            "expanding"
        } else if self.volatility_ratio < 0.05 {
            "compressed"
        } else {
            "normal"
        };
        (0..horizon)
            .map(|horizon_index| RegimeStep {
                horizon_index,
                label: if horizon_index > 0 && change_point_probability > 0.55 {
                    "transition_risk".into()
                } else {
                    primary.clone()
                },
                probability: (1.0
                    - change_point_probability * horizon_index as f32 / horizon.max(1) as f32)
                    .clamp(0.05, 1.0),
                change_point_probability,
                volatility_state: volatility_state.into(),
            })
            .collect()
    }

    fn seasonal_at(&self, step: usize) -> f32 {
        if self.seasonal_offsets.is_empty() {
            0.0
        } else {
            self.seasonal_offsets[(self.values.len() + step) % self.seasonal_offsets.len()]
        }
    }

    fn regime_probabilities(&self) -> BTreeMap<String, f32> {
        let trend_strength = (self.slope.abs() / self.residual_scale.max(1e-3)).min(1.0);
        let volatile = self.volatility_ratio.min(1.0);
        let seasonal = if self.seasonal_offsets.is_empty() {
            0.0
        } else {
            0.25
        };
        let shock = self.change_point_probability();
        let stable = (1.0 - trend_strength * 0.5 - volatile * 0.5 - shock * 0.4).max(0.05);
        normalize_probs(BTreeMap::from([
            ("stable".to_string(), stable),
            ("trend".to_string(), trend_strength.max(0.05)),
            ("seasonal".to_string(), seasonal),
            ("volatile".to_string(), volatile.max(0.05)),
            ("shock_recovery".to_string(), shock.max(0.02)),
        ]))
    }

    fn primary_regime(&self) -> String {
        self.regime_probabilities()
            .into_iter()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .map(|(label, _)| label)
            .unwrap_or_else(|| "stable".into())
    }

    fn uncertainty_driver(&self) -> String {
        if self.volatility_ratio > 0.25 {
            "recent residual volatility is high".into()
        } else if self.slope.abs() > self.residual_scale {
            "trend extrapolation dominates the horizon".into()
        } else if !self.seasonal_offsets.is_empty() {
            "seasonal residual offsets influence the horizon".into()
        } else {
            "recent residual scale is low".into()
        }
    }

    fn change_point_probability(&self) -> f32 {
        if self.values.len() < 4 {
            return 0.0;
        }
        let last_diff = self.values[self.values.len() - 1] - self.values[self.values.len() - 2];
        ((last_diff.abs() / (self.residual_scale * 3.0).max(1e-3)) - 0.1).clamp(0.0, 1.0)
    }
}

fn finite_values(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect()
}

fn estimate_slope(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let window = values.len().min(32);
    let start = values.len() - window;
    let y = &values[start..];
    let x_mean = (window - 1) as f32 / 2.0;
    let y_mean = y.iter().sum::<f32>() / window as f32;
    let numerator = y
        .iter()
        .enumerate()
        .map(|(index, value)| (index as f32 - x_mean) * (value - y_mean))
        .sum::<f32>();
    let denominator = (0..window)
        .map(|index| {
            let centered = index as f32 - x_mean;
            centered * centered
        })
        .sum::<f32>();
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn estimate_residual_scale(values: &[f32], slope: f32) -> f32 {
    if values.len() < 2 {
        return 1.0;
    }
    let residuals = values
        .windows(2)
        .map(|window| window[1] - window[0] - slope)
        .collect::<Vec<_>>();
    std_dev(&residuals).max(std_dev(values) * 0.05)
}

fn estimate_event_impulse(values: &[f32], slope: f32, residual_scale: f32) -> f32 {
    if values.len() < 3 {
        return 0.0;
    }
    let previous = values[values.len() - 2];
    let observed = values[values.len() - 1];
    let expected = previous + slope;
    let residual = observed - expected;
    if residual.abs() > residual_scale * 1.5 {
        residual * 0.5
    } else {
        0.0
    }
}

fn estimate_seasonal_offsets(values: &[f32], period: Option<usize>) -> Vec<f32> {
    let Some(period) = period else {
        return vec![];
    };
    if values.len() < period * 2 {
        return vec![];
    }
    let recent = &values[values.len() - period..];
    let recent_mean = recent.iter().sum::<f32>() / recent.len() as f32;
    recent
        .iter()
        .map(|value| (value - recent_mean) * 0.5)
        .collect()
}

fn seasonal_period(frequency: &Frequency) -> Option<usize> {
    match frequency {
        Frequency::Hourly => Some(24),
        Frequency::Daily => Some(7),
        Frequency::Weekly => Some(52),
        Frequency::Monthly => Some(12),
        _ => None,
    }
}

fn std_dev(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    (values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f32>()
        / values.len() as f32)
        .sqrt()
}

fn normal_score(quantile: f32) -> f32 {
    match quantile {
        q if (q - 0.01).abs() < 1e-4 => -2.326,
        q if (q - 0.05).abs() < 1e-4 => -1.645,
        q if (q - 0.10).abs() < 1e-4 => -1.282,
        q if (q - 0.25).abs() < 1e-4 => -0.674,
        q if (q - 0.50).abs() < 1e-4 => 0.0,
        q if (q - 0.75).abs() < 1e-4 => 0.674,
        q if (q - 0.90).abs() < 1e-4 => 1.282,
        q if (q - 0.95).abs() < 1e-4 => 1.645,
        q if (q - 0.99).abs() < 1e-4 => 2.326,
        q => (q / (1.0 - q)).ln() * 0.55,
    }
}

fn normalize_probs(mut values: BTreeMap<String, f32>) -> BTreeMap<String, f32> {
    let total = values.values().sum::<f32>().max(1e-6);
    for value in values.values_mut() {
        *value /= total;
    }
    values
}

fn sum_decomposition(decomposition: &ForecastDecomposition) -> Vec<f32> {
    let len = decomposition
        .baseline
        .len()
        .max(decomposition.seasonal.len())
        .max(decomposition.event.len())
        .max(decomposition.residual.len());
    (0..len)
        .map(|index| {
            decomposition
                .baseline
                .get(index)
                .copied()
                .unwrap_or_default()
                + decomposition
                    .seasonal
                    .get(index)
                    .copied()
                    .unwrap_or_default()
                + decomposition.event.get(index).copied().unwrap_or_default()
                + decomposition
                    .residual
                    .get(index)
                    .copied()
                    .unwrap_or_default()
        })
        .collect()
}

fn build_projection(options: &ForecastOptions, values: &[f32]) -> Option<ForecastConstraints> {
    let mut constraints = options.constraints.clone();
    if options.enforce_constraints && values.iter().all(|value| *value >= 0.0) {
        constraints.non_negative = true;
    }
    (options.enforce_constraints || constraints.has_any()).then_some(constraints)
}

#[derive(Clone, Debug)]
struct ImputedHistory {
    values: Vec<f32>,
    has_missing: bool,
}

fn impute_history(values: &[f32]) -> ImputedHistory {
    if values.is_empty() {
        return ImputedHistory {
            values: vec![0.0],
            has_missing: false,
        };
    }
    let mut imputed = values.to_vec();
    let has_missing = imputed.iter().any(|value| !value.is_finite());
    let mut index = 0;
    while index < imputed.len() {
        if imputed[index].is_finite() {
            index += 1;
            continue;
        }
        let start = index;
        while index < imputed.len() && !imputed[index].is_finite() {
            index += 1;
        }
        let end = index;
        let left = start
            .checked_sub(1)
            .and_then(|left| imputed.get(left))
            .copied();
        let right = imputed.get(end).copied();
        match (left, right) {
            (Some(left), Some(right)) if left.is_finite() && right.is_finite() => {
                let span = (end - start + 1) as f32;
                for (offset, value) in imputed[start..end].iter_mut().enumerate() {
                    let weight = (offset + 1) as f32 / span;
                    *value = left + (right - left) * weight;
                }
            }
            (Some(left), _) if left.is_finite() => {
                for value in &mut imputed[start..end] {
                    *value = left;
                }
            }
            (_, Some(right)) if right.is_finite() => {
                for value in &mut imputed[start..end] {
                    *value = right;
                }
            }
            _ => {
                for value in &mut imputed[start..end] {
                    *value = 0.0;
                }
            }
        }
    }
    ImputedHistory {
        values: imputed,
        has_missing,
    }
}
