use aionfm_utils::{
    EntityForecast, Explanation, ForecastEntity, ForecastOptions, Frequency, PredictionInterval,
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
        let values = finite_values(&entity.historical_values);
        let profile = SeriesProfile::from_values(&values, &entity.frequency);
        let mut point_forecast = profile.point_forecast(options.horizon);
        if options.enforce_constraints && values.iter().all(|value| *value >= 0.0) {
            clamp_nonnegative(&mut point_forecast);
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
            if options.enforce_constraints && values.iter().all(|value| *value >= 0.0) {
                clamp_nonnegative(&mut forecast);
            }
            quantiles.insert(quantile.key(), forecast);
        }
        let prediction_intervals =
            profile.prediction_intervals(&point_forecast, options.enforce_constraints);
        let scenario_paths = options.return_scenarios.then(|| {
            profile.scenarios(
                &point_forecast,
                options.scenario_count.unwrap_or(8),
                options.enforce_constraints,
            )
        });
        let regime_probabilities = options
            .return_regimes
            .then(|| profile.regime_probabilities());
        let current_regime = profile.primary_regime();
        EntityForecast {
            entity_id: entity.entity_id.clone(),
            forecast_horizon: options.horizon,
            frequency: entity.frequency.code().into(),
            target: entity.target.clone(),
            point_forecast,
            quantiles,
            prediction_intervals,
            scenario_paths,
            regime_probabilities,
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
        Self {
            values,
            last_value,
            slope,
            residual_scale,
            seasonal_offsets,
            volatility_ratio,
        }
    }

    fn point_forecast(&self, horizon: usize) -> Vec<f32> {
        (0..horizon)
            .map(|step| {
                let seasonal = if self.seasonal_offsets.is_empty() {
                    0.0
                } else {
                    self.seasonal_offsets[(self.values.len() + step) % self.seasonal_offsets.len()]
                };
                self.last_value + self.slope * (step + 1) as f32 + seasonal
            })
            .collect()
    }

    fn prediction_intervals(
        &self,
        point: &[f32],
        enforce_constraints: bool,
    ) -> BTreeMap<String, PredictionInterval> {
        let mut intervals = BTreeMap::new();
        for (coverage, score) in [("80", 1.2816_f32), ("95", 1.96_f32)] {
            let (mut lower, mut upper): (Vec<f32>, Vec<f32>) = point
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let spread = score * self.residual_scale * ((index + 1) as f32).sqrt();
                    (value - spread, value + spread)
                })
                .unzip();
            if enforce_constraints && self.values.iter().all(|value| *value >= 0.0) {
                clamp_nonnegative(&mut lower);
                clamp_nonnegative(&mut upper);
            }
            intervals.insert(coverage.into(), PredictionInterval { lower, upper });
        }
        intervals
    }

    fn scenarios(&self, point: &[f32], count: usize, enforce_constraints: bool) -> Vec<Vec<f32>> {
        let count = count.max(1);
        (0..count)
            .map(|scenario| {
                let centered = scenario as f32 - (count.saturating_sub(1)) as f32 / 2.0;
                let phase = (scenario + 1) as f32 * 0.73;
                let mut path = point
                    .iter()
                    .enumerate()
                    .map(|(step, point)| {
                        let drift =
                            centered * self.residual_scale * 0.35 * ((step + 1) as f32).sqrt();
                        let wave = ((step + 1) as f32 * phase).sin() * self.residual_scale * 0.25;
                        point + drift + wave
                    })
                    .collect::<Vec<_>>();
                if enforce_constraints && self.values.iter().all(|value| *value >= 0.0) {
                    clamp_nonnegative(&mut path);
                }
                path
            })
            .collect()
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

fn clamp_nonnegative(values: &mut [f32]) {
    for value in values {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
}
