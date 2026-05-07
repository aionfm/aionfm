use aionfm_utils::ForecastResponse;

/// Deterministic scenario sampler for responses that did not request model-side scenarios.
#[derive(Clone, Debug)]
pub struct ScenarioSampler {
    default_count: usize,
}

impl Default for ScenarioSampler {
    fn default() -> Self {
        Self { default_count: 8 }
    }
}

impl ScenarioSampler {
    pub fn ensure_scenarios(&self, response: &mut ForecastResponse) {
        for result in &mut response.results {
            if result.scenario_paths.is_none() {
                let scale = result
                    .prediction_intervals
                    .get("80")
                    .and_then(|interval| interval.upper.first().zip(interval.lower.first()))
                    .map(|(upper, lower)| ((upper - lower) / 2.0).abs())
                    .unwrap_or_else(|| {
                        result
                            .point_forecast
                            .iter()
                            .copied()
                            .map(f32::abs)
                            .fold(1.0, f32::max)
                            * 0.05
                    })
                    .max(1e-3);
                result.scenario_paths = Some(
                    (0..self.default_count)
                        .map(|scenario| {
                            let centered = scenario as f32 - (self.default_count - 1) as f32 / 2.0;
                            result
                                .point_forecast
                                .iter()
                                .enumerate()
                                .map(|(step, value)| {
                                    value
                                        + centered * scale * 0.25 * ((step + 1) as f32).sqrt()
                                        + ((scenario + 1) as f32 * (step + 1) as f32).sin()
                                            * scale
                                            * 0.1
                                })
                                .collect()
                        })
                        .collect(),
                );
            }
        }
    }

    pub fn apply_controls(
        &self,
        response: &mut ForecastResponse,
        scenario_type: Option<&str>,
        forced_regimes: &[String],
    ) {
        self.ensure_scenarios(response);
        let multiplier = match scenario_type.unwrap_or("median") {
            "optimistic" => 0.85,
            "conservative" => 1.10,
            "tail-risk" | "stress" => 1.35,
            _ => 1.0,
        };
        for result in &mut response.results {
            if let Some(paths) = &mut result.scenario_paths {
                for (path_index, path) in paths.iter_mut().enumerate() {
                    let center = result
                        .point_forecast
                        .get(path_index % result.point_forecast.len().max(1))
                        .copied()
                        .unwrap_or_default();
                    for value in path {
                        *value = center + (*value - center) * multiplier;
                    }
                }
            }
            if !forced_regimes.is_empty() {
                let timeline = result.regime_timeline.get_or_insert_with(|| {
                    (0..result.forecast_horizon)
                        .map(|horizon_index| aionfm_utils::RegimeStep {
                            horizon_index,
                            label: "forced".into(),
                            probability: 1.0,
                            change_point_probability: 0.0,
                            volatility_state: "controlled".into(),
                        })
                        .collect()
                });
                for (index, regime) in forced_regimes.iter().enumerate() {
                    if let Some(step) = timeline.get_mut(index) {
                        step.label = regime.clone();
                        step.probability = 1.0;
                        step.change_point_probability = 1.0;
                    }
                }
            }
        }
    }
}
