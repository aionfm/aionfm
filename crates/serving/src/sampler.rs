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

#[cfg(test)]
mod tests {
    use super::*;
    use aionfm_utils::EntityForecast;
    use std::collections::BTreeMap;

    #[test]
    fn generated_scenarios_honor_requested_count() {
        let mut response = response_with_point(vec![10.0, 12.0, 14.0]);

        ScenarioSampler::default().ensure_scenarios_with_count(&mut response, Some(3));

        assert_eq!(
            response.results[0].scenario_paths.as_ref().unwrap().len(),
            3
        );
    }

    #[test]
    fn stress_controls_scale_each_step_around_its_point_forecast() {
        let mut response = response_with_point(vec![10.0, 20.0]);
        response.results[0].scenario_paths = Some(vec![vec![11.0, 22.0]]);

        ScenarioSampler::default().apply_controls_with_count(
            &mut response,
            Some("stress"),
            &[],
            Some(1),
        );

        let path = &response.results[0].scenario_paths.as_ref().unwrap()[0];
        assert!((path[0] - 11.35).abs() < 1e-4);
        assert!((path[1] - 22.70).abs() < 1e-4);
    }

    fn response_with_point(point_forecast: Vec<f32>) -> ForecastResponse {
        ForecastResponse::new(
            "AionFM",
            "test",
            vec![EntityForecast {
                entity_id: "entity".into(),
                forecast_horizon: point_forecast.len(),
                frequency: "D".into(),
                target: "value".into(),
                point_forecast,
                quantiles: BTreeMap::new(),
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
            }],
        )
    }
}

impl ScenarioSampler {
    pub fn ensure_scenarios(&self, response: &mut ForecastResponse) {
        self.ensure_scenarios_with_count(response, None);
    }

    pub fn ensure_scenarios_with_count(
        &self,
        response: &mut ForecastResponse,
        requested_count: Option<usize>,
    ) {
        let scenario_count = requested_count.unwrap_or(self.default_count).max(1);
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
                    (0..scenario_count)
                        .map(|scenario| {
                            let centered = scenario as f32 - (scenario_count - 1) as f32 / 2.0;
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
        self.apply_controls_with_count(response, scenario_type, forced_regimes, None);
    }

    pub fn apply_controls_with_count(
        &self,
        response: &mut ForecastResponse,
        scenario_type: Option<&str>,
        forced_regimes: &[String],
        requested_count: Option<usize>,
    ) {
        self.ensure_scenarios_with_count(response, requested_count);
        let multiplier = match scenario_type.unwrap_or("median") {
            "optimistic" => 0.85,
            "conservative" => 1.10,
            "tail-risk" | "stress" => 1.35,
            _ => 1.0,
        };
        for result in &mut response.results {
            if let Some(paths) = &mut result.scenario_paths {
                for path in paths {
                    for (step, value) in path.iter_mut().enumerate() {
                        let center = result.point_forecast.get(step).copied().unwrap_or_default();
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
