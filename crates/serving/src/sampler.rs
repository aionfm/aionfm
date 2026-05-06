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
}
