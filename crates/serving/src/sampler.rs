use aionfm_utils::ForecastResponse;

/// Scenario sampler placeholder for regime/value two-stage generation.
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
                result.scenario_paths =
                    Some(vec![result.point_forecast.clone(); self.default_count]);
            }
        }
    }
}
