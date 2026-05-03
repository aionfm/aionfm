use aionfm_utils::{
    EntityForecast, Explanation, ForecastEntity, ForecastOptions, PredictionInterval,
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
            model_version: "0.1-skeleton".into(),
        }
    }
}

impl ForecastHeads {
    pub fn baseline_forecast(
        &self,
        entity: &ForecastEntity,
        options: &ForecastOptions,
    ) -> EntityForecast {
        let last_value = entity
            .historical_values
            .iter()
            .rev()
            .find(|value| value.is_finite())
            .copied()
            .unwrap_or_default();
        let point_forecast = vec![last_value; options.horizon];
        let mut quantiles = BTreeMap::new();
        for quantile in &options.quantiles {
            let spread = (quantile.0 - 0.5) * 0.1 * last_value.abs().max(1.0);
            quantiles.insert(quantile.key(), vec![last_value + spread; options.horizon]);
        }
        let mut prediction_intervals = BTreeMap::new();
        if let (Some(lower), Some(upper)) = (quantiles.get("0.10"), quantiles.get("0.90")) {
            prediction_intervals.insert(
                "80".into(),
                PredictionInterval {
                    lower: lower.clone(),
                    upper: upper.clone(),
                },
            );
        }
        let scenario_paths = if options.return_scenarios {
            let count = options.scenario_count.unwrap_or(1);
            Some(
                (0..count)
                    .map(|scenario| {
                        let offset = (scenario as f32 - count as f32 / 2.0)
                            * 0.01
                            * last_value.abs().max(1.0);
                        vec![last_value + offset; options.horizon]
                    })
                    .collect(),
            )
        } else {
            None
        };
        let regime_probabilities = options.return_regimes.then(|| {
            BTreeMap::from([
                ("stable".to_string(), 0.70),
                ("volatile".to_string(), 0.20),
                ("shock_recovery".to_string(), 0.10),
            ])
        });
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
                current_regime: Some("stable_baseline".into()),
                uncertainty_driver: Some("deterministic skeleton baseline".into()),
                change_point_probability: Some(0.0),
                notes: vec!["Replace ForecastHeads with learned decoders in production.".into()],
            }),
            metadata: entity.metadata.attributes.clone(),
        }
    }
}
