use aionfm_utils::EntityForecast;
use serde::{Deserialize, Serialize};

/// Post-hoc quantile calibration offsets.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct QuantileCalibration {
    pub offsets: std::collections::BTreeMap<String, f32>,
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
    }
}
