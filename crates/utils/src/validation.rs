use crate::{AionError, AionResult, BatchForecastRequest, ForecastOptions, QuantileLevel};

pub fn validate_forecast_options(options: &ForecastOptions) -> AionResult<()> {
    validate_horizon(options.horizon)?;
    validate_quantiles(&options.quantiles)?;
    if let Some(count) = options.scenario_count {
        if count == 0 {
            return Err(AionError::Validation(
                "scenario_count must be greater than zero".into(),
            ));
        }
    }
    validate_constraints(options)?;
    Ok(())
}

pub fn validate_batch_request(request: &BatchForecastRequest) -> AionResult<()> {
    validate_forecast_options(&request.forecast_options())?;
    if request.entities.is_empty() {
        return Err(AionError::Validation(
            "at least one entity is required".into(),
        ));
    }
    for entity in &request.entities {
        if entity.entity_id.trim().is_empty() {
            return Err(AionError::Validation("entity_id must not be empty".into()));
        }
        if entity.target.trim().is_empty() {
            return Err(AionError::Validation("target must not be empty".into()));
        }
        if entity.historical_values.is_empty() {
            return Err(AionError::Validation(format!(
                "entity {} requires historical_values",
                entity.entity_id
            )));
        }
    }
    Ok(())
}

fn validate_constraints(options: &ForecastOptions) -> AionResult<()> {
    if let (Some(min), Some(max)) = (options.constraints.min_value, options.constraints.max_value) {
        if min > max {
            return Err(AionError::Validation(
                "constraint min_value must be less than or equal to max_value".into(),
            ));
        }
    }
    for index in &options.constraints.closed_horizon_indices {
        if *index >= options.horizon {
            return Err(AionError::Validation(format!(
                "closed_horizon_indices contains {index}, outside horizon {}",
                options.horizon
            )));
        }
    }
    Ok(())
}

pub fn validate_horizon(horizon: usize) -> AionResult<()> {
    if horizon == 0 {
        return Err(AionError::Validation(
            "horizon must be greater than zero".into(),
        ));
    }
    Ok(())
}

pub fn validate_quantiles(levels: &[QuantileLevel]) -> AionResult<()> {
    for level in levels {
        if !(0.0..1.0).contains(&level.0) {
            return Err(AionError::Validation(format!(
                "quantile {} must be in the open interval (0, 1)",
                level.0
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_horizon() {
        assert!(validate_horizon(0).is_err());
    }

    #[test]
    fn rejects_invalid_constraint_range() {
        let options = ForecastOptions {
            horizon: 1,
            constraints: crate::ForecastConstraints {
                min_value: Some(5.0),
                max_value: Some(1.0),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(validate_forecast_options(&options).is_err());
    }
}
