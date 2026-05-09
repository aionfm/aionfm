use crate::{
    AionError, AionResult, BatchForecastRequest, ForecastOptions, QuantileLevel, RequestContext,
};
use std::collections::BTreeSet;

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
    validate_request_context(request.context.as_ref())?;
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
    validate_hierarchy(request)?;
    Ok(())
}

pub fn validate_request_context(context: Option<&RequestContext>) -> AionResult<()> {
    let Some(context) = context else {
        return Ok(());
    };
    validate_context_field("tenant_id", context.tenant_id.as_deref())?;
    validate_context_field("actor_id", context.actor_id.as_deref())?;
    validate_context_field("trace_id", context.trace_id.as_deref())?;
    validate_context_field("purpose", context.purpose.as_deref())?;
    Ok(())
}

fn validate_context_field(name: &str, value: Option<&str>) -> AionResult<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.trim().is_empty() {
        return Err(AionError::Validation(format!(
            "request context {name} must not be empty"
        )));
    }
    if value.len() > 128 {
        return Err(AionError::Validation(format!(
            "request context {name} must be 128 characters or fewer"
        )));
    }
    if !value.chars().all(|character| {
        character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.' | ':' | '/')
    }) {
        return Err(AionError::Validation(format!(
            "request context {name} contains unsupported characters"
        )));
    }
    Ok(())
}

fn validate_hierarchy(request: &BatchForecastRequest) -> AionResult<()> {
    let Some(hierarchy) = &request.options.hierarchy else {
        return Ok(());
    };
    let entity_ids = request
        .entities
        .iter()
        .map(|entity| entity.entity_id.as_str())
        .collect::<BTreeSet<_>>();
    for relation in &hierarchy.relations {
        if relation.parent_entity_id.trim().is_empty() {
            return Err(AionError::Validation(
                "hierarchy parent_entity_id must not be empty".into(),
            ));
        }
        if !entity_ids.contains(relation.parent_entity_id.as_str()) {
            return Err(AionError::Validation(format!(
                "hierarchy parent {} is not present in the forecast batch",
                relation.parent_entity_id
            )));
        }
        if relation.child_entity_ids.is_empty() {
            return Err(AionError::Validation(format!(
                "hierarchy parent {} requires at least one child",
                relation.parent_entity_id
            )));
        }
        let mut children = BTreeSet::new();
        for child in &relation.child_entity_ids {
            if child.trim().is_empty() {
                return Err(AionError::Validation(
                    "hierarchy child_entity_ids must not contain empty IDs".into(),
                ));
            }
            if child == &relation.parent_entity_id {
                return Err(AionError::Validation(format!(
                    "hierarchy parent {} cannot include itself as a child",
                    relation.parent_entity_id
                )));
            }
            if !entity_ids.contains(child.as_str()) {
                return Err(AionError::Validation(format!(
                    "hierarchy child {child} is not present in the forecast batch"
                )));
            }
            if !children.insert(child.as_str()) {
                return Err(AionError::Validation(format!(
                    "hierarchy child {child} is duplicated under parent {}",
                    relation.parent_entity_id
                )));
            }
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

    #[test]
    fn rejects_hierarchy_children_missing_from_batch() {
        let request = BatchForecastRequest {
            entities: vec![crate::ForecastEntity {
                entity_id: "parent".into(),
                target: "value".into(),
                historical_values: vec![1.0],
                frequency: Default::default(),
                covariates: vec![],
                metadata: Default::default(),
            }],
            context: None,
            horizon: 1,
            options: crate::RequestOptions {
                hierarchy: Some(crate::HierarchySpec {
                    relations: vec![crate::HierarchyRelation {
                        parent_entity_id: "parent".into(),
                        child_entity_ids: vec!["missing".into()],
                    }],
                    ..Default::default()
                }),
                ..Default::default()
            },
            quantiles: vec![QuantileLevel::Q50],
            request_id: Default::default(),
            scenario_count: None,
        };
        assert!(validate_batch_request(&request).is_err());
    }

    #[test]
    fn rejects_empty_tenant_context() {
        assert!(validate_request_context(Some(&RequestContext {
            tenant_id: Some(" ".into()),
            ..Default::default()
        }))
        .is_err());
    }
}
