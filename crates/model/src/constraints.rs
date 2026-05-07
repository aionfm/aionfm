use aionfm_utils::{ConstraintReport, ForecastConstraints};
use std::collections::BTreeSet;

/// Projects forecast trajectories into a configured feasible set.
#[derive(Clone, Debug)]
pub struct ConstraintProjector {
    constraints: ForecastConstraints,
}

impl ConstraintProjector {
    pub fn new(constraints: ForecastConstraints) -> Self {
        Self { constraints }
    }

    pub fn constraints(&self) -> &ForecastConstraints {
        &self.constraints
    }

    pub fn project(&self, values: &mut [f32], label: &str) -> ConstraintReport {
        let before = values.to_vec();
        let closed: BTreeSet<usize> = self
            .constraints
            .closed_horizon_indices
            .iter()
            .copied()
            .collect();
        let min_value = self
            .constraints
            .min_value
            .or_else(|| self.constraints.non_negative.then_some(0.0));
        for (index, value) in values.iter_mut().enumerate() {
            if let Some(closed_value) = self
                .constraints
                .closed_value
                .filter(|_| closed.contains(&index))
            {
                *value = closed_value;
            }
            if let Some(min_value) = min_value {
                *value = (*value).max(min_value);
            }
            if let Some(max_value) = self.constraints.max_value {
                *value = (*value).min(max_value);
            }
            if self.constraints.integer {
                *value = value.round();
            }
        }
        if self.constraints.monotonic_non_decreasing {
            enforce_monotonic(values);
        }
        report_delta(&before, values, label)
    }
}

fn enforce_monotonic(values: &mut [f32]) {
    let mut previous = None;
    for value in values {
        if let Some(previous_value) = previous {
            if *value < previous_value {
                *value = previous_value;
            }
        }
        previous = Some(*value);
    }
}

fn report_delta(before: &[f32], after: &[f32], label: &str) -> ConstraintReport {
    let adjusted_points = before
        .iter()
        .zip(after)
        .filter(|(left, right)| (*left - *right).abs() > 1e-6)
        .count();
    ConstraintReport {
        applied: adjusted_points > 0,
        adjusted_points,
        notes: if adjusted_points > 0 {
            vec![format!("projected {adjusted_points} values in {label}")]
        } else {
            vec![]
        },
    }
}

pub fn merge_reports(
    reports: impl IntoIterator<Item = ConstraintReport>,
) -> Option<ConstraintReport> {
    let mut merged = ConstraintReport::default();
    for report in reports {
        merged.applied |= report.applied;
        merged.adjusted_points += report.adjusted_points;
        merged.notes.extend(report.notes);
    }
    (merged.applied || merged.adjusted_points > 0 || !merged.notes.is_empty()).then_some(merged)
}

pub fn enforce_quantile_monotonicity(quantiles: &mut std::collections::BTreeMap<String, Vec<f32>>) {
    let mut keys = quantiles.keys().cloned().collect::<Vec<_>>();
    keys.sort_by(|left, right| {
        parse_quantile(left)
            .partial_cmp(&parse_quantile(right))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for horizon in 0..max_len(quantiles.values()) {
        let mut previous = None;
        for key in &keys {
            let Some(values) = quantiles.get_mut(key) else {
                continue;
            };
            let Some(value) = values.get_mut(horizon) else {
                continue;
            };
            if let Some(previous_value) = previous {
                if *value < previous_value {
                    *value = previous_value;
                }
            }
            previous = Some(*value);
        }
    }
}

fn parse_quantile(value: &str) -> f32 {
    value.parse::<f32>().unwrap_or(0.5)
}

fn max_len<'a>(values: impl Iterator<Item = &'a Vec<f32>>) -> usize {
    values.map(Vec::len).max().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projects_to_capacity_and_monotonicity() {
        let projector = ConstraintProjector::new(ForecastConstraints {
            min_value: Some(0.0),
            max_value: Some(10.0),
            monotonic_non_decreasing: true,
            ..Default::default()
        });
        let mut values = vec![3.0, -1.0, 12.0, 8.0];
        let report = projector.project(&mut values, "test");
        assert!(report.applied);
        assert_eq!(values, vec![3.0, 3.0, 10.0, 10.0]);
    }
}
