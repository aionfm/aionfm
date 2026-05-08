use aionfm_utils::{
    EntityForecast, ForecastResponse, HierarchySpec, ReconciliationMethod, ReconciliationReport,
};
use std::collections::BTreeMap;

/// Applies hierarchical reconciliation to a batch forecast response.
#[derive(Clone, Debug, Default)]
pub struct HierarchicalReconciler;

impl HierarchicalReconciler {
    pub fn reconcile(&self, response: &mut ForecastResponse, hierarchy: &HierarchySpec) {
        if hierarchy.relations.is_empty() {
            return;
        }
        match hierarchy.method {
            ReconciliationMethod::BottomUp => self.bottom_up(response, hierarchy),
            ReconciliationMethod::TopDown | ReconciliationMethod::MiddleOut => {
                self.bottom_up(response, hierarchy)
            }
        }
    }

    fn bottom_up(&self, response: &mut ForecastResponse, hierarchy: &HierarchySpec) {
        let index = response
            .results
            .iter()
            .enumerate()
            .map(|(index, forecast)| (forecast.entity_id.clone(), index))
            .collect::<BTreeMap<_, _>>();
        let mut adjusted_entities = Vec::new();
        for relation in &hierarchy.relations {
            let Some(parent_index) = index.get(&relation.parent_entity_id).copied() else {
                continue;
            };
            let children = relation
                .child_entity_ids
                .iter()
                .filter_map(|entity_id| {
                    index
                        .get(entity_id)
                        .and_then(|index| response.results.get(*index))
                })
                .cloned()
                .collect::<Vec<_>>();
            if children.is_empty() {
                continue;
            }
            let parent = &mut response.results[parent_index];
            reconcile_parent(parent, &children);
            adjusted_entities.push(parent.entity_id.clone());
        }
        if !adjusted_entities.is_empty() {
            response.reconciliation_report = Some(ReconciliationReport {
                method: hierarchy.method.clone(),
                adjusted_entities,
                notes: vec!["reconciled aggregate forecasts using bottom-up child sums".into()],
            });
        }
    }
}

fn reconcile_parent(parent: &mut EntityForecast, children: &[EntityForecast]) {
    parent.point_forecast =
        sum_vectors(children.iter().map(|child| child.point_forecast.as_slice()));
    let quantile_keys = parent.quantiles.keys().cloned().collect::<Vec<_>>();
    for key in quantile_keys {
        parent.quantiles.insert(
            key.clone(),
            sum_vectors(
                children
                    .iter()
                    .filter_map(|child| child.quantiles.get(&key).map(Vec::as_slice)),
            ),
        );
    }
    for (coverage, interval) in parent.prediction_intervals.iter_mut() {
        interval.lower = sum_vectors(children.iter().filter_map(|child| {
            child
                .prediction_intervals
                .get(coverage)
                .map(|interval| interval.lower.as_slice())
        }));
        interval.upper = sum_vectors(children.iter().filter_map(|child| {
            child
                .prediction_intervals
                .get(coverage)
                .map(|interval| interval.upper.as_slice())
        }));
    }
}

fn sum_vectors<'a>(vectors: impl IntoIterator<Item = &'a [f32]>) -> Vec<f32> {
    let vectors = vectors.into_iter().collect::<Vec<_>>();
    let len = vectors
        .iter()
        .map(|values| values.len())
        .max()
        .unwrap_or_default();
    (0..len)
        .map(|index| {
            vectors
                .iter()
                .map(|values| values.get(index).copied().unwrap_or_default())
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use aionfm_utils::{ForecastResponse, HierarchyRelation};

    #[test]
    fn bottom_up_reconciles_parent_point_forecast() {
        let mut response = ForecastResponse::new(
            "AionFM",
            "test",
            vec![
                entity("parent", vec![0.0, 0.0]),
                entity("child_a", vec![1.0, 2.0]),
                entity("child_b", vec![3.0, 4.0]),
            ],
        );
        HierarchicalReconciler.reconcile(
            &mut response,
            &HierarchySpec {
                method: ReconciliationMethod::BottomUp,
                relations: vec![HierarchyRelation {
                    parent_entity_id: "parent".into(),
                    child_entity_ids: vec!["child_a".into(), "child_b".into()],
                }],
            },
        );
        assert_eq!(response.results[0].point_forecast, vec![4.0, 6.0]);
        assert!(response.reconciliation_report.is_some());
    }

    fn entity(entity_id: &str, point_forecast: Vec<f32>) -> EntityForecast {
        EntityForecast {
            entity_id: entity_id.into(),
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
        }
    }
}
