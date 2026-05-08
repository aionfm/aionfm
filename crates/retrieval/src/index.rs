use crate::{similarity, RetrievalQuery, RetrievalSource, RetrievalWindow};
use aionfm_utils::RetrievalMatch;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RetrievalError {
    #[error("retrieval query context must not be empty")]
    EmptyContext,
    #[error("retrieval window length must be greater than zero")]
    InvalidWindow,
}

pub type RetrievalResult<T> = Result<T, RetrievalError>;

/// Search contract for exact, approximate, local, or remote retrieval indexes.
pub trait RetrievalIndex {
    fn search(&self, query: &RetrievalQuery) -> RetrievalResult<Vec<RetrievalMatch>>;
}

/// In-memory exact search index used by tests and local baseline serving.
#[derive(Clone, Debug, Default)]
pub struct InMemoryRetrievalIndex {
    windows: Vec<RetrievalWindow>,
}

impl InMemoryRetrievalIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.windows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    pub fn insert(&mut self, window: RetrievalWindow) -> RetrievalResult<()> {
        if window.values.is_empty() {
            return Err(RetrievalError::InvalidWindow);
        }
        self.windows.push(window);
        Ok(())
    }

    pub fn index_series(
        &mut self,
        entity_id: impl AsRef<str>,
        target: impl AsRef<str>,
        values: &[f32],
        window_len: usize,
        horizon: usize,
        regime_label: Option<&str>,
    ) -> RetrievalResult<()> {
        if window_len == 0 {
            return Err(RetrievalError::InvalidWindow);
        }
        let required = window_len + horizon.max(1);
        if values.len() < required {
            return Ok(());
        }
        let max_start = values.len() - required;
        for start in 0..=max_start {
            let outcome_start = start + window_len;
            let outcome_end = (outcome_start + horizon).min(values.len());
            let mut window = RetrievalWindow::new(
                entity_id.as_ref(),
                target.as_ref(),
                start,
                values[start..outcome_start].to_vec(),
                values[outcome_start..outcome_end].to_vec(),
            );
            if let Some(label) = regime_label {
                window = window.regime_label(label);
            }
            self.insert(window)?;
        }
        Ok(())
    }
}

impl RetrievalIndex for InMemoryRetrievalIndex {
    fn search(&self, query: &RetrievalQuery) -> RetrievalResult<Vec<RetrievalMatch>> {
        if query.context.is_empty() {
            return Err(RetrievalError::EmptyContext);
        }
        let mut matches = self
            .windows
            .iter()
            .filter(|window| matches_scope(window, query))
            .map(|window| RetrievalMatch {
                source_entity_id: window.entity_id.clone(),
                start_index: window.start_index,
                window_len: window.values.len(),
                similarity: similarity(query.metric, &query.context, &window.values),
                regime_label: window
                    .regime_label
                    .clone()
                    .or_else(|| query.regime_label.clone()),
                outcome_preview: window.outcome.iter().take(query.horizon).copied().collect(),
            })
            .collect::<Vec<_>>();
        matches.sort_by(|left, right| right.similarity.total_cmp(&left.similarity));
        matches.truncate(query.k.max(1));
        Ok(matches)
    }
}

fn matches_scope(window: &RetrievalWindow, query: &RetrievalQuery) -> bool {
    if query
        .tenant_id
        .as_deref()
        .is_some_and(|id| window.tenant_id.as_deref() != Some(id))
    {
        return false;
    }
    if query
        .domain
        .as_deref()
        .is_some_and(|domain| window.domain.as_deref() != Some(domain))
    {
        return false;
    }
    match query.source {
        RetrievalSource::InEntity => query
            .entity_id
            .as_deref()
            .is_some_and(|id| window.entity_id == id),
        RetrievalSource::SimilarEntities | RetrievalSource::DomainWide => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_index_returns_top_k_in_entity_matches() {
        let mut index = InMemoryRetrievalIndex::new();
        index
            .index_series(
                "store_42",
                "demand",
                &[1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 10.0, 11.0],
                3,
                2,
                Some("stable_growth"),
            )
            .unwrap();
        let matches = index
            .search(&RetrievalQuery::in_entity("store_42", vec![1.0, 2.0, 3.0], 2).top_k(2))
            .unwrap();
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].source_entity_id, "store_42");
        assert!(matches[0].similarity >= matches[1].similarity);
    }
}
