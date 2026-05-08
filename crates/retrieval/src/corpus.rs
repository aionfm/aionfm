use crate::SimilarityMetric;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Candidate source pool for retrieval.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalSource {
    InEntity,
    SimilarEntities,
    DomainWide,
}

impl Default for RetrievalSource {
    fn default() -> Self {
        Self::InEntity
    }
}

/// Query sent to a retrieval index.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievalQuery {
    pub context: Vec<f32>,
    #[serde(default)]
    pub horizon: usize,
    #[serde(default = "default_k")]
    pub k: usize,
    #[serde(default)]
    pub source: RetrievalSource,
    #[serde(default)]
    pub metric: SimilarityMetric,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regime_label: Option<String>,
}

impl RetrievalQuery {
    pub fn in_entity(entity_id: impl Into<String>, context: Vec<f32>, horizon: usize) -> Self {
        Self {
            context,
            horizon,
            k: default_k(),
            source: RetrievalSource::InEntity,
            metric: SimilarityMetric::default(),
            tenant_id: None,
            domain: None,
            entity_id: Some(entity_id.into()),
            regime_label: None,
        }
    }

    pub fn top_k(mut self, k: usize) -> Self {
        self.k = k.max(1);
        self
    }

    pub fn regime_label(mut self, label: impl Into<String>) -> Self {
        self.regime_label = Some(label.into());
        self
    }
}

fn default_k() -> usize {
    3
}

/// Historical segment stored in a retrieval corpus.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievalWindow {
    pub entity_id: String,
    pub target: String,
    pub start_index: usize,
    pub values: Vec<f32>,
    #[serde(default)]
    pub outcome: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regime_label: Option<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

impl RetrievalWindow {
    pub fn new(
        entity_id: impl Into<String>,
        target: impl Into<String>,
        start_index: usize,
        values: Vec<f32>,
        outcome: Vec<f32>,
    ) -> Self {
        Self {
            entity_id: entity_id.into(),
            target: target.into(),
            start_index,
            values,
            outcome,
            tenant_id: None,
            domain: None,
            regime_label: None,
            metadata: BTreeMap::new(),
        }
    }

    pub fn regime_label(mut self, label: impl Into<String>) -> Self {
        self.regime_label = Some(label.into());
        self
    }
}
