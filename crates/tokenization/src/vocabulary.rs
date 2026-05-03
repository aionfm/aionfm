use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Hierarchical token level from macro to micro behavior.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegimeLevel {
    Macro,
    Meso,
    Micro,
}

/// Discrete regime token definition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegimeToken {
    pub id: u32,
    pub level: RegimeLevel,
    pub label: String,
    #[serde(default)]
    pub description: Option<String>,
}

/// Vocabulary manager for interpretable regime mappings.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RegimeVocabulary {
    tokens: BTreeMap<u32, RegimeToken>,
    labels: BTreeMap<String, u32>,
}

impl RegimeVocabulary {
    pub fn insert(&mut self, token: RegimeToken) {
        self.labels.insert(token.label.clone(), token.id);
        self.tokens.insert(token.id, token);
    }

    pub fn token(&self, id: u32) -> Option<&RegimeToken> {
        self.tokens.get(&id)
    }

    pub fn id_for_label(&self, label: &str) -> Option<u32> {
        self.labels.get(label).copied()
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}
