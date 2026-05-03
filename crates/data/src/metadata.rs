use aionfm_utils::{EntityMetadata, Metadata};
use std::collections::BTreeMap;

/// Registry for static metadata used by loaders, adapters, and serving.
#[derive(Clone, Debug, Default)]
pub struct MetadataManager {
    entities: BTreeMap<String, EntityMetadata>,
}

impl MetadataManager {
    pub fn insert(&mut self, metadata: EntityMetadata) {
        self.entities.insert(metadata.entity_id.clone(), metadata);
    }

    pub fn get(&self, entity_id: &str) -> Option<&EntityMetadata> {
        self.entities.get(entity_id)
    }

    pub fn attributes_for(&self, entity_id: &str) -> Metadata {
        self.entities
            .get(entity_id)
            .map(|metadata| metadata.attributes.clone())
            .unwrap_or_default()
    }

    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }
}
