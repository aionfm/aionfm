use crate::AdapterConfig;
use std::collections::BTreeMap;

/// Registry mapping domains and names to adapter configs.
#[derive(Clone, Debug, Default)]
pub struct AdapterRegistry {
    adapters: BTreeMap<String, AdapterConfig>,
}

impl AdapterRegistry {
    pub fn register(&mut self, config: AdapterConfig) {
        self.adapters.insert(config.name.clone(), config);
    }

    pub fn get(&self, name: &str) -> Option<&AdapterConfig> {
        self.adapters.get(name)
    }

    pub fn for_domain(&self, domain: &str) -> Vec<&AdapterConfig> {
        self.adapters
            .values()
            .filter(|config| config.domain == domain)
            .collect()
    }
}
