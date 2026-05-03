use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Common forecast frequencies.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Frequency {
    Minutely,
    Hourly,
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Yearly,
    Custom(String),
}

impl Default for Frequency {
    fn default() -> Self {
        Self::Daily
    }
}

impl Frequency {
    /// Returns a compact schema code suitable for API responses.
    pub fn code(&self) -> &str {
        match self {
            Self::Minutely => "T",
            Self::Hourly => "H",
            Self::Daily => "D",
            Self::Weekly => "W",
            Self::Monthly => "M",
            Self::Quarterly => "Q",
            Self::Yearly => "Y",
            Self::Custom(code) => code.as_str(),
        }
    }
}

/// Arbitrary metadata associated with an entity or time series.
pub type Metadata = BTreeMap<String, serde_json::Value>;

/// Static metadata used for routing adapters, constraints, and UI context.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EntityMetadata {
    pub entity_id: String,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub frequency: Frequency,
    #[serde(default)]
    pub attributes: Metadata,
}

/// A single named covariate aligned to the target series.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CovariateSeries {
    pub name: String,
    pub values: Vec<f32>,
    #[serde(default)]
    pub known_future: bool,
}

/// Raw or normalized entity series.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntitySeries {
    pub entity_id: String,
    pub target: String,
    pub values: Vec<f32>,
    #[serde(default)]
    pub timestamps: Vec<String>,
    #[serde(default)]
    pub covariates: Vec<CovariateSeries>,
    #[serde(default)]
    pub metadata: EntityMetadata,
}

/// Boolean observation mask where `true` means the value is observed.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ObservationMask {
    pub observed: Vec<bool>,
}

/// Continuous value patch used by the value stream.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValuePatch {
    pub entity_id: String,
    pub start_index: usize,
    pub values: Vec<f32>,
    #[serde(default)]
    pub mask: ObservationMask,
}

/// Covariate patch aligned with a value patch.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CovariatePatch {
    pub entity_id: String,
    pub start_index: usize,
    pub covariates: BTreeMap<String, Vec<f32>>,
}

/// Mini-batch emitted by the data layer for training and inference.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PatchBatch {
    pub value_patches: Vec<ValuePatch>,
    pub covariate_patches: Vec<CovariatePatch>,
    pub metadata: Vec<EntityMetadata>,
}
