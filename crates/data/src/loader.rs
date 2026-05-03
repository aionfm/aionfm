use aionfm_utils::{AionError, AionResult, EntitySeries, Frequency};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Supported corpus storage formats from the spec.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataFormat {
    Parquet,
    ArrowIpc,
    Csv,
    ZippedCsv,
    MetadataCatalog,
}

/// Data-loading request used by corpus readers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoadRequest {
    pub path: PathBuf,
    pub format: DataFormat,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub frequency: Option<Frequency>,
    #[serde(default)]
    pub selected_columns: Vec<String>,
}

/// Abstract loader so parquet, arrow, CSV, and streaming backends are swappable.
#[async_trait]
pub trait DataLoader: Send + Sync {
    async fn load_series(&self, request: LoadRequest) -> AionResult<Vec<EntitySeries>>;
}

/// Placeholder loader that preserves the API surface while storage backends are added.
#[derive(Clone, Debug, Default)]
pub struct UnsupportedDataLoader;

#[async_trait]
impl DataLoader for UnsupportedDataLoader {
    async fn load_series(&self, request: LoadRequest) -> AionResult<Vec<EntitySeries>> {
        Err(AionError::Unsupported(format!(
            "{:?} loading is not implemented yet for {}",
            request.format,
            request.path.display()
        )))
    }
}
