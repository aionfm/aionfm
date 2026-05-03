use aionfm_utils::AionResult;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Metadata for model and optimizer checkpoints.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub path: PathBuf,
    pub epoch: usize,
    pub step: usize,
    pub model_version: String,
}

/// Storage abstraction for local, object-store, or registry-backed checkpoints.
#[async_trait]
pub trait CheckpointStore: Send + Sync {
    async fn save_metadata(&self, metadata: &CheckpointMetadata) -> AionResult<()>;
    async fn latest(&self) -> AionResult<Option<CheckpointMetadata>>;
}

/// In-memory checkpoint catalog useful for tests.
#[derive(Clone, Debug, Default)]
pub struct MemoryCheckpointStore {
    latest: std::sync::Arc<std::sync::Mutex<Option<CheckpointMetadata>>>,
}

#[async_trait]
impl CheckpointStore for MemoryCheckpointStore {
    async fn save_metadata(&self, metadata: &CheckpointMetadata) -> AionResult<()> {
        *self.latest.lock().expect("checkpoint mutex poisoned") = Some(metadata.clone());
        Ok(())
    }

    async fn latest(&self) -> AionResult<Option<CheckpointMetadata>> {
        Ok(self
            .latest
            .lock()
            .expect("checkpoint mutex poisoned")
            .clone())
    }
}
