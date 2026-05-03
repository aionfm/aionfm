use aionfm_utils::{AionResult, PatchBatch};
use async_trait::async_trait;
use std::collections::VecDeque;

/// Async stream of normalized patches ready for model consumption.
#[async_trait]
pub trait DataStream: Send {
    async fn next_batch(&mut self) -> AionResult<Option<PatchBatch>>;
}

/// In-memory stream used by tests and examples.
#[derive(Clone, Debug, Default)]
pub struct VecDataStream {
    batches: VecDeque<PatchBatch>,
}

impl VecDataStream {
    pub fn new(batches: impl IntoIterator<Item = PatchBatch>) -> Self {
        Self {
            batches: batches.into_iter().collect(),
        }
    }
}

#[async_trait]
impl DataStream for VecDataStream {
    async fn next_batch(&mut self) -> AionResult<Option<PatchBatch>> {
        Ok(self.batches.pop_front())
    }
}
