use crate::{LossAggregator, LossBreakdown, OptimizerConfig, OptimizerState};
use aionfm_data::DataStream;
use aionfm_model::ForecastModel;
use aionfm_utils::AionResult;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// High-level training configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub checkpoint_every_steps: usize,
    pub optimizer: OptimizerConfig,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 1,
            batch_size: 32,
            checkpoint_every_steps: 1_000,
            optimizer: OptimizerConfig::default(),
        }
    }
}

/// Training progress summary.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TrainingReport {
    pub epochs_completed: usize,
    pub steps_completed: usize,
    pub last_loss: f32,
}

/// Orchestrates data streaming, losses, optimizer state, and checkpoints.
pub struct Trainer<M> {
    pub config: TrainingConfig,
    pub losses: LossAggregator,
    pub optimizer_state: OptimizerState,
    _model: PhantomData<M>,
}

impl<M> Trainer<M>
where
    M: ForecastModel,
{
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            losses: LossAggregator::default(),
            optimizer_state: OptimizerState::default(),
            _model: PhantomData,
        }
    }

    pub async fn dry_run_epoch<S>(&mut self, stream: &mut S) -> AionResult<TrainingReport>
    where
        S: DataStream,
    {
        let mut report = TrainingReport::default();
        while let Some(_batch) = stream.next_batch().await? {
            let mut breakdown = LossBreakdown::default();
            breakdown.values.insert("next_patch".into(), 0.0);
            report.last_loss = self.losses.aggregate(&breakdown);
            self.optimizer_state.step += 1;
            report.steps_completed += 1;
        }
        self.optimizer_state.epoch += 1;
        report.epochs_completed = self.optimizer_state.epoch;
        Ok(report)
    }
}
