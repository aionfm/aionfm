use crate::{ForecastEvaluator, HierarchicalReconciler, InferenceMetrics, ScenarioSampler};
use aionfm_model::ForecastModel;
use aionfm_utils::{
    validation::validate_batch_request, AionResult, BatchForecastRequest, EvaluationReport,
    EvaluationRequest, ForecastRequest, ForecastResponse, InterpretationRequest, ScenarioRequest,
};
use std::sync::Arc;

/// Inference engine wrapping a concrete model implementation.
#[derive(Clone)]
pub struct InferenceEngine<M> {
    model: Arc<M>,
    metrics: InferenceMetrics,
    sampler: ScenarioSampler,
    reconciler: HierarchicalReconciler,
    evaluator: ForecastEvaluator,
}

impl<M> InferenceEngine<M>
where
    M: ForecastModel,
{
    pub fn new(model: M) -> Self {
        Self {
            model: Arc::new(model),
            metrics: InferenceMetrics::default(),
            sampler: ScenarioSampler::default(),
            reconciler: HierarchicalReconciler,
            evaluator: ForecastEvaluator::default(),
        }
    }

    pub fn forecast_one(&self, request: ForecastRequest) -> AionResult<ForecastResponse> {
        let started = std::time::Instant::now();
        let response = self.model.forecast(&request);
        self.metrics.record_latency(started.elapsed());
        response
    }

    pub fn forecast_batch(&self, request: BatchForecastRequest) -> AionResult<ForecastResponse> {
        validate_batch_request(&request)?;
        let options = request.forecast_options();
        let mut results = Vec::with_capacity(request.entities.len());
        for entity in request.entities {
            let response = self.forecast_one(ForecastRequest::new(entity, options.clone()))?;
            results.extend(response.results);
        }
        let mut response =
            ForecastResponse::new(self.model.model_name(), self.model.model_version(), results);
        if let Some(hierarchy) = &request.options.hierarchy {
            self.reconciler.reconcile(&mut response, hierarchy);
        }
        self.metrics.record_response(&response);
        Ok(response)
    }

    pub fn scenario(&self, request: ScenarioRequest) -> AionResult<ForecastResponse> {
        let mut forecast = self.forecast_batch(request.forecast)?;
        self.sampler.apply_controls(
            &mut forecast,
            request.scenario_type.as_deref(),
            &request.forced_regimes,
        );
        Ok(forecast)
    }

    pub fn interpretation(&self, request: InterpretationRequest) -> AionResult<ForecastResponse> {
        self.forecast_batch(request.forecast)
    }

    pub fn evaluate(&self, request: EvaluationRequest) -> AionResult<EvaluationReport> {
        let report = self.evaluator.evaluate(&request);
        self.metrics.record_evaluation(&report);
        Ok(report)
    }

    pub fn metrics(&self) -> InferenceMetrics {
        self.metrics.clone()
    }
}
