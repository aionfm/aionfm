use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Lightweight inference metrics hook.
#[derive(Clone, Debug, Default)]
pub struct InferenceMetrics {
    inner: Arc<Mutex<MetricsInner>>,
}

#[derive(Clone, Debug, Default)]
struct MetricsInner {
    request_count: usize,
    last_latency_ms: Option<f32>,
}

impl InferenceMetrics {
    pub fn record_latency(&self, duration: Duration) {
        let mut inner = self.inner.lock().expect("metrics mutex poisoned");
        inner.request_count += 1;
        inner.last_latency_ms = Some(duration.as_secs_f32() * 1_000.0);
    }

    pub fn request_count(&self) -> usize {
        self.inner
            .lock()
            .expect("metrics mutex poisoned")
            .request_count
    }

    pub fn last_latency_ms(&self) -> Option<f32> {
        self.inner
            .lock()
            .expect("metrics mutex poisoned")
            .last_latency_ms
    }
}
