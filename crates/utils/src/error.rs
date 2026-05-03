use thiserror::Error;

/// Result type used by AionFM crates.
pub type AionResult<T> = Result<T, AionError>;

/// Error categories shared across data, model, serving, API, and SDK layers.
#[derive(Debug, Error)]
pub enum AionError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("validation failed: {0}")]
    Validation(String),
    #[error("model unavailable: {0}")]
    ModelUnavailable(String),
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    #[error("backend error: {0}")]
    Backend(String),
    #[error("timeout: {0}")]
    Timeout(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}
