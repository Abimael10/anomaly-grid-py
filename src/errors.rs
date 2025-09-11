use pyo3::prelude::*;

/// Custom error type for anomaly grid operations
#[derive(Debug)]
pub enum PyAnomalyGridError {
    NotFitted,
    InvalidInput(String),
    TrainingFailed(String),
    DetectionFailed(String),
    ConfigurationError(String),
}

impl std::fmt::Display for PyAnomalyGridError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PyAnomalyGridError::NotFitted => write!(f, "Detector not fitted. Call fit() first."),
            PyAnomalyGridError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            PyAnomalyGridError::TrainingFailed(msg) => write!(f, "Training failed: {}", msg),
            PyAnomalyGridError::DetectionFailed(msg) => write!(f, "Detection failed: {}", msg),
            PyAnomalyGridError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for PyAnomalyGridError {}

impl From<PyAnomalyGridError> for PyErr {
    fn from(err: PyAnomalyGridError) -> PyErr {
        match err {
            PyAnomalyGridError::NotFitted => PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()),
            PyAnomalyGridError::InvalidInput(_) => PyErr::new::<pyo3::exceptions::PyTypeError, _>(err.to_string()),
            PyAnomalyGridError::TrainingFailed(_) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()),
            PyAnomalyGridError::DetectionFailed(_) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()),
            PyAnomalyGridError::ConfigurationError(_) => PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()),
        }
    }
}

impl PyAnomalyGridError {
    pub fn not_fitted() -> PyErr {
        PyAnomalyGridError::NotFitted.into()
    }
    
    pub fn invalid_input(msg: &str) -> PyErr {
        PyAnomalyGridError::InvalidInput(msg.to_string()).into()
    }
    
    pub fn training_failed(msg: &str) -> PyErr {
        PyAnomalyGridError::TrainingFailed(msg.to_string()).into()
    }
    
    pub fn detection_failed(msg: &str) -> PyErr {
        PyAnomalyGridError::DetectionFailed(msg.to_string()).into()
    }
    
    pub fn configuration_error(msg: &str) -> PyErr {
        PyAnomalyGridError::ConfigurationError(msg.to_string()).into()
    }
}

/// Convert anomaly-grid library errors to our custom error type
impl From<Box<dyn std::error::Error + Send + Sync>> for PyAnomalyGridError {
    fn from(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
        PyAnomalyGridError::DetectionFailed(err.to_string())
    }
}

/// Convert anomaly-grid AnomalyGridError to our custom error type
impl From<anomaly_grid::AnomalyGridError> for PyAnomalyGridError {
    fn from(err: anomaly_grid::AnomalyGridError) -> Self {
        PyAnomalyGridError::DetectionFailed(err.to_string())
    }
}