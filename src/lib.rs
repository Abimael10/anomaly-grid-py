use pyo3::prelude::*;
use std::collections::HashMap;

// Import the Rust anomaly-grid library
use anomaly_grid::{AnomalyDetector as RustAnomalyDetector};

/// Python wrapper for the Rust AnomalyDetector
#[pyclass]
struct AnomalyDetector {
    detector: RustAnomalyDetector,
}

#[pymethods]
impl AnomalyDetector {
    /// Create a new AnomalyDetector with specified maximum order
    #[new]
    fn new(max_order: Option<usize>) -> PyResult<Self> {
        let detector = RustAnomalyDetector::new(max_order.unwrap_or(3))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to create detector: {}", e)))?;
        Ok(AnomalyDetector { detector })
    }

    /// Train the detector with a sequence of events
    fn train(&mut self, events: Vec<String>) -> PyResult<()> {
        self.detector.train(&events)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Training failed: {}", e)))?;
        Ok(())
    }

    /// Detect anomalies in a sequence with given threshold
    fn detect(&self, events: Vec<String>, threshold: Option<f64>) -> PyResult<Vec<AnomalyInfo>> {
        let threshold = threshold.unwrap_or(0.1);
        let results = self.detector.detect_anomalies(&events, threshold)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Detection failed: {}", e)))?;
        
        let anomalies: Vec<AnomalyInfo> = results
            .into_iter()
            .enumerate()
            .map(|(position, result)| AnomalyInfo {
                position,
                sequence: result.sequence.join(","),
                likelihood: result.likelihood,
                anomaly_strength: result.anomaly_strength,
                is_anomaly: true, // All results from detect_anomalies are anomalies
            })
            .collect();
        
        Ok(anomalies)
    }

    /// Get the maximum order of the detector
    fn max_order(&self) -> usize {
        self.detector.max_order()
    }

    /// Get performance metrics
    fn get_performance_metrics(&self) -> PyResult<HashMap<String, u64>> {
        let metrics = self.detector.performance_metrics();
        let mut result = HashMap::new();
        result.insert("training_time_ms".to_string(), metrics.training_time_ms);
        result.insert("detection_time_ms".to_string(), metrics.detection_time_ms);
        result.insert("context_count".to_string(), metrics.context_count as u64);
        result.insert("estimated_memory_bytes".to_string(), metrics.estimated_memory_bytes as u64);
        Ok(result)
    }
}

/// Information about an anomaly detection result
#[pyclass]
#[derive(Clone)]
struct AnomalyInfo {
    #[pyo3(get)]
    position: usize,
    #[pyo3(get)]
    sequence: String,
    #[pyo3(get)]
    likelihood: f64,
    #[pyo3(get)]
    anomaly_strength: f64,
    #[pyo3(get)]
    is_anomaly: bool,
}

#[pymethods]
impl AnomalyInfo {
    fn __repr__(&self) -> String {
        format!(
            "AnomalyInfo(position={}, sequence='{}', likelihood={:.4}, anomaly_strength={:.4}, is_anomaly={})",
            self.position, self.sequence, self.likelihood, self.anomaly_strength, self.is_anomaly
        )
    }
}

/// The main Python module
#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AnomalyDetector>()?;
    m.add_class::<AnomalyInfo>()?;
    Ok(())
}