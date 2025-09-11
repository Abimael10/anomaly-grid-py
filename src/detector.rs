use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::PyArray1;
use anomaly_grid::{AnomalyDetector, AnomalyGridConfig};
use crate::arrays::{SequenceArray, scores_to_numpy, predictions_to_numpy};
use crate::errors::PyAnomalyGridError;

#[pyclass(name = "AnomalyDetector")]
pub struct PyAnomalyDetector {
    detector: Option<AnomalyDetector>,
    config: AnomalyGridConfig,
}

#[pymethods]
impl PyAnomalyDetector {
    #[new]
    fn new(max_order: Option<usize>) -> PyResult<Self> {
        let max_order = max_order.unwrap_or(3);
        let config = AnomalyGridConfig::default()
            .with_max_order(max_order)
            .map_err(PyAnomalyGridError::from)?;
        
        Ok(Self {
            detector: None,
            config,
        })
    }
    
    /// High-performance training with minimal overhead
    fn fit(&mut self, sequences: &PyAny) -> PyResult<()> {
        let seq_array = SequenceArray::from_python(sequences)?;
        seq_array.validate()?;
        
        let mut detector = AnomalyDetector::with_config(self.config.clone())
            .map_err(PyAnomalyGridError::from)?;
        
        detector.train_sequences(seq_array.as_slice())
            .map_err(PyAnomalyGridError::from)?;
        
        self.detector = Some(detector);
        Ok(())
    }
    
    /// Zero-copy prediction with NumPy output
    fn predict_proba(&self, sequences: &PyAny) -> PyResult<Py<PyArray1<f64>>> {
        let detector = self.detector.as_ref()
            .ok_or_else(|| PyAnomalyGridError::not_fitted())?;
        
        let seq_array = SequenceArray::from_python(sequences)?;
        seq_array.validate()?;
        
        let mut scores = Vec::with_capacity(seq_array.len());
        
        for sequence in seq_array.as_slice() {
            let anomalies = detector.detect_anomalies(sequence, 0.0)
                .map_err(PyAnomalyGridError::from)?;
            
            let score = if anomalies.is_empty() {
                0.0
            } else {
                anomalies.iter()
                    .map(|a| a.anomaly_strength)
                    .fold(0.0, f64::max)
            };
            
            scores.push(score);
        }
        
        Python::with_gil(|py| Ok(scores_to_numpy(py, scores)))
    }
    
    fn predict(&self, sequences: &PyAny, threshold: Option<f64>) -> PyResult<Py<PyArray1<bool>>> {
        let threshold = threshold.unwrap_or(0.1);
        let scores = self.predict_proba(sequences)?;
        
        Python::with_gil(|py| {
            let scores_array = scores.as_ref(py);
            let predictions: Vec<bool> = scores_array.readonly()
                .as_array()
                .iter()
                .map(|&score| score >= threshold)
                .collect();
            
            Ok(predictions_to_numpy(py, predictions))
        })
    }
    
    fn get_metrics(&self) -> PyResult<Py<PyDict>> {
        let detector = self.detector.as_ref()
            .ok_or_else(|| PyAnomalyGridError::not_fitted())?;
        
        let metrics = detector.performance_metrics();
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("training_time_ms", metrics.training_time_ms)?;
            dict.set_item("context_count", metrics.context_count)?;
            dict.set_item("memory_bytes", metrics.estimated_memory_bytes)?;
            Ok(dict.into())
        })
    }
}