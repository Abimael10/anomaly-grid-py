use pyo3::prelude::*;
use pyo3::types::PyModule;

mod arrays;
mod detector;
mod errors;

use detector::PyAnomalyDetector;

/// High-performance anomaly detection module
#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyAnomalyDetector>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}