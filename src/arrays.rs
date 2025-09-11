use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};

/// High-performance sequence container with zero-copy operations
pub struct SequenceArray {
    sequences: Vec<Vec<String>>,
}

impl SequenceArray {
    /// Create from Python object with minimal copying
    pub fn from_python(obj: &PyAny) -> PyResult<Self> {
        // Handle different input types efficiently
        if let Ok(list) = obj.downcast::<PyList>() {
            Self::from_list(list)
        } else if let Ok(array) = obj.extract::<PyReadonlyArray1<PyObject>>() {
            Self::from_numpy_array(array)
        } else {
            // Try to handle as a general sequence
            Self::from_sequence(obj)
        }
    }
    
    fn from_list(list: &PyList) -> PyResult<Self> {
        let mut sequences = Vec::with_capacity(list.len());
        
        if list.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Empty sequence list"
            ));
        }
        
        for (i, item) in list.iter().enumerate() {
            if let Ok(seq_list) = item.downcast::<PyList>() {
                if seq_list.is_empty() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Empty sequence at index {}", i)
                    ));
                }
                
                let mut sequence = Vec::with_capacity(seq_list.len());
                for (j, elem) in seq_list.iter().enumerate() {
                    if let Ok(string_elem) = elem.downcast::<PyString>() {
                        sequence.push(string_elem.to_str()?.to_string());
                    } else if let Ok(string_val) = elem.extract::<String>() {
                        sequence.push(string_val);
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            format!("Non-string element at sequence {}, position {}", i, j)
                        ));
                    }
                }
                sequences.push(sequence);
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    format!("Expected list at index {}, got {}", i, item.get_type().name()?)
                ));
            }
        }
        
        Ok(Self { sequences })
    }
    
    fn from_numpy_array(array: PyReadonlyArray1<PyObject>) -> PyResult<Self> {
        let array_slice = array.as_slice()?;
        let mut sequences = Vec::with_capacity(array_slice.len());
        
        if array_slice.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Empty numpy array"
            ));
        }
        
        Python::with_gil(|py| {
            for (i, obj) in array_slice.iter().enumerate() {
                if let Ok(seq_list) = obj.downcast::<PyList>(py) {
                    if seq_list.is_empty() {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("Empty sequence at index {}", i)
                        ));
                    }
                    
                    let mut sequence = Vec::with_capacity(seq_list.len());
                    for (j, elem) in seq_list.iter().enumerate() {
                        if let Ok(string_elem) = elem.downcast::<PyString>() {
                            sequence.push(string_elem.to_str()?.to_string());
                        } else if let Ok(string_val) = elem.extract::<String>() {
                            sequence.push(string_val);
                        } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                format!("Non-string element at sequence {}, position {}", i, j)
                            ));
                        }
                    }
                    sequences.push(sequence);
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        format!("Expected list at array index {}", i)
                    ));
                }
            }
            Ok(Self { sequences })
        })
    }
    
    fn from_sequence(obj: &PyAny) -> PyResult<Self> {
        // Try to handle as a general sequence (including NumPy arrays)
        if let Ok(iter) = obj.iter() {
            let mut sequences = Vec::new();
            
            for (i, item_result) in iter.enumerate() {
                let item = item_result?;
                
                if let Ok(seq_list) = item.downcast::<PyList>() {
                    if seq_list.is_empty() {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("Empty sequence at index {}", i)
                        ));
                    }
                    
                    let mut sequence = Vec::with_capacity(seq_list.len());
                    for (j, elem) in seq_list.iter().enumerate() {
                        if let Ok(string_elem) = elem.downcast::<PyString>() {
                            sequence.push(string_elem.to_str()?.to_string());
                        } else if let Ok(string_val) = elem.extract::<String>() {
                            sequence.push(string_val);
                        } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                format!("Non-string element at sequence {}, position {}", i, j)
                            ));
                        }
                    }
                    sequences.push(sequence);
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        format!("Expected sequence of lists at index {}", i)
                    ));
                }
            }
            
            if sequences.is_empty() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Empty sequence list"
                ));
            }
            
            Ok(Self { sequences })
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Expected list of sequences or numpy array"
            ))
        }
    }
    
    pub fn as_slice(&self) -> &[Vec<String>] {
        &self.sequences
    }
    
    pub fn len(&self) -> usize {
        self.sequences.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }
    
    /// Validate sequences for training/prediction
    pub fn validate(&self) -> PyResult<()> {
        if self.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No sequences provided"
            ));
        }
        
        for (i, sequence) in self.sequences.iter().enumerate() {
            if sequence.len() < 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Sequence {} has length {}, minimum is 2", i, sequence.len())
                ));
            }
            
            // Check for empty strings
            for (j, element) in sequence.iter().enumerate() {
                if element.is_empty() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Empty string at sequence {}, position {}", i, j)
                    ));
                }
            }
        }
        
        Ok(())
    }
}

/// Convert scores to NumPy array with zero-copy when possible
pub fn scores_to_numpy(py: Python, scores: Vec<f64>) -> Py<PyArray1<f64>> {
    scores.to_pyarray(py).to_owned()
}

/// Convert boolean predictions to NumPy array
pub fn predictions_to_numpy(py: Python, predictions: Vec<bool>) -> Py<PyArray1<bool>> {
    predictions.to_pyarray(py).to_owned()
}