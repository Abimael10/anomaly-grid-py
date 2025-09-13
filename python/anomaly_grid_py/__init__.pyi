"""Type stubs for anomaly_grid_py."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

class AnomalyDetector:
    """Simple anomaly detector for sequential data."""
    
    def __init__(self, max_order: int = 3) -> None:
        """
        Initialize anomaly detector.
        
        Parameters
        ----------
        max_order : int, default=3
            Maximum order of the Markov model.
        """
        ...

    def fit(self, X: List[List[str]]) -> "AnomalyDetector":
        """
        Train detector on normal sequences.
        
        Parameters
        ----------
        X : List[List[str]]
            Training sequences (normal data only).
            
        Returns
        -------
        self : AnomalyDetector
            Returns self for method chaining.
        """
        ...

    def predict_proba(self, X: List[List[str]]) -> np.ndarray:
        """
        Predict anomaly probabilities.
        
        Parameters
        ----------
        X : List[List[str]]
            Sequences to score.
            
        Returns
        -------
        scores : np.ndarray
            Anomaly probability scores [0, 1].
            Higher scores indicate more anomalous sequences.
        """
        ...

    def predict(self, X: List[List[str]], threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary anomaly labels.
        
        Parameters
        ----------
        X : List[List[str]]
            Sequences to classify.
        threshold : float, default=0.5
            Anomaly threshold.
            
        Returns
        -------
        predictions : np.ndarray
            Boolean array where True indicates anomalous sequences.
        """
        ...

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns
        -------
        metrics : Dict[str, Any]
            Performance metrics from the core detector.
        """
        ...

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get estimator parameters (scikit-learn compatibility)."""
        ...

    def set_params(self, **params) -> "AnomalyDetector":
        """Set estimator parameters (scikit-learn compatibility)."""
        ...

# Utility functions
def train_test_split(
    sequences: List[List[str]],
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Split sequences into training and test sets."""
    ...

def precision_recall_curve(
    y_true: Union[List[int], np.ndarray], 
    y_scores: Union[List[float], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate precision-recall curve for anomaly detection evaluation."""
    ...

def generate_sequences(
    n_sequences: int,
    seq_length: int,
    alphabet: List[str],
    anomaly_rate: float = 0.1,
    pattern_type: str = "random",
    random_state: Optional[int] = None,
) -> Tuple[List[List[str]], List[int]]:
    """Generate synthetic sequences for testing."""
    ...

def validate_sequences(
    sequences: List[List[str]], 
    min_length: int = 1
) -> None:
    """Validate sequence format and content."""
    ...

def calculate_sequence_stats(
    sequences: List[List[str]]
) -> Dict[str, Any]:
    """Calculate statistics for sequences."""
    ...

def memory_usage() -> float:
    """Get current memory usage in MB."""
    ...

class PerformanceTimer:
    """Lightweight performance timing utility."""

    def __init__(self) -> None: ...
    def __enter__(self) -> "PerformanceTimer": ...
    def __exit__(self, *args) -> None: ...
    def time_operation(self, name: str, func, *args, **kwargs) -> Any: ...
    def get_times(self) -> Dict[str, float]: ...
    def reset(self) -> None: ...

__version__: str
__all__: List[str]