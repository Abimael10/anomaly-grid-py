"""Type stubs for anomaly_grid_py."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

class AnomalyDetector:
    """High-performance anomaly detector for sequential data."""

    def __init__(self, max_order: int = ...) -> None: ...
    def fit(self, X: Union[List[List[str]], np.ndarray]) -> AnomalyDetector:
        """Fit detector on training sequences."""
        ...

    def predict_proba(self, X: Union[List[List[str]], np.ndarray]) -> np.ndarray:
        """Predict anomaly scores."""
        ...

    def predict(
        self, X: Union[List[List[str]], np.ndarray], threshold: float = ...
    ) -> np.ndarray:
        """Predict anomalies."""
        ...

    def detect_anomalies(
        self, X: Union[List[List[str]], np.ndarray], threshold: float = ...
    ) -> List[Any]:
        """Detect anomalies with detailed information."""
        ...

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        ...

    def get_params(self, deep: bool = ...) -> Dict[str, Any]:
        """Get estimator parameters."""
        ...

    def set_params(self, **params) -> AnomalyDetector:
        """Set estimator parameters."""
        ...

    @property
    def is_fitted(self) -> bool:
        """Check if detector is fitted."""
        ...

    @property
    def max_order(self) -> int:
        """Get maximum order."""
        ...

# Utility functions
def train_test_split(
    sequences: List[List[str]],
    test_size: float = ...,
    random_state: Optional[int] = ...,
) -> Tuple[List[List[str]], List[List[str]]]: ...
def cross_val_score(
    estimator: AnomalyDetector,
    X: List[List[str]],
    y: List[int],
    cv: int = ...,
    scoring: str = ...,
) -> np.ndarray: ...
def roc_auc_score(
    y_true: Union[List[int], np.ndarray], y_scores: Union[List[float], np.ndarray]
) -> float: ...
def precision_recall_curve(
    y_true: Union[List[int], np.ndarray], y_scores: Union[List[float], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def generate_sequences(
    n_sequences: int,
    seq_length: int,
    alphabet: List[str],
    anomaly_rate: float = ...,
    pattern_type: str = ...,
    random_state: Optional[int] = ...,
) -> Tuple[List[List[str]], List[int]]: ...
def memory_usage() -> float: ...
def validate_sequences(sequences: List[List[str]], min_length: int = ...) -> None: ...
def calculate_sequence_stats(sequences: List[List[str]]) -> Dict[str, Any]: ...

class PerformanceTimer:
    """Lightweight performance timing utility."""

    def __init__(self) -> None: ...
    def __enter__(self) -> PerformanceTimer: ...
    def __exit__(self, *args) -> None: ...
    def time_operation(self, name: str, func, *args, **kwargs) -> Any: ...
    def get_times(self) -> Dict[str, float]: ...
    def reset(self) -> None: ...

__version__: str
__all__: List[str]
