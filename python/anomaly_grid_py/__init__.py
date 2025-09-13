"""
Anomaly Grid: High-performance sequence anomaly detection.

Simple, clean API built on solid Rust core implementation.
"""

# Import ONLY the core Rust module and essential utilities
from . import _core
from .utils import (
    train_test_split,
    precision_recall_curve,
    generate_sequences,
    validate_sequences,
    calculate_sequence_stats,
    PerformanceTimer,
    memory_usage,
)

# Simple wrapper around the core for convenience
class AnomalyDetector:
    """
    Simple anomaly detector for sequential data.
    
    Built on high-performance Rust core with minimal Python overhead.
    
    Parameters
    ----------
    max_order : int, default=3
        Maximum order of the Markov model. Higher values capture 
        longer-range dependencies but require more training data.
    
    Examples
    --------
    >>> detector = AnomalyDetector(max_order=3)
    >>> training_data = [['A', 'B', 'C'], ['A', 'B', 'D']] * 100
    >>> detector.fit(training_data)
    >>> scores = detector.predict_proba([['A', 'B', 'X'], ['X', 'Y', 'Z']])
    >>> print(scores)  # [0.25, 0.67] - higher scores = more anomalous
    """
    
    def __init__(self, max_order=3):
        # Validate parameters early
        if not isinstance(max_order, int):
            raise TypeError(f"max_order must be an integer, got {type(max_order).__name__}")
        if max_order <= 0:
            raise ValueError(f"max_order must be greater than 0, got {max_order}")
        
        self._core_detector = _core.AnomalyDetector(max_order=max_order)
        self._fitted = False
        self.max_order = max_order
    
    def fit(self, X):
        """
        Train the detector on normal sequences.
        
        Parameters
        ----------
        X : list of list of str
            Training sequences. Each sequence should be a list of strings.
            Only normal (non-anomalous) sequences should be provided.
        
        Returns
        -------
        self : AnomalyDetector
            Returns self for method chaining.
        
        Examples
        --------
        >>> detector = AnomalyDetector()
        >>> normal_sequences = [
        ...     ['LOGIN', 'BALANCE', 'LOGOUT'],
        ...     ['LOGIN', 'TRANSFER', 'LOGOUT'],
        ... ] * 100  # Repeat for sufficient training data
        >>> detector.fit(normal_sequences)
        """
        validate_sequences(X, min_length=2)
        self._core_detector.fit(X)
        self._fitted = True
        return self
    
    def predict_proba(self, X):
        """
        Predict anomaly probabilities for sequences.
        
        Parameters
        ----------
        X : list of list of str
            Sequences to score for anomalies.
        
        Returns
        -------
        scores : numpy.ndarray
            Anomaly probability scores between 0 and 1.
            Higher scores indicate more anomalous sequences.
        
        Examples
        --------
        >>> scores = detector.predict_proba([
        ...     ['LOGIN', 'BALANCE', 'LOGOUT'],  # Normal
        ...     ['HACK', 'EXPLOIT', 'STEAL'],    # Anomalous
        ... ])
        >>> print(scores)  # [0.24, 0.67] - second sequence is more anomalous
        """
        if not self._fitted:
            raise ValueError("Detector is not fitted. Call fit() first.")
        
        validate_sequences(X, min_length=1)
        return self._core_detector.predict_proba(X)
    
    def predict(self, X, threshold=0.5):
        """
        Predict binary anomaly labels for sequences.
        
        Parameters
        ----------
        X : list of list of str
            Sequences to classify.
        threshold : float, default=0.5
            Anomaly threshold. Sequences with scores >= threshold 
            are classified as anomalous.
        
        Returns
        -------
        predictions : numpy.ndarray of bool
            Boolean array where True indicates anomalous sequences.
        
        Examples
        --------
        >>> # Set threshold based on training data
        >>> training_scores = detector.predict_proba(validation_normal)
        >>> threshold = np.max(training_scores) + 0.1
        >>> predictions = detector.predict(test_sequences, threshold)
        """
        if not self._fitted:
            raise ValueError("Detector is not fitted. Call fit() first.")
        
        scores = self.predict_proba(X)
        import numpy as np
        return scores >= threshold
    
    def get_performance_metrics(self):
        """
        Get performance metrics from the core detector.
        
        Returns
        -------
        metrics : dict
            Dictionary containing performance metrics like training time,
            memory usage, vocabulary size, etc.
        """
        if not self._fitted:
            raise ValueError("Detector is not fitted. Call fit() first.")
        
        return self._core_detector.get_metrics()
    
    def get_params(self, deep=True):
        """Get parameters for this estimator (scikit-learn compatibility)."""
        return {'max_order': self.max_order}
    
    def set_params(self, **params):
        """Set parameters for this estimator (scikit-learn compatibility)."""
        for key, value in params.items():
            if key == 'max_order':
                self.max_order = value
                self._core_detector = _core.AnomalyDetector(max_order=value)
                self._fitted = False
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self

__version__ = "0.4.0"

__all__ = [
    # Core detector
    "AnomalyDetector",
    
    # Essential utilities
    "train_test_split",
    "precision_recall_curve", 
    "generate_sequences",
    "validate_sequences",
    "calculate_sequence_stats",
    "PerformanceTimer",
    "memory_usage",
]