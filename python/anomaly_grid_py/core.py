"""
High-performance anomaly detection with minimal dependencies.
Only requires numpy for array operations.
"""

import numpy as np
from typing import Union, List, Tuple, Dict, Any, Optional
from . import _core


class AnomalyDetector:
    """
    Minimal, high-performance anomaly detector.

    Designed for maximum speed with minimal external dependencies.
    Only requires numpy for efficient array operations.
    """

    def __init__(self, max_order: int = 3):
        """
        Initialize detector.

        Parameters
        ----------
        max_order : int, default=3
            Maximum context order. Higher = more memory, better accuracy.
        """
        self._detector = _core.AnomalyDetector(max_order)
        self._fitted = False

    def fit(self, X: Union[List[List[str]], np.ndarray]) -> "AnomalyDetector":
        """
        Fit detector on training sequences.

        Parameters
        ----------
        X : list of lists or numpy array
            Training sequences. Each sequence must have at least 2 elements.

        Returns
        -------
        self : AnomalyDetector
        """
        # Validate input efficiently
        sequences = self._validate_sequences(X)

        # Direct call to Rust - no intermediate processing
        self._detector.fit(sequences)
        self._fitted = True

        return self

    def predict_proba(self, X: Union[List[List[str]], np.ndarray]) -> np.ndarray:
        """
        Predict anomaly scores.

        Parameters
        ----------
        X : list of lists or numpy array
            Test sequences. Each sequence must have at least 2 elements.

        Returns
        -------
        scores : numpy.ndarray
            Anomaly scores [0, 1].
        """
        self._check_fitted()
        sequences = self._validate_sequences(X)

        # Direct Rust call returns NumPy array - zero copy
        return self._detector.predict_proba(sequences)

    def predict(
        self, X: Union[List[List[str]], np.ndarray], threshold: float = 0.1
    ) -> np.ndarray:
        """
        Predict anomalies.

        Parameters
        ----------
        X : list of lists or numpy array
            Test sequences. Each sequence must have at least 2 elements.
        threshold : float, default=0.1
            Detection threshold.

        Returns
        -------
        predictions : numpy.ndarray
            Boolean anomaly predictions.
        """
        self._check_fitted()
        sequences = self._validate_sequences(X)

        # Direct Rust call returns NumPy array
        return self._detector.predict(sequences, threshold)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from last training."""
        self._check_fitted()
        return self._detector.get_metrics()

    def predict_proba_with_padding(
        self, X: Union[List[List[str]], np.ndarray], padding_token: str = "<PAD>"
    ) -> np.ndarray:
        """
        Predict anomaly scores with automatic padding for short sequences.

        This method automatically pads single-element sequences to meet
        the minimum length requirement.

        Parameters
        ----------
        X : list of lists or numpy array
            Test sequences. Single-element sequences will be padded.
        padding_token : str, default="<PAD>"
            Token to use for padding short sequences.

        Returns
        -------
        scores : numpy.ndarray
            Anomaly scores [0, 1].
        """
        self._check_fitted()

        # Convert to list format for processing
        if isinstance(X, np.ndarray):
            sequences = X.tolist()
        else:
            sequences = X.copy()

        # Pad sequences that are too short
        padded_sequences = []
        for seq in sequences:
            if len(seq) < 2:
                # Pad with padding token
                padded_seq = seq + [padding_token] * (2 - len(seq))
                padded_sequences.append(padded_seq)
            else:
                padded_sequences.append(seq)

        # Validate and predict
        validated_sequences = self._validate_sequences(padded_sequences)
        return self._detector.predict_proba(validated_sequences)

    def _validate_sequences(
        self, X: Union[List[List[str]], np.ndarray], allow_single_element: bool = False
    ) -> Union[List[List[str]], np.ndarray]:
        """Validate input sequences with flexible length requirements."""
        if isinstance(X, np.ndarray):
            if X.ndim != 1:
                raise ValueError("NumPy array must be 1-dimensional array of sequences")
            return X

        if not isinstance(X, list):
            raise TypeError("Input must be list or numpy array")

        if not X:
            raise ValueError("Empty sequence list")

        # Quick validation of first sequence only (performance optimization)
        if not isinstance(X[0], (list, tuple)):
            raise TypeError("Sequences must be lists or tuples")

        # Check minimum length requirements
        min_length = 1 if allow_single_element else 2
        if len(X[0]) < min_length:
            if allow_single_element:
                raise ValueError(f"Sequences must have at least {min_length} element")
            else:
                raise ValueError(
                    f"Sequences must have at least {min_length} elements. "
                    f"Single-element sequences are not supported for anomaly detection "
                    f"as they provide insufficient context for pattern analysis."
                )

        return X

    def _check_fitted(self):
        """Check if detector is fitted."""
        if not self._fitted:
            raise ValueError("Detector not fitted. Call fit() first.")


# Custom lightweight estimator interface (replaces scikit-learn dependency)
class BaseEstimator:
    """Minimal estimator interface for compatibility."""

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get estimator parameters."""
        return {key: getattr(self, key) for key in self._get_param_names()}

    def set_params(self, **params) -> "BaseEstimator":
        """Set estimator parameters."""
        valid_params = self.get_params()

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter: {key}")
            setattr(self, key, value)

        return self

    def _get_param_names(self) -> List[str]:
        """Get parameter names from __init__ signature."""
        import inspect

        signature = inspect.signature(self.__init__)
        return [
            p.name
            for p in signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]


# Extend AnomalyDetector with minimal estimator interface
class AnomalyDetector(AnomalyDetector, BaseEstimator):
    """High-performance detector with minimal estimator interface."""

    pass
