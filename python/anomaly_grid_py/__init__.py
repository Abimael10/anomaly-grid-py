"""
Anomaly Grid: High-performance sequence anomaly detection.

Minimal dependency implementation focusing on speed and efficiency.
Only requires numpy for array operations.
"""

from .core import AnomalyDetector
from .utils import (
    train_test_split,
    cross_val_score, 
    roc_auc_score,
    precision_recall_curve,
    generate_sequences,
    PerformanceTimer,
    memory_usage,
    validate_sequences,
    calculate_sequence_stats
)

__version__ = "0.3.0"
__all__ = [
    "AnomalyDetector",
    "train_test_split",
    "cross_val_score",
    "roc_auc_score", 
    "precision_recall_curve",
    "generate_sequences",
    "PerformanceTimer",
    "memory_usage",
    "validate_sequences",
    "calculate_sequence_stats"
]

# Performance-focused configuration
import numpy as np
np.seterr(all='raise')  # Catch numerical errors early