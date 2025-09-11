"""
Anomaly Grid: High-performance sequence anomaly detection.

Minimal dependency implementation focusing on speed and efficiency.
Only requires numpy for array operations.
"""

from .core import AnomalyDetector
from .utils import (
    PerformanceTimer,
    calculate_sequence_stats,
    cross_val_score,
    generate_sequences,
    memory_usage,
    precision_recall_curve,
    roc_auc_score,
    train_test_split,
    validate_sequences,
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
    "calculate_sequence_stats",
]

# Performance-focused configuration
import numpy as np

np.seterr(all="raise")  # Catch numerical errors early
