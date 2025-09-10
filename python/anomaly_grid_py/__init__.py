"""
Anomaly Grid Python - Python bindings for the anomaly-grid Rust library

A simple and efficient anomaly detection library for sequential data.
"""

from ._core import AnomalyDetector, AnomalyInfo

__version__ = "0.1.0"
__all__ = ["AnomalyDetector", "AnomalyInfo"]
