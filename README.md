# 🔍 Anomaly Grid Python

[![PyPI version](https://badge.fury.io/py/anomaly-grid-py.svg)](https://badge.fury.io/py/anomaly-grid-py)
[![Python versions](https://img.shields.io/pypi/pyversions/anomaly-grid-py.svg)](https://pypi.org/project/anomaly-grid-py/)
[![Downloads](https://img.shields.io/pypi/dm/anomaly-grid-py.svg)](https://pypi.org/project/anomaly-grid-py/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/abimael10/anomaly-grid-py/workflows/CI/badge.svg)](https://github.com/abimael10/anomaly-grid-py/actions)
[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://www.rust-lang.org/)

High-performance sequence anomaly detection using Markov models. Minimal dependencies, scikit-learn style API.

## Installation

```bash
pip install anomaly-grid-py
```

## Quick Start

```python
from anomaly_grid_py import AnomalyDetector

# Create detector
detector = AnomalyDetector(max_order=3)

# Train on normal sequences
normal_patterns = [
    ['LOGIN', 'BALANCE', 'LOGOUT'],
    ['LOGIN', 'WITHDRAW', 'LOGOUT'],
    ['LOGIN', 'TRANSFER', 'LOGOUT']
] * 100

detector.fit(normal_patterns)

# Detect anomalies
test_sequences = [
    ['LOGIN', 'BALANCE', 'LOGOUT'],      # Normal
    ['HACK', 'EXPLOIT', 'STEAL'],        # Anomalous
]

# Get anomaly scores (0-1, higher = more anomalous)
scores = detector.predict_proba(test_sequences)
print(f"Scores: {scores}")

# Get binary predictions
anomalies = detector.predict(test_sequences, threshold=0.1)
print(f"Anomalies: {anomalies}")
```

## API Reference

### AnomalyDetector(max_order=3)

**Methods:**
- `fit(X)` - Train on list of sequences
- `predict_proba(X)` - Get anomaly scores (0-1)
- `predict(X, threshold=0.1)` - Get binary predictions
- `get_performance_metrics()` - Get training metrics

**Parameters:**
- `X`: List of sequences, each sequence is a list of strings
- `threshold`: Detection threshold (0-1)

## Examples

See [`example.py`](example.py) for complete examples:

```bash
python example.py
```

## License

MIT License - see [LICENSE](LICENSE) file for details.