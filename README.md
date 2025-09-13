# Anomaly Grid Python

[![PyPI version](https://badge.fury.io/py/anomaly-grid-py.svg)](https://badge.fury.io/py/anomaly-grid-py)
[![Python versions](https://img.shields.io/pypi/pyversions/anomaly-grid-py.svg)](https://pypi.org/project/anomaly-grid-py/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/anomaly-grid-py.svg)](https://pypi.org/project/anomaly-grid-py/)

Sequence anomaly detection using Markov models.

## Installation

```bash
pip install anomaly-grid-py
```

## Usage

```python
import anomaly_grid_py

# Create detector
detector = anomaly_grid_py.AnomalyDetector(max_order=3)

# Train on normal sequences
normal_sequences = [
    ['A', 'B', 'C'],
    ['A', 'B', 'D'],
    ['A', 'C', 'D']
] * 50  # Need enough data for training

detector.fit(normal_sequences)

# Get anomaly scores
test_sequences = [
    ['A', 'B', 'C'],  # Normal
    ['X', 'Y', 'Z']   # Anomalous
]

scores = detector.predict_proba(test_sequences)
print(scores)  # [0.26, 0.62] - higher = more anomalous

# Binary predictions
anomalies = detector.predict(test_sequences, threshold=0.5)
print(anomalies)  # [False, True]
```

## API

```python
# Create detector
AnomalyDetector(max_order=3)

# Train on normal sequences
detector.fit(sequences)

# Get anomaly scores [0,1]
detector.predict_proba(sequences)

# Get binary predictions
detector.predict(sequences, threshold=0.5)

# Get model info
detector.get_performance_metrics()
```

## Requirements

- Python 3.8+
- NumPy

## Development

```bash
git clone https://github.com/abimael10/anomaly-grid-py.git
cd anomaly-grid-py
./setup.sh
source venv/bin/activate
pytest tests/
```

## License

MIT