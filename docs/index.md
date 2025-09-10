# Anomaly Grid Python Documentation

Welcome to the documentation for anomaly-grid-py, Python bindings for anomaly detection using Markov models.

## Quick Start

```python
import anomaly_grid_py

# Create detector
detector = anomaly_grid.AnomalyDetector(max_order=3)

# Train on normal patterns
detector.train(["login", "access", "logout"] * 10)

# Detect anomalies
anomalies = detector.detect(["login", "admin_escalation", "logout"], threshold=0.1)

for anomaly in anomalies:
    print(f"Anomaly at position {anomaly.position}: {anomaly.sequence}")
```

## Contents

```{toctree}
:maxdepth: 2

installation
api
examples
contributing
changelog
```

## Features

- **Fast**: Rust-powered implementation for high performance
- **Simple**: Clean Python API with minimal dependencies
- **Flexible**: Configurable Markov model parameters
- **Typed**: Full type hint support for better development experience

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
