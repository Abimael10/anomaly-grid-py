# Anomaly Grid Python

[![PyPI version](https://badge.fury.io/py/anomaly-grid-py.svg)](https://badge.fury.io/py/anomaly-grid-py)
[![Python versions](https://img.shields.io/pypi/pyversions/anomaly-grid-py.svg)](https://pypi.org/project/anomaly-grid-py/)
[![Downloads](https://pepy.tech/badge/anomaly-grid-py)](https://pepy.tech/project/anomaly-grid-py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/abimael10/anomaly-grid-py/workflows/CI/badge.svg)](https://github.com/abimael10/anomaly-grid-py/actions)
[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://www.rust-lang.org/)

Python bindings for anomaly detection using Markov models. Train on sequential data to detect unusual patterns.

## Installation

### From PyPI

```bash
pip install anomaly-grid-py
```

### From Source

For development or latest features:

```bash
# Clone the repository
git clone https://github.com/abimael10/anomaly-grid-py
cd anomaly-grid-py

# Set up development environment
./setup.sh
source venv/bin/activate

# Build the package
maturin develop
```

**Note**: Requires Rust toolchain for building. Dependencies are downloaded automatically.

## Quick Start

```python
import anomaly_grid_py

# Create detector for web server log analysis
detector = anomaly_grid_py.AnomalyDetector(max_order=3)

# Train with normal web server patterns
normal_logs = [
    "GET", "/", "200", "GET", "/login", "200", "POST", "/login", "302",
    "GET", "/dashboard", "200", "GET", "/profile", "200", "POST", "/logout", "302"
] * 50  # 50 user sessions

detector.train(normal_logs)

# Detect suspicious activity
suspicious_activity = [
    "GET", "/", "200", "GET", "/admin", "403", "GET", "/admin/users", "403",
    "POST", "/admin/delete", "403", "GET", "/../etc/passwd", "404"
]

anomalies = detector.detect(suspicious_activity, threshold=0.1)
print(f"🚨 Detected {len(anomalies)} suspicious patterns")

for anomaly in anomalies[:3]:  # Show first 3
    print(f"Alert: '{anomaly.sequence}' (confidence: {anomaly.anomaly_strength:.1%})")
```

## Detailed Example

See [`example.py`](example.py) for a complete working example:

```bash
python example.py
```

## API Reference

### AnomalyDetector

The main class for anomaly detection.

#### Constructor
- `AnomalyDetector(max_order=3)`: Create a new detector with specified maximum order

#### Methods
- `train(events)`: Train the detector with a list of events
- `detect(events, threshold=0.1)`: Detect anomalies in a sequence
- `get_performance_metrics()`: Get performance metrics as a dictionary
- `max_order()`: Get the maximum order of the detector

### AnomalyInfo

Information about an anomaly detection result.

#### Properties
- `position`: Position in the sequence (int)
- `sequence`: The sequence window that was analyzed (string)
- `likelihood`: Likelihood of the sequence under the model (float)
- `anomaly_strength`: Anomaly strength score [0,1] (float)
- `is_anomaly`: Whether this sequence is considered an anomaly (bool)

## Development

### Building from Source

```bash
# Install development dependencies
pip install maturin pytest

# On Linux, also install patchelf
pip install patchelf  # Linux only

# Build in development mode
maturin develop

# Run tests
pytest tests/
```

### Development Dependencies

For a complete development environment:

```bash
# Install all development dependencies
pip install -e .[dev]

# Or install specific dependency groups
pip install -e .[test]  # Testing dependencies
pip install -e .[docs]  # Documentation dependencies
```

### Project Structure

```
anomaly-grid-py/
├── .github/workflows/          # CI/CD configuration
├── docs/                       # Documentation
├── python/anomaly_grid_py/     # Python module
├── src/lib.rs                  # PyO3 bindings
├── tests/                      # Test suite
├── build.sh                    # Build script
├── setup.sh                    # Environment setup
├── example.py                  # Usage example
├── pyproject.toml              # Python package config
├── Cargo.toml                  # Rust extension config
├── CHANGELOG.md                # Version history
└── LICENSE                     # MIT license
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_anomaly_detector.py
```

### Code Quality

This project includes configuration for several code quality tools:

- **Black**: Code formatting
- **Ruff**: Linting and code analysis
- **MyPy**: Type checking
- **Pre-commit**: Git hooks for quality checks

```bash
# Install pre-commit hooks (if pre-commit is installed)
pre-commit install

# Run all quality checks (if tools are installed)
pre-commit run --all-files
```

## Use Cases

Suitable for sequential data analysis:

- **Log Analysis**: HTTP requests, application events, system logs
- **User Behavior**: Login patterns, navigation sequences, action flows
- **Network Traffic**: Connection patterns, protocol sequences
- **Sensor Data**: IoT readings, equipment status changes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.
