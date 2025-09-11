# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-09-11

- Adapting the bindings to the new version of the Rust library

## [0.1.0] - 2025-09-10

### Added
- Initial release of anomaly-grid Python bindings
- `AnomalyDetector` class for training and detection
- `AnomalyInfo` class for anomaly results
- Support for Python 3.8+ on Linux, Windows, macOS
- Comprehensive test suite (22 tests)
- Complete documentation and realistic examples
- Type hints and stub files for IDE support

### Features
- Train models on sequential string data
- Detect anomalies with configurable thresholds
- Performance metrics tracking
- Clean Python API over high-performance Rust implementation
- Real-world examples: web logs, user behavior, IoT sensors, network traffic
- Cross-platform wheel distribution
- Uses published anomaly-grid crate v0.2.2 from crates.io
