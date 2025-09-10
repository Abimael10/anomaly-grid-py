# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Uses published anomaly-grid crate v0.2.2 from crates.io
- Improved PyPI readiness with proper licensing and metadata

## [0.1.0] - 2024-09-10

### Added
- Initial release of anomaly-grid Python bindings
- `AnomalyDetector` class for training and detection
- `AnomalyInfo` class for anomaly results
- Support for Python 3.8+
- Basic test suite
- Documentation and examples

### Features
- Train models on sequential string data
- Detect anomalies with configurable thresholds
- Performance metrics tracking
- Clean Python API over Rust implementation
