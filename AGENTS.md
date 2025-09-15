# Repository Guidelines

## Project Structure & Module Organization

Source code is organized in a hybrid Rust-Python architecture:
- `src/` - Core Rust implementation for high-performance anomaly detection
- `python/anomaly_grid_py/` - Python wrapper and utilities
- `tests/` - Comprehensive test suite with performance benchmarks
- `benchmarks/` - Performance evaluation tools and scripts

## Build, Test, and Development Commands

```bash
# Setup development environment
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Build the package (Rust + Python)
maturin develop

# Run tests
pytest tests/

# Run example
python example.py

# Format code
black python/anomaly_grid_py/ tests/
ruff check python/anomaly_grid_py/ tests/
```

## Coding Style & Naming Conventions

- **Indentation**: 4 spaces (Python), standard Rust formatting
- **File naming**: Snake_case for Python modules, kebab-case for Rust
- **Function/variable naming**: Snake_case for Python, snake_case for Rust
- **Linting**: Black (formatting), Ruff (linting), Cargo fmt (Rust)

## Testing Guidelines

- **Framework**: pytest for Python tests, built-in Rust testing
- **Test files**: `test_*.py` in `tests/` directory
- **Running tests**: `pytest tests/` or `python -m pytest tests/`
- **Coverage**: Focus on core functionality and edge cases

## Commit & Pull Request Guidelines

- **Commit format**: Descriptive messages, examples from repo:
  - "Prepare for release v0.4.0"
  - "fixed the same wheel compatibility issue in the performance benchmark step"
  - "implemented a pragmatic solution that acknowledges the PyPy 3.11 issue"
- **PR process**: CI must pass (tests, linting, security scans, wheel builds)
- **Branch naming**: Feature branches, merge to master

---

# Repository Tour

## 🎯 What This Repository Does

anomaly-grid-py is a high-performance Python library for sequence anomaly detection using variable-order Markov chains in finite alphabet sequences. It combines a Rust core for computational efficiency with a clean Python API for ease of use.

**Key responsibilities:**
- Detect anomalous patterns in sequential data (logs, user behavior, system events)
- Provide unsupervised learning for normal pattern recognition
- Deliver production-ready performance with minimal dependencies

---

## 🏗️ Architecture Overview

### System Context
```
[Training Data] → [AnomalyDetector] → [Anomaly Scores/Predictions]
                        ↓
                   [Rust Core Engine]
                        ↓
                   [Statistical Models]
```

### Key Components
- **Python Wrapper** (`python/anomaly_grid_py/`) - Clean API with scikit-learn compatibility
- **Rust Core** (`src/`) - High-performance anomaly detection engine using variable-order Markov chains
- **Utilities** (`utils.py`) - Lightweight helper functions with no external dependencies
- **Type System** - Full type hints and stub files for IDE support

### Data Flow
1. **Training**: Normal sequences → Statistical model building → Transition probability matrices
2. **Detection**: Test sequences → Likelihood calculation → Anomaly probability scoring
3. **Prediction**: Probability scores → Threshold application → Binary anomaly classification

---

## 📁 Project Structure [Partial Directory Tree]

```
anomaly-grid-py/
├── src/                        # Rust core implementation
│   ├── lib.rs                  # PyO3 module definition
│   ├── detector.rs             # Main anomaly detection logic
│   ├── arrays.rs               # Sequence array handling
│   └── errors.rs               # Error handling and conversion
├── python/anomaly_grid_py/     # Python package
│   ├── __init__.py             # Main API and AnomalyDetector class
│   ├── __init__.pyi            # Type stubs for IDE support
│   ├── utils.py                # Utility functions (no external deps)
│   └── _core.*.so              # Compiled Rust extension
├── tests/                      # Test suite
│   ├── test_anomaly_detector.py # Core functionality tests
│   ├── test_edge_cases.py      # Edge case handling
│   └── test_performance.py     # Performance benchmarks
├── benchmarks/                 # Performance evaluation tools
├── .github/workflows/          # CI/CD configuration
│   ├── ci.yml                  # Main CI pipeline
│   └── release.yml             # Release automation
├── Cargo.toml                  # Rust dependencies and build config
├── pyproject.toml              # Python package configuration
└── setup.sh                   # Development environment setup
```

### Key Files to Know

| File | Purpose | When You'd Touch It |
|------|---------|---------------------|
| `python/anomaly_grid_py/__init__.py` | Main Python API | Adding new features or API changes |
| `src/detector.rs` | Core Rust detection logic | Performance optimizations or algorithm changes |
| `Cargo.toml` | Rust dependencies | Adding Rust dependencies |
| `pyproject.toml` | Python package config | Changing Python dependencies or metadata |
| `setup.sh` | Development setup | Modifying build process |
| `example.py` | Usage examples | Understanding API or adding examples |
| `.github/workflows/ci.yml` | CI configuration | Changing build/test process |

---

## 🔧 Technology Stack

### Core Technologies
- **Language**: Python 3.8+ with Rust core - Combines Python ease-of-use with Rust performance
- **Framework**: PyO3 for Python-Rust bindings - Enables zero-copy operations and native performance
- **Build System**: Maturin - Specialized for Rust-Python hybrid packages
- **Dependencies**: Minimal by design - Only NumPy for Python, essential Rust crates

### Key Libraries
- **PyO3** - Python bindings for Rust, enables high-performance integration
- **NumPy** - Array operations and zero-copy data exchange
- **anomaly-grid** - Core Rust library for Markov chain anomaly detection
- **maturin** - Build tool for Python extensions written in Rust

### Development Tools
- **pytest** - Testing framework with benchmark support
- **black + ruff** - Code formatting and linting
- **maturin** - Build and development tool for Rust-Python packages

---

## 🌐 External Dependencies

### Required Services
- **anomaly-grid (Rust crate)** - Core anomaly detection algorithms, critical for functionality
- **NumPy** - Array operations and data interchange, essential for performance

### Optional Integrations
- **psutil** - Memory monitoring for benchmarks, fallback available
- **matplotlib** - Plotting for benchmarks, optional visualization

---

## 🔄 Common Workflows

### Basic Anomaly Detection Workflow
1. **Create detector**: `detector = AnomalyDetector(max_order=3)`
2. **Train on normal data**: `detector.fit(normal_sequences)`
3. **Set threshold**: Based on validation data, not arbitrary values
4. **Detect anomalies**: `scores = detector.predict_proba(test_sequences)`
5. **Make decisions**: `predictions = detector.predict(test_sequences, threshold)`

**Code path:** `__init__.py` → `_core.so` → `detector.rs` → `anomaly-grid` crate

### Development Workflow
1. **Setup environment**: `./setup.sh && source venv/bin/activate`
2. **Make changes**: Edit Python or Rust code
3. **Rebuild**: `maturin develop` (rebuilds Rust extension)
4. **Test**: `pytest tests/` or `python example.py`
5. **Format**: `black .` and `ruff check .`

**Code path:** Source changes → `maturin develop` → Compiled extension → Tests

---

## 📈 Performance & Scale

### Performance Characteristics
- **Throughput**: 1,000-20,000 sequences/second depending on configuration
- **Memory**: <10 MB for typical models with 1000+ training sequences
- **Training**: <100ms for datasets with 1000+ sequences
- **Scalability**: Tested up to 100K+ sequences

### Monitoring
- **Metrics**: Built-in performance metrics via `get_performance_metrics()`
- **Benchmarks**: Comprehensive benchmark suite in `tests/test_performance.py`
- **CI**: Automated performance regression testing

---

## 🚨 Things to Be Careful About

### 🔒 Security Considerations
- **Input validation**: All sequence inputs are validated for type and format
- **Memory safety**: Rust core provides memory safety guarantees
- **Dependencies**: Minimal dependency surface reduces attack vectors

### ⚠️ Important Implementation Notes
- **Threshold setting**: Always base thresholds on training/validation data, never use arbitrary values
- **Training data**: Use only normal (non-anomalous) sequences for training (unsupervised learning)
- **Evaluation metrics**: Use PR-AUC for evaluation, not ROC-AUC (more appropriate for anomaly detection)
- **Sequence length**: Minimum 2 elements required for pattern analysis
- **Memory usage**: Monitor memory with large vocabularies or high max_order values

### 🔧 Development Considerations
- **Rust changes**: Require `maturin develop` to rebuild the extension
- **CI compatibility**: PyPy 3.11 has known compatibility issues, handled gracefully in CI
- **Cross-platform**: Builds tested on Linux, macOS, and Windows
- **Version compatibility**: Python 3.8+ required, Rust stable toolchain

*Updated at: 2025-01-27 UTC*