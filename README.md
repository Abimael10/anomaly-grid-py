# anomaly-grid-py

[![CI](https://github.com/abimael10/anomaly-grid-py/workflows/CI/badge.svg)](https://github.com/abimael10/anomaly-grid-py/actions)
[![PyPI version](https://img.shields.io/pypi/v/anomaly-grid-py.svg)](https://pypi.org/project/anomaly-grid-py/)
[![Python versions](https://img.shields.io/pypi/pyversions/anomaly-grid-py.svg)](https://pypi.org/project/anomaly-grid-py/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/anomaly-grid-py.svg)](https://pypi.org/project/anomaly-grid-py/)

**Sequence deviation detection using variable-order Markov chains for finite alphabet sequences.**

## 🎯 Niche Focus & Strengths

This library excels at detecting **temporal pattern violations** in sequences with **small, finite alphabets** (mostly ≤20 symbols but can be tested for more). Based on comprehensive benchmarking, it shows clear advantages over traditional ML when:

- **Sequential order matters** - State machines, protocols, biological sequences (this one in particular I have to test it further because depending on how I try to create the synthetic data for this particular one, higher orders than 2 perform way worse and have significant overheaD)
- **Finite vocabularies** - Network states (12-16 symbols), amino acids (20), communication protocols
- **Multi-order dependencies** - Patterns that span 2-4 sequence elements
- **Subtle deviations** - Violations of learned transition rules

### ✅ **Proven Performance Advantages**
- **Protocol State Machines** (16 states): **+6.2% F1 improvement** over traditional ML (F1: 0.71 vs 0.67)
- **Communication Protocols** (12 symbols): **+3.0% F1 improvement** over traditional ML (F1: 0.52 vs 0.50)
- **Temporal pattern recognition**: Consistently outperforms on sequence-dependent anomalies

To confirm these results that were taken today Sep 16th, 2025, run this notebook: [Open notebook](benchmark.ipynb) 

### ⚠️ **Realistic Limitations**
- **Moderate performance ceiling**: F1 scores typically 0.45-0.71 on challenging datasets where the ideal solution would be to use other algorithms like isolation forest, etc.
- **Small alphabet requirement**: Performance advantage diminishes with >20 symbols but sometimes with the ideal batch processing this can be handled or sometimes it is not necessary to go beyond 20 states, and that is where our approach can handle it better. It is always best to keep comparing for results.
- **Training data needs**: Requires 100+ normal sequences for stable performance
- **Subtle anomalies**: Performance degrades with very low contamination rates (<2%)

## Installation

```bash
pip install anomaly-grid-py
```

## Quick Start

```python
import anomaly_grid_py

# Create detector with appropriate order for your alphabet size
detector = anomaly_grid_py.AnomalyDetector(max_order=3)

# Train on normal sequences only (unsupervised learning)
normal_sequences = [
    ['INIT', 'LISTEN', 'SYN_RECV', 'ESTABLISHED', 'DATA_XFER', 'CLOSED'],
    ['INIT', 'SYN_SENT', 'ESTABLISHED', 'AUTH', 'DATA_XFER', 'CLOSED'],
    ['INIT', 'LISTEN', 'SYN_RECV', 'ESTABLISHED', 'CLOSE_WAIT', 'CLOSED']
] * 100  # Need sufficient training data (typically 100+ sequences)

detector.fit(normal_sequences)

# Detect anomalies in test sequences
test_sequences = [
    ['INIT', 'LISTEN', 'SYN_RECV', 'ESTABLISHED', 'DATA_XFER', 'CLOSED'],  # Normal
    ['INIT', 'ESTABLISHED', 'DATA_XFER', 'CLOSED'],  # Anomalous: skipped states
    ['INIT', 'LISTEN', 'ERROR', 'RESET', 'CLOSED']   # Anomalous: unexpected error
]

# Get anomaly scores [0,1] - higher means more anomalous
scores = detector.predict_proba(test_sequences)
print(f"Anomaly scores: {scores}")

# Get binary predictions with optimized threshold
anomalies = detector.predict(test_sequences, threshold=0.5)
print(f"Anomalies detected: {anomalies}")
```

## Example: Network Protocol Analysis

```python
import anomaly_grid_py

# Network connection state sequences (16-state protocol)
normal_connections = [
    ['INIT', 'SYN_SENT', 'ESTABLISHED', 'DATA_XFER', 'FIN_WAIT1', 'CLOSED'],
    ['INIT', 'LISTEN', 'SYN_RECV', 'ESTABLISHED', 'DATA_XFER', 'CLOSE_WAIT', 'CLOSED'],
    ['INIT', 'SYN_SENT', 'ESTABLISHED', 'AUTH', 'DATA_XFER', 'DATA_XFER', 'CLOSED']
] * 200

# Train detector with higher order for complex state dependencies
detector = anomaly_grid_py.AnomalyDetector(max_order=4)  # Higher order for 16-state alphabet
detector.fit(normal_connections)

# Test sequences with potential attacks
test_connections = [
    ['INIT', 'SYN_SENT', 'ESTABLISHED', 'DATA_XFER', 'CLOSED'],        # Normal
    ['INIT', 'ESTABLISHED', 'DATA_XFER', 'CLOSED'],                    # SYN flood attack
    ['INIT', 'SYN_SENT', 'RESET', 'INIT', 'SYN_SENT', 'RESET'],      # Connection reset attack
    ['INIT', 'LISTEN', 'SYN_RECV', 'ERROR', 'CLOSED']                 # Protocol violation
]

scores = detector.predict_proba(test_connections)
print("Connection anomaly scores:", scores)
# Expected: Normal sequences ~0.2, attacks ~0.6-0.8
```

## Benchmarked Performance

Based on rigorous evaluation across finite alphabet datasets with 3-fold cross-validation:

| Dataset Type | Alphabet Size | Sequence-Based F1 | Traditional ML F1 | Advantage |
|--------------|---------------|-------------------|-------------------|-----------|
| **Protocol State Machines** | 16 symbols | **0.71** | 0.67 | **+6.2%** |
| **Communication Protocols** | 12 symbols | **0.52** | 0.50 | **+3.0%** |
| **Biological Sequences** | 20 symbols | 0.45-0.60* | 0.45-0.55* | **+2-5%** |

*Performance varies significantly based on sequence complexity and contamination rate (2-3%).

### Key Performance Insights

- **Best performance**: 16-state protocols with order=4 (F1: 0.71, AUC: 0.90)
- **Order selection matters**: Higher orders (3-4) work better for larger alphabets (16-20 symbols)
- **Realistic expectations**: Some edge cases might underperform

## API Reference

### AnomalyDetector

```python
# Initialize detector
detector = AnomalyDetector(max_order=3)
```

**Parameters:**
- `max_order` (int): Maximum n-gram order (1-4). Higher orders capture longer dependencies but need more training data.

**Recommended orders by alphabet size:**
- 8-12 symbols: `max_order=2-3`
- 13-16 symbols: `max_order=3-4` (best performance with order=4)
- 17-20 symbols: `max_order=3-4`
- >20 symbols: Consider other algorithms

### Methods

```python
# Train on normal sequences (unsupervised)
detector.fit(sequences)

# Get anomaly probability scores [0,1]
scores = detector.predict_proba(sequences)

# Get binary anomaly predictions
predictions = detector.predict(sequences, threshold=0.5)

# Get model performance metrics
metrics = detector.get_performance_metrics()
```

### Threshold Selection

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

# Use validation data to find optimal threshold
val_scores = detector.predict_proba(validation_sequences)
precision, recall, thresholds = precision_recall_curve(val_labels, val_scores)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
optimal_threshold = thresholds[np.argmax(f1_scores)]

# Apply to test data
predictions = detector.predict(test_sequences, threshold=optimal_threshold)
```

## Best Practices

### 1. **Data Requirements**
```python
# Ensure sufficient training data
assert len(normal_sequences) >= 100, "Need at least 100 training sequences"

# Check sequence lengths (based on benchmark averages)
avg_length = np.mean([len(seq) for seq in normal_sequences])
assert avg_length >= 50, "Sequences should be at least 50 elements for good performance"

# Verify alphabet size
alphabet = set()
for seq in normal_sequences:
    alphabet.update(seq)
assert len(alphabet) <= 20, f"Alphabet size {len(alphabet)} may be too large for optimal performance"
```

### 2. **Order Selection Strategy**
```python
# Test different orders based on alphabet size
alphabet_size = len(set(symbol for seq in train_sequences for symbol in seq))

if alphabet_size <= 12:
    test_orders = [2, 3]
elif alphabet_size <= 16:
    test_orders = [3, 4]  # Order 4 showed best results for 16-state protocols
else:
    test_orders = [3, 4]

best_f1 = 0
best_order = 2

for order in test_orders:
    detector = AnomalyDetector(max_order=order)
    detector.fit(train_sequences)
    scores = detector.predict_proba(val_sequences)
    # Calculate F1 and select best order
```

### 3. **Performance Evaluation**
```python
# Use appropriate metrics for imbalanced data (typical contamination: 2-3%)
from sklearn.metrics import classification_report, average_precision_score

scores = detector.predict_proba(test_sequences)
predictions = detector.predict(test_sequences, threshold=optimal_threshold)

print(classification_report(test_labels, predictions))
print(f"Average Precision: {average_precision_score(test_labels, scores):.3f}")
print(f"ROC AUC: {roc_auc_score(test_labels, scores):.3f}")
```

## When to Use This Library

### ✅ **Ideal Use Cases**
- **Network protocol monitoring**: State machine violations, unexpected transitions
- **System workflow validation**: Process step anomalies, sequence deviations
- **Communication analysis**: Protocol timing attacks, message flow anomalies
- **Quality control**: Manufacturing step violations, procedure deviations

### ⚠️ **Consider Alternatives When**
- **Large vocabularies** (>20 symbols): Traditional ML may perform better
- **Continuous data**: Requires discretization which may lose information
- **Very high-dimensional features**: Feature-based approaches more suitable
- **Real-time constraints**: May need optimization for high-throughput scenarios

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
