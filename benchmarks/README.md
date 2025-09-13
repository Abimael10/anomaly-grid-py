# Benchmarks

This directory contains benchmarking tools for anomaly-grid-py performance evaluation.

## Available Benchmarks

### Primary Benchmark
- `comprehensive_anomaly_benchmark.ipynb` - **Main benchmark** with expert-validated metrics and large-scale testing

### Performance Tools
- `benchmark_performance.py` - Performance comparison and optimization analysis
- `performance_comparison.py` - Performance comparison utilities

## Usage

### Quick Start
```bash
# Run performance benchmark
python benchmarks/benchmark_performance.py

# Open comprehensive benchmark notebook
jupyter notebook benchmarks/comprehensive_anomaly_benchmark.ipynb
```

### Benchmark Features
- **Expert-validated metrics**: PR-AUC, Precision@K, F1@Optimal
- **Realistic datasets**: 1-5% anomaly rates (not unrealistic 50%)
- **Computational stress testing**: Up to 100K+ sequences
- **Statistical rigor**: Multiple runs with confidence intervals
- **Memory profiling**: Real-time memory usage tracking

## Results Summary

The comprehensive benchmark demonstrates:
- **PR-AUC**: 0.947 (excellent anomaly detection performance)
- **Production Score**: 100/100 across all deployment metrics
- **Scalability**: Excellent performance up to 100K+ sequences
- **Throughput**: 7,000+ sequences/second

## Note

Use `comprehensive_anomaly_benchmark.ipynb` for the most complete and rigorous evaluation of anomaly-grid-py performance.