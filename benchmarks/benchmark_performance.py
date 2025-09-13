#!/usr/bin/env python3
"""
Performance benchmark script for anomaly detection.

This script compares the performance of different detector configurations
and demonstrates the impact of optimizations.
"""

import time
import numpy as np
from anomaly_grid_py import (
    AnomalyDetector, 
    generate_sequences,
    PerformanceTimer
)
from anomaly_grid_py.optimized import OptimizedAnomalyDetector, PerformanceComparator


def generate_test_data(n_sequences=1000, seq_length=10):
    """Generate test data for benchmarking."""
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    sequences, _ = generate_sequences(
        n_sequences=n_sequences,
        seq_length=seq_length,
        alphabet=alphabet,
        anomaly_rate=0.1
    )
    return sequences


def benchmark_basic_operations():
    """Benchmark basic detector operations."""
    print("🔥 PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    # Generate test data
    print("📊 Generating test data...")
    training_data = generate_test_data(n_sequences=500, seq_length=8)
    test_data = generate_test_data(n_sequences=200, seq_length=8)
    
    print(f"Training data: {len(training_data)} sequences")
    print(f"Test data: {len(test_data)} sequences")
    print()
    
    # Create detectors
    detectors = {
        'Standard': AnomalyDetector(max_order=3),
        'Optimized_No_Cache': OptimizedAnomalyDetector(
            max_order=3, 
            enable_caching=False,
            enable_profiling=True
        ),
        'Optimized_With_Cache': OptimizedAnomalyDetector(
            max_order=3, 
            enable_caching=True,
            cache_size=500,
            enable_profiling=True
        )
    }
    
    # Run comparison
    print("🚀 Running performance comparison...")
    comparison_results = PerformanceComparator.compare_detectors(
        detectors=detectors,
        training_data=training_data,
        test_data=test_data,
        iterations=5
    )
    
    # Display results
    print("\n📈 PERFORMANCE RESULTS")
    print("-" * 50)
    
    for name, results in comparison_results['detectors'].items():
        stats = results['benchmark_results']['statistics']
        print(f"\n{name}:")
        print(f"  Training time: {results['training_time_seconds']:.3f}s")
        print(f"  Avg prediction time: {stats['avg_time_seconds']:.3f}s")
        print(f"  Throughput: {stats['avg_throughput_seq_per_sec']:.1f} seq/sec")
        
        # Show cache stats for optimized detectors
        if 'Optimized' in name:
            detector = detectors[name]
            metrics = detector.get_performance_metrics()
            
            if 'cache_stats' in metrics:
                cache_stats = metrics['cache_stats']
                print(f"  Cache hit rate: {cache_stats['hit_rate']:.2%}")
                print(f"  Cache size: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
    
    # Show summary
    summary = comparison_results['summary']
    print(f"\n🏆 SUMMARY")
    print(f"Fastest detector: {summary['fastest_detector']}")
    print(f"Highest throughput: {summary['highest_throughput_detector']}")
    print(f"Performance ranking: {' > '.join(summary['performance_ranking'])}")
    
    return comparison_results


def benchmark_caching_impact():
    """Benchmark the impact of caching on repeated predictions."""
    print("\n🔄 CACHING IMPACT BENCHMARK")
    print("=" * 50)
    
    # Create test data with some repeated sequences
    base_sequences = generate_test_data(n_sequences=50, seq_length=6)
    
    # Create repeated test data (simulating real-world repeated patterns)
    repeated_test_data = []
    for _ in range(100):
        repeated_test_data.extend(base_sequences[:10])  # Repeat first 10 sequences
    
    print(f"Test data: {len(repeated_test_data)} sequences (with repetitions)")
    
    # Test with and without caching
    detector_no_cache = OptimizedAnomalyDetector(
        max_order=3, 
        enable_caching=False,
        enable_profiling=True
    )
    
    detector_with_cache = OptimizedAnomalyDetector(
        max_order=3, 
        enable_caching=True,
        cache_size=100,
        enable_profiling=True
    )
    
    # Train both detectors
    training_data = generate_test_data(n_sequences=100, seq_length=6)
    detector_no_cache.fit(training_data)
    detector_with_cache.fit(training_data)
    
    # Benchmark repeated predictions
    print("\n🔄 Testing repeated predictions...")
    
    # No cache
    start_time = time.perf_counter()
    scores_no_cache = detector_no_cache.predict_proba(repeated_test_data)
    time_no_cache = time.perf_counter() - start_time
    
    # With cache
    start_time = time.perf_counter()
    scores_with_cache = detector_with_cache.predict_proba(repeated_test_data)
    time_with_cache = time.perf_counter() - start_time
    
    # Results
    print(f"\nNo cache: {time_no_cache:.3f}s")
    print(f"With cache: {time_with_cache:.3f}s")
    print(f"Speedup: {time_no_cache / time_with_cache:.2f}x")
    
    # Cache statistics
    cache_stats = detector_with_cache.get_performance_metrics()['cache_stats']
    print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
    
    # Verify results are the same
    max_diff = np.max(np.abs(scores_no_cache - scores_with_cache))
    print(f"Max score difference: {max_diff:.6f} (should be ~0)")


def benchmark_batch_processing():
    """Benchmark batch processing performance."""
    print("\n📦 BATCH PROCESSING BENCHMARK")
    print("=" * 50)
    
    # Generate large test dataset
    large_test_data = generate_test_data(n_sequences=1000, seq_length=8)
    training_data = generate_test_data(n_sequences=200, seq_length=8)
    
    print(f"Large test dataset: {len(large_test_data)} sequences")
    
    # Test different batch sizes
    batch_sizes = [50, 100, 200, 500]
    
    detector = OptimizedAnomalyDetector(
        max_order=3,
        enable_caching=False,  # Focus on batching
        enable_profiling=True
    )
    detector.fit(training_data)
    
    print("\n📊 Testing different batch sizes...")
    
    for batch_size in batch_sizes:
        detector.batch_size = batch_size
        
        start_time = time.perf_counter()
        scores = detector.predict_proba(large_test_data, use_batching=True)
        batch_time = time.perf_counter() - start_time
        
        throughput = len(large_test_data) / batch_time
        print(f"Batch size {batch_size:3d}: {batch_time:.3f}s ({throughput:.1f} seq/sec)")
    
    # Compare with no batching
    start_time = time.perf_counter()
    scores_no_batch = detector.predict_proba(large_test_data, use_batching=False)
    no_batch_time = time.perf_counter() - start_time
    
    no_batch_throughput = len(large_test_data) / no_batch_time
    print(f"No batching  : {no_batch_time:.3f}s ({no_batch_throughput:.1f} seq/sec)")


def main():
    """Run all performance benchmarks."""
    print("🎯 ANOMALY DETECTION PERFORMANCE BENCHMARKS")
    print("=" * 60)
    
    try:
        # Basic operations benchmark
        basic_results = benchmark_basic_operations()
        
        # Caching impact benchmark
        benchmark_caching_impact()
        
        # Batch processing benchmark
        benchmark_batch_processing()
        
        print("\n✅ All benchmarks completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()