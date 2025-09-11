"""
Performance comparison: minimal deps vs full deps implementation.
"""
import time
import numpy as np
from anomaly_grid_py import AnomalyDetector, generate_sequences, PerformanceTimer

def benchmark_training(n_sequences_list, seq_length=20, alphabet_size=10):
    """Benchmark training performance."""
    alphabet = [f"state_{i}" for i in range(alphabet_size)]
    results = {}
    
    for n_sequences in n_sequences_list:
        print(f"Benchmarking {n_sequences} sequences...")
        
        # Generate data
        sequences, _ = generate_sequences(n_sequences, seq_length, alphabet)
        
        # Benchmark training
        detector = AnomalyDetector(max_order=3)
        
        with PerformanceTimer() as timer:
            detector.fit(sequences)
        
        metrics = detector.get_performance_metrics()
        
        results[n_sequences] = {
            'training_time': timer.elapsed,
            'memory_bytes': metrics['memory_bytes'],
            'context_count': metrics['context_count']
        }
        
        print(f"  Training time: {timer.elapsed:.3f}s")
        print(f"  Memory usage: {metrics['memory_bytes'] / 1024 / 1024:.1f} MB")
        print(f"  Contexts: {metrics['context_count']}")
    
    return results

def benchmark_prediction(n_test_sequences_list, seq_length=20):
    """Benchmark prediction performance."""
    alphabet = [f"state_{i}" for i in range(10)]
    
    # Train once
    train_sequences, _ = generate_sequences(1000, seq_length, alphabet)
    detector = AnomalyDetector(max_order=3)
    detector.fit(train_sequences)
    
    results = {}
    
    for n_test in n_test_sequences_list:
        print(f"Benchmarking {n_test} test sequences...")
        
        # Generate test data
        test_sequences, _ = generate_sequences(n_test, seq_length, alphabet)
        
        # Benchmark prediction
        with PerformanceTimer() as timer:
            scores = detector.predict_proba(test_sequences)
        
        throughput = n_test / timer.elapsed
        
        results[n_test] = {
            'prediction_time': timer.elapsed,
            'throughput': throughput,
            'scores_shape': scores.shape
        }
        
        print(f"  Prediction time: {timer.elapsed:.3f}s")
        print(f"  Throughput: {throughput:.0f} sequences/second")
    
    return results

def memory_efficiency_test():
    """Test memory efficiency."""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory: {initial_memory:.1f} MB")
        
        # Create large dataset
        alphabet = [f"state_{i}" for i in range(20)]
        sequences, _ = generate_sequences(10000, 50, alphabet)
        
        detector = AnomalyDetector(max_order=4)
        detector.fit(sequences)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        metrics = detector.get_performance_metrics()
        estimated_memory = metrics['memory_bytes'] / 1024 / 1024  # MB
        
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"Memory increase: {memory_increase:.1f} MB")
        print(f"Estimated by library: {estimated_memory:.1f} MB")
        print(f"Efficiency ratio: {estimated_memory / memory_increase:.2f}")
    except ImportError:
        print("psutil not available - skipping memory efficiency test")

if __name__ == "__main__":
    print("=== Anomaly Grid Performance Benchmarks ===")
    print()
    
    print("1. Training Performance")
    training_results = benchmark_training([100, 500, 1000, 5000])
    print()
    
    print("2. Prediction Performance") 
    prediction_results = benchmark_prediction([100, 500, 1000, 5000])
    print()
    
    print("3. Memory Efficiency")
    memory_efficiency_test()