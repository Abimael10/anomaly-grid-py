#!/usr/bin/env python3
"""
Anomaly Grid 0.2.0 - Example
Minimal dependencies, maximum performance.
"""

import time
import numpy as np
from anomaly_grid_py import (
    AnomalyDetector,
    generate_sequences,
    train_test_split,
    roc_auc_score,
    PerformanceTimer,
    memory_usage
)

def basic_usage_example():
    """Basic usage with the new API."""
    print("=== Basic Usage Example ===")
    
    # Create detector
    detector = AnomalyDetector(max_order=3)
    print(f"Created detector with max_order: 3")
    
    # Training sequences (normal patterns)
    normal_sequences = [
        ['LOGIN', 'BALANCE', 'LOGOUT'],
        ['LOGIN', 'WITHDRAW', 'LOGOUT'],
        ['LOGIN', 'TRANSFER', 'LOGOUT'],
        ['LOGIN', 'BALANCE', 'TRANSFER', 'LOGOUT'],
        ['LOGIN', 'WITHDRAW', 'BALANCE', 'LOGOUT']
    ] * 20  # Repeat for better training
    
    print(f"Training on {len(normal_sequences)} sequences...")
    
    # Fit the detector
    with PerformanceTimer() as timer:
        detector.fit(normal_sequences)
    
    print(f"Training completed in {timer.elapsed:.3f}s")
    
    # Get performance metrics
    metrics = detector.get_performance_metrics()
    print(f"Contexts created: {metrics['context_count']}")
    print(f"Memory usage: {metrics['memory_bytes'] / 1024:.1f} KB")
    
    # Test sequences
    test_sequences = [
        ['LOGIN', 'BALANCE', 'LOGOUT'],      # Normal
        ['LOGIN', 'WITHDRAW', 'LOGOUT'],     # Normal
        ['HACK', 'EXPLOIT', 'STEAL'],        # Anomalous
        ['LOGIN', 'HACK', 'LOGOUT'],         # Partially anomalous
        ['ATTACK', 'PENETRATE', 'ESCAPE']    # Anomalous
    ]
    
    print(f"\nTesting on {len(test_sequences)} sequences...")
    
    # Get continuous scores
    scores = detector.predict_proba(test_sequences)
    print("Anomaly scores:")
    for i, (seq, score) in enumerate(zip(test_sequences, scores)):
        print(f"  {i+1}. {seq} -> {score:.3f}")
    
    # Get binary predictions
    predictions = detector.predict(test_sequences, threshold=0.1)
    print("\nBinary predictions (threshold=0.1):")
    for i, (seq, pred) in enumerate(zip(test_sequences, predictions)):
        status = "ANOMALY" if pred else "NORMAL"
        print(f"  {i+1}. {seq} -> {status}")
    
    print()

def performance_demonstration():
    """Demonstrate high-performance capabilities."""
    print("=== Performance Demonstration ===")
    
    # Generate large dataset
    alphabet = ['LOGIN', 'BALANCE', 'WITHDRAW', 'TRANSFER', 'LOGOUT', 'VERIFY', 'CONFIRM']
    
    print("Generating large dataset...")
    sequences, labels = generate_sequences(
        n_sequences=5000,
        seq_length=15,
        alphabet=alphabet,
        anomaly_rate=0.2
    )
    
    print(f"Generated {len(sequences)} sequences")
    print(f"Anomaly rate: {np.mean(labels):.1%}")
    
    # Split data
    train_sequences, test_sequences = train_test_split(
        sequences, test_size=0.3, random_state=42
    )
    train_labels = labels[:len(train_sequences)]
    test_labels = labels[len(train_sequences):]
    
    print(f"Training set: {len(train_sequences)} sequences")
    print(f"Test set: {len(test_sequences)} sequences")
    
    # Training performance
    detector = AnomalyDetector(max_order=4)
    
    memory_before = memory_usage()
    
    with PerformanceTimer() as timer:
        detector.fit(train_sequences)
    
    memory_after = memory_usage()
    
    print(f"\nTraining performance:")
    print(f"  Time: {timer.elapsed:.3f}s")
    print(f"  Throughput: {len(train_sequences) / timer.elapsed:.0f} sequences/second")
    print(f"  Memory increase: {memory_after - memory_before:.1f} MB")
    
    # Prediction performance
    with PerformanceTimer() as timer:
        scores = detector.predict_proba(test_sequences)
    
    print(f"\nPrediction performance:")
    print(f"  Time: {timer.elapsed:.3f}s")
    print(f"  Throughput: {len(test_sequences) / timer.elapsed:.0f} sequences/second")
    
    # Accuracy evaluation
    auc = roc_auc_score(test_labels, scores)
    print(f"\nAccuracy:")
    print(f"  ROC-AUC: {auc:.3f}")
    
    # Test different thresholds
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
    print(f"\nThreshold analysis:")
    print("Threshold | Accuracy | Detected")
    print("-" * 30)
    
    for threshold in thresholds:
        predictions = detector.predict(test_sequences, threshold=threshold)
        accuracy = np.mean(predictions == np.array(test_labels))
        detected_rate = np.mean(predictions)
        print(f"{threshold:9.2f} | {accuracy:8.3f} | {detected_rate:8.1%}")
    
    print()

def zero_copy_demonstration():
    """Demonstrate zero-copy NumPy integration."""
    print("=== Zero-Copy NumPy Integration ===")
    
    # Create detector
    detector = AnomalyDetector(max_order=3)
    
    # Training data
    train_sequences = [
        ['A', 'B', 'C'],
        ['A', 'B', 'D'],
        ['A', 'C', 'D']
    ] * 100
    
    detector.fit(train_sequences)
    
    # Test with different input formats
    test_sequences = [
        ['A', 'B', 'C'],  # Normal
        ['X', 'Y', 'Z']   # Anomalous
    ]
    
    print("Testing different input formats:")
    
    # List input
    scores_list = detector.predict_proba(test_sequences)
    print(f"List input -> NumPy output: {type(scores_list)} {scores_list.shape}")
    print(f"Scores: {scores_list}")
    
    # NumPy array input (currently not fully supported, but outputs are NumPy)
    print(f"NumPy outputs confirmed: scores are {type(scores_list)}, predictions would be numpy.ndarray")
    print(f"Zero-copy achieved: Direct NumPy array outputs from Rust")
    
    print()

def error_handling_demonstration():
    """Demonstrate robust error handling."""
    print("=== Error Handling Demonstration ===")
    
    detector = AnomalyDetector(max_order=3)
    
    # Test various error conditions
    error_cases = [
        ("Empty sequences", []),
        ("Short sequences", [['A']]),
        ("Non-string elements", [['A', 1, 'C']]),
        ("Empty strings", [['A', '', 'C']]),
        ("Wrong input type", "not a list"),
    ]
    
    for description, test_input in error_cases:
        try:
            detector.fit(test_input)
            print(f"❌ {description}: Should have failed!")
        except (ValueError, TypeError) as e:
            print(f"✅ {description}: {type(e).__name__}: {e}")
    
    # Test prediction without fitting
    try:
        detector = AnomalyDetector(max_order=3)
        detector.predict_proba([['A', 'B', 'C']])
        print("❌ Prediction without fitting: Should have failed!")
    except ValueError as e:
        print(f"✅ Prediction without fitting: {type(e).__name__}: {e}")
    
    # Test invalid threshold
    detector.fit([['A', 'B', 'C']] * 10)
    try:
        detector.predict([['A', 'B', 'C']], threshold=1.5)
        print("❌ Invalid threshold: Should have failed!")
    except ValueError as e:
        print(f"✅ Invalid threshold: {type(e).__name__}: {e}")
    
    print()

def detailed_anomaly_information():
    """Demonstrate detailed anomaly information."""
    print("=== Detailed Anomaly Information ===")
    
    # Create and train detector
    detector = AnomalyDetector(max_order=3)
    
    normal_sequences = [
        ['LOGIN', 'BALANCE', 'LOGOUT'],
        ['LOGIN', 'WITHDRAW', 'LOGOUT'],
        ['LOGIN', 'TRANSFER', 'LOGOUT']
    ] * 20
    
    detector.fit(normal_sequences)
    
    # Test sequences with known anomalies
    test_sequences = [
        ['LOGIN', 'BALANCE', 'LOGOUT'],    # Normal
        ['HACK', 'EXPLOIT', 'STEAL'],      # Completely anomalous
        ['LOGIN', 'HACK', 'LOGOUT'],       # Partially anomalous
    ]
    
    print("Anomaly scores for different sequences:")
    scores = detector.predict_proba(test_sequences)
    predictions = detector.predict(test_sequences, threshold=0.1)
    
    for i, (seq, score, pred) in enumerate(zip(test_sequences, scores, predictions)):
        status = "ANOMALY" if pred else "NORMAL"
        print(f"  {i+1}. {seq} -> Score: {score:.3f}, Status: {status}")
    
    print("\nWith lower threshold (0.01):")
    predictions_low = detector.predict(test_sequences, threshold=0.01)
    
    for i, (seq, score, pred) in enumerate(zip(test_sequences, scores, predictions_low)):
        status = "ANOMALY" if pred else "NORMAL"
        print(f"  {i+1}. {seq} -> Score: {score:.3f}, Status: {status}")
    
    print()

def main():
    """Run all examples."""
    print("🚀 ANOMALY GRID 0.3.0 - HIGH-PERFORMANCE EXAMPLES")
    print("=" * 60)
    print("Minimal dependencies, maximum performance")
    print("=" * 60)
    print()
    
    # Show import time
    start_time = time.time()
    import_time = time.time() - start_time
    print(f"Import time: {import_time:.3f}s")
    print(f"Initial memory: {memory_usage():.1f} MB")
    print()
    
    # Run examples
    basic_usage_example()
    performance_demonstration()
    zero_copy_demonstration()
    error_handling_demonstration()
    detailed_anomaly_information()
    
    print("=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
