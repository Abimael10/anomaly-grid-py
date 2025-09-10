"""
Performance tests for the AnomalyDetector class
"""

import pytest
import time
import anomaly_grid_py


class TestPerformance:
    """Performance benchmarks for anomaly detection."""

    def test_training_performance(self):
        """Test training performance."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=3)
        training_data = ["A", "B", "C"] * 1000  # 3000 events

        import time

        start_time = time.time()
        detector.train(training_data)
        end_time = time.time()

        training_time = end_time - start_time
        assert training_time < 1.0  # Should complete in under 1 second
        print(f"Training 3000 events took {training_time:.4f}s")

    def test_detection_performance(self):
        """Test detection performance."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=3)
        training_data = ["A", "B", "C"] * 100
        detector.train(training_data)

        test_data = ["A", "B", "X", "Y"] * 100  # 400 events

        import time

        start_time = time.time()
        result = detector.detect(test_data, 0.1)
        end_time = time.time()

        detection_time = end_time - start_time
        assert isinstance(result, list)
        assert detection_time < 1.0  # Should complete in under 1 second
        print(f"Detection of 400 events took {detection_time:.4f}s")

    def test_memory_usage(self):
        """Test memory usage with large datasets."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=5)

        # Train with large dataset
        large_dataset = [f"event_{i % 100}" for i in range(10000)]
        detector.train(large_dataset)

        # Check metrics
        metrics = detector.get_performance_metrics()
        assert metrics["estimated_memory_bytes"] > 0
        assert metrics["context_count"] > 0

    def test_scalability(self):
        """Test scalability with increasing data sizes."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=3)

        sizes = [100, 500, 1000, 2000]
        times = []

        for size in sizes:
            data = ["A", "B", "C"] * (size // 3)

            start_time = time.time()
            detector.train(data)
            end_time = time.time()

            times.append(end_time - start_time)

        # Training time should scale reasonably
        assert all(t < 1.0 for t in times)  # Should be under 1 second
