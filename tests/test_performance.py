"""Performance tests for the AnomalyDetector class."""

import time

import anomaly_grid_py
import numpy as np


class TestPerformance:
    """Performance benchmarks for anomaly detection."""

    def test_training_performance(self):
        """Test training performance."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=3)

        # Create sequences for training (new API expects list of sequences)
        training_sequences = []
        for i in range(1000):  # 1000 sequences
            training_sequences.append(["A", "B", "C"])

        start_time = time.time()
        detector.fit(training_sequences)
        end_time = time.time()

        training_time = end_time - start_time
        assert training_time < 2.0  # Should complete in under 2 seconds
        print(f"Training 1000 sequences took {training_time:.4f}s")

    def test_detection_performance(self):
        """Test detection performance."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=3)

        # Train with sequences
        training_sequences = [["A", "B", "C"]] * 100
        detector.fit(training_sequences)

        # Create test sequences
        test_sequences = []
        for i in range(400):  # 400 test sequences
            if i % 4 == 0:
                test_sequences.append(["A", "B", "X"])  # Some anomalous
            else:
                test_sequences.append(["A", "B", "C"])  # Mostly normal

        start_time = time.time()
        scores = detector.predict_proba(test_sequences)
        end_time = time.time()

        detection_time = end_time - start_time
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 400
        assert detection_time < 1.0  # Should complete in under 1 second
        print(f"Detection of 400 sequences took {detection_time:.4f}s")

    def test_memory_usage(self):
        """Test memory usage with large datasets."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=5)

        # Train with large dataset of sequences
        large_sequences = []
        for i in range(2000):  # 2000 sequences
            # Create varied sequences
            seq = [f"event_{i % 100}", f"event_{(i+1) % 100}", f"event_{(i+2) % 100}"]
            large_sequences.append(seq)

        detector.fit(large_sequences)

        # Check metrics
        metrics = detector.get_performance_metrics()
        assert "memory_bytes" in metrics
        assert "context_count" in metrics
        assert metrics["memory_bytes"] >= 0
        assert metrics["context_count"] >= 0

    def test_scalability(self):
        """Test scalability with increasing data sizes."""
        sizes = [100, 500, 1000, 2000]
        times = []

        for size in sizes:
            detector = anomaly_grid_py.AnomalyDetector(max_order=3)

            # Create sequences of the specified size
            sequences = []
            for i in range(size):
                sequences.append(["A", "B", "C"])

            start_time = time.time()
            detector.fit(sequences)
            end_time = time.time()

            times.append(end_time - start_time)

        # Training time should scale reasonably
        assert all(t < 2.0 for t in times)  # Should be under 2 seconds
        print(f"Training times for sizes {sizes}: {[f'{t:.4f}s' for t in times]}")

    def test_prediction_scalability(self):
        """Test prediction scalability with increasing test sizes."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=3)

        # Train once
        training_sequences = [["A", "B", "C"]] * 100
        detector.fit(training_sequences)

        test_sizes = [100, 500, 1000, 2000]
        times = []

        for size in test_sizes:
            # Create test sequences
            test_sequences = []
            for i in range(size):
                if i % 10 == 0:
                    test_sequences.append(["X", "Y", "Z"])  # Some anomalous
                else:
                    test_sequences.append(["A", "B", "C"])  # Mostly normal

            start_time = time.time()
            scores = detector.predict_proba(test_sequences)
            end_time = time.time()

            assert len(scores) == size
            times.append(end_time - start_time)

        # Prediction time should scale reasonably
        assert all(t < 1.0 for t in times)  # Should be under 1 second
        print(
            f"Prediction times for sizes {test_sizes}: {[f'{t:.4f}s' for t in times]}"
        )

    def test_large_vocabulary(self):
        """Test performance with large vocabulary."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=2)

        # Create sequences with large vocabulary
        vocab_size = 1000
        sequences = []
        for i in range(500):  # 500 sequences
            seq = [f"word_{i % vocab_size}", f"word_{(i+1) % vocab_size}"]
            sequences.append(seq)

        start_time = time.time()
        detector.fit(sequences)
        training_time = time.time() - start_time

        # Test prediction with new vocabulary
        test_sequences = [[f"word_{vocab_size + 1}", f"word_{vocab_size + 2}"]]

        start_time = time.time()
        scores = detector.predict_proba(test_sequences)
        prediction_time = time.time() - start_time

        assert isinstance(scores, np.ndarray)
        assert len(scores) == 1
        assert training_time < 2.0
        assert prediction_time < 0.1

        print(
            f"Large vocabulary: training={training_time:.4f}s, prediction={prediction_time:.4f}s"
        )
