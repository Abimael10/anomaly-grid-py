"""Edge case tests for the AnomalyDetector class."""

import pytest
import anomaly_grid_py


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_training_data(self):
        """Test training with empty data."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=2)

        with pytest.raises(RuntimeError):
            detector.train([])

    def test_empty_detection_data(self):
        """Test detection with empty data."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=2)
        detector.train(["A", "B", "A", "B"])

        results = detector.detect([], threshold=0.1)
        assert results == []

    def test_single_event_training(self):
        """Test training with single event."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=2)

        # Single event should raise an error
        with pytest.raises(RuntimeError, match="Sequence too short"):
            detector.train(["A"])

        # But minimum required data should work
        detector.train(["A", "B"])
        results = detector.detect(["A", "B", "C"], threshold=0.1)
        assert isinstance(results, list)

    def test_very_large_max_order(self):
        """Test with very large max_order."""
        # Should handle large max_order
        detector = anomaly_grid_py.AnomalyDetector(max_order=100)
        assert detector.max_order() == 100

    def test_zero_max_order(self):
        """Test with zero max_order."""
        with pytest.raises(ValueError):
            anomaly_grid_py.AnomalyDetector(max_order=0)

    def test_negative_max_order(self):
        """Test with negative max_order."""
        with pytest.raises(OverflowError):
            anomaly_grid_py.AnomalyDetector(max_order=-1)

    def test_invalid_threshold_values(self):
        """Test detection with invalid threshold values."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=2)
        detector.train(["A", "B", "A", "B"])

        # Test negative threshold
        with pytest.raises(RuntimeError):
            detector.detect(["A", "B"], threshold=-0.1)

        # Test threshold > 1
        with pytest.raises(RuntimeError):
            detector.detect(["A", "B"], threshold=1.5)

        # Test NaN threshold
        with pytest.raises(RuntimeError):
            detector.detect(["A", "B"], threshold=float("nan"))

    def test_unicode_events(self):
        """Test with unicode event strings."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=2)

        unicode_events = ["🔥", "💧", "🌪️", "🔥", "💧"]
        detector.train(unicode_events)

        results = detector.detect(["🔥", "⚡"], threshold=0.1)
        assert isinstance(results, list)

    def test_very_long_event_strings(self):
        """Test with very long event strings."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=2)

        long_event = "A" * 1000
        detector.train([long_event, "B", long_event])

        results = detector.detect([long_event, "X"], threshold=0.1)
        assert isinstance(results, list)

    def test_many_unique_events(self):
        """Test with many unique events."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=2)

        # Create 1000 unique events
        unique_events = [f"event_{i}" for i in range(1000)]
        detector.train(unique_events)

        # Should handle gracefully
        metrics = detector.get_performance_metrics()
        assert metrics["context_count"] > 0

    def test_detection_before_training(self):
        """Test detection before training."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=2)

        with pytest.raises(RuntimeError):
            detector.detect(["A", "B"], threshold=0.1)

    def test_repeated_training(self):
        """Test training multiple times."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=2)

        # First training
        detector.train(["A", "B", "A", "B"])
        metrics1 = detector.get_performance_metrics()

        # Second training (should accumulate)
        detector.train(["C", "D", "C", "D"])
        metrics2 = detector.get_performance_metrics()

        # Context count should increase
        assert metrics2["context_count"] >= metrics1["context_count"]
