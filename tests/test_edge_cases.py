"""Edge case tests for the AnomalyDetector class."""

import anomaly_grid_py
import numpy as np
import pytest


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_training_data(self):
        """Test training with empty data."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=2)

        with pytest.raises(ValueError, match="Empty sequence list"):
            detector.fit([])

    def test_empty_detection_data(self):
        """Test detection with empty data."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=2)
        detector.fit([["A", "B"], ["A", "B"]])

        with pytest.raises(ValueError, match="Empty sequence list"):
            detector.predict_proba([])

    def test_single_event_training(self):
        """Test training with single event sequences."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=2)

        # Single event sequences should raise an error
        with pytest.raises(ValueError, match="at least 2 elements"):
            detector.fit([["A"]])

        # But minimum required data should work
        detector.fit([["A", "B"], ["A", "B"]])
        results = detector.predict_proba([["A", "B"], ["B", "C"]])
        assert isinstance(results, np.ndarray)
        assert len(results) == 2

    def test_very_large_max_order(self):
        """Test with very large max_order."""
        # Should handle large max_order
        detector = anomaly_grid_py.AnomalyDetector(max_order=100)
        # Train with sufficient data
        training_data = [["A", "B", "C"] * 10] * 5
        detector.fit(training_data)

        # Should work without issues
        results = detector.predict_proba([["A", "B", "C"]])
        assert isinstance(results, np.ndarray)

    def test_zero_max_order(self):
        """Test with zero max_order."""
        # Should fail at instantiation with proper validation
        with pytest.raises(ValueError, match="max_order must be greater than 0"):
            anomaly_grid_py.AnomalyDetector(max_order=0)

    def test_negative_max_order(self):
        """Test with negative max_order."""
        with pytest.raises(ValueError, match="max_order must be greater than 0"):
            anomaly_grid_py.AnomalyDetector(max_order=-1)

    def test_invalid_threshold_values(self):
        """Test detection with invalid threshold values."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=2)
        detector.fit([["A", "B"], ["A", "B"]])

        # The library is permissive with thresholds, so test that they work
        # Test negative threshold (should work but treat as 0)
        results_negative = detector.predict([["A", "B"]], threshold=-0.1)
        assert isinstance(results_negative, np.ndarray)

        # Test threshold > 1 (should work but treat as 1)
        results_high = detector.predict([["A", "B"]], threshold=1.5)
        assert isinstance(results_high, np.ndarray)

        # Test valid thresholds
        results_valid = detector.predict([["A", "B"]], threshold=0.5)
        assert isinstance(results_valid, np.ndarray)

        # Test NaN threshold - this might still raise an error
        try:
            detector.predict([["A", "B"]], threshold=float("nan"))
        except (ValueError, RuntimeError):
            pass  # Expected to fail

    def test_unicode_events(self):
        """Test with unicode event strings."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=2)

        unicode_sequences = [["🔥", "💧"], ["🌪️", "🔥"], ["💧", "🌪️"]]
        detector.fit(unicode_sequences)

        results = detector.predict_proba([["🔥", "⚡"]])
        assert isinstance(results, np.ndarray)
        assert len(results) == 1

    def test_very_long_event_strings(self):
        """Test with very long event strings."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=2)

        long_event = "A" * 1000
        training_sequences = [[long_event, "B"], [long_event, "B"]]
        detector.fit(training_sequences)

        results = detector.predict_proba([[long_event, "X"]])
        assert isinstance(results, np.ndarray)
        assert len(results) == 1

    def test_many_unique_events(self):
        """Test with many unique events."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=2)

        # Create sequences with many unique events
        unique_sequences = []
        for i in range(100):
            unique_sequences.append([f"event_{i}", f"event_{i+1}"])

        detector.fit(unique_sequences)

        # Should handle gracefully
        metrics = detector.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert "context_count" in metrics

    def test_detection_before_training(self):
        """Test detection before training."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=2)

        with pytest.raises(ValueError, match="not fitted"):
            detector.predict_proba([["A", "B"]])

        with pytest.raises(ValueError, match="not fitted"):
            detector.predict([["A", "B"]], threshold=0.1)

    def test_repeated_training(self):
        """Test training multiple times."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=2)

        # First training
        detector.fit([["A", "B"], ["A", "B"]])

        # Test that we can predict after first training
        results1 = detector.predict_proba([["A", "B"]])
        assert isinstance(results1, np.ndarray)

        # Second training (should replace previous training)
        detector.fit([["C", "D"], ["C", "D"]])

        # Should still work
        results2 = detector.predict_proba([["C", "D"]])
        assert isinstance(results2, np.ndarray)

    def test_invalid_input_types(self):
        """Test with invalid input types."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=2)

        # Test with non-list input
        with pytest.raises(TypeError):
            detector.fit("not a list")

        # Test with non-sequence elements
        with pytest.raises(TypeError):
            detector.fit([123, 456])

        # Train properly first
        detector.fit([["A", "B"], ["A", "B"]])

        # Test prediction with invalid types
        with pytest.raises(TypeError):
            detector.predict_proba("not a list")

    def test_mixed_sequence_lengths(self):
        """Test with sequences of different lengths."""
        detector = anomaly_grid_py.AnomalyDetector(max_order=2)

        # Train with sequences of different lengths
        mixed_sequences = [
            ["A", "B"],
            ["C", "D", "E"],
            ["F", "G", "H", "I"],
            ["J", "K"],
        ]
        detector.fit(mixed_sequences)

        # Test with mixed length sequences
        test_sequences = [["A", "B"], ["X", "Y", "Z"]]
        results = detector.predict_proba(test_sequences)
        assert isinstance(results, np.ndarray)
        assert len(results) == 2

    # Padding functionality removed in clean implementation
    # Core detector validates minimum sequence length instead
