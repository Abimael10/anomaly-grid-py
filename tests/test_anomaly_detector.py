"""Tests for the AnomalyDetector class."""

import anomaly_grid_py
import numpy as np


def test_anomaly_detector_creation():
    """Test creating an AnomalyDetector instance"""
    detector = anomaly_grid_py.AnomalyDetector()
    assert detector is not None

    # Test with custom max_order
    detector_custom = anomaly_grid_py.AnomalyDetector(max_order=5)
    assert detector_custom is not None


def test_fit_and_predict():
    """Test basic training and detection functionality"""
    detector = anomaly_grid_py.AnomalyDetector(max_order=3)

    # Train with sequences (new API expects list of sequences)
    training_data = [
        ["A", "B", "C"],
        ["A", "B", "C"],
        ["A", "B", "C"],
    ] * 3
    detector.fit(training_data)

    # Test normal sequence
    normal_sequences = [["A", "B", "C"], ["A", "B", "C"]]
    scores = detector.predict_proba(normal_sequences)

    # Should return numpy array of scores
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 2
    assert all(isinstance(score, (float, np.floating)) for score in scores)
    assert all(0 <= score <= 1 for score in scores)

    # Test binary predictions
    predictions = detector.predict(normal_sequences, threshold=0.1)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 2
    assert all(isinstance(pred, (bool, np.bool_)) for pred in predictions)

    # Test anomalous sequence
    anomalous_sequences = [["A", "B", "X"], ["X", "Y", "Z"]]
    anomaly_scores = detector.predict_proba(anomalous_sequences)

    # Anomalous sequences should generally have higher scores
    assert isinstance(anomaly_scores, np.ndarray)
    assert len(anomaly_scores) == 2


def test_predict_proba_output():
    """Test predict_proba returns correct format"""
    detector = anomaly_grid_py.AnomalyDetector(max_order=2)

    # Train with simple pattern
    training_data = [["A", "B"], ["A", "B"], ["A", "B"]] * 5
    detector.fit(training_data)

    # Test prediction
    test_sequences = [["A", "B"], ["X", "Y"]]
    scores = detector.predict_proba(test_sequences)

    # Check output format
    assert isinstance(scores, np.ndarray)
    assert scores.dtype == np.float64
    assert scores.shape == (2,)
    assert all(0 <= score <= 1 for score in scores)


def test_predict_output():
    """Test predict returns correct format"""
    detector = anomaly_grid_py.AnomalyDetector(max_order=2)

    # Train with simple pattern
    training_data = [["A", "B"], ["A", "B"], ["A", "B"]] * 5
    detector.fit(training_data)

    # Test prediction
    test_sequences = [["A", "B"], ["X", "Y"]]
    predictions = detector.predict(test_sequences, threshold=0.1)

    # Check output format
    assert isinstance(predictions, np.ndarray)
    assert predictions.dtype == np.bool_
    assert predictions.shape == (2,)


def test_get_performance_metrics():
    """Test getting detector performance metrics"""
    detector = anomaly_grid_py.AnomalyDetector(max_order=2)

    # Train with sequences
    training_data = [["A", "B"], ["A", "B"]] * 5
    detector.fit(training_data)

    metrics = detector.get_performance_metrics()
    assert isinstance(metrics, dict)

    # Check expected metric keys (updated for new API)
    assert "training_time_ms" in metrics
    assert "context_count" in metrics
    assert "memory_bytes" in metrics

    # Check metric types
    assert isinstance(metrics["training_time_ms"], (int, float))
    assert isinstance(metrics["context_count"], int)
    assert isinstance(metrics["memory_bytes"], int)


def test_threshold_parameter():
    """Test different threshold values"""
    detector = anomaly_grid_py.AnomalyDetector(max_order=2)

    # Train with pattern
    training_data = [["A", "B"], ["A", "B"]] * 5
    detector.fit(training_data)

    # Test with different thresholds
    test_sequences = [["A", "B"], ["X", "Y"]]

    results_low = detector.predict(test_sequences, threshold=0.01)
    results_high = detector.predict(test_sequences, threshold=0.9)

    # Both should return boolean arrays
    assert isinstance(results_low, np.ndarray)
    assert isinstance(results_high, np.ndarray)
    assert results_low.dtype == np.bool_
    assert results_high.dtype == np.bool_

    # Lower threshold should generally detect more anomalies
    assert len(results_low) == len(results_high) == 2


def test_scikit_learn_style_api():
    """Test scikit-learn style API compatibility"""
    detector = anomaly_grid_py.AnomalyDetector(max_order=2)

    # Test method chaining
    training_data = [["A", "B"], ["B", "C"], ["C", "A"]] * 3
    result = detector.fit(training_data)

    # fit should return self for method chaining
    assert result is detector

    # Test that we can call predict after fit
    test_data = [["A", "B"], ["X", "Y"]]
    scores = detector.predict_proba(test_data)
    predictions = detector.predict(test_data)

    assert isinstance(scores, np.ndarray)
    assert isinstance(predictions, np.ndarray)


def test_error_handling():
    """Test error handling for invalid inputs"""
    detector = anomaly_grid_py.AnomalyDetector(max_order=2)

    # Test prediction before fitting
    try:
        detector.predict_proba([["A", "B"]])
        assert False, "Should raise error when predicting before fitting"
    except ValueError as e:
        assert "not fitted" in str(e).lower()

    # Test empty training data
    try:
        detector.fit([])
        assert False, "Should raise error for empty training data"
    except ValueError:
        pass

    # Test sequences that are too short
    try:
        detector.fit([["A"]])  # Single element sequences not allowed
        assert False, "Should raise error for single-element sequences"
    except ValueError as e:
        assert "at least 2 elements" in str(e)


# Padding functionality removed in clean implementation
# Core detector handles sequences directly without padding
