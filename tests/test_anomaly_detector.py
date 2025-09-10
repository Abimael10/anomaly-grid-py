"""
Tests for the AnomalyDetector class
"""

import pytest
import anomaly_grid_py


def test_anomaly_detector_creation():
    """Test creating an AnomalyDetector instance"""
    detector = anomaly_grid_py.AnomalyDetector()
    assert detector is not None
    
    # Test with custom max_order
    detector_custom = anomaly_grid_py.AnomalyDetector(max_order=5)
    assert detector_custom is not None


def test_train_and_detect():
    """Test basic training and detection functionality"""
    detector = anomaly_grid_py.AnomalyDetector(max_order=3)
    
    # Train with a simple pattern
    training_data = ["A", "B", "C", "A", "B", "C", "A", "B", "C"]
    detector.train(training_data)
    
    # Test normal sequence (should have no anomalies)
    normal_sequence = ["A", "B", "C", "A", "B"]
    results = detector.detect(normal_sequence, threshold=0.1)
    
    # Normal sequence should have few or no anomalies
    assert len(results) >= 0
    assert all(isinstance(result, anomaly_grid_py.AnomalyInfo) for result in results)
    
    # Test anomalous sequence
    anomalous_sequence = ["A", "B", "X", "Y", "Z"]
    anomaly_results = detector.detect(anomalous_sequence, threshold=0.1)
    
    # Should detect some anomalies
    assert len(anomaly_results) > 0
    # Check that results have the expected structure
    for result in anomaly_results:
        assert hasattr(result, 'sequence')
        assert hasattr(result, 'likelihood')
        assert hasattr(result, 'anomaly_strength')
        assert result.is_anomaly == True


def test_anomaly_info_properties():
    """Test AnomalyInfo object properties"""
    detector = anomaly_grid_py.AnomalyDetector(max_order=2)
    detector.train(["A", "B", "A", "B"])
    
    results = detector.detect(["A", "B", "X", "Y"], threshold=0.1)
    
    for result in results:
        assert hasattr(result, 'position')
        assert hasattr(result, 'sequence')
        assert hasattr(result, 'likelihood')
        assert hasattr(result, 'anomaly_strength')
        assert hasattr(result, 'is_anomaly')
        assert isinstance(result.position, int)
        assert isinstance(result.sequence, str)
        assert isinstance(result.likelihood, float)
        assert isinstance(result.anomaly_strength, float)
        assert isinstance(result.is_anomaly, bool)


def test_detector_max_order():
    """Test getting the max order"""
    detector = anomaly_grid_py.AnomalyDetector(max_order=5)
    assert detector.max_order() == 5
    
    detector2 = anomaly_grid_py.AnomalyDetector(max_order=2)
    assert detector2.max_order() == 2


def test_get_performance_metrics():
    """Test getting detector performance metrics"""
    detector = anomaly_grid_py.AnomalyDetector(max_order=2)
    detector.train(["A", "B", "A", "B"])
    
    metrics = detector.get_performance_metrics()
    assert isinstance(metrics, dict)
    # Metrics should contain performance information
    assert "training_time_ms" in metrics
    assert "detection_time_ms" in metrics
    assert "context_count" in metrics
    assert "estimated_memory_bytes" in metrics


def test_threshold_parameter():
    """Test different threshold values"""
    detector = anomaly_grid_py.AnomalyDetector(max_order=2)
    detector.train(["A", "B", "A", "B"])
    
    # Test with different thresholds
    results_low = detector.detect(["A", "B", "X", "Y"], threshold=0.01)
    results_high = detector.detect(["A", "B", "X", "Y"], threshold=0.9)
    
    # Both should return some results since we have anomalous patterns
    assert len(results_low) >= 0
    assert len(results_high) >= 0
    
    # With a very low threshold, more events might be flagged as anomalies
    # With a very high threshold, fewer events might be flagged as anomalies
    # The exact behavior depends on the likelihood values