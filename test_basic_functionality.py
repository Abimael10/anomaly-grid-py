#!/usr/bin/env python3
"""
Basic functionality test for anomaly-grid-py.
This script tests the core functionality with the new API.
"""

import sys
import traceback

def test_basic_functionality():
    """Test basic functionality with new API."""
    try:
        print("[TEST] Testing anomaly-grid-py basic functionality...")
        
        # Test import
        print("1. Testing import...")
        import anomaly_grid_py
        print("   [OK] Package imported successfully")
        
        # Test detector creation
        print("2. Testing detector creation...")
        detector = anomaly_grid_py.AnomalyDetector(max_order=1)
        print("   [OK] Detector created successfully")
        
        # Test training
        print("3. Testing training...")
        training_data = [
            ['A', 'B', 'C'],
            ['A', 'B', 'C'],
            ['A', 'B', 'C']
        ] * 10
        detector.fit(training_data)
        print("   [OK] Training completed successfully")
        
        # Test prediction
        print("4. Testing prediction...")
        test_data = [
            ['A', 'B', 'C'],  # Normal
            ['X', 'Y', 'Z']   # Anomalous
        ]
        scores = detector.predict_proba(test_data)
        print(f"   [OK] Prediction completed: scores = {scores}")
        
        # Test binary prediction
        print("5. Testing binary prediction...")
        predictions = detector.predict(test_data, threshold=0.1)
        print(f"   [OK] Binary prediction completed: predictions = {predictions}")
        
        # Test metrics
        print("6. Testing metrics...")
        metrics = detector.get_performance_metrics()
        print(f"   [OK] Metrics retrieved: {metrics}")
        
        print("\n[SUCCESS] All tests passed! The package is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)