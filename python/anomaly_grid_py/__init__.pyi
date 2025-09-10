"""
Type stubs for anomaly_grid_py package
"""

from typing import Dict, List, Optional

__version__: str

class AnomalyInfo:
    """Information about an anomaly detection result."""
    
    position: int
    sequence: str
    likelihood: float
    anomaly_strength: float
    is_anomaly: bool
    
    def __repr__(self) -> str: ...

class AnomalyDetector:
    """Python wrapper for the Rust AnomalyDetector."""
    
    def __init__(self, max_order: Optional[int] = None) -> None:
        """Create a new AnomalyDetector with specified maximum order.
        
        Args:
            max_order: Maximum order for the Markov model (default: 3)
        """
        ...
    
    def train(self, events: List[str]) -> None:
        """Train the detector with a sequence of events.
        
        Args:
            events: List of event strings to train on
            
        Raises:
            RuntimeError: If training fails
        """
        ...
    
    def detect(self, events: List[str], threshold: Optional[float] = None) -> List[AnomalyInfo]:
        """Detect anomalies in a sequence with given threshold.
        
        Args:
            events: List of event strings to analyze
            threshold: Anomaly detection threshold (default: 0.1)
            
        Returns:
            List of AnomalyInfo objects for detected anomalies
            
        Raises:
            RuntimeError: If detection fails
        """
        ...
    
    def get_performance_metrics(self) -> Dict[str, int]:
        """Get performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        ...
    
    def max_order(self) -> int:
        """Get the maximum order of the detector.
        
        Returns:
            Maximum order value
        """
        ...

__all__ = ["AnomalyDetector", "AnomalyInfo"]