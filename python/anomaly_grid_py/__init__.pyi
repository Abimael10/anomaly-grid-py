__version__: str

class AnomalyInfo:
    position: int
    sequence: str
    likelihood: float
    anomaly_strength: float
    is_anomaly: bool

class AnomalyDetector:
    def __init__(self, max_order: int | None = None) -> None: ...
    def train(self, events: list[str]) -> None: ...
    def detect(
        self, events: list[str], threshold: float | None = None,
    ) -> list[AnomalyInfo]: ...
    def get_performance_metrics(self) -> dict[str, int]: ...
    def max_order(self) -> int: ...

__all__ = ["AnomalyDetector", "AnomalyInfo"]
