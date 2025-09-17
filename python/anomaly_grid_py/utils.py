"""Utilities."""

from typing import Any, Dict, List, Tuple

import numpy as np


def train_test_split(
    sequences: List[List[str]],
    test_size: float = 0.2,
    random_state: int = None,  # type: ignore
) -> Tuple[List[List[str]], List[List[str]]]:
    """Train/test split."""
    if random_state is not None:
        np.random.seed(random_state)

    n_sequences = len(sequences)
    n_test = int(n_sequences * test_size)

    # Shuffle indices
    indices = np.random.permutation(n_sequences)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    train_sequences = [sequences[i] for i in train_indices]
    test_sequences = [sequences[i] for i in test_indices]

    return train_sequences, test_sequences


def cross_val_score(
    estimator, X: List[List[str]], y: List[int], cv: int = 3
) -> np.ndarray:
    """Cross-validation."""
    n_samples = len(X)
    fold_size = n_samples // cv
    scores = []

    for i in range(cv):
        # Create train/test split for this fold
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < cv - 1 else n_samples

        test_indices = list(range(start_idx, end_idx))
        train_indices = [j for j in range(n_samples) if j not in test_indices]

        X_train = [X[j] for j in train_indices]
        X_test = [X[j] for j in test_indices]
        y_test = [y[j] for j in test_indices]

        # Fit and predict
        estimator_copy = type(estimator)(**estimator.get_params())
        estimator_copy.fit(X_train)
        predictions = estimator_copy.predict(X_test)

        # Calculate accuracy
        accuracy = np.mean(predictions == np.array(y_test))
        scores.append(accuracy)

    return np.array(scores)


def generate_sequences(
    n_sequences: int, seq_length: int, alphabet: List[str], anomaly_rate: float = 0.1
) -> Tuple[List[List[str]], List[int]]:
    """Generate synthetic sequences."""
    sequences = []
    labels = []

    n_anomalies = int(n_sequences * anomaly_rate)
    n_normal = n_sequences - n_anomalies

    # Generate normal sequences (pattern: A->B->C->A->B->C...)
    normal_pattern = alphabet[:3] if len(alphabet) >= 3 else alphabet
    for _ in range(n_normal):
        sequence = []
        for i in range(seq_length):
            sequence.append(normal_pattern[i % len(normal_pattern)])
        sequences.append(sequence)
        labels.append(0)

    # Generate anomalous sequences (random)
    for _ in range(n_anomalies):
        sequence = [np.random.choice(alphabet) for _ in range(seq_length)]
        sequences.append(sequence)
        labels.append(1)

    # Shuffle
    indices = np.random.permutation(n_sequences)
    sequences = [sequences[i] for i in indices]
    labels = [labels[i] for i in indices]

    return sequences, labels


def precision_recall_curve(
    y_true: np.ndarray, y_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate precision-recall curve."""
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    # Sort by scores (descending)
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    y_scores_sorted = y_scores[desc_score_indices]

    # Calculate precision and recall
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)

    precision = tp / (tp + fp)
    recall = tp / np.sum(y_true)

    # Handle division by zero
    precision = np.nan_to_num(precision)

    return precision, recall, y_scores_sorted


class PerformanceTimer:
    """Performance timer."""

    def __init__(self):
        self.times = {}
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        import time

        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        import time

        self.elapsed = time.perf_counter() - self.start_time

    def time_operation(self, name: str, func, *args, **kwargs):
        """Time function."""
        import time

        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        self.times[name] = elapsed
        return result

    def get_times(self) -> dict:
        """Get times."""
        return self.times.copy()

    def reset(self):
        """Reset."""
        self.times.clear()
        self.start_time = None
        self.elapsed = None


def memory_usage() -> float:
    """Get memory usage in MB."""
    try:
        import os

        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        # Fallback if psutil not available
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def validate_sequences(sequences: List[List[str]], min_length: int = 2) -> None:
    """Validate sequences."""
    if not sequences:
        raise ValueError("Empty sequence list")

    for i, sequence in enumerate(sequences):
        if not isinstance(sequence, (list, tuple)):
            raise TypeError(f"Sequence {i} must be a list or tuple")

        if len(sequence) < min_length:
            raise ValueError(
                f"Sequence {i} has length {len(sequence)}, minimum is {min_length}. "
                f"Sequences need at least {min_length} elements for pattern analysis."
            )

        for j, element in enumerate(sequence):
            if not isinstance(element, str):
                raise TypeError(
                    f"Element at sequence {i}, position {j} must be a string"
                )

            if not element or not element.strip():
                raise ValueError(
                    f"Empty or whitespace-only element at sequence {i}, position {j}"
                )


def calculate_sequence_stats(sequences: List[List[str]]) -> Dict[str, Any]:
    """Calculate sequence stats."""
    if not sequences:
        return {}

    lengths = [len(seq) for seq in sequences]
    all_elements = [elem for seq in sequences for elem in seq]
    unique_elements = set(all_elements)

    return {
        "n_sequences": len(sequences),
        "total_elements": len(all_elements),
        "unique_elements": len(unique_elements),
        "vocabulary": sorted(unique_elements),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
    }
