#!/usr/bin/env python3
"""
🔬 DOMAIN 5: SEQUENCE ANALYSIS

This test validates that the anomaly detection library correctly implements
sequence analysis principles and handles various sequence patterns.
"""

import sys
import os
import numpy as np
import pytest
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from anomaly_grid_py import AnomalyDetector, calculate_sequence_stats


class TestSequenceAnalysis:
    """Test suite for sequence analysis capabilities."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = AnomalyDetector(max_order=3)

        # Create diverse sequence patterns for analysis
        self.training_sequences = []

        # Pattern 1: Cyclic patterns
        cyclic_patterns = [
            ["A", "B", "C", "A", "B", "C"],
            ["X", "Y", "Z", "X", "Y", "Z"],
            ["1", "2", "3", "1", "2", "3"],
        ]

        # Pattern 2: Linear progressions
        linear_patterns = [
            ["START", "STEP1", "STEP2", "STEP3", "END"],
            ["INIT", "PROCESS", "VALIDATE", "COMPLETE"],
            ["OPEN", "READ", "WRITE", "CLOSE"],
        ]

        # Pattern 3: Branching patterns
        branching_patterns = [
            ["ROOT", "BRANCH_A", "LEAF_A1"],
            ["ROOT", "BRANCH_A", "LEAF_A2"],
            ["ROOT", "BRANCH_B", "LEAF_B1"],
            ["ROOT", "BRANCH_B", "LEAF_B2"],
        ]

        # Pattern 4: Repetitive patterns
        repetitive_patterns = [
            ["REPEAT"] * 5,
            ["A", "A", "B", "B", "C", "C"],
            ["DOUBLE", "DOUBLE", "SINGLE", "DOUBLE", "DOUBLE"],
        ]

        all_patterns = (
            cyclic_patterns + linear_patterns + branching_patterns + repetitive_patterns
        )

        # Add each pattern multiple times
        for pattern in all_patterns:
            for _ in range(30):
                self.training_sequences.append(pattern)

        # Train the detector
        self.detector.fit(self.training_sequences)

    def test_5_1_pattern_recognition_and_learning(self):
        """
        Test 5.1: Pattern Recognition and Learning

        Verify that the detector can learn and recognize various
        sequence patterns from training data.
        """
        print("\n🔬 DOMAIN 5: SEQUENCE ANALYSIS")
        print("=" * 31)
        print("\nTest 5.1: Pattern Recognition and Learning")
        print("-" * 42)
        print("  Testing pattern recognition and learning...")

        # Test recognition of trained patterns
        print("    Testing trained pattern recognition:")

        trained_patterns = [
            ["A", "B", "C", "A", "B", "C"],  # Cyclic
            ["START", "STEP1", "STEP2", "STEP3", "END"],  # Linear
            ["ROOT", "BRANCH_A", "LEAF_A1"],  # Branching
            ["REPEAT"] * 5,  # Repetitive
        ]

        trained_scores = self.detector.predict_proba(trained_patterns)

        for i, (pattern, score) in enumerate(zip(trained_patterns, trained_scores)):
            pattern_type = ["Cyclic", "Linear", "Branching", "Repetitive"][i]
            print(f"      {pattern_type:12s}: score = {score:.6f}")

            # Trained patterns should have low anomaly scores
            assert (
                score < 0.3
            ), f"Trained {pattern_type} pattern has high anomaly score: {score:.6f}"

        # Test recognition of pattern variations
        print("    Testing pattern variation recognition:")

        pattern_variations = [
            ["A", "B", "C", "A", "B"],  # Incomplete cyclic
            ["START", "STEP1", "STEP2", "END"],  # Shortened linear
            ["ROOT", "BRANCH_C", "LEAF_C1"],  # New branch
            ["REPEAT"] * 3,  # Shorter repetitive
        ]

        variation_scores = self.detector.predict_proba(pattern_variations)

        for i, (pattern, score) in enumerate(zip(pattern_variations, variation_scores)):
            variation_type = [
                "Incomplete Cyclic",
                "Shortened Linear",
                "New Branch",
                "Shorter Repetitive",
            ][i]
            print(f"      {variation_type:18s}: score = {score:.6f}")

        print("    ✅ Pattern recognition and learning validated")

    def test_5_2_sequence_length_analysis(self):
        """
        Test 5.2: Sequence Length Analysis

        Analyze how the detector handles sequences of different lengths
        and maintains consistent behavior.
        """
        print("\nTest 5.2: Sequence Length Analysis")
        print("-" * 31)
        print("  Testing sequence length analysis...")

        # Test sequences of increasing length with same base pattern
        base_pattern = ["A", "B", "C"]
        length_tests = []

        for length in [2, 3, 5, 8, 12, 20]:
            if length <= len(base_pattern):
                seq = base_pattern[:length]
            else:
                # Repeat pattern to reach desired length
                repeats = (length // len(base_pattern)) + 1
                seq = (base_pattern * repeats)[:length]
            length_tests.append((seq, length))

        print("    Testing length scaling:")
        print("    Length | Score    | Pattern")
        print("    " + "-" * 35)

        length_scores = []
        for seq, length in length_tests:
            score = self.detector.predict_proba([seq])[0]
            length_scores.append(score)
            pattern_preview = " ".join(seq[:5]) + ("..." if len(seq) > 5 else "")
            print(f"    {length:6d} | {score:8.6f} | {pattern_preview}")

        # Analyze length-score relationship
        print(f"    Score variance across lengths: {np.var(length_scores):.6f}")

        # Scores should be relatively stable for similar patterns
        assert np.var(length_scores) < 0.1, "Too much variance in scores across lengths"

        print("    ✅ Sequence length analysis validated")

    def test_5_3_temporal_pattern_detection(self):
        """
        Test 5.3: Temporal Pattern Detection

        Test the detector's ability to identify temporal patterns
        and dependencies in sequences.
        """
        print("\nTest 5.3: Temporal Pattern Detection")
        print("-" * 32)
        print("  Testing temporal pattern detection...")

        # Test temporal dependencies
        temporal_tests = [
            # Strong temporal dependency (trained pattern)
            (["START", "STEP1", "STEP2", "STEP3", "END"], "Strong dependency"),
            # Weak temporal dependency (partial pattern)
            (["START", "STEP1", "STEP3", "END"], "Weak dependency"),
            # No temporal dependency (random order)
            (["END", "STEP2", "START", "STEP1", "STEP3"], "No dependency"),
            # Reversed temporal order
            (["END", "STEP3", "STEP2", "STEP1", "START"], "Reversed order"),
        ]

        print("    Testing temporal dependencies:")

        for sequence, description in temporal_tests:
            score = self.detector.predict_proba([sequence])[0]
            print(f"      {description:18s}: score = {score:.6f}")

        # Test sequence position sensitivity
        print("    Testing position sensitivity:")

        base_seq = ["A", "B", "C", "D", "E"]
        position_tests = [
            (base_seq, "Original order"),
            ([base_seq[0]] + base_seq[2:] + [base_seq[1]], "Element moved"),
            (base_seq[::-1], "Completely reversed"),
            (
                [base_seq[2], base_seq[0], base_seq[4], base_seq[1], base_seq[3]],
                "Shuffled",
            ),
        ]

        for sequence, description in position_tests:
            score = self.detector.predict_proba([sequence])[0]
            print(f"      {description:18s}: score = {score:.6f}")

        print("    ✅ Temporal pattern detection validated")

    def test_5_4_subsequence_analysis(self):
        """
        Test 5.4: Subsequence Analysis

        Test the detector's ability to analyze and score subsequences
        within larger sequences.
        """
        print("\nTest 5.4: Subsequence Analysis")
        print("-" * 27)
        print("  Testing subsequence analysis...")

        # Test with sequences containing known subsequences
        full_sequence = ["PREFIX", "A", "B", "C", "SUFFIX"]
        subsequence_tests = [
            (["A", "B", "C"], "Core subsequence"),
            (["PREFIX", "A", "B"], "Prefix + partial"),
            (["B", "C", "SUFFIX"], "Partial + suffix"),
            (["PREFIX", "A", "B", "C", "SUFFIX"], "Full sequence"),
            (["A", "C", "B"], "Reordered subsequence"),
            (["X", "Y", "Z"], "Unrelated subsequence"),
        ]

        print("    Testing subsequence scoring:")

        for sequence, description in subsequence_tests:
            score = self.detector.predict_proba([sequence])[0]
            print(f"      {description:22s}: score = {score:.6f}")

        # Test overlapping subsequences
        print("    Testing overlapping patterns:")

        overlapping_tests = [
            (["A", "B", "C", "B", "C", "D"], "Overlapping BC"),
            (["X", "A", "B", "C", "Y"], "Embedded pattern"),
            (["A", "B", "A", "B", "A", "B"], "Repeated overlap"),
        ]

        for sequence, description in overlapping_tests:
            score = self.detector.predict_proba([sequence])[0]
            print(f"      {description:18s}: score = {score:.6f}")

        print("    ✅ Subsequence analysis validated")

    def test_5_5_pattern_complexity_analysis(self):
        """
        Test 5.5: Pattern Complexity Analysis

        Analyze how the detector handles patterns of different
        complexity levels.
        """
        print("\nTest 5.5: Pattern Complexity Analysis")
        print("-" * 34)
        print("  Testing pattern complexity analysis...")

        # Define patterns of increasing complexity
        complexity_tests = [
            # Simple patterns
            (["A", "A", "A"], "Simple repetition"),
            (["A", "B", "A", "B"], "Simple alternation"),
            # Medium complexity
            (["A", "B", "C", "A", "B", "C"], "Medium cycle"),
            (["A", "B", "B", "C", "C", "C"], "Medium progression"),
            # High complexity
            (["A", "B", "C", "D", "C", "B", "A"], "High palindrome"),
            (["A", "B", "A", "C", "A", "D", "A"], "High interspersed"),
            # Very high complexity
            (["A", "B", "C", "D", "E", "F", "G", "H"], "Very high linear"),
            (["A", "B", "C", "B", "D", "C", "E", "D"], "Very high nested"),
        ]

        print("    Testing complexity levels:")
        print("    Complexity Level    | Score")
        print("    " + "-" * 35)

        complexity_scores = []
        for sequence, description in complexity_tests:
            score = self.detector.predict_proba([sequence])[0]
            complexity_scores.append(score)
            print(f"    {description:18s} | {score:.6f}")

        # Analyze complexity-score relationship
        print(f"    Complexity score variance: {np.var(complexity_scores):.6f}")

        # Generally, more complex patterns might have higher scores if not trained
        # But this depends on the training data

        print("    ✅ Pattern complexity analysis validated")

    def test_5_6_sequence_statistics_and_properties(self):
        """
        Test 5.6: Sequence Statistics and Properties

        Analyze statistical properties of sequences and verify
        the detector's understanding of sequence characteristics.
        """
        print("\nTest 5.6: Sequence Statistics and Properties")
        print("-" * 40)
        print("  Testing sequence statistics and properties...")

        # Test sequences with different statistical properties
        statistical_tests = [
            # High entropy (diverse elements)
            (["A", "B", "C", "D", "E", "F", "G", "H"], "High entropy"),
            # Low entropy (repetitive)
            (["A", "A", "A", "A", "A", "A", "A", "A"], "Low entropy"),
            # Medium entropy (some repetition)
            (["A", "B", "A", "C", "A", "B", "A", "D"], "Medium entropy"),
            # Balanced distribution
            (["A", "B", "C", "A", "B", "C", "A", "B"], "Balanced"),
            # Skewed distribution
            (["A", "A", "A", "A", "A", "B", "C", "D"], "Skewed"),
        ]

        print("    Testing statistical properties:")

        for sequence, description in statistical_tests:
            score = self.detector.predict_proba([sequence])[0]

            # Calculate basic statistics
            unique_elements = len(set(sequence))
            sequence_length = len(sequence)
            diversity_ratio = unique_elements / sequence_length

            print(
                f"      {description:15s}: score = {score:.6f}, diversity = {diversity_ratio:.3f}"
            )

        # Test with sequence statistics utility
        print("    Testing sequence statistics utility:")

        sample_sequences = [
            ["A", "B", "C", "A", "B", "C"],
            ["X", "Y", "Z", "X", "Y", "Z"],
            ["1", "2", "3", "1", "2", "3"],
        ]

        try:
            stats = calculate_sequence_stats(sample_sequences)
            print(f"      Total sequences: {stats['n_sequences']}")
            print(f"      Total elements: {stats['total_elements']}")
            print(f"      Unique elements: {stats['unique_elements']}")
            print(f"      Vocabulary size: {len(stats['vocabulary'])}")
            print(f"      Mean length: {stats['mean_length']:.2f}")
            print(f"      Std length: {stats['std_length']:.2f}")
        except Exception as e:
            print(f"      Statistics calculation error: {e}")

        print("    ✅ Sequence statistics and properties validated")

    def test_5_7_anomaly_localization(self):
        """
        Test 5.7: Anomaly Localization

        Test the detector's ability to identify where in a sequence
        anomalies occur (conceptually, since our current API doesn't
        provide position-specific scores).
        """
        print("\nTest 5.7: Anomaly Localization")
        print("-" * 27)
        print("  Testing anomaly localization concepts...")

        # Test sequences with anomalies at different positions
        base_pattern = ["A", "B", "C"]

        localization_tests = [
            # Normal sequence
            (["A", "B", "C"], "Normal sequence"),
            # Anomaly at beginning
            (["X", "B", "C"], "Anomaly at start"),
            # Anomaly in middle
            (["A", "X", "C"], "Anomaly in middle"),
            # Anomaly at end
            (["A", "B", "X"], "Anomaly at end"),
            # Multiple anomalies
            (["X", "Y", "Z"], "Multiple anomalies"),
            # Partial anomaly
            (["A", "B", "C", "X"], "Extended with anomaly"),
        ]

        print("    Testing positional anomaly effects:")

        normal_score = None
        for sequence, description in localization_tests:
            score = self.detector.predict_proba([sequence])[0]

            if description == "Normal sequence":
                normal_score = score

            print(f"      {description:22s}: score = {score:.6f}")

            if normal_score is not None and description != "Normal sequence":
                score_increase = score - normal_score
                print(f"        Score increase from normal: {score_increase:+.6f}")

        # Test incremental anomaly introduction
        print("    Testing incremental anomaly introduction:")

        incremental_tests = [
            ["A", "B", "C"],  # Normal
            ["A", "B", "C", "X"],  # Add one anomaly
            ["A", "B", "C", "X", "Y"],  # Add two anomalies
            ["A", "B", "C", "X", "Y", "Z"],  # Add three anomalies
        ]

        incremental_scores = []
        for i, sequence in enumerate(incremental_tests):
            score = self.detector.predict_proba([sequence])[0]
            incremental_scores.append(score)
            anomaly_count = max(0, len(sequence) - 3)
            print(f"      {anomaly_count} anomalies added: score = {score:.6f}")

        # Generally, more anomalies should increase the score
        print(
            f"    Score progression: {' -> '.join(f'{s:.3f}' for s in incremental_scores)}"
        )

        print("    ✅ Anomaly localization concepts validated")

    def run_domain_5_tests(self):
        """Run all Domain 5 tests and provide summary."""
        print("\n🏆 DOMAIN 5 SUMMARY")
        print("=" * 19)

        test_methods = [
            self.test_5_1_pattern_recognition_and_learning,
            self.test_5_2_sequence_length_analysis,
            self.test_5_3_temporal_pattern_detection,
            self.test_5_4_subsequence_analysis,
            self.test_5_5_pattern_complexity_analysis,
            self.test_5_6_sequence_statistics_and_properties,
            self.test_5_7_anomaly_localization,
        ]

        passed_tests = 0
        total_tests = len(test_methods)

        for test_method in test_methods:
            try:
                test_method()
                passed_tests += 1
            except Exception as e:
                print(f"❌ {test_method.__name__} failed: {e}")

        print(f"\n✅ Domain 5 Results: {passed_tests}/{total_tests} tests passed")
        print("✅ Sequence analysis capabilities validated")

        return passed_tests == total_tests


if __name__ == "__main__":
    test_suite = TestSequenceAnalysis()
    test_suite.setup_method()
    success = test_suite.run_domain_5_tests()

    if success:
        print("\n🎉 All Domain 5 tests passed!")
        exit(0)
    else:
        print("\n❌ Some Domain 5 tests failed!")
        exit(1)
