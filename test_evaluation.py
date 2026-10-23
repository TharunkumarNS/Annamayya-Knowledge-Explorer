"""Tests for evaluation metric conventions and source-derived case generation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_retrieval import prepare_cases, ranking_metrics


class EvaluationTests(unittest.TestCase):
    def test_metrics_for_single_and_multiple_relevant_results(self) -> None:
        metrics = ranking_metrics(["A", "C"], ["B", "C", "A"])
        self.assertEqual(metrics["top1"], 0)
        self.assertEqual(metrics["reciprocal_rank"], 0.5)
        self.assertEqual(metrics["recall_at_3"], 1.0)
        self.assertEqual(metrics["recall_at_5"], 1.0)

    def test_missing_relevant_result_has_zero_metrics(self) -> None:
        metrics = ranking_metrics(["A"], [])
        self.assertEqual(metrics["top1"], 0)
        self.assertEqual(metrics["reciprocal_rank"], 0.0)
        self.assertEqual(metrics["recall_at_3"], 0.0)

    def test_prepared_cases_are_real_in_sample_audio_text_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.csv"
            count = prepare_cases(path)
            contents = path.read_text(encoding="utf-8-sig")
        self.assertEqual(count, 13)
        self.assertIn("dataset_association_only", contents)
        self.assertIn("same_recording_in_index", contents)
        self.assertIn("is_tuning_case", contents)
        self.assertNotIn("ANN-025", contents)


if __name__ == "__main__":
    unittest.main()
