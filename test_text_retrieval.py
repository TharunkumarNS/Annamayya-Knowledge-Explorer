"""Unit tests for text_retrieval; these use a deterministic mock encoder."""

import unittest

import numpy as np
import pandas as pd

from text_retrieval import TextRetrievalEngine, preprocess_text


class MockEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, sentences, **_kwargs):
        self.calls += 1
        vectors = []
        for sentence in sentences:
            text = sentence.casefold()
            vectors.append(
                [
                    float("sacred" in text or "దైవ" in text),
                    float("mountain" in text or "కొండ" in text),
                    float("love" in text or "ప్రేమ" in text),
                ]
            )
        return np.asarray(vectors, dtype=np.float32) + 0.01


def sample_frames():
    compositions = pd.DataFrame(
        [
            {"composition_id": "A1", "canonical_title": "Sacred Hill"},
            {"composition_id": "A2", "canonical_title": "Love Song"},
        ]
    )
    text_records = pd.DataFrame(
        [
            {"text_id": "T1", "composition_id": "A1", "text_type": "lyrics", "language": "Telugu", "content": "దైవ కొండ"},
            {"text_id": "T2", "composition_id": "A1", "text_type": "translation", "language": "English", "content": "sacred mountain"},
            {"text_id": "T3", "composition_id": "A2", "text_type": "translation", "language": "English", "content": "love and devotion"},
            {"text_id": "T4", "composition_id": "A2", "text_type": "lyrics", "language": "Telugu", "content": "   "},
        ]
    )
    return text_records, compositions


class TextRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.text_records, self.compositions = sample_frames()
        self.encoder = MockEncoder()
        self.engine = TextRetrievalEngine(self.text_records, self.compositions, encoder=self.encoder)

    def test_preprocessing_and_indexing(self):
        self.assertEqual(preprocess_text("  తెలుగు\n పాట  "), "తెలుగు పాట")
        self.assertEqual(self.engine.indexed_text_count, 3)
        self.assertEqual(self.engine.indexed_composition_count, 2)
        self.assertEqual(self.encoder.calls, 1)

    def test_results_are_composition_ranked_with_evidence(self):
        results = self.engine.search_text("sacred mountain", top_k=2)
        self.assertEqual([result["composition_id"] for result in results], ["A1", "A2"])
        self.assertEqual(results[0]["best_match"]["text_id"], "T1")
        self.assertTrue(np.isfinite(results[0]["score"]))
        self.assertEqual(len({result["composition_id"] for result in results}), len(results))
        self.assertEqual(self.encoder.calls, 2)

    def test_filters_and_validation(self):
        results = self.engine.search_text("sacred", languages="Telugu", text_types="lyrics")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["best_match"]["text_id"], "T1")
        self.assertEqual(self.engine.search_text("sacred", languages="Romanized Telugu"), [])
        with self.assertRaises(ValueError):
            self.engine.search_text("   ")
        with self.assertRaises(ValueError):
            self.engine.search_text("sacred", top_k=0)

    def test_required_columns_and_empty_dataset(self):
        with self.assertRaisesRegex(ValueError, "missing required"):
            TextRetrievalEngine(pd.DataFrame(), self.compositions, encoder=MockEncoder())
        unknown_parent = self.text_records.copy()
        unknown_parent.loc[0, "composition_id"] = "MISSING"
        with self.assertRaisesRegex(ValueError, "missing from Compositions"):
            TextRetrievalEngine(unknown_parent, self.compositions, encoder=MockEncoder())
        empty = self.text_records.iloc[0:0]
        engine = TextRetrievalEngine(empty, self.compositions, encoder=MockEncoder())
        self.assertEqual(engine.search_text("sacred"), [])


if __name__ == "__main__":
    unittest.main()
