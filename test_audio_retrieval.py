"""Synthetic-WAV tests for the audio retrieval baseline."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import soundfile as sf

from audio_retrieval import AudioRetrievalEngine, extract_audio_features


def write_tone(path: Path, frequency: float, duration: float = 1.0) -> None:
    sample_rate = 22_050
    time = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    waveform = 0.5 * np.sin(2 * np.pi * frequency * time)
    sf.write(path, waveform, sample_rate)


def make_data(root: Path):
    first = root / "tone_a.wav"
    second = root / "tone_a_long.wav"
    third = root / "tone_b.wav"
    write_tone(first, 440.0, 1.0)
    write_tone(second, 440.0, 1.7)
    write_tone(third, 880.0, 1.1)
    records = pd.DataFrame(
        [
            {"audio_id": "R1", "composition_id": "A1", "audio_path": str(first), "recording_title": "First"},
            {"audio_id": "R2", "composition_id": "A1", "audio_path": str(second), "recording_title": "Second take"},
            {"audio_id": "R3", "composition_id": "A2", "audio_path": str(third), "recording_title": "Other"},
        ]
    )
    compositions = pd.DataFrame(
        [
            {"composition_id": "A1", "canonical_title": "A tone"},
            {"composition_id": "A2", "canonical_title": "B tone"},
        ]
    )
    return records, compositions, first, second


class AudioRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.records, self.compositions, self.query_path, self.second_path = make_data(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_features_are_fixed_size_finite_and_duration_independent(self):
        short = extract_audio_features(self.query_path)
        long = extract_audio_features(self.second_path)
        self.assertEqual(short.shape, (64,))
        self.assertEqual(long.shape, (64,))
        self.assertTrue(np.isfinite(short).all())
        self.assertTrue(np.isfinite(long).all())
        self.assertAlmostEqual(float(np.linalg.norm(short)), 1.0, places=5)

    def test_index_stats_retrieval_mapping_and_composition_aggregation(self):
        engine = AudioRetrievalEngine(self.records, self.compositions)
        stats = engine.build_index()
        self.assertEqual((stats["encountered"], stats["indexed"], stats["skipped"], stats["failed"]), (3, 3, 0, 0))
        results = engine.search_audio(self.query_path, top_k=10)
        self.assertEqual(results[0]["composition_id"], "A1")
        self.assertEqual(results[0]["best_matching_audio_id"], "R1")
        self.assertEqual(results[0]["score_type"], "cosine_similarity")
        self.assertAlmostEqual(results[0]["score"], 1.0, places=5)
        self.assertEqual(len({result["composition_id"] for result in results}), len(results))
        self.assertEqual(len(results[0]["recording_evidence"]), 2)

    def test_missing_empty_and_corrupt_recordings_are_reported(self):
        empty = self.root / "empty.wav"
        empty.write_bytes(b"")
        corrupt = self.root / "broken.mp3"
        corrupt.write_bytes(b"not an audio stream")
        records = pd.concat(
            [
                self.records,
                pd.DataFrame(
                    [
                        {"audio_id": "R4", "composition_id": "A2", "audio_path": str(self.root / 'missing.wav')},
                        {"audio_id": "R5", "composition_id": "A2", "audio_path": str(empty)},
                        {"audio_id": "R6", "composition_id": "A2", "audio_path": str(corrupt)},
                    ]
                ),
            ],
            ignore_index=True,
        )
        stats = AudioRetrievalEngine(records, self.compositions).build_index()
        self.assertEqual(stats["encountered"], 6)
        self.assertEqual(stats["indexed"], 3)
        self.assertEqual(stats["skipped"], 1)
        self.assertEqual(stats["failed"], 2)
        self.assertIn("R4", stats["skipped_audio_ids"])
        self.assertIn("R5", stats["failures"])
        self.assertIn("R6", stats["failures"])

    def test_empty_index_bad_query_top_k_and_feature_failure(self):
        empty_records = self.records.iloc[0:0]
        engine = AudioRetrievalEngine(empty_records, self.compositions)
        self.assertEqual(engine.search_audio(self.query_path), [])
        with self.assertRaises(ValueError):
            engine.search_audio(self.query_path, top_k=0)
        with self.assertRaises(ValueError):
            engine.search_audio("   ")
        with self.assertRaises(FileNotFoundError):
            engine.search_audio(self.root / "not_here.wav")

        with patch("audio_retrieval.extract_audio_features", side_effect=ValueError("test extraction error")):
            failed_engine = AudioRetrievalEngine(self.records, self.compositions)
            stats = failed_engine.build_index()
        self.assertEqual(stats["indexed"], 0)
        self.assertEqual(stats["failed"], 3)
        self.assertEqual(failed_engine.search_audio(self.query_path), [])

    def test_input_schema_and_unknown_parent_validation(self):
        with self.assertRaisesRegex(ValueError, "missing required"):
            AudioRetrievalEngine(pd.DataFrame(), self.compositions)
        bad_compositions = pd.DataFrame([{"composition_id": "OTHER", "canonical_title": "Other"}])
        engine = AudioRetrievalEngine(self.records, bad_compositions)
        stats = engine.build_index()
        self.assertEqual(stats["failed"], 3)


if __name__ == "__main__":
    unittest.main()
