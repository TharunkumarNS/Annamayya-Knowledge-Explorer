"""Deterministic tests for weighted reciprocal-rank fusion."""

import unittest
from pathlib import Path

from multimodal_retrieval import MultimodalRetrievalEngine, RetrievalBranchError


class FakeTextEngine:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    def search_text(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if self.error:
            raise self.error
        return self.results


class FakeAudioEngine:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    def search_audio(self, path, **kwargs):
        self.calls.append((path, kwargs))
        if self.error:
            raise self.error
        return self.results


def text_result(composition_id, score="missing"):
    result = {
        "composition_id": composition_id,
        "canonical_title": f"Title {composition_id}",
        "score_type": "cosine_similarity",
        "best_match": {"text_id": f"T-{composition_id}", "content": "text proof"},
        "matching_text_records": [],
    }
    if score != "missing":
        result["score"] = score
    return result


def audio_result(composition_id, score, audio_id):
    return {
        "composition_id": composition_id,
        "canonical_title": f"Title {composition_id}",
        "score": score,
        "score_type": "cosine_distance",
        "best_matching_audio_id": audio_id,
        "recording_evidence": [{"audio_id": audio_id, "score": score}],
    }


class MultimodalRetrievalTests(unittest.TestCase):
    def setUp(self):
        # The audio engine's scores here are lower-is-better distances. RRF uses
        # each branch's returned ordering, not the raw incomparable score values.
        self.text = FakeTextEngine(
            [text_result("C1", 0.91), text_result("C2", 0.72), text_result("C4")]
        )
        self.audio = FakeAudioEngine(
            [audio_result("C2", 0.04, "R2"), audio_result("C3", 0.8, "R3")]
        )
        self.engine = MultimodalRetrievalEngine(
            self.text, self.audio, text_weight=0.5, audio_weight=0.5, rrf_constant=60
        )

    def test_text_only_and_audio_only(self):
        text_results = self.engine.search(text_query="meaning", top_k=2)
        self.assertEqual([row["composition_id"] for row in text_results], ["C1", "C2"])
        self.assertEqual(text_results[0]["fused_score"], 1.0)
        self.assertIsNone(text_results[0]["audio_score"])
        self.assertEqual(text_results[0]["modalities"], ["text"])

        audio_results = self.engine.search(audio_query_path=Path("query.wav"), top_k=2)
        self.assertEqual([row["composition_id"] for row in audio_results], ["C2", "C3"])
        self.assertEqual(audio_results[0]["fused_score"], 1.0)
        self.assertEqual(self.audio.calls[-1][0], Path("query.wav"))

    def test_combined_rrf_merges_compositions_and_keeps_branch_evidence(self):
        results = self.engine.search("meaning", "query.wav", top_k=10)
        self.assertEqual([row["composition_id"] for row in results], ["C2", "C1", "C3", "C4"])
        self.assertEqual(len({row["composition_id"] for row in results}), len(results))
        c2 = results[0]
        self.assertEqual(c2["text_rank"], 2)
        self.assertEqual(c2["audio_rank"], 1)
        self.assertEqual(c2["text_score"], 0.72)
        self.assertEqual(c2["audio_score"], 0.04)
        self.assertEqual(c2["best_matching_audio_id"], "R2")
        self.assertEqual(c2["modalities"], ["text", "audio"])
        self.assertIn("T-C2", str(c2["text_evidence"]))
        self.assertEqual(c2["fused_score"], (61 / 62 + 1) / 2)
        c4 = next(row for row in results if row["composition_id"] == "C4")
        self.assertIsNone(c4["text_score"])

    def test_configurable_weights_and_candidate_deduplication(self):
        self.text.results = [text_result("C1", 0.4), text_result("C2", 0.3), text_result("C2", 0.2)]
        weighted = MultimodalRetrievalEngine(
            self.text,
            FakeAudioEngine([audio_result("C2", 0.01, "R2")]),
            text_weight=0.8,
            audio_weight=0.2,
            rrf_constant=0,
        )
        results = weighted.search("phrase", "audio.mp3")
        self.assertEqual([row["composition_id"] for row in results], ["C1", "C2"])
        self.assertAlmostEqual(results[0]["fused_score"], 0.8)
        self.assertAlmostEqual(results[1]["fused_score"], 0.8 / 2 + 0.2)
        self.assertEqual(results[1]["text_rank"], 2)

    def test_ties_have_deterministic_composition_id_order(self):
        engine = MultimodalRetrievalEngine(
            FakeTextEngine([text_result("Z"), text_result("A")]),
            FakeAudioEngine([audio_result("A", 0.1, "RA"), audio_result("Z", 0.9, "RZ")]),
            text_weight=0.5,
            audio_weight=0.5,
            rrf_constant=0,
        )
        first = engine.search("query", "audio.wav")
        second = engine.search("query", "audio.wav")
        self.assertEqual([row["composition_id"] for row in first], ["A", "Z"])
        self.assertEqual(first, second)

    def test_single_success_with_other_failure_and_all_failures(self):
        failing_audio = FakeAudioEngine(error=RuntimeError("decode failed"))
        partial = MultimodalRetrievalEngine(self.text, failing_audio)
        rows = partial.search("query", "bad.wav")
        self.assertTrue(rows)
        self.assertEqual(partial.last_branch_errors, {"audio": "decode failed"})
        failed = MultimodalRetrievalEngine(
            FakeTextEngine(error=ValueError("bad model")), failing_audio
        )
        with self.assertRaises(RetrievalBranchError) as raised:
            failed.search("query", "bad.wav")
        self.assertIn("text", raised.exception.errors)
        self.assertIn("audio", raised.exception.errors)
        self.assertEqual(failed.search("query", "bad.wav", raise_on_branch_error=False), [])

    def test_valid_empty_branch_and_empty_union(self):
        engine = MultimodalRetrievalEngine(
            FakeTextEngine([]), FakeAudioEngine([audio_result("A", 0.2, "R")]),
            text_weight=0.5, audio_weight=0.5,
        )
        result = engine.search("valid but no hit", "audio.wav")
        self.assertEqual(result[0]["fused_score"], 0.5)
        empty = MultimodalRetrievalEngine(FakeTextEngine([]), FakeAudioEngine([]))
        self.assertEqual(empty.search("q", "p"), [])

    def test_graph_enrichment_never_changes_fused_score(self):
        class Graph:
            def get_composition_details(self, composition_id):
                return {"composition": {"composition_id": composition_id}, "extra": "verified"}

        engine = MultimodalRetrievalEngine(self.text, knowledge_graph=Graph())
        result = engine.search("text only")[0]
        self.assertEqual(result["graph_data"]["extra"], "verified")
        self.assertEqual(result["fused_score"], 1.0)

        class BrokenGraph:
            def get_composition_details(self, _composition_id):
                raise RuntimeError("offline")

        engine.knowledge_graph = BrokenGraph()
        result = engine.search("text only")[0]
        self.assertIsNone(result["graph_data"])
        self.assertEqual(engine.last_graph_error, "offline")

    def test_query_weight_and_top_k_validation(self):
        with self.assertRaises(ValueError):
            self.engine.search()
        with self.assertRaises(ValueError):
            self.engine.search("  ")
        with self.assertRaises(ValueError):
            self.engine.search(audio_query_path=" ")
        with self.assertRaises(ValueError):
            self.engine.search("q", top_k=0)
        with self.assertRaises(ValueError):
            MultimodalRetrievalEngine(text_weight=0, audio_weight=0)


if __name__ == "__main__":
    unittest.main()
