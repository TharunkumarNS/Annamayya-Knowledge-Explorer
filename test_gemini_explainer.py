"""Unit tests for evidence packaging and mocked Gemini generation."""

import json
import os
import unittest
from unittest.mock import patch

import pandas as pd

from gemini_explainer import (
    DEFAULT_MODEL,
    SYSTEM_INSTRUCTION,
    GeminiExplainer,
    GeminiExplanationError,
    GeminiNotConfiguredError,
    InsufficientEvidenceError,
    build_prompt,
    evidence_fingerprint,
    prepare_evidence,
)


def sample_result(composition_id="ANN-001"):
    text_best = {
        "text_id": "T-001",
        "composition_id": composition_id,
        "text_type": "lyrics",
        "language": "Telugu",
        "content": "దేవుని స్తుతించే పాట",
        "score": 0.87,
    }
    text_branch = {
        "score_type": "cosine_similarity",
        "best_match": text_best,
        "matching_text_records": [text_best],
    }
    audio_recording = {
        "audio_id": "R-001",
        "composition_id": composition_id,
        "score": 0.93,
        "audio_path": "Recordings/audio001.mp3",
        "resolved_audio_path": "C:/private/audio001.mp3",
    }
    audio_branch = {
        "score_type": "cosine_similarity",
        "best_matching_audio_id": "R-001",
        "recording_evidence": [audio_recording],
    }
    return {
        "composition_id": composition_id,
        "canonical_title": "Adivo Alladivo",
        "fused_score": 0.96,
        "text_score": 0.87,
        "audio_score": 0.93,
        "text_rank": 1,
        "audio_rank": 2,
        "best_matching_audio_id": "R-001",
        "modalities": ["text", "audio"],
        "metadata": {"singer": "Singer A", "raga": None, "tala": "Adi"},
        "text_evidence": {
            "best_match": text_best,
            "matching_text_records": [text_best],
            "branch_result": text_branch,
        },
        "audio_evidence": {
            "best_matching_audio_id": "R-001",
            "recording_evidence": [audio_recording],
            "branch_result": audio_branch,
        },
        "graph_data": {
            "composition": {"composition_id": composition_id, "canonical_title": "Adivo Alladivo"},
            "singers": ["Singer A"],
            "ragas": [],
            "talas": ["Adi"],
            "text_records": [{"text_id": "T-001", "verified": False}],
            "audio_records": [{"audio_id": "R-001"}],
        },
    }


class FakeModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.models = FakeModels(response=response, error=error)


class FakeResponse:
    def __init__(self, text, usage_metadata=None):
        self.text = text
        self.usage_metadata = usage_metadata


def sample_frames():
    compositions = pd.DataFrame([{
        "composition_id": "ANN-001", "canonical_title": "Adivo Alladivo",
        "singer": "Singer A", "raga": None, "tala": "Adi",
        "notes": "Source note", "text_source_url": "https://example.test/lyrics",
    }])
    texts = pd.DataFrame([
        {"composition_id": "ANN-001", "text_id": "T-001", "text_type": "lyrics",
         "language": "Telugu", "content": "దేవుని స్తుతించే పాట", "source_url": "https://example.test/telugu",
         "verified": False},
        {"composition_id": "ANN-001", "text_id": "T-002", "text_type": "translation",
         "language": "English", "content": "A supplied English rendering.", "source_url": "https://example.test/translation"},
    ])
    audio = pd.DataFrame([{
        "composition_id": "ANN-001", "audio_id": "R-001", "recording_title": "Recording A",
        "singer": "Singer A", "audio_file_exists": True, "source_url": "https://example.test/audio",
        "audio_path": "Recordings/audio001.mp3",
    }])
    return compositions, texts, audio


class EvidencePreparationTests(unittest.TestCase):
    def setUp(self):
        self.compositions, self.texts, self.audio = sample_frames()

    def test_only_available_source_fields_and_provenance_are_packaged(self):
        package = prepare_evidence(
            [sample_result()], compositions=self.compositions,
            text_records=self.texts, audio_records=self.audio,
        )
        candidate = package["candidates"][0]
        self.assertNotIn("raga", candidate["composition"])
        self.assertEqual(candidate["composition"]["singer"], "Singer A")
        self.assertEqual(candidate["text_records"][0]["sources"]["source_url"], "https://example.test/telugu")
        self.assertIs(candidate["text_records"][0]["verified"], False)
        self.assertTrue(candidate["audio_recordings"][0]["processed_in_audio_search"])
        self.assertNotIn("audio_path", json.dumps(package))
        self.assertIn("text_source_url", candidate["composition"])

    def test_candidate_limit_selected_composition_and_missing_data(self):
        results = [sample_result("ANN-001"), sample_result("ANN-002"), sample_result("ANN-003"), sample_result("ANN-004")]
        package = prepare_evidence(results, max_candidates=2)
        self.assertEqual(len(package["candidates"]), 2)
        selected = prepare_evidence(results, selected_composition_id="ANN-003")
        self.assertEqual([row["composition"]["composition_id"] for row in selected["candidates"]], ["ANN-003"])
        empty = prepare_evidence([{"composition_id": "ANN-9", "fused_score": 0.5}])
        self.assertEqual(empty["candidates"][0]["available_evidence"]["text_records"], 0)
        self.assertNotIn("text_records", empty["candidates"][0])

    def test_long_text_is_transparently_excerpted_and_total_size_bounded(self):
        long = sample_result()
        long["text_evidence"]["best_match"]["content"] = "క" * 10000
        long["text_evidence"]["matching_text_records"] = [long["text_evidence"]["best_match"]]
        _, texts, audio = sample_frames()
        texts.loc[0, "content"] = "క" * 20000
        package = prepare_evidence(
            [long], text_records=texts, audio_records=audio, max_evidence_chars=1200,
        )
        serialized = json.dumps(package, ensure_ascii=False, separators=(",", ":"))
        self.assertLessEqual(len(serialized), 1200)
        records = package["candidates"][0].get("text_records", [])
        if records and "content" in records[0]:
            self.assertTrue(records[0].get("content_truncated"))

    def test_prompt_separates_question_history_and_untrusted_evidence(self):
        package = prepare_evidence([sample_result()])
        prompt = build_prompt(
            "Why was this composition returned?", package,
            [{"question": "Earlier?", "answer": "Earlier answer."}],
        )
        self.assertIn("Why was this composition returned?", prompt)
        self.assertIn("T-001", prompt)
        self.assertIn("Earlier answer.", prompt)
        self.assertIn("untrusted source data", prompt)
        self.assertIn("not factual evidence", prompt)
        self.assertIn("not probabilities", SYSTEM_INSTRUCTION)
        self.assertIn("insufficient", SYSTEM_INSTRUCTION)
        with self.assertRaises(InsufficientEvidenceError):
            build_prompt("What is this about?", {"candidates": []})

    def test_evidence_fingerprint_is_stable_and_sensitive_to_changes(self):
        evidence = prepare_evidence([sample_result()])
        self.assertEqual(evidence_fingerprint(evidence), evidence_fingerprint(evidence))
        changed = {**evidence, "candidates": [{"composition": {"composition_id": "other"}}]}
        self.assertNotEqual(evidence_fingerprint(evidence), evidence_fingerprint(changed))


class GeminiClientTests(unittest.TestCase):
    def setUp(self):
        self.evidence = prepare_evidence([sample_result()])

    def test_successful_mocked_response_and_usage(self):
        client = FakeClient(FakeResponse("Evidence-grounded answer.", type("Usage", (), {
            "prompt_token_count": 20, "candidates_token_count": 13, "total_token_count": 33,
        })()))
        service = GeminiExplainer(client=client, model="test-model")
        result = service.explain("Summarize this song", self.evidence)
        self.assertEqual(result["answer"], "Evidence-grounded answer.")
        self.assertEqual(result["model"], "test-model")
        self.assertEqual(result["usage"]["total_tokens"], 33)
        self.assertEqual(result["evidence_fingerprint"], evidence_fingerprint(self.evidence))
        self.assertEqual(len(client.models.calls), 1)
        call = client.models.calls[0]
        self.assertIn("Summarize this song", call["contents"])
        self.assertEqual(call["model"], "test-model")

    def test_service_enforces_candidate_bound_without_mutating_input(self):
        evidence = prepare_evidence([sample_result(f"ANN-{index}") for index in range(1, 5)])
        original_count = len(evidence["candidates"])
        client = FakeClient(FakeResponse("Bounded answer."))
        service = GeminiExplainer(client=client, max_candidates=1)
        service.explain("Compare candidates", evidence)
        prompt = client.models.calls[0]["contents"]
        self.assertEqual(len(evidence["candidates"]), original_count)
        self.assertIn('"composition_id":"ANN-1"', prompt)
        self.assertNotIn('"composition_id":"ANN-2"', prompt)

    def test_missing_api_key_does_not_call_google(self):
        with patch.dict(os.environ, {}, clear=True):
            service = GeminiExplainer()
            self.assertEqual(service.model, DEFAULT_MODEL)
            with self.assertRaises(GeminiNotConfiguredError):
                service.explain("Question", self.evidence)

    def test_empty_evidence_is_rejected_before_the_api_call(self):
        client = FakeClient(FakeResponse("Should never be used."))
        service = GeminiExplainer(client=client)
        with self.assertRaises(InsufficientEvidenceError):
            service.explain("Question", {"candidates": []})
        self.assertEqual(client.models.calls, [])

    def test_api_failure_is_sanitized_and_malformed_response_is_handled(self):
        failed = GeminiExplainer(api_key="secret-token", client=FakeClient(error=RuntimeError("bad secret-token")))
        with self.assertRaises(GeminiExplanationError) as raised:
            failed.explain("Question", self.evidence)
        self.assertNotIn("secret-token", str(raised.exception))
        malformed = GeminiExplainer(client=FakeClient(FakeResponse(None)))
        with self.assertRaisesRegex(GeminiExplanationError, "no usable text"):
            malformed.explain("Question", self.evidence)

    def test_question_validation_and_model_configuration(self):
        with patch.dict(os.environ, {"GEMINI_MODEL": "configured-model"}, clear=False):
            self.assertEqual(GeminiExplainer(client=FakeClient()).model, "configured-model")
        with self.assertRaises(ValueError):
            build_prompt(" ", self.evidence)
        with self.assertRaises(ValueError):
            build_prompt("q" * 5001, self.evidence)


if __name__ == "__main__":
    unittest.main()
