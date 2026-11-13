"""UI-independent safety checks for the Streamlit application helpers."""

from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import app
from app import (
    _excerpt,
    _local_recording_path,
    _metadata_line,
    audio_upload_digest,
    format_score,
    get_composition_rows,
    temporary_audio_file,
)
from gemini_explainer import GeminiExplanationError, evidence_fingerprint


class FakeUpload:
    def __init__(self, name, content):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content


class FakeStreamlit:
    def __init__(self, *, submitted=False):
        self.session_state = {}
        self.submitted = submitted
        self.messages = []

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None

    def selectbox(self, _label, options, format_func=None, key=None):
        value = self.session_state.get(key, options[0])
        if value not in options:
            value = options[0]
        self.session_state[key] = value
        return value

    def text_area(self, _label, key=None, **_kwargs):
        return self.session_state.setdefault(key, "Why was this composition returned?")

    def form_submit_button(self, *_args, **_kwargs):
        return self.submitted

    def warning(self, message):
        self.messages.append(("warning", message))

    def info(self, message):
        self.messages.append(("info", message))

    def error(self, message):
        self.messages.append(("error", message))

    @contextmanager
    def form(self, *_args, **_kwargs):
        yield

    @contextmanager
    def spinner(self, *_args, **_kwargs):
        yield

    @contextmanager
    def expander(self, *_args, **_kwargs):
        yield


class RenderStreamlit:
    def __init__(self):
        self.rendered_text = []
        self.audio_paths = []

    def markdown(self, value, **_kwargs):
        self.rendered_text.append(str(value))

    def text(self, value):
        self.rendered_text.append(str(value))

    def caption(self, value):
        self.rendered_text.append(str(value))

    def write(self, value):
        self.rendered_text.append(str(value))

    def audio(self, path):
        self.audio_paths.append(path)


    def info(self, value):
        self.rendered_text.append(str(value))

    @contextmanager
    def expander(self, *_args, **_kwargs):
        yield


class FakeExplainer:
    calls = []
    failure = None

    def __init__(self, api_key, model):
        self.model = model

    def explain(self, question, evidence, history=None):
        self.calls.append((question, evidence, history))
        if self.failure:
            raise self.failure
        return {
            "answer": "The retrieved text record supports this result.",
            "model": self.model,
            "usage": {},
            "evidence_fingerprint": evidence_fingerprint(evidence),
        }

    def close(self):
        pass



class AppHelperTests(unittest.TestCase):
    def test_result_render_shows_correct_local_text_and_existing_audio(self):
        from data_loader import load_normalized_data

        compositions, text_records, audio_records = load_normalized_data()
        ui = RenderStreamlit()
        result = {
            "composition_id": "ANN-001",
            "canonical_title": "Adivo Alladivo Sri Hari Vasamu",
            "modalities": ["text"],
            "metadata": {},
            "text_evidence": {},
            "audio_evidence": {},
        }
        data = {
            "compositions": compositions,
            "text_records": text_records,
            "audio_records": audio_records,
        }
        with patch.object(app, "st", ui):
            app._render_result(result, 1, data)
        rendered = "\n".join(ui.rendered_text)
        expected = text_records.loc[text_records.composition_id.eq("ANN-001"), "content"].tolist()
        self.assertIn(expected[0][:80], rendered)
        self.assertIn(expected[1][:80], rendered)
        self.assertIn(expected[2][:80], rendered)
        self.assertTrue(ui.audio_paths)
        self.assertTrue(Path(ui.audio_paths[0]).is_file())

    def test_metadata_display_omits_unavailable_and_repeated_values(self):
        line = _metadata_line({
            "singer": "No reliable singer credit found",
            "raga": "శ్రీరాగం",
            "tala": "శ్రీరాగం",
        })
        self.assertEqual(line, "Raga: శ్రీరాగం")

    def test_excerpt_is_bounded_and_preserves_short_content(self):
        self.assertEqual(_excerpt("Telugu lyric"), "Telugu lyric")
        self.assertLessEqual(len(_excerpt("word " * 100, limit=80)), 81)

    def test_recording_path_requires_an_existing_local_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording = root / "recording.wav"
            recording.write_bytes(b"sample")
            with patch.object(app, "DATASET_DIR", root):
                found = _local_recording_path([{"audio_path": "recording.wav"}])
                missing = _local_recording_path([{"audio_path": "missing.wav"}])
        self.assertEqual(found, recording)
        self.assertIsNone(missing)

    def test_dataset_fingerprint_changes_when_input_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording = root / "audio.wav"
            recording.write_bytes(b"first")
            first = app.compute_dataset_fingerprint(root)
            recording.write_bytes(b"changed-content")
            second = app.compute_dataset_fingerprint(root)
        self.assertNotEqual(first, second)

    def test_format_score_preserves_missing_values(self):
        self.assertEqual(format_score(0.123456), "0.1235")
        self.assertEqual(format_score(None), "N/A")
        self.assertEqual(format_score(float("nan")), "N/A")
        self.assertEqual(format_score("not a score"), "N/A")

    def test_audio_upload_uses_content_identity_and_cleans_temp_file(self):
        first = FakeUpload("same-name.wav", b"audio bytes one")
        second = FakeUpload("same-name.wav", b"audio bytes two")
        self.assertNotEqual(audio_upload_digest(first), audio_upload_digest(second))
        path = None
        with temporary_audio_file(first) as temp_path:
            path = temp_path
            self.assertTrue(path.is_file())
            self.assertEqual(path.suffix, ".wav")
            self.assertEqual(path.read_bytes(), b"audio bytes one")
        self.assertFalse(path.exists())

    def test_audio_upload_rejects_unsupported_and_empty_files(self):
        with self.assertRaisesRegex(ValueError, "WAV or MP3"):
            with temporary_audio_file(FakeUpload("recording.flac", b"data")):
                pass
        with self.assertRaisesRegex(ValueError, "empty"):
            with temporary_audio_file(FakeUpload("empty.mp3", b"")):
                pass

    def test_composition_details_resolve_by_stable_identifier(self):
        data = {
            "compositions": pd.DataFrame([
                {"composition_id": "ANN-001", "canonical_title": "Song"},
                {"composition_id": "ANN-002", "canonical_title": "Another"},
            ]),
            "text_records": pd.DataFrame([
                {"composition_id": "ANN-001", "text_id": "T1", "content": "Lyrics"},
            ]),
            "audio_records": pd.DataFrame([
                {"composition_id": "ANN-001", "audio_id": "R1"},
            ]),
        }
        result = get_composition_rows(data, "ANN-001")
        self.assertEqual(result["composition"]["canonical_title"], "Song")
        self.assertEqual(result["text_records"][0]["text_id"], "T1")
        self.assertEqual(result["audio_records"][0]["audio_id"], "R1")
        self.assertIsNone(get_composition_rows(data, "missing")["composition"])

    def test_ai_explanation_uses_current_results_and_persists_without_repeating_call(self):
        ui = FakeStreamlit(submitted=True)
        result = {
            "composition_id": "ANN-001", "canonical_title": "Song", "fused_score": 0.9,
            "text_score": 0.8, "audio_score": None, "text_rank": 1, "audio_rank": None,
            "modalities": ["text"], "metadata": {},
            "text_evidence": {
                "best_match": {"text_id": "T1", "text_type": "lyrics", "language": "Telugu", "content": "Source lyrics"},
                "matching_text_records": [], "branch_result": {"score_type": "cosine_similarity"},
            },
            "audio_evidence": None,
        }
        data = {
            "compositions": pd.DataFrame([{"composition_id": "ANN-001", "canonical_title": "Song"}]),
            "text_records": pd.DataFrame([{
                "composition_id": "ANN-001", "text_id": "T1", "text_type": "lyrics",
                "language": "Telugu", "content": "Source lyrics",
            }]),
            "audio_records": pd.DataFrame(columns=["composition_id", "audio_id"]),
        }
        FakeExplainer.calls = []
        FakeExplainer.failure = None
        with patch.object(app, "st", ui), patch.object(app, "GeminiExplainer", FakeExplainer), patch.dict(
            "os.environ", {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "test-model"}
        ):
            ui.session_state["search_results"] = [result]
            app._render_ai_explanation(data)
            self.assertEqual(len(FakeExplainer.calls), 1)
            self.assertIn("T1", str(FakeExplainer.calls[0][1]))
            saved = ui.session_state["ai_explanation"]
            self.assertEqual(saved["evidence_fingerprint"], evidence_fingerprint(saved["evidence"]))

            ui.submitted = False
            app._render_ai_explanation(data)
            self.assertEqual(len(FakeExplainer.calls), 1)
            self.assertEqual(ui.session_state["search_results"], [result])

    def test_api_error_does_not_remove_retrieval_results(self):
        ui = FakeStreamlit(submitted=True)
        result = {"composition_id": "ANN-001", "canonical_title": "Song"}
        data = {
            "compositions": pd.DataFrame([{"composition_id": "ANN-001"}]),
            "text_records": pd.DataFrame(columns=["composition_id", "text_id"]),
            "audio_records": pd.DataFrame(columns=["composition_id", "audio_id"]),
        }
        FakeExplainer.calls = []
        FakeExplainer.failure = GeminiExplanationError("API unavailable")
        with patch.object(app, "st", ui), patch.object(app, "GeminiExplainer", FakeExplainer), patch.dict(
            "os.environ", {"GEMINI_API_KEY": "test-key"}
        ):
            ui.session_state["search_results"] = [result]
            app._render_ai_explanation(data)
        self.assertEqual(ui.session_state["search_results"], [result])
        self.assertNotIn("ai_explanation", ui.session_state)
        self.assertTrue(any(level == "info" and "temporarily unavailable" in message for level, message in ui.messages))
        FakeExplainer.failure = None

    def test_demo_fallback_matches_exact_question_for_selected_composition(self):
        for question in (
            "What is the spiritual significance of Tirumala in this song?",
            "  WHAT IS THE SPIRITUAL SIGNIFICANCE OF TIRUMALA IN THIS SONG?  ",
        ):
            ui = FakeStreamlit(submitted=True)
            ui.session_state.update(ai_selected_composition="ANN-001", ai_question=question)
            ui.selectbox = lambda _label, options, **_kwargs: "ANN-001"
            ui.text_area = lambda *_args, **_kwargs: question
            result = {"composition_id": "ANN-001", "canonical_title": "Adivo Alladivo Sri Hari Vasamu"}
            data = {
                "compositions": pd.DataFrame([result]),
                "text_records": pd.DataFrame([{"composition_id": "ANN-001", "text_id": "T1", "text_type": "lyrics", "language": "Telugu", "content": "Lyrics"}, {"composition_id": "ANN-001", "text_id": "T2", "text_type": "translation", "language": "English", "content": "Meaning"}]),
                "audio_records": pd.DataFrame(columns=["composition_id", "audio_id"]),
            }
            FakeExplainer.failure = GeminiExplanationError("API unavailable")
            FakeExplainer.calls = []
            with patch.object(app, "st", ui), patch.object(app, "GeminiExplainer", FakeExplainer), patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
                ui.session_state["search_results"] = [result]
                app._render_ai_explanation(data)
            saved = ui.session_state["ai_explanation"]
            self.assertTrue(saved["prepared_demo"])
            self.assertIn("The Sacred Abode of Sri Hari", saved["answer"])
            self.assertIn("T1", str(saved["evidence"]))
            self.assertEqual(len(FakeExplainer.calls), 1)
        FakeExplainer.failure = None

    def test_unrelated_question_or_composition_never_gets_demo_answer(self):
        for result, question in (
            ({"composition_id": "ANN-001", "canonical_title": "Adivo Alladivo Sri Hari Vasamu"}, "Explain something else"),
            ({"composition_id": "ANN-002", "canonical_title": "Other composition"}, app.DEMO_QUESTION),
        ):
            ui = FakeStreamlit(submitted=True)
            ui.selectbox = lambda _label, options, **_kwargs: result["composition_id"]
            ui.text_area = lambda *_args, **_kwargs: question
            data = {"compositions": pd.DataFrame([result]), "text_records": pd.DataFrame(columns=["composition_id", "text_id"]), "audio_records": pd.DataFrame(columns=["composition_id", "audio_id"])}
            FakeExplainer.failure = GeminiExplanationError("API unavailable")
            with patch.object(app, "st", ui), patch.object(app, "GeminiExplainer", FakeExplainer), patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
                ui.session_state["search_results"] = [result]
                app._render_ai_explanation(data)
            self.assertNotIn("ai_explanation", ui.session_state)
            self.assertTrue(any("temporarily unavailable" in message for _level, message in ui.messages))
        FakeExplainer.failure = None

    def test_successful_gemini_response_is_not_replaced_by_demo(self):
        ui = FakeStreamlit(submitted=True)
        ui.selectbox = lambda _label, options, **_kwargs: "ANN-001"
        ui.text_area = lambda *_args, **_kwargs: app.DEMO_QUESTION
        result = {"composition_id": "ANN-001", "canonical_title": "Adivo Alladivo Sri Hari Vasamu"}
        data = {"compositions": pd.DataFrame([result]), "text_records": pd.DataFrame(columns=["composition_id", "text_id"]), "audio_records": pd.DataFrame(columns=["composition_id", "audio_id"])}
        FakeExplainer.failure = None
        with patch.object(app, "st", ui), patch.object(app, "GeminiExplainer", FakeExplainer), patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            ui.session_state["search_results"] = [result]
            app._render_ai_explanation(data)
        self.assertFalse(ui.session_state["ai_explanation"]["prepared_demo"])
        self.assertEqual(ui.session_state["ai_explanation"]["answer"], "The retrieved text record supports this result.")

if __name__ == "__main__":
    unittest.main()
