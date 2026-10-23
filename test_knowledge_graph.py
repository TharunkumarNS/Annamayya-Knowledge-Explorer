"""Mocked Neo4j tests; no live Neo4j server is required."""

import os
import unittest
from unittest.mock import patch

import pandas as pd

from knowledge_graph import AnnamayyaKnowledgeGraph, SCHEMA_QUERIES


class FakeRecord:
    def __init__(self, values=None):
        self.values = values

    def data(self):
        return self.values or {}


class FakeResult:
    def __init__(self, records=None, single=None):
        self._records = records or []
        self._single = single

    def __iter__(self):
        return iter(self._records)

    def single(self):
        return self._single


class FakeSession:
    def __init__(self, driver):
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def run(self, query, **parameters):
        self.driver.calls.append((query, parameters))
        return self.driver.result


class FakeDriver:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
        self.closed = False
        self.result = FakeResult(
            records=[FakeRecord({"composition": {"composition_id": "A1"}})],
            single=FakeRecord({"composition": {"composition_id": "A1"}, "text_records": []}),
        )

    def verify_connectivity(self):
        if self.fail:
            raise OSError("server unreachable")

    def session(self):
        return FakeSession(self)

    def close(self):
        self.closed = True


def sample_frames():
    compositions = pd.DataFrame(
        [{"composition_id": "A1", "canonical_title": "Song", "singer": "Singer One", "raga": None, "tala": float("nan"), "notes": "verified"}]
    )
    texts = pd.DataFrame(
        [{"text_id": "T1", "composition_id": "A1", "text_type": "lyrics", "language": "Telugu", "content": "పాట", "source_url": "https://example.test"}]
    )
    audio = pd.DataFrame(
        [{"audio_id": "R1", "composition_id": "A1", "audio_path": "Recordings/audio001.mp3", "audio_file_exists": True}]
    )
    return compositions, texts, audio


class KnowledgeGraphTests(unittest.TestCase):
    def setUp(self):
        self.driver = FakeDriver()
        self.graph = AnnamayyaKnowledgeGraph(driver=self.driver)
        self.compositions, self.texts, self.audio = sample_frames()

    def test_schema_constraints_and_connection(self):
        self.graph.connect()
        self.graph.initialize_schema()
        queries = [query for query, _ in self.driver.calls]
        self.assertEqual(queries, list(SCHEMA_QUERIES))
        self.assertTrue(all("IF NOT EXISTS" in query for query in queries))

    def test_dataset_import_is_parameterized_and_idempotent(self):
        first = self.graph.import_dataset(self.compositions, self.texts, self.audio)
        first_count = len(self.driver.calls)
        second = self.graph.import_dataset(self.compositions, self.texts, self.audio)
        self.assertEqual(first, {"compositions": 1, "text_records": 1, "audio_records": 1, "metadata_links": 1})
        self.assertEqual(first, second)
        self.assertEqual(len(self.driver.calls), first_count * 2)
        upserts = [(query, params) for query, params in self.driver.calls if "MERGE" in query]
        self.assertTrue(upserts)
        for query, params in upserts:
            self.assertIn("MERGE", query)
            self.assertNotIn("Song", query)
            if "TextRecord" in query:
                self.assertEqual(params["properties"]["content"], "పాట")
            if "AudioRecording" in query:
                self.assertIs(params["properties"]["audio_file_exists"], True)

    def test_optional_metadata_and_queries_for_known_unknown_ids(self):
        result = self.graph.get_composition_details("A1")
        self.assertEqual(result["composition"]["composition_id"], "A1")
        self.assertEqual(self.graph.get_recordings("A1")[0]["composition"]["composition_id"], "A1")
        self.assertTrue(self.graph.find_compositions("singer", "Singer One"))
        query, parameters = self.driver.calls[-1]
        self.assertIn("$value", query)
        self.assertEqual(parameters, {"value": "Singer One"})

        self.driver.result = FakeResult(single=None)
        self.assertIsNone(self.graph.get_composition_details("unknown"))
        self.driver.result = FakeResult(records=[])
        self.assertEqual(self.graph.get_recordings("unknown"), [])
        with self.assertRaises(ValueError):
            self.graph.find_compositions("arbitrary", "x")

    def test_missing_configuration_failure_and_close(self):
        with patch.dict(os.environ, {}, clear=True):
            graph = AnnamayyaKnowledgeGraph()
            with self.assertRaisesRegex(RuntimeError, "NEO4J_URI"):
                graph.connect()
        bad_driver = FakeDriver(fail=True)
        graph = AnnamayyaKnowledgeGraph(driver=bad_driver)
        with self.assertRaisesRegex(RuntimeError, "server unreachable"):
            graph.connect()
        self.graph.close()
        self.assertTrue(self.driver.closed)


if __name__ == "__main__":
    unittest.main()
