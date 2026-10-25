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
        if self.driver.results:
            return self.driver.results.pop(0)
        return self.driver.result


class FakeDriver:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
        self.closed = False
        self.results = []
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

    def test_composition_choices_and_neighborhood_are_parameterized(self):
        self.driver.result = FakeResult(records=[
            FakeRecord({"composition": {"composition_id": "A1", "canonical_title": "Song"}}),
        ])
        self.assertEqual(self.graph.get_compositions()[0]["composition"]["composition_id"], "A1")
        self.assertIn("MATCH (c:Composition)", self.driver.calls[-1][0])

        self.driver.result = FakeResult(records=[
            FakeRecord({"id": "c1", "labels": ["Composition"], "properties": {"composition_id": "A1"}}),
            FakeRecord({"id": "t1", "labels": ["TextRecord"], "properties": {"text_id": "T1"}}),
        ])
        neighborhood_nodes = self.graph.get_composition_neighborhood("A1", hops=1)
        self.assertEqual([node["id"] for node in neighborhood_nodes["nodes"]], ["c1", "t1"])
        node_query, node_parameters = self.driver.calls[-2]
        self.assertIn("[*0..1]", node_query)
        self.assertEqual(node_parameters, {"composition_id": "A1", "limit": 250})

        self.driver.result = FakeResult(records=[FakeRecord({"source": "t1", "target": "c1", "type": "BELONGS_TO"})])
        neighborhood_edges = self.graph.get_composition_neighborhood("A1", hops=2)
        self.assertEqual(neighborhood_edges["relationships"][0]["type"], "BELONGS_TO")
        edge_query, edge_parameters = self.driver.calls[-1]
        self.assertIn("[*1..2]", edge_query)
        self.assertEqual(edge_parameters["composition_id"], "A1")

    def test_neighborhood_validates_parameters(self):
        with self.assertRaises(ValueError):
            self.graph.get_composition_neighborhood("")
        with self.assertRaises(ValueError):
            self.graph.get_composition_neighborhood("A1", hops=3)
        with self.assertRaises(ValueError):
            self.graph.get_composition_neighborhood("A1", limit=0)

    def test_full_graph_loads_all_supported_nodes_and_relationships(self):
        self.driver.results = [
            FakeResult(records=[
                FakeRecord({"id": "c1", "labels": ["Composition"], "properties": {"composition_id": "A1"}}),
                FakeRecord({"id": "t1", "labels": ["TextRecord"], "properties": {"text_id": "T1"}}),
                FakeRecord({"id": "s1", "labels": ["Singer"], "properties": {"name": "Singer One"}}),
            ]),
            FakeResult(records=[
                FakeRecord({"source": "t1", "target": "c1", "type": "BELONGS_TO", "properties": {}}),
                FakeRecord({"source": "c1", "target": "s1", "type": "PERFORMED_BY", "properties": {}}),
            ]),
        ]
        graph = self.graph.get_full_graph_snapshot()
        self.assertEqual(len(graph["nodes"]), 3)
        self.assertEqual({edge["type"] for edge in graph["relationships"]}, {"BELONGS_TO", "PERFORMED_BY"})
        node_query, node_parameters = self.driver.calls[-2]
        relationship_query, relationship_parameters = self.driver.calls[-1]
        for label in ("Composition", "TextRecord", "AudioRecording", "Singer", "Raga", "Tala"):
            self.assertIn(label, node_parameters["node_labels"])
        self.assertNotIn("LIMIT", node_query)
        self.assertNotIn("LIMIT", relationship_query)
        self.assertEqual(node_parameters, relationship_parameters)

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
