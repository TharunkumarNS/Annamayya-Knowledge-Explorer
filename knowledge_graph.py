"""Optional, idempotent Neo4j representation of normalized Annamayya data."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd


SCHEMA_QUERIES = (
    "CREATE CONSTRAINT composition_id IF NOT EXISTS FOR (n:Composition) REQUIRE n.composition_id IS UNIQUE",
    "CREATE CONSTRAINT text_id IF NOT EXISTS FOR (n:TextRecord) REQUIRE n.text_id IS UNIQUE",
    "CREATE CONSTRAINT audio_id IF NOT EXISTS FOR (n:AudioRecording) REQUIRE n.audio_id IS UNIQUE",
    "CREATE CONSTRAINT singer_name IF NOT EXISTS FOR (n:Singer) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT raga_name IF NOT EXISTS FOR (n:Raga) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT tala_name IF NOT EXISTS FOR (n:Tala) REQUIRE n.name IS UNIQUE",
)


def _value(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _props(row: pd.Series, excluded: set[str]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for key, value in row.items():
        if key in excluded or value is None or pd.isna(value):
            continue
        if isinstance(value, str):
            properties[str(key)] = value.strip()
        elif hasattr(value, "item"):
            # Convert NumPy scalar wrappers (e.g. numpy.bool_) to values the
            # Neo4j driver can serialize as native Python types.
            properties[str(key)] = value.item()
        else:
            properties[str(key)] = value
    return properties


class AnnamayyaKnowledgeGraph:
    """Manage stable composition, text, audio, and metadata nodes in Neo4j.

    Relationships point from child record to composition for ``BELONGS_TO``
    and ``RECORDING_OF``. Metadata relationships point from composition to
    ``Singer``, ``Raga``, and ``Tala`` nodes.
    """

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        *,
        driver: Any | None = None,
    ) -> None:
        self.uri = uri or os.getenv("NEO4J_URI")
        self.username = username or os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.driver = driver

    def connect(self) -> None:
        """Create and verify a driver using environment settings or injected driver."""
        if self.driver is not None:
            try:
                self.driver.verify_connectivity()
                return
            except Exception as exc:
                raise RuntimeError(f"Could not connect to Neo4j: {exc}") from exc
        if not self.uri or not self.username or not self.password:
            raise RuntimeError(
                "Neo4j is not configured. Set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD."
            )
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("Neo4j Python driver is missing. Install it with: python -m pip install neo4j") from exc
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            self.driver.verify_connectivity()
        except Exception as exc:
            self.close()
            raise RuntimeError(f"Could not connect to Neo4j at {self.uri}: {exc}") from exc

    def _require_driver(self) -> Any:
        if self.driver is None:
            raise RuntimeError("Not connected to Neo4j. Call connect() first.")
        return self.driver

    def initialize_schema(self) -> None:
        """Create idempotent uniqueness constraints for graph entity keys."""
        driver = self._require_driver()
        with driver.session() as session:
            for query in SCHEMA_QUERIES:
                session.run(query)

    def import_dataset(
        self,
        compositions: pd.DataFrame,
        text_records: pd.DataFrame,
        audio_records: pd.DataFrame,
    ) -> dict[str, int]:
        """Upsert normalized rows and only the parent/metadata links they support."""
        required = {
            "compositions": (compositions, {"composition_id"}),
            "text_records": (text_records, {"text_id", "composition_id"}),
            "audio_records": (audio_records, {"audio_id", "composition_id"}),
        }
        for name, (frame, columns) in required.items():
            missing = sorted(columns - set(frame.columns))
            if missing:
                raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")
        composition_ids = {_value(value) for value in compositions["composition_id"]}
        composition_ids.discard(None)
        for name, frame in (("text_records", text_records), ("audio_records", audio_records)):
            orphaned = {
                _value(value) for value in frame["composition_id"]
                if _value(value) is not None and _value(value) not in composition_ids
            }
            if orphaned:
                raise ValueError(
                    f"{name} refers to unknown composition IDs: {', '.join(sorted(orphaned))}"
                )
        driver = self._require_driver()
        counts = {"compositions": 0, "text_records": 0, "audio_records": 0, "metadata_links": 0}
        self.initialize_schema()
        with driver.session() as session:
            for _, row in compositions.iterrows():
                composition_id = _value(row["composition_id"])
                if composition_id is None:
                    continue
                props = _props(row, {"composition_id"})
                session.run(
                    "MERGE (c:Composition {composition_id: $composition_id}) SET c += $properties",
                    composition_id=composition_id,
                    properties=props,
                )
                counts["compositions"] += 1
                for field, label, relationship in (
                    ("singer", "Singer", "PERFORMED_BY"),
                    ("raga", "Raga", "USES_RAGA"),
                    ("tala", "Tala", "USES_TALA"),
                ):
                    name = _value(row.get(field))
                    if name is None:
                        continue
                    query = (
                        f"MERGE (entity:{label} {{name: $name}}) "
                        "WITH entity MATCH (c:Composition {composition_id: $composition_id}) "
                        f"MERGE (c)-[:{relationship}]->(entity)"
                    )
                    session.run(query, name=name, composition_id=composition_id)
                    counts["metadata_links"] += 1

            for _, row in text_records.iterrows():
                text_id = _value(row["text_id"])
                composition_id = _value(row["composition_id"])
                if text_id is None or composition_id is None:
                    continue
                session.run(
                    "MERGE (t:TextRecord {text_id: $text_id}) SET t += $properties "
                    "WITH t MATCH (c:Composition {composition_id: $composition_id}) "
                    "MERGE (t)-[:BELONGS_TO]->(c)",
                    text_id=text_id,
                    composition_id=composition_id,
                    properties=_props(row, {"text_id", "composition_id"}),
                )
                counts["text_records"] += 1

            for _, row in audio_records.iterrows():
                audio_id = _value(row["audio_id"])
                composition_id = _value(row["composition_id"])
                if audio_id is None or composition_id is None:
                    continue
                session.run(
                    "MERGE (a:AudioRecording {audio_id: $audio_id}) SET a += $properties "
                    "WITH a MATCH (c:Composition {composition_id: $composition_id}) "
                    "MERGE (a)-[:RECORDING_OF]->(c)",
                    audio_id=audio_id,
                    composition_id=composition_id,
                    properties=_props(row, {"audio_id", "composition_id"}),
                )
                counts["audio_records"] += 1
        return counts

    @staticmethod
    def _records(result: Any) -> list[dict[str, Any]]:
        return [record.data() for record in result]

    def get_composition_details(self, composition_id: str) -> dict[str, Any] | None:
        """Return one composition and its associated text/audio/metadata entities."""
        if not _value(composition_id):
            raise ValueError("composition_id must be non-empty.")
        query = (
            "MATCH (c:Composition {composition_id: $composition_id}) "
            "OPTIONAL MATCH (t:TextRecord)-[:BELONGS_TO]->(c) "
            "OPTIONAL MATCH (a:AudioRecording)-[:RECORDING_OF]->(c) "
            "OPTIONAL MATCH (c)-[:PERFORMED_BY]->(s:Singer) "
            "OPTIONAL MATCH (c)-[:USES_RAGA]->(r:Raga) "
            "OPTIONAL MATCH (c)-[:USES_TALA]->(tl:Tala) "
            "RETURN properties(c) AS composition, "
            "collect(DISTINCT properties(t)) AS text_records, "
            "collect(DISTINCT properties(a)) AS audio_records, "
            "collect(DISTINCT s.name) AS singers, collect(DISTINCT r.name) AS ragas, "
            "collect(DISTINCT tl.name) AS talas"
        )
        with self._require_driver().session() as session:
            record = session.run(query, composition_id=str(composition_id)).single()
        return record.data() if record else None

    def get_recordings(self, composition_id: str) -> list[dict[str, Any]]:
        """Return known recording nodes for a composition, or an empty list."""
        query = (
            "MATCH (a:AudioRecording)-[:RECORDING_OF]->(c:Composition {composition_id: $composition_id}) "
            "RETURN properties(a) AS recording ORDER BY a.audio_id"
        )
        with self._require_driver().session() as session:
            return self._records(session.run(query, composition_id=str(composition_id)))

    def find_compositions(self, field: str, value: str) -> list[dict[str, Any]]:
        """Find compositions by verified singer, raga, or tala metadata."""
        if field not in {"singer", "raga", "tala"}:
            raise ValueError("field must be one of: singer, raga, tala")
        cleaned = _value(value)
        if cleaned is None:
            raise ValueError("value must be non-empty.")
        label, relationship = {
            "singer": ("Singer", "PERFORMED_BY"),
            "raga": ("Raga", "USES_RAGA"),
            "tala": ("Tala", "USES_TALA"),
        }[field]
        query = (
            f"MATCH (c:Composition)-[:{relationship}]->(m:{label} {{name: $value}}) "
            "RETURN properties(c) AS composition ORDER BY c.composition_id"
        )
        with self._require_driver().session() as session:
            return self._records(session.run(query, value=cleaned))

    def get_graph_snapshot(self, limit: int = 500) -> dict[str, list[dict[str, Any]]]:
        """Return a bounded set of nodes and relationships for graph visualization."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        query = (
            "MATCH (source)-[relationship]->(target) "
            "RETURN elementId(source) AS source_id, labels(source) AS source_labels, "
            "properties(source) AS source_properties, type(relationship) AS relationship, "
            "elementId(target) AS target_id, labels(target) AS target_labels, "
            "properties(target) AS target_properties "
            "ORDER BY source_id, target_id, relationship LIMIT $limit"
        )
        with self._require_driver().session() as session:
            records = self._records(session.run(query, limit=limit))

        nodes: dict[str, dict[str, Any]] = {}
        relationships: list[dict[str, Any]] = []
        for row in records:
            for side in ("source", "target"):
                node_id = row.get(f"{side}_id")
                if node_id is not None:
                    nodes[str(node_id)] = {
                        "id": str(node_id),
                        "labels": row.get(f"{side}_labels") or [],
                        "properties": row.get(f"{side}_properties") or {},
                    }
            if row.get("source_id") is not None and row.get("target_id") is not None:
                relationships.append({
                    "source": str(row["source_id"]),
                    "target": str(row["target_id"]),
                    "type": row.get("relationship") or "RELATED_TO",
                })
        return {"nodes": list(nodes.values()), "relationships": relationships}

    def get_compositions(self) -> list[dict[str, Any]]:
        """Return composition records for choosing a graph neighborhood."""
        query = (
            "MATCH (c:Composition) RETURN properties(c) AS composition "
            "ORDER BY c.canonical_title, c.composition_id"
        )
        with self._require_driver().session() as session:
            return self._records(session.run(query))

    def get_full_graph_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        """Load every supported knowledge-graph node and all links between them."""
        node_labels = ["Composition", "TextRecord", "AudioRecording", "Singer", "Raga", "Tala"]
        node_query = (
            "MATCH (node) WHERE any(label IN labels(node) WHERE label IN $node_labels) "
            "RETURN elementId(node) AS id, labels(node) AS labels, properties(node) AS properties "
            "ORDER BY id"
        )
        relationship_query = (
            "MATCH (source)-[relationship]->(target) "
            "WHERE any(label IN labels(source) WHERE label IN $node_labels) "
            "AND any(label IN labels(target) WHERE label IN $node_labels) "
            "RETURN elementId(source) AS source, elementId(target) AS target, "
            "type(relationship) AS type, properties(relationship) AS properties "
            "ORDER BY source, target, type"
        )
        with self._require_driver().session() as session:
            node_rows = self._records(session.run(node_query, node_labels=node_labels))
            relationship_rows = self._records(session.run(relationship_query, node_labels=node_labels))
        return {
            "nodes": [
                {"id": str(row["id"]), "labels": row.get("labels") or [], "properties": row.get("properties") or {}}
                for row in node_rows if row.get("id") is not None
            ],
            "relationships": [
                {
                    "source": str(row["source"]),
                    "target": str(row["target"]),
                    "type": row.get("type") or "RELATED_TO",
                    "properties": row.get("properties") or {},
                }
                for row in relationship_rows if row.get("source") is not None and row.get("target") is not None
            ],
        }

    def get_composition_neighborhood(
        self, composition_id: str, *, hops: int = 2, limit: int = 250,
    ) -> dict[str, list[dict[str, Any]]]:
        """Load a bounded composition neighborhood and its real Neo4j relationships."""
        cleaned = _value(composition_id)
        if cleaned is None:
            raise ValueError("composition_id must be non-empty")
        if isinstance(hops, bool) or not isinstance(hops, int) or hops not in (1, 2):
            raise ValueError("hops must be 1 or 2")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")

        node_query = (
            "MATCH (c:Composition {composition_id: $composition_id})-[*0.." + str(hops) + "]-(n) "
            "RETURN DISTINCT elementId(n) AS id, labels(n) AS labels, properties(n) AS properties "
            "LIMIT $limit"
        )
        relationship_query = (
            "MATCH p=(c:Composition {composition_id: $composition_id})-[*1.." + str(hops) + "]-(n) "
            "UNWIND relationships(p) AS relationship "
            "RETURN DISTINCT elementId(startNode(relationship)) AS source, "
            "elementId(endNode(relationship)) AS target, type(relationship) AS type "
            "LIMIT $limit"
        )
        with self._require_driver().session() as session:
            node_rows = self._records(session.run(node_query, composition_id=cleaned, limit=limit))
            relationship_rows = self._records(session.run(relationship_query, composition_id=cleaned, limit=limit))
        return {
            "nodes": [
                {"id": str(row["id"]), "labels": row.get("labels") or [], "properties": row.get("properties") or {}}
                for row in node_rows if row.get("id") is not None
            ],
            "relationships": [
                {"source": str(row["source"]), "target": str(row["target"]), "type": row.get("type") or "RELATED_TO"}
                for row in relationship_rows if row.get("source") is not None and row.get("target") is not None
            ],
        }

    def close(self) -> None:
        """Close and release the underlying Neo4j driver, if any."""
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    def __enter__(self) -> "AnnamayyaKnowledgeGraph":
        self.connect()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
