"""Multilingual semantic retrieval over Phase 1 Annamayya text records.

The default encoder is ``sentence-transformers/LaBSE``.  Document and query
embeddings are L2-normalized, so reported scores are cosine similarities.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd

from data_loader import load_normalized_data


DEFAULT_MODEL_NAME = "sentence-transformers/LaBSE"
REQUIRED_TEXT_COLUMNS = {"text_id", "composition_id", "text_type", "language", "content"}
REQUIRED_COMPOSITION_COLUMNS = {"composition_id", "canonical_title"}


class Encoder(Protocol):
    """Minimal interface implemented by sentence-transformers encoders."""

    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any:
        """Return one embedding vector per input sentence."""


def preprocess_text(value: object) -> str | None:
    """Conservatively normalize whitespace while preserving Unicode and punctuation."""
    if value is None or pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _normalise_rows(embeddings: Any) -> np.ndarray:
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError("The encoder returned invalid embeddings.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("The encoder returned a zero-length embedding.")
    return vectors / norms


def load_labse_model(model_name: str = DEFAULT_MODEL_NAME) -> Encoder:
    """Load LaBSE once for an index, with a clear dependency/load failure message."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for semantic retrieval. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load multilingual model '{model_name}'. "
            "The first run needs internet access to download model files."
        ) from exc


@dataclass(frozen=True)
class IndexedRecord:
    text_id: str
    composition_id: str
    text_type: str
    language: str
    content: str


class TextRetrievalEngine:
    """Build and query an in-memory semantic index over normalized text records."""

    def __init__(
        self,
        text_records: pd.DataFrame,
        compositions: pd.DataFrame,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        encoder: Encoder | None = None,
    ) -> None:
        self._validate_inputs(text_records, compositions)
        self.model_name = model_name
        self._encoder = encoder if encoder is not None else load_labse_model(model_name)
        self._composition_metadata = compositions.drop_duplicates("composition_id").set_index("composition_id")
        self._records = self._prepare_records(text_records)
        unknown_parents = {record.composition_id for record in self._records} - set(
            self._composition_metadata.index
        )
        if unknown_parents:
            raise ValueError(
                "TextRecords contains composition IDs missing from Compositions: "
                + ", ".join(sorted(unknown_parents))
            )
        self._embeddings = self._encode_documents()

    @staticmethod
    def _validate_inputs(text_records: pd.DataFrame, compositions: pd.DataFrame) -> None:
        missing_text = sorted(REQUIRED_TEXT_COLUMNS - set(text_records.columns))
        missing_compositions = sorted(REQUIRED_COMPOSITION_COLUMNS - set(compositions.columns))
        if missing_text:
            raise ValueError("TextRecords is missing required column(s): " + ", ".join(missing_text))
        if missing_compositions:
            raise ValueError("Compositions is missing required column(s): " + ", ".join(missing_compositions))
        if not compositions["composition_id"].is_unique:
            raise ValueError("Compositions must contain unique composition_id values.")

    @staticmethod
    def _prepare_records(text_records: pd.DataFrame) -> list[IndexedRecord]:
        seen_text_ids: set[str] = set()
        records: list[IndexedRecord] = []
        for _, row in text_records.iterrows():
            text_id = preprocess_text(row["text_id"])
            composition_id = preprocess_text(row["composition_id"])
            content = preprocess_text(row["content"])
            if text_id is None or composition_id is None or content is None:
                continue
            if text_id in seen_text_ids:
                raise ValueError(f"TextRecords contains duplicate text_id: {text_id}")
            seen_text_ids.add(text_id)
            records.append(
                IndexedRecord(
                    text_id=text_id,
                    composition_id=composition_id,
                    text_type=preprocess_text(row["text_type"]) or "unknown",
                    language=preprocess_text(row["language"]) or "unknown",
                    content=content,
                )
            )
        return records

    def _encode_documents(self) -> np.ndarray:
        if not self._records:
            return np.empty((0, 0), dtype=np.float32)
        content = [record.content for record in self._records]
        return _normalise_rows(self._encoder.encode(content, show_progress_bar=False))

    @property
    def indexed_text_count(self) -> int:
        """Number of usable text records in the in-memory index."""
        return len(self._records)

    @property
    def indexed_composition_count(self) -> int:
        """Number of distinct compositions represented by indexed text records."""
        return len({record.composition_id for record in self._records})

    def search_text(
        self,
        query: str,
        top_k: int = 10,
        *,
        languages: str | Sequence[str] | None = None,
        text_types: str | Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to ``top_k`` compositions ranked by maximum cosine similarity.

        Filters apply before similarity ranking.  Each result contains its highest
        scoring text record as evidence and all matching eligible records.
        """
        cleaned_query = preprocess_text(query)
        if cleaned_query is None:
            raise ValueError("Query must contain non-whitespace text.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if not self._records:
            return []

        allowed_languages = _normalise_filter(languages)
        allowed_types = _normalise_filter(text_types)
        eligible = [
            index
            for index, record in enumerate(self._records)
            if (allowed_languages is None or record.language.casefold() in allowed_languages)
            and (allowed_types is None or record.text_type.casefold() in allowed_types)
        ]
        if not eligible:
            return []

        query_embedding = _normalise_rows(self._encoder.encode([cleaned_query], show_progress_bar=False))[0]
        similarities = self._embeddings[eligible] @ query_embedding
        per_composition: dict[str, list[tuple[float, IndexedRecord]]] = defaultdict(list)
        for index, score in zip(eligible, similarities, strict=True):
            per_composition[self._records[index].composition_id].append((float(score), self._records[index]))

        results: list[dict[str, Any]] = []
        for composition_id, matches in per_composition.items():
            matches.sort(key=lambda item: (-item[0], item[1].text_id))
            best_score, best_record = matches[0]
            metadata = self._composition_metadata.loc[composition_id].to_dict()
            evidence = [
                {
                    "text_id": record.text_id,
                    "score": score,
                    "language": record.language,
                    "text_type": record.text_type,
                    "content": record.content,
                }
                for score, record in matches
            ]
            results.append(
                {
                    "composition_id": composition_id,
                    "canonical_title": metadata.get("canonical_title"),
                    "score": best_score,
                    "score_type": "cosine_similarity",
                    "best_match": evidence[0],
                    "matching_text_records": evidence,
                    "metadata": metadata,
                }
            )
        return sorted(results, key=lambda item: (-item["score"], item["composition_id"]))[:top_k]


def _normalise_filter(values: str | Sequence[str] | None) -> set[str] | None:
    if values is None:
        return None
    raw_values = [values] if isinstance(values, str) else values
    normalised = {value.casefold() for value in raw_values if preprocess_text(value) is not None}
    return normalised


def build_engine(
    text_records: pd.DataFrame,
    compositions: pd.DataFrame,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    encoder: Encoder | None = None,
) -> TextRetrievalEngine:
    """Build a reusable semantic retrieval engine from Phase 1 DataFrames."""
    return TextRetrievalEngine(
        text_records, compositions, model_name=model_name, encoder=encoder
    )


def _print_results(results: list[dict[str, Any]]) -> None:
    if not results:
        print("No matching compositions were found.")
        return
    for position, result in enumerate(results, start=1):
        evidence = result["best_match"]
        snippet = evidence["content"].replace("\n", " ")[:180]
        print(
            f"{position}. {result['canonical_title']} ({result['composition_id']})\n"
            f"   cosine similarity: {result['score']:.4f}; "
            f"{evidence['language']} {evidence['text_type']} [{evidence['text_id']}]\n"
            f"   {snippet}"
        )


def main() -> None:
    """Run a small command-line demonstration against the packaged dataset."""
    parser = argparse.ArgumentParser(description="Search Annamayya text records with LaBSE.")
    parser.add_argument("query", nargs="?", default="sacred abode of Sri Hari")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--language", action="append", dest="languages")
    parser.add_argument("--text-type", action="append", dest="text_types")
    arguments = parser.parse_args()

    compositions, text_records, _ = load_normalized_data()
    engine = build_engine(text_records, compositions)
    print(
        f"Indexed {engine.indexed_text_count} text records across "
        f"{engine.indexed_composition_count} compositions using {engine.model_name}."
    )
    _print_results(
        engine.search_text(
            arguments.query,
            top_k=arguments.top_k,
            languages=arguments.languages,
            text_types=arguments.text_types,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"Text retrieval could not run: {exc}") from exc
