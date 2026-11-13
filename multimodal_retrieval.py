"""Rank composition candidates from the existing text and audio engines.

Fusion uses weighted reciprocal rank fusion (RRF), which avoids treating text
and audio cosine scores as if their numerical ranges were calibrated alike.
The source branch scores and evidence remain in each result unchanged.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import math
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from data_loader import load_normalized_data
from text_retrieval import build_engine as build_text_engine
from audio_retrieval import AudioRetrievalEngine


class RetrievalBranchError(RuntimeError):
    """Raised when every requested modality fails during retrieval."""

    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        details = "; ".join(f"{branch}: {message}" for branch, message in errors.items())
        super().__init__(f"All requested retrieval branches failed ({details}).")


class MultimodalRetrievalEngine:
    """Fuse ranked composition candidates from text and/or audio retrieval."""

    def __init__(
        self,
        text_engine: Any | None = None,
        audio_engine: Any | None = None,
        compositions: pd.DataFrame | None = None,
        *,
        text_weight: float = 0.6,
        audio_weight: float = 0.4,
        rrf_constant: int = 60,
        candidate_k: int = 100,
        knowledge_graph: Any | None = None,
    ) -> None:
        if (
            not math.isfinite(text_weight)
            or not math.isfinite(audio_weight)
            or text_weight < 0
            or audio_weight < 0
            or text_weight + audio_weight <= 0
        ):
            raise ValueError("Modality weights must be non-negative and have a positive sum.")
        if isinstance(rrf_constant, bool) or not isinstance(rrf_constant, int) or rrf_constant < 0:
            raise ValueError("rrf_constant must be a non-negative integer.")
        if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k <= 0:
            raise ValueError("candidate_k must be a positive integer.")
        self.text_engine = text_engine
        self.audio_engine = audio_engine
        self.compositions = compositions
        self.text_weight = float(text_weight)
        self.audio_weight = float(audio_weight)
        self.rrf_constant = rrf_constant
        self.candidate_k = candidate_k
        self.knowledge_graph = knowledge_graph
        self.last_branch_errors: dict[str, str] = {}
        self.last_graph_error: str | None = None

    @staticmethod
    def _normalise_results(results: Sequence[dict[str, Any]]) -> OrderedDict[str, dict[str, Any]]:
        """Deduplicate composition results by retaining the first ranked item."""
        unique: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for result in results or []:
            composition_id = result.get("composition_id")
            if composition_id is None or pd.isna(composition_id):
                continue
            key = str(composition_id)
            unique.setdefault(key, result)
        return unique

    def search(
        self,
        text_query: str | None = None,
        audio_query_path: str | Path | None = None,
        top_k: int = 10,
        *,
        languages: str | Sequence[str] | None = None,
        text_types: str | Sequence[str] | None = None,
        raise_on_branch_error: bool = True,
    ) -> list[dict[str, Any]]:
        """Search one or both modalities and return fused composition candidates.

        RRF contribution is ``(rrf_constant + 1) / (rrf_constant + rank)``.
        Successful supplied modalities' configured weights are renormalized to
        sum to one. A branch failure is recorded; an optional successful branch
        can still return candidates. If every requested branch fails, the method
        raises :class:`RetrievalBranchError` unless ``raise_on_branch_error`` is
        false, in which case it returns an empty list and exposes errors on
        ``last_branch_errors``.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        has_text = text_query is not None
        has_audio = audio_query_path is not None
        if not has_text and not has_audio:
            raise ValueError("Provide text_query, audio_query_path, or both.")
        if has_text and (not isinstance(text_query, str) or not text_query.strip()):
            raise ValueError("text_query must contain non-whitespace text when supplied.")
        if has_audio and (not isinstance(audio_query_path, (str, Path)) or not str(audio_query_path).strip()):
            raise ValueError("audio_query_path must contain a non-empty path when supplied.")

        self.last_branch_errors = {}
        self.last_graph_error = None
        branch_results: dict[str, OrderedDict[str, dict[str, Any]]] = {}
        if has_text:
            if self.text_engine is None:
                self.last_branch_errors["text"] = "No text retrieval engine was configured."
            else:
                try:
                    kwargs: dict[str, Any] = {"top_k": self.candidate_k}
                    if languages is not None:
                        kwargs["languages"] = languages
                    if text_types is not None:
                        kwargs["text_types"] = text_types
                    raw = self.text_engine.search_text(text_query, **kwargs)
                    branch_results["text"] = self._normalise_results(raw)
                except Exception as exc:
                    self.last_branch_errors["text"] = str(exc)
        if has_audio:
            if self.audio_engine is None:
                self.last_branch_errors["audio"] = "No audio retrieval engine was configured."
            else:
                try:
                    raw = self.audio_engine.search_audio(audio_query_path, top_k=self.candidate_k)
                    branch_results["audio"] = self._normalise_results(raw)
                except Exception as exc:
                    self.last_branch_errors["audio"] = str(exc)

        if not branch_results:
            if raise_on_branch_error:
                raise RetrievalBranchError(self.last_branch_errors)
            return []

        active_weights = {
            "text": self.text_weight if "text" in branch_results else 0.0,
            "audio": self.audio_weight if "audio" in branch_results else 0.0,
        }
        total_weight = sum(active_weights.values())
        if total_weight <= 0:
            raise ValueError("All successfully evaluated modalities have zero weight.")
        effective_weights = {name: value / total_weight for name, value in active_weights.items()}

        all_ids = set().union(*(set(results) for results in branch_results.values()))
        fused: list[dict[str, Any]] = []
        for composition_id in all_ids:
            text_result = branch_results.get("text", {}).get(composition_id)
            audio_result = branch_results.get("audio", {}).get(composition_id)
            text_rank = (
                list(branch_results["text"]).index(composition_id) + 1
                if text_result is not None
                else None
            )
            audio_rank = (
                list(branch_results["audio"]).index(composition_id) + 1
                if audio_result is not None
                else None
            )
            contributions: dict[str, float] = {}
            if text_rank is not None:
                contributions["text"] = effective_weights["text"] * (
                    self.rrf_constant + 1
                ) / (self.rrf_constant + text_rank)
            if audio_rank is not None:
                contributions["audio"] = effective_weights["audio"] * (
                    self.rrf_constant + 1
                ) / (self.rrf_constant + audio_rank)

            metadata = {}
            if text_result:
                metadata.update(text_result.get("metadata") or {})
            if audio_result:
                metadata.update(audio_result.get("composition_metadata") or {})
            canonical_title = (
                (text_result or {}).get("canonical_title")
                or (audio_result or {}).get("canonical_title")
                or metadata.get("canonical_title")
            )
            modalities = [name for name, result in (("text", text_result), ("audio", audio_result)) if result]
            fused.append(
                {
                    "composition_id": composition_id,
                    "canonical_title": canonical_title,
                    "fused_score": float(sum(contributions.values())),
                    "text_score": text_result.get("score") if text_result else None,
                    "audio_score": audio_result.get("score") if audio_result else None,
                    "text_rank": text_rank,
                    "audio_rank": audio_rank,
                    "best_matching_audio_id": (
                        audio_result.get("best_matching_audio_id") if audio_result else None
                    ),
                    "modalities": modalities,
                    "modality_contributions": contributions,
                    "metadata": metadata,
                    "text_evidence": (
                        {
                            "best_match": text_result.get("best_match"),
                            "matching_text_records": text_result.get("matching_text_records", []),
                            "branch_result": text_result,
                        }
                        if text_result
                        else None
                    ),
                    "audio_evidence": (
                        {
                            "best_matching_audio_id": audio_result.get("best_matching_audio_id"),
                            "recording_evidence": audio_result.get("recording_evidence", []),
                            "branch_result": audio_result,
                        }
                        if audio_result
                        else None
                    ),
                }
            )

        fused.sort(key=lambda item: (-item["fused_score"], item["composition_id"]))
        results = fused[:top_k]
        self._enrich_with_graph(results)
        return results

    def _enrich_with_graph(self, results: list[dict[str, Any]]) -> None:
        """Attach graph details without changing fused scores or blocking retrieval."""
        if self.knowledge_graph is None:
            return
        for result in results:
            try:
                result["graph_data"] = self.knowledge_graph.get_composition_details(
                    result["composition_id"]
                )
            except Exception as exc:
                self.last_graph_error = str(exc)
                result["graph_data"] = None


def _make_demo(args: argparse.Namespace) -> tuple[MultimodalRetrievalEngine, Any | None]:
    compositions, text_records, audio_records = load_normalized_data()
    text_engine = build_text_engine(text_records, compositions) if args.text else None
    audio_engine = AudioRetrievalEngine(audio_records, compositions) if args.audio else None
    if audio_engine is not None:
        print("Audio index:", audio_engine.build_index())

    graph = None
    graph_requested = args.graph or args.import_graph
    if graph_requested:
        try:
            from knowledge_graph import AnnamayyaKnowledgeGraph

            graph = AnnamayyaKnowledgeGraph()
            graph.connect()
            graph.initialize_schema()
            if args.import_graph:
                print("Graph import:", graph.import_dataset(compositions, text_records, audio_records))
        except Exception as exc:
            if graph is not None:
                graph.close()
            graph = None
            print(f"Neo4j unavailable; continuing without graph enrichment: {exc}")

    engine = MultimodalRetrievalEngine(
        text_engine=text_engine,
        audio_engine=audio_engine,
        compositions=compositions,
        text_weight=args.text_weight,
        audio_weight=args.audio_weight,
        knowledge_graph=graph,
    )
    return engine, graph


def main() -> None:
    """Run text-only, audio-only, or combined retrieval from the command line."""
    parser = argparse.ArgumentParser(description="Fuse Annamayya text and audio retrieval.")
    parser.add_argument("--text", help="Text query")
    parser.add_argument("--audio", help="Query audio path")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--text-weight", type=float, default=0.6)
    parser.add_argument("--audio-weight", type=float, default=0.4)
    parser.add_argument("--graph", action="store_true", help="Enrich results from configured Neo4j")
    parser.add_argument("--import-graph", action="store_true", help="Import the normalized dataset before searching")
    args = parser.parse_args()
    if not args.text and not args.audio and not args.import_graph:
        parser.error("provide --text, --audio, or --import-graph")

    engine, graph = _make_demo(args)
    try:
        results = engine.search(args.text, args.audio, args.top_k)
        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. {result['canonical_title'] or result['composition_id']} "
                f"({result['composition_id']}) fused={result['fused_score']:.4f} "
                f"text={result['text_score']} audio={result['audio_score']} "
                f"via={','.join(result['modalities'])}"
            )
            if result["text_evidence"]:
                evidence = result["text_evidence"]["best_match"]
                if evidence:
                    print(f"   text: {evidence['text_id']} {evidence['language']} {evidence['text_type']}")
            if result["audio_evidence"]:
                print(f"   audio: {result['best_matching_audio_id']}")
        if engine.last_branch_errors:
            print("Retrieval branch errors:", engine.last_branch_errors)
        if graph is not None:
            if engine.last_graph_error:
                print("Graph enrichment error:", engine.last_graph_error)
            graph.close()
    except Exception:
        if graph is not None:
            graph.close()
        raise


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"Multimodal retrieval could not run: {exc}") from exc
