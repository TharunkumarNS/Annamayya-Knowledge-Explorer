"""Run a reproducible, source-associated retrieval smoke evaluation.

The packaged data has no independent query set. Prepared cases therefore use
the same indexed record as the query and must not be interpreted as general
retrieval accuracy or as held-out evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from audio_retrieval import AudioRetrievalEngine  # noqa: E402
from data_loader import DATASET_DIR, load_normalized_data  # noqa: E402
from multimodal_retrieval import MultimodalRetrievalEngine  # noqa: E402
from text_retrieval import DEFAULT_MODEL_NAME, build_engine as build_text_engine  # noqa: E402

EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
DEFAULT_CASES = EVALUATION_DIR / "evaluation_queries.csv"
DEFAULT_RESULTS = EVALUATION_DIR / "evaluation_results.csv"
DEFAULT_SUMMARY = EVALUATION_DIR / "evaluation_summary.json"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "evaluation_report.md"
METHODS = ("metadata_baseline", "text_only", "audio_only", "multimodal")
CASE_COLUMNS = (
    "query_id",
    "query_mode",
    "query_text",
    "query_audio_path",
    "expected_composition_ids",
    "match_type",
    "source_reference",
    "verification_status",
    "is_tuning_case",
    "is_held_out",
    "notes",
)


def _dataset_fingerprint() -> str:
    """Return a reproducible fingerprint from packaged input metadata."""
    digest = hashlib.sha256()
    for path in sorted(item for item in DATASET_DIR.rglob("*") if item.is_file()):
        stat = path.stat()
        digest.update(
            f"{path.relative_to(DATASET_DIR).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8")
        )
    return digest.hexdigest()


def ranking_metrics(
    expected_ids: Iterable[str], ranked_ids: Iterable[str], *, ks: tuple[int, ...] = (3, 5)
) -> dict[str, float | int]:
    """Calculate Top-1, Recall@k and reciprocal rank for one query."""
    expected = {str(item) for item in expected_ids if str(item)}
    ranked = [str(item) for item in ranked_ids]
    first_relevant = next((index + 1 for index, item in enumerate(ranked) if item in expected), None)
    metrics: dict[str, float | int] = {
        "top1": int(bool(ranked and ranked[0] in expected)),
        "reciprocal_rank": 1.0 / first_relevant if first_relevant else 0.0,
    }
    for k in ks:
        metrics[f"recall_at_{k}"] = float(any(item in expected for item in ranked[:k]))
    return metrics


def prepare_cases(output_path: Path = DEFAULT_CASES) -> int:
    """Create paired, source-associated exact-recording cases from real records."""
    compositions, texts, audios = load_normalized_data()
    title_by_id = compositions.set_index("composition_id")["canonical_title"].to_dict()
    usable_text = texts.loc[texts["text_type"].isin(["lyrics", "transliteration"])]
    text_by_id = {
        str(composition_id): group.sort_values("text_id").iloc[0]
        for composition_id, group in usable_text.groupby("composition_id", sort=True)
    }
    existing = audios.loc[audios["audio_file_exists"]].sort_values("audio_id")
    rows: list[dict[str, Any]] = []
    for _, audio in existing.iterrows():
        composition_id = str(audio["composition_id"])
        text = text_by_id.get(composition_id)
        if text is None:
            continue
        source_values = [
            str(value).strip()
            for value in (audio.get("source_url"), text.get("source_url"))
            if pd.notna(value) and str(value).strip()
        ]
        rows.append(
            {
                "query_id": f"direct-{audio['audio_id']}",
                "query_mode": "text_audio",
                "query_text": str(text["content"])[:240],
                "query_audio_path": str(audio["audio_path"]),
                "expected_composition_ids": composition_id,
                "match_type": "same_recording_in_index",
                "source_reference": " | ".join(dict.fromkeys(source_values)),
                "verification_status": "dataset_association_only",
                "is_tuning_case": "true",
                "is_held_out": "false",
                "notes": (
                    f"Query uses indexed recording {audio['audio_id']} and its dataset-associated "
                    f"text for {title_by_id.get(composition_id) or composition_id}; in-sample "
                    "smoke case, not independent ground truth."
                ),
            }
        )
    if not rows:
        raise RuntimeError("No present audio recordings have a linked usable text record.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _metadata_rank(query: str, compositions: pd.DataFrame) -> list[dict[str, Any]]:
    """Strict normalized substring lookup over explicit composition metadata only."""
    normalized_query = " ".join("".join(ch.casefold() if ch.isalnum() else " " for ch in query).split())
    matches: list[tuple[int, int, str, dict[str, Any]]] = []
    for _, row in compositions.iterrows():
        fields = [row.get(field) for field in ("canonical_title", "singer", "raga", "tala")]
        normalized_fields = [
            " ".join("".join(ch.casefold() if ch.isalnum() else " " for ch in str(value)).split())
            for value in fields
            if pd.notna(value) and str(value).strip()
        ]
        hits = [value for value in normalized_fields if value and value in normalized_query]
        if hits:
            matches.append(
                (len(hits), sum(len(value) for value in hits), str(row["composition_id"]), row.to_dict())
            )
    matches.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [
        {"composition_id": composition_id, "score": float(hit_count), "score_type": "metadata_field_hits", "metadata": meta}
        for hit_count, _length, composition_id, meta in matches
    ]


def _read_cases(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(CASE_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Evaluation CSV is missing columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
    for row in rows:
        if row["query_mode"] not in {"text", "audio", "text_audio"}:
            raise ValueError(f"Unsupported query_mode for {row['query_id']}: {row['query_mode']}")
        expected = [part.strip() for part in row["expected_composition_ids"].split("|") if part.strip()]
        if not expected:
            raise ValueError(f"Evaluation case {row['query_id']} has no expected composition IDs.")
    return rows


def _metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = len(records)
    successful = [record for record in records if record["status"] == "success"]
    denominator = eligible
    return {
        "eligible_cases": eligible,
        "successfully_evaluated": len(successful),
        "failed_cases": eligible - len(successful),
        "metric_denominator": denominator,
        "denominator_policy": "all eligible cases; failed or empty rankings contribute zero",
        "top1": sum(record["top1"] or 0 for record in records) / denominator if denominator else None,
        "recall_at_3": sum(record["recall_at_3"] or 0 for record in records) / denominator if denominator else None,
        "recall_at_5": sum(record["recall_at_5"] or 0 for record in records) / denominator if denominator else None,
        "mrr": sum(record["reciprocal_rank"] or 0 for record in records) / denominator if denominator else None,
    }


def _format_metric(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.3f}"


def run_evaluation(
    cases_path: Path = DEFAULT_CASES,
    results_path: Path = DEFAULT_RESULTS,
    summary_path: Path = DEFAULT_SUMMARY,
    report_path: Path = DEFAULT_REPORT,
    *,
    top_k: int = 5,
    skip_text: bool = False,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    compositions, text_records, audio_records = load_normalized_data()
    cases = _read_cases(cases_path)
    known_ids = set(compositions["composition_id"].astype(str))
    for case in cases:
        unexpected = set(case["expected_composition_ids"].split("|")) - known_ids
        if unexpected:
            raise ValueError(f"{case['query_id']} references unknown composition IDs: {sorted(unexpected)}")

    text_engine = None
    text_init_error = "Text engine initialization explicitly skipped by operator." if skip_text else None
    if not skip_text:
        try:
            text_engine = build_text_engine(text_records, compositions)
        except Exception as exc:
            text_init_error = f"{type(exc).__name__}: {exc}"

    audio_engine = AudioRetrievalEngine(audio_records, compositions=compositions, dataset_dir=DATASET_DIR)
    audio_stats = audio_engine.build_index()
    multimodal = MultimodalRetrievalEngine(
        text_engine=text_engine,
        audio_engine=audio_engine,
        compositions=compositions,
    )
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    output_rows: list[dict[str, Any]] = []

    for case in cases:
        expected = [item.strip() for item in case["expected_composition_ids"].split("|") if item.strip()]
        query_audio = case["query_audio_path"] or None
        query_text = case["query_text"] or None
        requests = {
            "metadata_baseline": lambda: _metadata_rank(query_text or "", compositions),
            "text_only": lambda: text_engine.search_text(query_text, top_k=top_k),
            "audio_only": lambda: audio_engine.search_audio(DATASET_DIR / query_audio, top_k=top_k),
            "multimodal": lambda: multimodal.search(
                text_query=query_text,
                audio_query_path=DATASET_DIR / query_audio,
                top_k=top_k,
            ),
        }
        for method in METHODS:
            record: dict[str, Any] = {
                "query_id": case["query_id"],
                "query_mode": case["query_mode"],
                "retrieval_method": method,
                "expected_composition_ids": "|".join(expected),
                "ranked_composition_ids": "",
                "top_ranked_prediction": "",
                "status": "success",
                "failure_or_exclusion_reason": "",
                "top1": 0,
                "recall_at_3": 0.0,
                "recall_at_5": 0.0,
                "reciprocal_rank": 0.0,
                "scores_json": "{}",
                "is_tuning_case": case["is_tuning_case"],
                "is_held_out": case["is_held_out"],
            }
            if method in {"text_only", "multimodal"} and text_init_error:
                record["status"] = "failed"
                record["failure_or_exclusion_reason"] = "Text engine initialization failed: " + text_init_error
            elif method == "multimodal" and not query_audio:
                record["status"] = "excluded"
                record["failure_or_exclusion_reason"] = "Multimodal requires a paired audio recording."
            elif method == "audio_only" and not query_audio:
                record["status"] = "excluded"
                record["failure_or_exclusion_reason"] = "No audio query supplied."
            elif method == "text_only" and not query_text:
                record["status"] = "excluded"
                record["failure_or_exclusion_reason"] = "No text query supplied."
            else:
                try:
                    ranked = requests[method]()
                    branch_errors = getattr(multimodal, "last_branch_errors", {}) if method == "multimodal" else {}
                    if branch_errors:
                        raise RuntimeError("; ".join(f"{branch}: {message}" for branch, message in branch_errors.items()))
                    ids = [str(item["composition_id"]) for item in ranked]
                    metrics = ranking_metrics(expected, ids)
                    record.update(
                        ranked_composition_ids="|".join(ids),
                        top_ranked_prediction=ids[0] if ids else "",
                        top1=metrics["top1"],
                        recall_at_3=metrics["recall_at_3"],
                        recall_at_5=metrics["recall_at_5"],
                        reciprocal_rank=metrics["reciprocal_rank"],
                        scores_json=json.dumps(
                            [
                                {
                                    "composition_id": item.get("composition_id"),
                                    "score": item.get("fused_score", item.get("score")),
                                    "text_score": item.get("text_score"),
                                    "audio_score": item.get("audio_score"),
                                    "fused_score": item.get("fused_score"),
                                    "text_rank": item.get("text_rank"),
                                    "audio_rank": item.get("audio_rank"),
                                }
                                for item in ranked
                            ],
                            ensure_ascii=False,
                        ),
                    )
                except Exception as exc:
                    record["status"] = "failed"
                    record["failure_or_exclusion_reason"] = f"{type(exc).__name__}: {exc}"
            by_method[method].append(record)
            output_rows.append(record)

    metrics = {method: _metric_summary(records) for method, records in by_method.items()}
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_set": str(cases_path.relative_to(PROJECT_ROOT)) if cases_path.is_relative_to(PROJECT_ROOT) else str(cases_path),
        "evaluation_design": "exploratory in-sample direct-recording smoke set; not held out",
        "ground_truth_status": "dataset-associated IDs; not independently or manually confirmed",
        "all_cases_marked_tuning": all(row["is_tuning_case"].casefold() == "true" for row in cases),
        "held_out_case_count": sum(row["is_held_out"].casefold() == "true" for row in cases),
        "distinct_compositions": len({part.strip() for row in cases for part in row["expected_composition_ids"].split("|") if part.strip()}),
        "common_case_set": True,
        "top_k": top_k,
        "metrics": metrics,
        "audio_index": audio_stats,
        "text_engine_initialization_error": text_init_error,
        "configuration": {
            "text_model": DEFAULT_MODEL_NAME,
            "audio_features": "22.05 kHz mono; 20 MFCC-like + 12 chroma means/std; cosine similarity",
            "fusion": "weighted reciprocal rank fusion; text 0.6, audio 0.4, rrf_constant 60",
            "metadata_baseline": "case-insensitive alphanumeric substring match against title, singer, raga, tala; deterministic hit-count/length/ID tie-break",
            "ranking_depth": top_k,
            "dataset_source": "annamayya_dataset_package/Annamayya_25_Song_Dataset.xlsx",
            "dataset_fingerprint": _dataset_fingerprint(),
            "random_seed": None,
        },
        "limitations": [
            "Each query recording is already part of the indexed set; identity retrieval is expected and is not cross-performance evidence.",
            "The dataset does not provide an independent held-out query set or independently verified relevance judgments.",
            "Failures and empty rankings contribute zero to all aggregate metrics; denominator is all eligible cases.",
            "No weights or thresholds were tuned against a held-out set; all generated cases are explicitly marked as tuning/in-sample.",
        ],
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]) if output_rows else [])
        writer.writeheader()
        writer.writerows(output_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    table = [
        "| Method | Eligible | Succeeded | Failed | Top-1 | Recall@3 | Recall@5 | MRR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        value = metrics[method]
        table.append(
            f"| {method.replace('_', ' ').title()} | {value['eligible_cases']} | {value['successfully_evaluated']} | {value['failed_cases']} | "
            f"{_format_metric(value['top1'])} | {_format_metric(value['recall_at_3'])} | {_format_metric(value['recall_at_5'])} | {_format_metric(value['mrr'])} |"
        )
    report = "\n".join(
        [
            "# Retrieval evaluation report",
            "",
            "> Exploratory in-sample smoke test only. This is not an independent accuracy or generalization benchmark.",
            "",
            f"Generated: `{summary['generated_at_utc']}`",
            f"Cases: **{len(cases)}**; distinct compositions: **{summary['distinct_compositions']}**; held-out: **0**; all cases are tuning/in-sample.",
            "",
            "## Comparison",
            "",
            *table,
            "",
            "Metrics use every eligible case as the denominator. Failed queries and empty result lists contribute zero. Recall@k is 1 when any expected composition occurs in the top k; MRR is the reciprocal rank of the first relevant result.",
            "",
            "## Data and method limits",
            "",
            "The query audio files are exact recordings already present in the index. Correct direct self-matches verify decoding, feature extraction, ranking, and composition mapping, but do not test alternate performances. Ground truth is the workbook's recording-to-composition association, not independently checked. The same small package supplies both queries and indexed content; no held-out set exists.",
            "",
            "Text engine initialization: " + ("successful." if text_init_error is None else f"failed: `{text_init_error}`"),
            f"Audio index: {audio_stats['indexed']} indexed, {audio_stats['skipped']} missing, {audio_stats['failed']} decode/feature failures.",
            "",
            "Metadata baseline performs strict case-insensitive substring matching on canonical title, singer, raga, and tala only; tie-breaks use number of matching fields, total matched text length, then composition ID. It uses no embeddings, audio, graph, or fusion.",
            "",
            "Per-query detail: [`../data/evaluation/evaluation_results.csv`](../data/evaluation/evaluation_results.csv). Machine-readable summary: [`../data/evaluation/evaluation_summary.json`](../data/evaluation/evaluation_summary.json).",
            "",
            "## Per-method failures",
            "",
        ]
    )
    for method in METHODS:
        failures = [row for row in by_method[method] if row["status"] != "success"]
        report += f"- **{method.replace('_', ' ').title()}**: {len(failures)} failures/exclusions.\n"
        for failure in failures[:8]:
            report += f"  - `{failure['query_id']}`: {failure['failure_or_exclusion_reason']}\n"
    report += "\n## Incorrect top-ranked cases\n\n"
    for method in METHODS:
        incorrect = [row for row in by_method[method] if row["status"] == "success" and not row["top1"]]
        report += f"- **{method.replace('_', ' ').title()}**: {len(incorrect)} successful case(s) did not rank a relevant composition first.\n"
        for row in incorrect[:8]:
            report += (
                f"  - `{row['query_id']}` expected `{row['expected_composition_ids']}`, "
                f"top result `{row['top_ranked_prediction'] or 'no result'}`.\n"
            )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Regenerate source-associated exact-recording cases first.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--skip-text",
        action="store_true",
        help="Do not load LaBSE (for offline/no-cache runs); text and multimodal rows are recorded as failed.",
    )
    args = parser.parse_args()
    try:
        if args.prepare:
            print(f"Prepared {prepare_cases(args.cases)} source-associated cases at {args.cases}.")
        summary = run_evaluation(
            args.cases,
            args.results,
            args.summary,
            args.report,
            top_k=args.top_k,
            skip_text=args.skip_text,
        )
    except Exception as exc:
        print(f"Evaluation failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Per-query results: {args.results}")
    print(f"Summary: {args.summary}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
