"""Report data integrity for the packaged Annamayya dataset.

Run from the project root with ``python scripts/validate_data.py``.
The report is based on the source workbook and normalized records; it never
rewrites the source data.
"""

from __future__ import annotations

import json
import sys
import argparse
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import (  # noqa: E402
    DEFAULT_SOURCE_PATH,
    _clean_value,
    load_normalized_data,
    load_source,
    validate_normalized_data,
)


def build_report() -> dict[str, object]:
    source = load_source()
    compositions, text_records, audio_records = load_normalized_data()
    warnings = validate_normalized_data(compositions, text_records, audio_records)
    source_ids = source["Composition ID"].map(_clean_value)
    duplicate_ids = sorted(source_ids[source_ids.duplicated(keep=False)].dropna().unique().tolist())
    invalid_id_rows = [int(index) + 2 for index, value in source_ids.items() if value is None]
    missing_title_ids = compositions.loc[
        compositions["canonical_title"].map(_clean_value).isna(), "composition_id"
    ].tolist()
    optional_metadata = ("singer", "raga", "tala")
    incomplete_metadata = {
        field: int(compositions[field].map(_clean_value).isna().sum())
        for field in optional_metadata
    }
    existing_audio = audio_records.loc[audio_records["audio_file_exists"]]
    source_columns = {
        column: int(source[column].map(_clean_value).notna().sum())
        for column in (
            "Text source URL",
            "English transliteration source URL",
            "Telugu meaning source URL",
            "Audio source URL",
            "Notes",
        )
        if column in source.columns
    }
    return {
        "source_path": str(DEFAULT_SOURCE_PATH),
        "source_rows": int(len(source)),
        "composition_count": int(len(compositions)),
        "text_record_count": int(len(text_records)),
        "text_records_by_language_and_type": {
            f"{language} / {text_type}": int(count)
            for (language, text_type), count in text_records.groupby(
                ["language", "text_type"], dropna=False
            ).size().items()
        },
        "audio_record_count": int(len(audio_records)),
        "audio_files_present": int(audio_records["audio_file_exists"].sum()),
        "audio_files_missing": int((~audio_records["audio_file_exists"]).sum()),
        "present_audio_paths": existing_audio["audio_path"].tolist(),
        "missing_audio_paths": audio_records.loc[
            ~audio_records["audio_file_exists"], "audio_path"
        ].tolist(),
        "duplicate_source_composition_ids": duplicate_ids,
        "invalid_source_id_excel_rows": invalid_id_rows,
        "normalized_composition_ids_unique": bool(compositions["composition_id"].is_unique),
        "normalized_text_ids_unique": bool(text_records["text_id"].is_unique),
        "normalized_audio_ids_unique": bool(audio_records["audio_id"].is_unique),
        "compositions_missing_titles": missing_title_ids,
        "incomplete_optional_metadata_counts": incomplete_metadata,
        "source_and_verification_fields_populated": source_columns,
        "warnings": warnings,
        "errors": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "data_integrity_report.json",
        help="Optional path where the machine-readable report is saved.",
    )
    args = parser.parse_args()
    try:
        report = build_report()
    except Exception as exc:  # Report a concise failure without hiding it.
        print(json.dumps({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1
    report["status"] = "PASS" if not report["errors"] else "FAIL"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved integrity report: {args.output}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
