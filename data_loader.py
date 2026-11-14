"""Normalize the Annamayya dataset into composition, text, and audio tables.

The packaged source is an Excel workbook rather than ``metadata.csv``.  This
module therefore defaults to ``annamayya_dataset_package/Annamayya_25_Song_Dataset.xlsx``
and also accepts CSV files when a future dataset uses that format.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any
import warnings

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "annamayya_dataset_package"
DEFAULT_SOURCE_PATH = DATASET_DIR / "Annamayya_25_Song_Dataset.xlsx"

COMPOSITION_COLUMNS = [
    "composition_id",
    "canonical_title",
    "singer",
    "raga",
    "tala",
    "text_source_url",
    "transliteration_source_url",
    "translation_source_url",
    "audio_source_url",
    "notes",
]
TEXT_COLUMNS = [
    "text_id",
    "composition_id",
    "text_type",
    "language",
    "content",
    "source_url",
    "source_column",
    "notes",
]
AUDIO_COLUMNS = [
    "audio_id",
    "composition_id",
    "audio_path",
    "audio_file_exists",
    "recording_title",
    "singer",
    "raga",
    "tala",
    "source_url",
]

_REQUIRED_SOURCE_COLUMNS = {"Composition ID", "Song title"}
_TEXT_MAPPINGS = (
    ("Telugu lyrics", "lyrics", "Telugu", "Text source URL"),
    (
        "English transliteration",
        "transliteration",
        "Romanized Telugu",
        "English transliteration source URL",
    ),
    (
        "English meaning/translation",
        "translation",
        "English",
        "Telugu meaning source URL",
    ),
)
_PROVISIONAL_WORD = re.compile(r"\bprovisional\b", re.IGNORECASE)


def _clean_value(value: Any) -> str | None:
    """Return a stripped non-empty string, or ``None`` for any missing value."""
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def _optional_value(row: pd.Series, column: str) -> str | None:
    return _clean_value(row[column]) if column in row.index else None


def _require_columns(source: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns - set(source.columns))
    if missing:
        raise ValueError("Source dataset is missing required column(s): " + ", ".join(missing))


def load_source(source_path: str | Path | None = None) -> pd.DataFrame:
    """Load the packaged workbook (or a supplied CSV/XLSX source) into a DataFrame."""
    path = Path(source_path) if source_path is not None else DEFAULT_SOURCE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Dataset source was not found: {path}")

    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            source = pd.read_csv(path)
        elif suffix in {".xlsx", ".xls"}:
            source = pd.read_excel(path)
        else:
            raise ValueError("Unsupported source format. Use a .csv, .xlsx, or .xls file.")
    except Exception as exc:
        raise ValueError(f"Could not read dataset source '{path}': {exc}") from exc

    if source.empty:
        raise ValueError(f"Dataset source is empty: {path}")
    _require_columns(source, _REQUIRED_SOURCE_COLUMNS)
    return source


def extract_compositions(source: pd.DataFrame) -> pd.DataFrame:
    """Return one normalized row per unique source composition."""
    _require_columns(source, _REQUIRED_SOURCE_COLUMNS)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for index, row in source.iterrows():
        composition_id = _clean_value(row["Composition ID"])
        if composition_id is None:
            raise ValueError(f"Row {index} has no usable Composition ID; parent links are unsafe.")
        if composition_id in seen_ids:
            continue
        seen_ids.add(composition_id)
        rows.append(
            {
                "composition_id": composition_id,
                "canonical_title": _optional_value(row, "Song title"),
                "singer": _optional_value(row, "Singer"),
                "raga": _optional_value(row, "Raga"),
                "tala": _optional_value(row, "Tala"),
                "text_source_url": _optional_value(row, "Text source URL"),
                "transliteration_source_url": _optional_value(
                    row, "English transliteration source URL"
                ),
                "translation_source_url": _optional_value(row, "Telugu meaning source URL"),
                "audio_source_url": _optional_value(row, "Audio source URL"),
                "notes": _optional_value(row, "Notes"),
            }
        )
    return pd.DataFrame(rows, columns=COMPOSITION_COLUMNS)


def extract_text_records(source: pd.DataFrame) -> pd.DataFrame:
    """Unpivot available lyric, transliteration, and translation columns into rows."""
    _require_columns(source, _REQUIRED_SOURCE_COLUMNS)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str | None]] = set()

    for index, row in source.iterrows():
        composition_id = _clean_value(row["Composition ID"])
        if composition_id is None:
            raise ValueError(f"Row {index} has no usable Composition ID; text parent is unsafe.")
        for source_column, text_type, language, url_column in _TEXT_MAPPINGS:
            if source_column not in source.columns:
                continue
            content = _optional_value(row, source_column)
            if content is None:
                continue
            if text_type == "translation" and _PROVISIONAL_WORD.search(content):
                continue
            source_url = _optional_value(row, url_column)
            dedupe_key = (composition_id, text_type, content, source_url)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            records.append(
                {
                    "composition_id": composition_id,
                    "text_type": text_type,
                    "language": language,
                    "content": content,
                    "source_url": source_url,
                    "source_column": source_column,
                    "notes": _optional_value(row, "Notes"),
                }
            )

    for number, record in enumerate(records, start=1):
        record["text_id"] = f"T{number:03d}"
    return pd.DataFrame(records, columns=TEXT_COLUMNS)


def extract_audio_records(source: pd.DataFrame, source_path: str | Path | None = None) -> pd.DataFrame:
    """Return normalized recording rows for nonblank ``Audio file name`` values.

    Audio filenames are made package-relative (``Recordings/<filename>``).  The
    ``audio_file_exists`` flag reports local package availability without
    dropping a valid source reference whose file is absent.
    """
    _require_columns(source, _REQUIRED_SOURCE_COLUMNS)
    if "Audio file name" not in source.columns:
        return pd.DataFrame(columns=AUDIO_COLUMNS)

    dataset_dir = (Path(source_path).parent if source_path is not None else DATASET_DIR)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, row in source.iterrows():
        composition_id = _clean_value(row["Composition ID"])
        if composition_id is None:
            raise ValueError(f"Row {index} has no usable Composition ID; audio parent is unsafe.")
        filename = _optional_value(row, "Audio file name")
        if filename is None:
            continue
        audio_path = Path("Recordings") / filename
        dedupe_key = (composition_id, audio_path.as_posix())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        records.append(
            {
                "composition_id": composition_id,
                "audio_path": audio_path.as_posix(),
                "audio_file_exists": (dataset_dir / audio_path).is_file(),
                "recording_title": _optional_value(row, "Song title"),
                "singer": _optional_value(row, "Singer"),
                "raga": _optional_value(row, "Raga"),
                "tala": _optional_value(row, "Tala"),
                "source_url": _optional_value(row, "Audio source URL"),
            }
        )
    for number, record in enumerate(records, start=1):
        record["audio_id"] = f"R{number:03d}"
    return pd.DataFrame(records, columns=AUDIO_COLUMNS)


def validate_normalized_data(
    compositions: pd.DataFrame, text_records: pd.DataFrame, audio_records: pd.DataFrame
) -> list[str]:
    """Return data-quality warnings; raise ``ValueError`` for integrity failures."""
    if not compositions["composition_id"].is_unique:
        raise ValueError("Composition IDs are not unique.")
    if not text_records["text_id"].is_unique:
        raise ValueError("Text IDs are not unique.")
    if not audio_records["audio_id"].is_unique:
        raise ValueError("Audio IDs are not unique.")

    parent_ids = set(compositions["composition_id"])
    for name, frame in (("TextRecords", text_records), ("AudioRecords", audio_records)):
        unknown = set(frame["composition_id"]) - parent_ids
        if unknown:
            raise ValueError(f"{name} contains unknown composition IDs: {sorted(unknown)}")

    if text_records["content"].map(_clean_value).isna().any():
        raise ValueError("TextRecords contains empty content.")
    translations = text_records.loc[text_records["text_type"] == "translation", "content"]
    if translations.str.contains(_PROVISIONAL_WORD, na=False).any():
        raise ValueError("A prohibited provisional translation remains.")
    if audio_records["audio_path"].map(_clean_value).isna().any():
        raise ValueError("AudioRecords contains a blank audio path.")

    warnings_found: list[str] = []
    if "audio_file_exists" in audio_records and (~audio_records["audio_file_exists"]).any():
        missing = audio_records.loc[~audio_records["audio_file_exists"], "audio_path"].tolist()
        warnings_found.append("Audio source references not present in the package: " + ", ".join(missing))
    return warnings_found


def load_normalized_data(
    source_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the source and return ``(Compositions, TextRecords, AudioRecords)``."""
    source = load_source(source_path)
    compositions = extract_compositions(source)
    text_records = extract_text_records(source)
    audio_records = extract_audio_records(source, source_path)
    for message in validate_normalized_data(compositions, text_records, audio_records):
        warnings.warn(message, UserWarning, stacklevel=2)
    return compositions, text_records, audio_records


def _main() -> None:
    """Run an inspectable direct-execution verification against the default source."""
    source = load_source()
    compositions = extract_compositions(source)
    text_records = extract_text_records(source)
    audio_records = extract_audio_records(source)
    quality_warnings = validate_normalized_data(compositions, text_records, audio_records)

    print(f"Source: {DEFAULT_SOURCE_PATH}")
    print("Source columns:", list(source.columns))
    for name, frame in (
        ("Compositions", compositions),
        ("TextRecords", text_records),
        ("AudioRecords", audio_records),
    ):
        print(f"\n{name}: shape={frame.shape}")
        print("Columns:", list(frame.columns))
        print(frame.head().to_string(index=False))

    provisional_removed = int(
        source.get("English meaning/translation", pd.Series(dtype="object"))
        .map(_clean_value)
        .map(lambda text: bool(text and _PROVISIONAL_WORD.search(text)))
        .sum()
    )
    blank_audio_excluded = int(
        source.get("Audio file name", pd.Series(dtype="object")).map(_clean_value).isna().sum()
    )
    print(f"\nTranslations removed for 'Provisional': {provisional_removed}")
    print(f"Audio rows excluded for blank paths: {blank_audio_excluded}")
    print("Integrity checks: PASS (unique IDs, valid parent references, valid text and audio rows)")
    for message in quality_warnings:
        print("Data-quality warning:", message)


if __name__ == "__main__":
    _main()
