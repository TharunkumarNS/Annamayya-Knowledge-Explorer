"""Database access for the standalone merge-ready duet catalog."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
PAIR_DATABASE_PATH = PROJECT_ROOT / "generational_duet" / "data" / "duet_pairs.sqlite3"


def _connection() -> sqlite3.Connection:
    if not PAIR_DATABASE_PATH.is_file():
        raise FileNotFoundError(f"Duet pair database was not found: {PAIR_DATABASE_PATH}")
    connection = sqlite3.connect(PAIR_DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _decode_pair(row: sqlite3.Row) -> dict[str, Any]:
    pair = dict(row)
    pair["vintage_anchors_ms"] = json.loads(pair["vintage_anchors_ms"])
    pair["modern_anchors_ms"] = json.loads(pair["modern_anchors_ms"])
    pair["vintage_audio_path"] = (PAIR_DATABASE_PATH.parent / pair["vintage_audio_path"]).resolve()
    pair["modern_audio_path"] = (PAIR_DATABASE_PATH.parent / pair["modern_audio_path"]).resolve()
    pair["bell_audio_path"] = (PAIR_DATABASE_PATH.parent / pair["bell_audio_path"]).resolve()
    return pair


def list_available_pairs() -> list[dict[str, Any]]:
    """Return catalog pairs whose two recordings and bell asset are present."""
    with _connection() as connection:
        rows = connection.execute(
            "SELECT * FROM audio_pairs WHERE is_available = 1 ORDER BY display_name"
        ).fetchall()
    pairs = []
    for row in rows:
        pair = _decode_pair(row)
        if all(pair[key].is_file() for key in ("vintage_audio_path", "modern_audio_path", "bell_audio_path")):
            pairs.append(pair)
    return pairs


def get_pair(pair_id: int) -> dict[str, Any] | None:
    """Read one enabled pair by ID, returning None if it is absent or incomplete."""
    with _connection() as connection:
        row = connection.execute(
            "SELECT * FROM audio_pairs WHERE pair_id = ? AND is_available = 1", (pair_id,)
        ).fetchone()
    if row is None:
        return None
    pair = _decode_pair(row)
    if not all(pair[key].is_file() for key in ("vintage_audio_path", "modern_audio_path", "bell_audio_path")):
        return None
    return pair
