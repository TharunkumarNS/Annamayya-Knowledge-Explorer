"""SQLite persistence for the normalized composition, text, and audio tables."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from data_loader import DEFAULT_SOURCE_PATH, load_normalized_data

PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = Path(__import__("os").environ.get("ANNAMAYYA_DATABASE", PROJECT_ROOT / "data" / "annamayya.sqlite3"))


def _connect(path: str | Path = DATABASE_PATH) -> sqlite3.Connection:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(path: str | Path = DATABASE_PATH, force: bool = False) -> Path:
    """Create or refresh SQLite from the packaged workbook's normalized rows."""
    target = Path(path)
    refresh = force or not target.is_file()
    if not refresh and DEFAULT_SOURCE_PATH.is_file():
        refresh = DEFAULT_SOURCE_PATH.stat().st_mtime_ns > target.stat().st_mtime_ns
    if not refresh:
        return target

    compositions, texts, audio = load_normalized_data()
    connection = _connect(target)
    try:
        connection.executescript("""
            DROP TABLE IF EXISTS TextRecords;
            DROP TABLE IF EXISTS AudioRecords;
            DROP TABLE IF EXISTS Compositions;
            CREATE TABLE Compositions (
                composition_id TEXT PRIMARY KEY, canonical_title TEXT, singer TEXT, raga TEXT, tala TEXT,
                text_source_url TEXT, transliteration_source_url TEXT, translation_source_url TEXT,
                audio_source_url TEXT, notes TEXT
            );
            CREATE TABLE TextRecords (
                text_id TEXT PRIMARY KEY, composition_id TEXT NOT NULL, text_type TEXT, language TEXT,
                content TEXT, source_url TEXT, source_column TEXT, notes TEXT,
                FOREIGN KEY(composition_id) REFERENCES Compositions(composition_id)
            );
            CREATE TABLE AudioRecords (
                audio_id TEXT PRIMARY KEY, composition_id TEXT NOT NULL, audio_path TEXT NOT NULL,
                audio_file_exists INTEGER NOT NULL, recording_title TEXT, singer TEXT, raga TEXT, tala TEXT,
                source_url TEXT,
                FOREIGN KEY(composition_id) REFERENCES Compositions(composition_id)
            );
            CREATE INDEX idx_text_composition ON TextRecords(composition_id);
            CREATE INDEX idx_audio_composition ON AudioRecords(composition_id);
        """)
        compositions.to_sql("Compositions", connection, if_exists="append", index=False)
        texts.to_sql("TextRecords", connection, if_exists="append", index=False)
        audio.to_sql("AudioRecords", connection, if_exists="append", index=False)
        connection.commit()
    finally:
        connection.close()
    return target


def load_sql_data(path: str | Path = DATABASE_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read normalized SQL tables into the DataFrames expected by retrieval."""
    target = initialize_database(path)
    connection = _connect(target)
    try:
        compositions = pd.read_sql_query("SELECT * FROM Compositions ORDER BY composition_id", connection)
        texts = pd.read_sql_query("SELECT * FROM TextRecords ORDER BY text_id", connection)
        audio = pd.read_sql_query("SELECT * FROM AudioRecords ORDER BY audio_id", connection)
    finally:
        connection.close()
    if "audio_file_exists" in audio:
        audio["audio_file_exists"] = audio["audio_file_exists"].astype(bool)
    return compositions, texts, audio


def database_stats(path: str | Path = DATABASE_PATH) -> dict[str, Any]:
    connection = _connect(initialize_database(path))
    try:
        counts = {
            table.lower(): int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("Compositions", "TextRecords", "AudioRecords")
        }
    finally:
        connection.close()
    return counts
