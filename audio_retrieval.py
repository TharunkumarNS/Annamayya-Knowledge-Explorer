"""Audio similarity retrieval for the Annamayya recording package.

Each file is decoded by SoundFile to mono 22.05 kHz audio. NumPy computes the
MFCC-like mel cepstral and chroma frame features; means and standard deviations
produce one fixed 64-value vector per recording. L2-normalized vectors are
ranked by cosine similarity (higher is closer). This baseline describes timbre
and harmony, not singing transcription or guaranteed alternate-performance
recognition. NumPy feature calculations avoid a SciPy/Librosa import stall in
the current environment while remaining compatible with the Phase 3 baseline.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from data_loader import DATASET_DIR, PROJECT_ROOT, load_normalized_data


SAMPLE_RATE = 22_050
N_FFT = 2_048
HOP_LENGTH = 512
N_MFCC = 20
N_CHROMA = 12
FEATURE_DIMENSION = N_MFCC * 2 + N_CHROMA * 2
REQUIRED_AUDIO_COLUMNS = {"audio_id", "composition_id", "audio_path"}
REQUIRED_COMPOSITION_COLUMNS = {"composition_id", "canonical_title"}


def _load_soundfile():
    """Import SoundFile lazily, so the module remains importable without audio extras."""
    try:
        import soundfile
    except ImportError as exc:
        raise RuntimeError(
            "SoundFile is required for audio decoding. "
            "Install them with: python -m pip install -r requirements.txt"
        ) from exc
    return soundfile


def _hz_to_mel(frequency: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def _mel_filterbank(sample_rate: int, bins: int, filters: int = 40) -> np.ndarray:
    """Build triangular mel filters for an rFFT power spectrum."""
    mel_points = np.linspace(_hz_to_mel(0.0), _hz_to_mel(sample_rate / 2), filters + 2)
    hz_points = 700.0 * (10 ** (mel_points / 2595.0) - 1.0)
    fft_frequencies = np.linspace(0.0, sample_rate / 2, bins)
    bank = np.zeros((filters, bins), dtype=np.float32)
    for index in range(filters):
        left, center, right = hz_points[index : index + 3]
        bank[index] = np.maximum(
            0.0,
            np.minimum(
                (fft_frequencies - left) / max(center - left, 1e-12),
                (right - fft_frequencies) / max(right - center, 1e-12),
            ),
        )
    return bank


def _resample_mono(waveform: np.ndarray, source_rate: int) -> np.ndarray:
    """Convert mono waveform to SAMPLE_RATE using block averaging or interpolation."""
    if source_rate == SAMPLE_RATE:
        return waveform.astype(np.float32, copy=False)
    if source_rate % SAMPLE_RATE == 0:
        factor = source_rate // SAMPLE_RATE
        usable_length = len(waveform) - len(waveform) % factor
        if usable_length:
            return waveform[:usable_length].reshape(-1, factor).mean(axis=1).astype(np.float32)
    output_length = max(1, round(len(waveform) * SAMPLE_RATE / source_rate))
    source_positions = np.arange(len(waveform), dtype=np.float64)
    target_positions = np.arange(output_length, dtype=np.float64) * source_rate / SAMPLE_RATE
    return np.interp(target_positions, source_positions, waveform).astype(np.float32)


def extract_audio_features(audio_path: str | Path) -> np.ndarray:
    """Decode one recording and return its normalized, fixed-length feature vector."""
    soundfile = _load_soundfile()
    path = Path(audio_path)
    try:
        waveform, source_rate = soundfile.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError(f"Could not decode audio file '{path}': {exc}") from exc
    if waveform.size:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    waveform = _resample_mono(waveform, int(source_rate))
    if waveform.size < N_FFT or not np.isfinite(waveform).all():
        raise ValueError(f"Audio file '{path}' is empty, too short, or contains invalid samples.")
    if not np.any(waveform):
        raise ValueError(f"Audio file '{path}' contains only silence.")

    try:
        frames = np.lib.stride_tricks.sliding_window_view(waveform, N_FFT)[::HOP_LENGTH]
        window = np.hanning(N_FFT).astype(np.float32)
        mel_bank = _mel_filterbank(SAMPLE_RATE, N_FFT // 2 + 1)
        dct = np.cos(
            (np.pi / 40.0)
            * (np.arange(40, dtype=np.float32) + 0.5)[None, :]
            * np.arange(N_MFCC, dtype=np.float32)[:, None]
        ).astype(np.float32)
        dct[0] *= np.sqrt(1.0 / 40.0)
        dct[1:] *= np.sqrt(2.0 / 40.0)
        frequencies = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE)
        pitch_classes = np.full(frequencies.shape, -1, dtype=np.int8)
        positive = frequencies > 0
        midi_notes = np.rint(69 + 12 * np.log2(frequencies[positive] / 440.0)).astype(int)
        pitch_classes[positive] = np.mod(midi_notes, N_CHROMA)
        mfcc_batches: list[np.ndarray] = []
        chroma_batches: list[np.ndarray] = []
        for start in range(0, len(frames), 512):
            frame_batch = frames[start : start + 512] * window
            power = (np.abs(np.fft.rfft(frame_batch, axis=1)) ** 2 / N_FFT).astype(np.float32).T
            log_mel = np.log(np.maximum(mel_bank @ power, 1e-10))
            mfcc_batches.append(dct @ log_mel)
            chroma = np.zeros((N_CHROMA, power.shape[1]), dtype=np.float32)
            for pitch_class in range(N_CHROMA):
                selected = pitch_classes == pitch_class
                if selected.any():
                    chroma[pitch_class] = power[selected].sum(axis=0)
            chroma /= np.maximum(chroma.sum(axis=0, keepdims=True), 1e-10)
            chroma_batches.append(chroma)
        mfcc = np.concatenate(mfcc_batches, axis=1)
        chroma = np.concatenate(chroma_batches, axis=1)
        vector = np.concatenate(
            [mfcc.mean(axis=1), mfcc.std(axis=1), chroma.mean(axis=1), chroma.std(axis=1)]
        ).astype(np.float32)
    except Exception as exc:
        raise ValueError(f"Could not extract audio features from '{path}': {exc}") from exc
    norm = float(np.linalg.norm(vector))
    if vector.shape != (FEATURE_DIMENSION,) or not np.isfinite(vector).all() or norm == 0:
        raise ValueError(f"Audio feature extraction produced an invalid vector for '{path}'.")
    return vector / norm


@dataclass
class IndexStats:
    """Counts and diagnostics for the latest index build."""

    encountered: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    skipped_audio_ids: list[str] | None = None
    failures: dict[str, str] | None = None


class AudioRetrievalEngine:
    """Build an in-memory audio-feature index and retrieve similar recordings."""

    def __init__(
        self,
        audio_records: pd.DataFrame,
        compositions: pd.DataFrame | None = None,
        *,
        dataset_dir: str | Path = DATASET_DIR,
        feature_extractor: Callable[[str | Path], np.ndarray] | None = None,
    ) -> None:
        missing = sorted(REQUIRED_AUDIO_COLUMNS - set(audio_records.columns))
        if missing:
            raise ValueError("AudioRecords is missing required column(s): " + ", ".join(missing))
        if audio_records["audio_id"].duplicated().any():
            raise ValueError("AudioRecords must contain unique audio_id values.")
        if compositions is not None:
            missing_compositions = sorted(REQUIRED_COMPOSITION_COLUMNS - set(compositions.columns))
            if missing_compositions:
                raise ValueError(
                    "Compositions is missing required column(s): " + ", ".join(missing_compositions)
                )
            if compositions["composition_id"].duplicated().any():
                raise ValueError("Compositions must contain unique composition_id values.")
        self._audio_records = audio_records.copy().reset_index(drop=True)
        self._compositions = (
            compositions.drop_duplicates("composition_id").set_index("composition_id")
            if compositions is not None
            else pd.DataFrame()
        )
        self.dataset_dir = Path(dataset_dir).resolve()
        self._feature_extractor = feature_extractor or extract_audio_features
        self._features: dict[str, np.ndarray] = {}
        self._indexed_records: dict[str, dict[str, Any]] = {}
        self._stats = IndexStats(skipped_audio_ids=[], failures={})
        self._built = False

    def _resolve_path(self, value: object) -> Path | None:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        path = Path(text)
        if path.is_absolute():
            return path
        candidates = (self.dataset_dir / path, PROJECT_ROOT / path)
        return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])

    def build_index(self) -> dict[str, Any]:
        """Extract and cache features for every decodable recording in AudioRecords."""
        self._features.clear()
        self._indexed_records.clear()
        self._stats = IndexStats(encountered=len(self._audio_records), skipped_audio_ids=[], failures={})

        known_compositions = set(self._compositions.index) if not self._compositions.empty else None
        for _, row in self._audio_records.iterrows():
            audio_id_value = row["audio_id"]
            composition_value = row["composition_id"]
            audio_id = "" if audio_id_value is None or pd.isna(audio_id_value) else str(audio_id_value).strip()
            composition_id = (
                "" if composition_value is None or pd.isna(composition_value)
                else str(composition_value).strip()
            )
            path = self._resolve_path(row["audio_path"])
            if not audio_id or not composition_id:
                self._stats.failed += 1
                self._stats.failures[audio_id or "<missing audio_id>"] = "Missing audio_id or composition_id."
                continue
            if known_compositions is not None and composition_id not in known_compositions:
                self._stats.failed += 1
                self._stats.failures[audio_id] = f"Unknown composition_id: {composition_id}"
                continue
            if path is None or not path.is_file():
                self._stats.skipped += 1
                self._stats.skipped_audio_ids.append(audio_id)
                continue
            try:
                features = np.asarray(self._feature_extractor(path), dtype=np.float32).reshape(-1)
                if features.shape != (FEATURE_DIMENSION,) or not np.isfinite(features).all():
                    raise ValueError(
                        f"Feature extractor must return {FEATURE_DIMENSION} finite values."
                    )
                norm = float(np.linalg.norm(features))
                if norm == 0:
                    raise ValueError("Feature extractor returned a zero vector.")
                self._features[audio_id] = features / norm
                self._indexed_records[audio_id] = {
                    **row.to_dict(),
                    "resolved_audio_path": str(path),
                    "composition_id": composition_id,
                }
                self._stats.indexed += 1
            except Exception as exc:
                self._stats.failed += 1
                self._stats.failures[audio_id] = str(exc)

        self._built = True
        return self.get_index_stats()

    def get_index_stats(self) -> dict[str, Any]:
        """Return counts of encountered, indexed, skipped, and failed recordings."""
        return asdict(self._stats)

    def search_audio(self, query_audio_path: str | Path, top_k: int = 10) -> list[dict[str, Any]]:
        """Rank distinct compositions by their best recording's cosine similarity."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if query_audio_path is None or pd.isna(query_audio_path) or not str(query_audio_path).strip():
            raise ValueError("query_audio_path must identify a non-empty audio path.")
        query_path = Path(query_audio_path)
        if not query_path.is_absolute():
            resolved = self._resolve_path(query_audio_path)
            query_path = resolved if resolved is not None else query_path
        if not query_path.is_file():
            raise FileNotFoundError(f"Query audio file was not found: {query_path}")
        if not self._built:
            self.build_index()
        if not self._features:
            return []

        query_features = np.asarray(self._feature_extractor(query_path), dtype=np.float32).reshape(-1)
        if query_features.shape != (FEATURE_DIMENSION,) or not np.isfinite(query_features).all():
            raise ValueError(f"Query audio must produce {FEATURE_DIMENSION} finite features.")
        query_norm = float(np.linalg.norm(query_features))
        if query_norm == 0:
            raise ValueError("Query audio produced a zero feature vector.")
        query_features /= query_norm

        recording_results: list[dict[str, Any]] = []
        for audio_id, indexed in self._indexed_records.items():
            candidate_features = self._features[audio_id]
            if candidate_features.shape != query_features.shape:
                continue
            score = float(np.dot(query_features, candidate_features))
            if np.isfinite(score):
                recording_results.append(
                    {
                        "audio_id": audio_id,
                        "composition_id": indexed["composition_id"],
                        "score": score,
                        "audio_path": indexed["audio_path"],
                        "resolved_audio_path": indexed["resolved_audio_path"],
                        "metadata": {
                            key: indexed.get(key)
                            for key in ("recording_title", "singer", "raga", "tala", "source_url")
                        },
                    }
                )
        recording_results.sort(key=lambda item: (-item["score"], item["audio_id"]))

        by_composition: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in recording_results:
            by_composition[result["composition_id"]].append(result)

        composition_results: list[dict[str, Any]] = []
        for composition_id, matches in by_composition.items():
            best = matches[0]
            composition_metadata = (
                self._compositions.loc[composition_id].to_dict()
                if not self._compositions.empty
                else {}
            )
            composition_results.append(
                {
                    "composition_id": composition_id,
                    "canonical_title": composition_metadata.get("canonical_title"),
                    "score": best["score"],
                    "score_type": "cosine_similarity",
                    "best_matching_audio_id": best["audio_id"],
                    "composition_metadata": composition_metadata,
                    "recording_evidence": matches,
                }
            )
        composition_results.sort(key=lambda item: (-item["score"], item["composition_id"]))
        return composition_results[:top_k]


def _print_results(results: list[dict[str, Any]]) -> None:
    if not results:
        print("No indexed recordings are available for comparison.")
        return
    for position, result in enumerate(results, start=1):
        evidence = result["recording_evidence"][0]
        print(
            f"{position}. {result['canonical_title'] or result['composition_id']} "
            f"({result['composition_id']}) | cosine similarity {result['score']:.4f} "
            f"| audio {evidence['audio_id']}"
        )


def main() -> None:
    """Run the audio engine against the packaged recordings."""
    parser = argparse.ArgumentParser(description="Retrieve Annamayya compositions by audio similarity.")
    parser.add_argument("query_audio", nargs="?", help="Query recording path")
    parser.add_argument("--top-k", type=int, default=10)
    arguments = parser.parse_args()

    compositions, _, audio_records = load_normalized_data()
    engine = AudioRetrievalEngine(audio_records, compositions=compositions)
    stats = engine.build_index()
    print("Index statistics:", stats)
    if not arguments.query_audio:
        parser.error("provide a query audio path, for example Recordings/audio001.mp3")
    results = engine.search_audio(arguments.query_audio, top_k=arguments.top_k)
    _print_results(results)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"Audio retrieval could not run: {exc}") from exc
