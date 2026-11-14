"""Build a multi-transition duet from a database-provided recording pair."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from typing import Any

import imageio_ffmpeg
from pydub import AudioSegment
from pydub.effects import normalize


SAMPLE_RATE = 44_100
CHANNELS = 2
TRACK_GAIN_DB = -4.0
TARGET_HEADROOM_DB = 0.5


def _load_audio(path: Path, label: str) -> AudioSegment:
    if not path.is_file():
        raise FileNotFoundError(f"{label} audio file is missing: {path}")
    try:
        decoded = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path), "-f", "wav",
             "-acodec", "pcm_s16le", "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE), "-"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        audio = AudioSegment.from_file(io.BytesIO(decoded.stdout), format="wav")
    except Exception as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise ValueError(f"Could not decode {label} audio '{path}': {detail}") from exc
    if not len(audio):
        raise ValueError(f"{label} audio file is empty: {path}")
    return audio.set_channels(CHANNELS).set_frame_rate(SAMPLE_RATE).set_sample_width(2)


def _anchors(value: Any, label: str) -> list[int]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} anchors must be a non-empty list of millisecond timestamps.")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{label} anchors must contain integer milliseconds.")
    if value != sorted(value) or value[0] <= 0:
        raise ValueError(f"{label} anchors must be positive and in ascending order.")
    return value


def _make_slices(vintage: AudioSegment, modern: AudioSegment,
                 vintage_anchors: list[int], modern_anchors: list[int],
                 crossfade_ms: int) -> list[AudioSegment]:
    if len(vintage_anchors) != 3 or len(modern_anchors) != 3:
        raise ValueError("Each recording must have exactly three verse-boundary anchors.")
    for label, audio, anchors in (("Vintage", vintage, vintage_anchors), ("Modern", modern, modern_anchors)):
        if anchors[-1] >= len(audio):
            raise ValueError(f"{label} final anchor ({anchors[-1]} ms) must be before its duration ({len(audio)} ms).")
    slices = [
        vintage[:vintage_anchors[0]],
        modern[modern_anchors[0]:modern_anchors[1]],
        vintage[vintage_anchors[1]:vintage_anchors[2]],
        modern[modern_anchors[2]:],
    ]
    for index, segment in enumerate(slices, 1):
        if len(segment) <= crossfade_ms:
            raise ValueError(f"Duet section {index} must be longer than the {crossfade_ms} ms crossfade.")
    return slices


def create_duet_audio(pair: dict[str, Any], crossfade_ms: int | None = None,
                      bell_gain_db: float = -4.0) -> bytes:
    """Create and return a normalized stereo 44.1 kHz 16-bit PCM WAV payload."""
    if crossfade_ms is None:
        crossfade_ms = int(pair["crossfade_duration_ms"])
    if isinstance(crossfade_ms, bool) or not isinstance(crossfade_ms, int) or crossfade_ms <= 0:
        raise ValueError("Crossfade duration must be a positive integer number of milliseconds.")
    try:
        bell_gain_db = float(bell_gain_db)
    except (TypeError, ValueError) as exc:
        raise ValueError("Bell gain must be a number of decibels.") from exc

    vintage = _load_audio(Path(pair["vintage_audio_path"]), "Vintage").apply_gain(TRACK_GAIN_DB)
    modern = _load_audio(Path(pair["modern_audio_path"]), "Modern").apply_gain(TRACK_GAIN_DB)
    bell = _load_audio(Path(pair["bell_audio_path"]), "Temple Bell").apply_gain(bell_gain_db)
    slices = _make_slices(vintage, modern, _anchors(pair["vintage_anchors_ms"], "Vintage"),
                          _anchors(pair["modern_anchors_ms"], "Modern"), crossfade_ms)

    merged = slices[0]
    for next_slice in slices[1:]:
        prior_length = len(merged)
        midpoint_ms = prior_length - crossfade_ms + crossfade_ms // 2
        merged = merged.append(next_slice, crossfade=crossfade_ms)
        bell_clip = bell[:len(merged) - midpoint_ms]
        merged = merged.overlay(bell_clip, position=midpoint_ms)

    if merged.max_dBFS == float("-inf"):
        raise ValueError("The merged audio is silent and cannot be normalized.")
    final_audio = normalize(merged, headroom=TARGET_HEADROOM_DB)
    buffer = io.BytesIO()
    final_audio.set_channels(CHANNELS).set_frame_rate(SAMPLE_RATE).set_sample_width(2).export(buffer, format="wav")
    return buffer.getvalue()
