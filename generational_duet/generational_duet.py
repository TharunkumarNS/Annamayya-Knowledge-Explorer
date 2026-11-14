"""Create, master, and export a vintage/modern duet with a Temple Bell overlay.

Pydub requires FFmpeg (or avlib) to decode formats such as MP3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.exceptions import CouldntDecodeError
from pydub.silence import detect_silences


AudioPath = Union[str, Path]
TARGET_CHANNELS = 2
TARGET_SAMPLE_RATE = 44_100
GAIN_REDUCTION_DB = -4.0
DEFAULT_CROSSFADE_MS = 5_000
TARGET_PEAK_DBFS = -0.5


class AudioLoadError(RuntimeError):
    """Raised when an existing audio file cannot be decoded."""


def validate_file_path(file_path: AudioPath, track_name: str) -> Path:
    """Check that a path points to an existing file and return it as a Path."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{track_name} audio file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{track_name} audio path is not a file: {path}")
    return path


def find_pallavi_end_automatically(audio_path: AudioPath) -> int | None:
    """Return the start of the first qualifying silence in the 1–2.5 min window.

    This follows the requested silence heuristic; it does not understand lyrics
    or musical form. Returns ``None`` if the window contains no detected silence.
    """
    audio = AudioSegment.from_wav(audio_path)
    search_window = audio[60_000:150_000]
    if len(search_window) == 0:
        return None

    silences = detect_silences(
        search_window,
        min_silence_len=800,
        silence_thresh=-35,
    )
    if not silences:
        return None

    first_silence_start_ms = silences[0][0]
    return int(60_000 + first_silence_start_ms)


def load_audio(file_path: AudioPath, track_name: str = "Audio") -> AudioSegment:
    """Load an audio file with pydub, reporting decoding problems clearly."""
    path = validate_file_path(file_path, track_name)
    try:
        segment = AudioSegment.from_file(path)
    except (CouldntDecodeError, OSError, FileNotFoundError) as exc:
        raise AudioLoadError(
            f"Could not load {track_name.lower()} audio from '{path}'. "
            "Check that the file is valid and that FFmpeg is installed for this format."
        ) from exc
    except Exception as exc:
        raise AudioLoadError(f"Could not load {track_name.lower()} audio from '{path}': {exc}") from exc

    if len(segment) <= 0:
        raise AudioLoadError(f"{track_name} audio file contains no audio: {path}")
    return segment


def validate_pallavi_timestamp(
    pallavi_end_ms: int,
    duration_ms: int,
    track_name: str,
    *,
    allow_zero: bool,
) -> None:
    """Ensure a pallavi anchor is an integer and lies within the track."""
    if isinstance(pallavi_end_ms, bool) or not isinstance(pallavi_end_ms, int):
        raise TypeError(f"{track_name} pallavi_end_ms must be an integer number of milliseconds.")
    minimum = 0 if allow_zero else 1
    if pallavi_end_ms < minimum:
        qualifier = "0 or greater" if allow_zero else "greater than 0"
        raise ValueError(f"{track_name} pallavi_end_ms must be {qualifier}.")
    if pallavi_end_ms > duration_ms:
        raise ValueError(
            f"{track_name} pallavi_end_ms ({pallavi_end_ms} ms) exceeds "
            f"the audio duration ({duration_ms} ms)."
        )
    if not allow_zero and pallavi_end_ms == duration_ms:
        raise ValueError(
            f"{track_name} pallavi_end_ms must be before the end of the audio "
            "so the modern slice contains audio."
        )


def prepare_audio(audio: AudioSegment, gain_db: float = GAIN_REDUCTION_DB) -> AudioSegment:
    """Convert audio to stereo/44.1 kHz and lower its level by gain_db."""
    return (
        audio.set_channels(TARGET_CHANNELS)
        .set_frame_rate(TARGET_SAMPLE_RATE)
        .set_sample_width(2)
        .apply_gain(gain_db)
    )


def prepare_tracks(
    vintage_audio: AudioSegment,
    modern_audio: AudioSegment,
    *,
    vintage_high_pass_hz: int | None = None,
) -> tuple[AudioSegment, AudioSegment]:
    """Standardize both tracks and optionally high-pass-filter the vintage one."""
    if vintage_high_pass_hz is not None:
        if isinstance(vintage_high_pass_hz, bool) or not isinstance(vintage_high_pass_hz, int):
            raise TypeError("vintage_high_pass_hz must be an integer or None.")
        if vintage_high_pass_hz <= 0 or vintage_high_pass_hz >= TARGET_SAMPLE_RATE // 2:
            raise ValueError("vintage_high_pass_hz must be between 0 and the Nyquist frequency (22,050 Hz).")

    vintage_prepared = prepare_audio(vintage_audio)
    modern_prepared = prepare_audio(modern_audio)
    if vintage_high_pass_hz is not None:
        vintage_prepared = vintage_prepared.high_pass_filter(vintage_high_pass_hz)
    return vintage_prepared, modern_prepared


def slice_tracks(
    vintage_audio: AudioSegment,
    modern_audio: AudioSegment,
    vintage_pallavi_end_ms: int,
    modern_pallavi_end_ms: int,
) -> tuple[AudioSegment, AudioSegment]:
    """Return vintage intro-through-pallavi and modern audio after its pallavi."""
    validate_pallavi_timestamp(
        vintage_pallavi_end_ms, len(vintage_audio), "Vintage", allow_zero=False
    )
    validate_pallavi_timestamp(
        modern_pallavi_end_ms, len(modern_audio), "Modern", allow_zero=True
    )
    vintage_slice = vintage_audio[:vintage_pallavi_end_ms]
    modern_slice = modern_audio[modern_pallavi_end_ms:]
    return vintage_slice, modern_slice


def prepare_generational_duet(
    vintage_file_path: AudioPath,
    modern_file_path: AudioPath,
    vintage_pallavi_end_ms: int,
    modern_pallavi_end_ms: int,
    *,
    vintage_high_pass_hz: int | None = None,
) -> tuple[AudioSegment, AudioSegment]:
    """Load, validate, prepare, and slice two recordings for later merging.

    Returns ``(vintage_slice, modern_slice)``. No mixing or crossfade is done.
    """
    vintage_path = validate_file_path(vintage_file_path, "Vintage")
    modern_path = validate_file_path(modern_file_path, "Modern")
    vintage_loaded = load_audio(vintage_path, "Vintage")
    modern_loaded = load_audio(modern_path, "Modern")

    # Validate anchors against the original source durations before processing.
    validate_pallavi_timestamp(
        vintage_pallavi_end_ms, len(vintage_loaded), "Vintage", allow_zero=False
    )
    validate_pallavi_timestamp(
        modern_pallavi_end_ms, len(modern_loaded), "Modern", allow_zero=True
    )

    vintage_prepared, modern_prepared = prepare_tracks(
        vintage_loaded,
        modern_loaded,
        vintage_high_pass_hz=vintage_high_pass_hz,
    )
    return slice_tracks(
        vintage_prepared,
        modern_prepared,
        vintage_pallavi_end_ms,
        modern_pallavi_end_ms,
    )


def _validate_prepared_pair(vintage: AudioSegment, modern: AudioSegment) -> None:
    """Ensure input slices use the shared format expected by the merger."""
    for name, audio in (("Vintage", vintage), ("Modern", modern)):
        if len(audio) <= 0:
            raise ValueError(f"{name} slice must contain audio.")
        if audio.channels != TARGET_CHANNELS or audio.frame_rate != TARGET_SAMPLE_RATE:
            raise ValueError(
                f"{name} slice must be stereo at {TARGET_SAMPLE_RATE} Hz; "
                f"got {audio.channels} channel(s) at {audio.frame_rate} Hz."
            )


def crossfade_slices(
    vintage_slice: AudioSegment,
    modern_slice: AudioSegment,
    crossfade_duration_ms: int = DEFAULT_CROSSFADE_MS,
) -> AudioSegment:
    """Join slices with an exact-length linear-amplitude overlap.

    The overlap replaces the end of the vintage slice and beginning of the
    modern slice, so output duration is ``len(vintage) + len(modern) - overlap``.
    """
    if isinstance(crossfade_duration_ms, bool) or not isinstance(crossfade_duration_ms, int):
        raise TypeError("Crossfade duration must be an integer number of milliseconds.")
    if crossfade_duration_ms <= 0:
        raise ValueError("Crossfade duration must be positive.")
    _validate_prepared_pair(vintage_slice, modern_slice)
    if len(vintage_slice) <= crossfade_duration_ms:
        raise ValueError("Vintage slice must be longer than the crossfade duration.")
    if len(modern_slice) <= crossfade_duration_ms:
        raise ValueError("Modern slice must be longer than the crossfade duration.")

    # Pydub uses millisecond slicing. Convert the requested window to its exact
    # number of frames at 44.1 kHz, then reuse that same window for both tracks.
    overlap_frames = round(crossfade_duration_ms * TARGET_SAMPLE_RATE / 1000)
    samples_per_frame = TARGET_CHANNELS
    vintage_samples = np.frombuffer(vintage_slice.set_sample_width(2).raw_data, dtype=np.int16).astype(np.float32)
    modern_samples = np.frombuffer(modern_slice.set_sample_width(2).raw_data, dtype=np.int16).astype(np.float32)
    overlap_samples = overlap_frames * samples_per_frame

    vintage_prefix = vintage_samples[:-overlap_samples]
    modern_tail = modern_samples[overlap_samples:]
    vintage_overlap = vintage_samples[-overlap_samples:].reshape(overlap_frames, samples_per_frame)
    modern_overlap = modern_samples[:overlap_samples].reshape(overlap_frames, samples_per_frame)

    # Linear amplitude ramps: vintage is 1→0 while modern is 0→1.
    vintage_gain = np.linspace(1.0, 0.0, overlap_frames, endpoint=True, dtype=np.float32)[:, None]
    modern_gain = np.linspace(0.0, 1.0, overlap_frames, endpoint=True, dtype=np.float32)[:, None]
    mixed_overlap = vintage_overlap * vintage_gain + modern_overlap * modern_gain
    mixed_overlap = np.clip(np.rint(mixed_overlap), -32768, 32767).astype(np.int16).reshape(-1)

    output_samples = np.concatenate((vintage_prefix.astype(np.int16), mixed_overlap, modern_tail.astype(np.int16)))
    return AudioSegment(
        data=output_samples.tobytes(),
        sample_width=2,
        frame_rate=TARGET_SAMPLE_RATE,
        channels=TARGET_CHANNELS,
    )


def load_temple_bell(bell_wav_path: AudioPath, bell_gain_db: float = 0.0) -> AudioSegment:
    """Load a WAV bell and standardize it to stereo 44.1 kHz."""
    if not isinstance(bell_gain_db, (int, float)) or isinstance(bell_gain_db, bool):
        raise TypeError("Bell gain adjustment must be a number of decibels.")
    path = validate_file_path(bell_wav_path, "Temple Bell")
    try:
        bell = AudioSegment.from_wav(path)
    except Exception as exc:
        raise AudioLoadError(f"Could not load Temple Bell WAV from '{path}': {exc}") from exc
    if len(bell) <= 0:
        raise AudioLoadError(f"Temple Bell WAV contains no audio: {path}")
    return (
        bell.set_channels(TARGET_CHANNELS)
        .set_frame_rate(TARGET_SAMPLE_RATE)
        .set_sample_width(2)
        .apply_gain(float(bell_gain_db))
    )


def overlay_temple_bell(
    merged_music: AudioSegment,
    bell_wav_path: AudioPath,
    strike_position_ms: int,
    *,
    bell_gain_db: float = 0.0,
) -> AudioSegment:
    """Overlay a bell whose file begins with the strike at the overlap midpoint.

    If the bell's reverberation would pass the music's end, it is trimmed to
    the remaining duration so the overlay never lengthens the music.
    """
    if isinstance(strike_position_ms, bool) or not isinstance(strike_position_ms, int):
        raise TypeError("Bell strike position must be an integer number of milliseconds.")
    if strike_position_ms < 0:
        raise ValueError("Bell strike position cannot be negative.")
    if len(merged_music) <= 0:
        raise ValueError("Merged music must contain audio.")
    if merged_music.channels != TARGET_CHANNELS or merged_music.frame_rate != TARGET_SAMPLE_RATE:
        raise ValueError(f"Merged music must be stereo at {TARGET_SAMPLE_RATE} Hz.")

    if strike_position_ms >= len(merged_music):
        raise ValueError("Temple Bell strike position falls outside the merged audio.")
    bell = load_temple_bell(bell_wav_path, bell_gain_db)
    # Keep the bell (including as much of its reverb as possible) within the
    # merged result; pydub's overlay alone would also cap output to music length.
    bell = bell[: len(merged_music) - strike_position_ms]
    return merged_music.overlay(bell, position=strike_position_ms)


def merge_with_temple_bell(
    vintage_slice: AudioSegment,
    modern_slice: AudioSegment,
    bell_wav_path: AudioPath,
    crossfade_duration_ms: int = DEFAULT_CROSSFADE_MS,
    *,
    bell_gain_db: float = 0.0,
) -> AudioSegment:
    """Crossfade the prepared slices and overlay the bell at the midpoint."""
    music = crossfade_slices(vintage_slice, modern_slice, crossfade_duration_ms)
    overlap_start_ms = len(vintage_slice) - crossfade_duration_ms
    midpoint_ms = overlap_start_ms + crossfade_duration_ms // 2
    return overlay_temple_bell(
        music,
        bell_wav_path,
        midpoint_ms,
        bell_gain_db=bell_gain_db,
    )


def normalize_peak(audio: AudioSegment, target_peak_dbfs: float = TARGET_PEAK_DBFS) -> AudioSegment:
    """Apply one global gain change so the signal peak reaches target_peak_dbfs.

    Applying the same gain to the whole mix raises or lowers the music and bell
    together, preserving their relative balance.
    """
    if len(audio) <= 0:
        raise ValueError("Cannot normalize empty audio.")
    peak_dbfs = audio.max_dBFS
    if peak_dbfs == float("-inf"):
        raise ValueError("Cannot normalize silent audio; the merged recording contains no signal.")
    return normalize(audio, headroom=-target_peak_dbfs)


def export_wav(audio: AudioSegment, output_path: AudioPath = "generational_duet.wav") -> Path:
    """Export stereo 44.1 kHz 16-bit PCM WAV and return its path."""
    path = Path(output_path)
    if path.suffix.lower() != ".wav":
        raise ValueError(f"Output filename must end in .wav: {path}")
    if path.exists() and path.is_dir():
        raise ValueError(f"Output path points to a directory, not a WAV file: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        wav_audio = (
            audio.set_channels(TARGET_CHANNELS)
            .set_frame_rate(TARGET_SAMPLE_RATE)
            .set_sample_width(2)
        )
        wav_audio.export(path, format="wav")
    except Exception as exc:
        raise OSError(f"Could not export the Generational Duet WAV to '{path}': {exc}") from exc
    return path


def create_generational_duet(
    vintage_audio_path: AudioPath,
    modern_audio_path: AudioPath,
    vintage_pallavi_end_ms: int,
    modern_pallavi_end_ms: int,
    temple_bell_audio_path: AudioPath,
    crossfade_duration_ms: int = DEFAULT_CROSSFADE_MS,
    bell_gain_db: float = 0.0,
    output_path: AudioPath = "generational_duet.wav",
) -> tuple[AudioSegment, Path]:
    """Run all processing stages, normalize the finished mix, and export WAV."""
    vintage_slice, modern_slice = prepare_generational_duet(
        vintage_file_path=vintage_audio_path,
        modern_file_path=modern_audio_path,
        vintage_pallavi_end_ms=vintage_pallavi_end_ms,
        modern_pallavi_end_ms=modern_pallavi_end_ms,
    )
    merged = merge_with_temple_bell(
        vintage_slice,
        modern_slice,
        bell_wav_path=temple_bell_audio_path,
        crossfade_duration_ms=crossfade_duration_ms,
        bell_gain_db=bell_gain_db,
    )
    mastered = normalize_peak(merged, TARGET_PEAK_DBFS)
    exported_path = export_wav(mastered, output_path)
    return mastered, exported_path


if __name__ == "__main__":
    detected_end_ms = find_pallavi_end_automatically("test_track.wav")
    if detected_end_ms is None:
        print("No qualifying silence was found in the 1:00–2:30 search window.")
    else:
        print(f"Estimated pallavi_end_ms: {detected_end_ms}")
