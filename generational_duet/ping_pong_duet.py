"""Create a four-part ping-pong duet from two WAV recordings."""

from pathlib import Path
import io
import subprocess

import imageio_ffmpeg
from pydub import AudioSegment
from pydub.effects import normalize


BASE_DIR = Path(__file__).resolve().parent
track_3_path = str(BASE_DIR / "songs" / "audio001_old.mpeg")
track_4_path = str(BASE_DIR / "songs" / "audio001_new.mp3")
bell_path = str(BASE_DIR / "temple_bell.wav")
crossfade_duration = 5000

track_3_anchors = [65280, 124560, 178810]
track_4_anchors = [57730, 129390, 194440]

OUTPUT_PATH = str(BASE_DIR / "ping_pong_duet.wav")
SAMPLE_RATE = 44100
CHANNELS = 2
TRACK_GAIN_DB = -4.0
BELL_GAIN_DB = -4.0


def load_wav(path: str, label: str) -> AudioSegment:
    """Load and standardize an audio file, raising a readable error on failure."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"{label} file was not found: {file_path.resolve()}")
    try:
        # Decode through the bundled FFmpeg executable. This avoids Pydub's
        # separate ffprobe lookup, which imageio-ffmpeg does not bundle.
        decoded = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v", "error",
                "-i", str(file_path),
                "-f", "wav",
                "-acodec", "pcm_s16le",
                "-ac", str(CHANNELS),
                "-ar", str(SAMPLE_RATE),
                "-",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        audio = AudioSegment.from_file(io.BytesIO(decoded.stdout), format="wav")
    except Exception as exc:
        raise RuntimeError(f"Could not load {label} audio '{file_path}': {exc}") from exc
    if len(audio) == 0:
        raise ValueError(f"{label} WAV contains no audio: {file_path}")
    return audio.set_channels(CHANNELS).set_frame_rate(SAMPLE_RATE).set_sample_width(2)


def validate_anchors(audio: AudioSegment, anchors: list[int], label: str) -> None:
    """Ensure the timestamp anchors are ordered and within their track."""
    if any(isinstance(value, bool) or not isinstance(value, int) for value in anchors):
        raise TypeError(f"All {label} anchors must be integer milliseconds.")
    if anchors != sorted(anchors) or any(value <= 0 for value in anchors):
        raise ValueError(f"{label} anchors must be positive and in ascending order.")
    if anchors[-1] >= len(audio):
        raise ValueError(
            f"Last {label} anchor ({anchors[-1]} ms) must be before the "
            f"track end ({len(audio)} ms)."
        )


def make_slices(track_3: AudioSegment, track_4: AudioSegment) -> list[AudioSegment]:
    """Build the requested four alternating verse slices."""
    validate_anchors(track_3, track_3_anchors, "Track 3")
    validate_anchors(track_4, track_4_anchors, "Track 4")
    slices = [
        track_3[: track_3_anchors[0]],
        track_4[track_4_anchors[0] : track_4_anchors[1]],
        track_3[track_3_anchors[1] : track_3_anchors[2]],
        track_4[track_4_anchors[2] :],
    ]
    for index, audio in enumerate(slices, start=1):
        if len(audio) <= crossfade_duration:
            raise ValueError(
                f"Slice {index} must be longer than the {crossfade_duration} ms crossfade."
            )
    return slices


def create_ping_pong_duet() -> Path:
    """Load sources, alternate slices with bell-masked crossfades, and export."""
    print("Loading audio files...")
    track_3 = load_wav(track_3_path, "Track 3").apply_gain(TRACK_GAIN_DB)
    track_4 = load_wav(track_4_path, "Track 4").apply_gain(TRACK_GAIN_DB)
    bell = load_wav(bell_path, "Temple Bell").apply_gain(BELL_GAIN_DB)

    slices = make_slices(track_3, track_4)
    merged_audio = slices[0]

    for transition_number, next_slice in enumerate(slices[1:], start=1):
        previous_length_ms = len(merged_audio)
        if previous_length_ms <= crossfade_duration:
            raise ValueError("Current merged audio is too short for the requested crossfade.")

        # Pydub overlaps the final crossfade_duration ms of the base with the
        # beginning of next_slice. Save the overlap midpoint before appending.
        overlap_start_ms = previous_length_ms - crossfade_duration
        bell_position_ms = overlap_start_ms + crossfade_duration // 2

        print(f"Transition {transition_number}/3: adding slice {transition_number + 1}...")
        merged_audio = merged_audio.append(next_slice, crossfade=crossfade_duration)
        bell_to_overlay = bell[: len(merged_audio) - bell_position_ms]
        merged_audio = merged_audio.overlay(bell_to_overlay, position=bell_position_ms)
        print(f"Transition {transition_number}/3 complete (bell at {bell_position_ms} ms).")

    print("Peak-normalizing final mix to -0.5 dBFS...")
    if merged_audio.max_dBFS == float("-inf"):
        raise ValueError("The merged audio is silent, so it cannot be peak-normalized.")
    final_audio = normalize(merged_audio, headroom=0.5)
    final_audio = final_audio.set_channels(CHANNELS).set_frame_rate(SAMPLE_RATE).set_sample_width(2)

    output_file = Path(OUTPUT_PATH)
    try:
        final_audio.export(output_file, format="wav")
    except Exception as exc:
        raise RuntimeError(f"Could not export WAV '{output_file}': {exc}") from exc
    print(f"Finished: {output_file.resolve()}")
    return output_file


if __name__ == "__main__":
    try:
        create_ping_pong_duet()
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as error:
        print(f"Could not create ping-pong duet: {error}")
