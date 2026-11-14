# Generational Duet

This standalone module loads two recordings, converts both to stereo at 44.1 kHz, applies -4 dB gain, optionally high-pass-filters the vintage track, extracts the anchored slices, crossfades them, and overlays a Temple Bell WAV at the overlap midpoint. The main `create_generational_duet()` function then peak-normalizes the complete mix to -0.5 dBFS and exports stereo, 44.1 kHz, 16-bit PCM WAV.

## Setup

Install dependencies with `python -m pip install -r requirements.txt`. Pydub needs FFmpeg available on `PATH` to decode MP3 and many other compressed formats. WAV input may work without FFmpeg.

A short, synthesized 4-second bell ding is included at `assets/temple_bell.wav` for the example. Replace it with a real temple bell recording if desired.

## Complete runnable example

Update the example paths and timestamps at the bottom of `generational_duet.py`, then run `python generational_duet.py` from this directory. The equivalent function call is:

```python
from generational_duet import create_generational_duet

final_audio, output_file = create_generational_duet(
    vintage_audio_path="songs/vintage.mp3",
    modern_audio_path="songs/modern.mp3",
    vintage_pallavi_end_ms=30_000,
    modern_pallavi_end_ms=28_500,
    temple_bell_audio_path="assets/temple_bell.wav",
    crossfade_duration_ms=5_000,
    bell_gain_db=-2.0,
    output_path="generational_duet.wav",
)
print(f"Saved {output_file}")
```

The function returns `(final_audio, output_file)`: the mastered `pydub.AudioSegment` and the path to the exported WAV. Output defaults to `generational_duet.wav` in the current directory.

Peak normalization applies one gain adjustment to the completed mix so its loudest peak reaches -0.5 dBFS. This may raise or lower overall loudness while preserving the bell-to-music level relationship; it does not compress or independently rebalance either source.

The bell WAV is assumed to begin with the strike. It is positioned at the overlap midpoint and trimmed at the merged audio's end so its reverb can continue after the crossfade without extending the result duration.

## Separate pair catalog

`data/duet_pairs.sqlite3` is a separate SQLite catalog for merge-ready pairs; it is not connected to the main song database. The `audio_pairs` table currently contains the Adivo old/new pair, its track paths, anchor arrays, bell path, and crossfade duration. Paths are stored relative to this feature's `data` directory. The source recordings live under `songs/`, and the bell WAV lives in this folder, so all merge assets are inside the project.
