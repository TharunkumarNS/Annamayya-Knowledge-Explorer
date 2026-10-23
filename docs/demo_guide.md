# Seven-minute demonstration guide

This guide uses the actual local package. It does not assume that optional hosted services are configured.

## Before the demo

From the project root, install `requirements.txt`, then run:

```powershell
python scripts/validate_data.py
python -m unittest -v
python -m streamlit run app.py
```

The first LaBSE use needs model weights available locally or internet access. This project's current environment did not have Streamlit, cached LaBSE weights, a Gemini API key/SDK, or Neo4j configuration, so those workflows were not browser/live-service verified here. If those are unavailable during a presentation, use the backend commands below and describe them as a CLI fallback, not as UI proof.

## 0:00–0:45 — Problem and architecture

Explain that the application organizes a 25-composition dataset into source-aware text and audio records. Text retrieval uses LaBSE; audio retrieval computes actual MFCC-like/chroma features; multimodal results use weighted reciprocal-rank fusion. Neo4j and Gemini are optional enrichments and do not determine the core candidates.

## 0:45–1:45 — Text retrieval

Use a real lyric snippet for `ANN-001` (Adivo Alladivo):

```powershell
python text_retrieval.py "adivō alladivō harivāsamu" --top-k 5
```

The source workbook associates Romanized lyrics with Vignanam and Telugu lyrics with TemplesinIndiaInfo. Inspect the returned text record and cosine score as a similarity—not a probability. If model weights cannot load, show the unit test instead and disclose the missing model runtime; do not show made-up results.

## 1:45–2:45 — Audio retrieval

Use a recording that exists in the package:

```powershell
python audio_retrieval.py "Recordings/audio001.mp3" --top-k 5
```

The current verified exact-file smoke check decoded the MP3, built a one-recording real feature index, and returned `ANN-001` / `R001` at cosine similarity 1.0. Explain that this proves the decode-feature-rank-ID path for an exact recording only; the fixed statistics do not establish cross-performance music recognition.

## 2:45–3:45 — Multimodal retrieval

With LaBSE weights available, search both actual modalities:

```powershell
python multimodal_retrieval.py --text "adivō alladivō harivāsamu" --audio "Recordings/audio001.mp3" --top-k 5
```

Inspect each branch rank/evidence and the fused score. The default weights (`text=0.6`, `audio=0.4`) are starting settings, not tuned accuracy guarantees. If text initialization fails, the app should identify that branch failure; do not call an audio-only fallback “multimodal.”

## 3:45–4:30 — Knowledge graph

If Neo4j is running and credentials are configured, set `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD`; use `python multimodal_retrieval.py --text "..." --import-graph` to import supported dataset relationships, then add `--graph` to retrieve graph enrichment. The graph tests are mocked in this environment; a live relationship should only be demonstrated after an actual connection and query.

Without Neo4j, show the local composition metadata and report that graph enrichment is unavailable. Core search remains local.

## 4:30–5:15 — Gemini explanation

Set `GEMINI_API_KEY` in the launching shell, search first, select an evidence set, then explicitly submit an explanation question in the app. Only selected retrieved evidence and the question are sent to Google's Gemini API. The generated explanation is interpretation, not verified translation or historical fact. Mocked tests cover the client behavior; this environment has no key or SDK, so a live request was not tested.

## 5:15–6:15 — Evaluation

Run:

```powershell
python scripts/evaluate_retrieval.py --prepare --skip-text
```

Open `docs/evaluation_report.md` and the per-query CSV. The 13 audio cases use exact recordings already present in the index; the audio self-match result is intentionally labeled in-sample and cannot be presented as a general accuracy score. LaBSE-dependent rows are explicitly failures when skipped. The package contains no independent, held-out query recordings.

## 6:15–7:00 — Limitations and close

Disclose the one missing audio file (`audio025.mp3`), sparse English translations (10 records), missing tala for 24 compositions, small coverage, audio feature limitations, and lack of independent holdout. Neo4j and Gemini are optional and need live configuration. The app/browser workflow was not exercised in the current environment because Streamlit is not installed.
