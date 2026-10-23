# Multimodal Annamayya Knowledge Explorer

## Phase 1: dataset normalization

`data_loader.py` reads `annamayya_dataset_package/Annamayya_25_Song_Dataset.xlsx` and produces normalized Compositions, TextRecords, and AudioRecords DataFrames.

Run:

```powershell
python data_loader.py
```

## Phase 2: semantic text retrieval

`text_retrieval.py` reuses `load_normalized_data()` from Phase 1. It embeds usable TextRecords with `sentence-transformers/LaBSE`, L2-normalizes the vectors, and calculates cosine similarity with the query. Text-level matches are grouped by `composition_id`; the highest matching text score becomes the composition score. Results include the matching text record as evidence.

Optional `--language` and `--text-type` filters apply before ranking. The engine keeps embeddings only in memory for the current process, so it does not use a persistent embedding cache.

Install dependencies. The first semantic run downloads LaBSE and therefore requires internet access:

```powershell
python -m pip install -r requirements.txt
```

Run a search:

```powershell
python text_retrieval.py "sacred abode of Sri Hari" --top-k 5
python text_retrieval.py "అదివో అల్లదివో" --language Telugu --text-type lyrics
```

Run the unit tests:

```powershell
python -m unittest -v test_text_retrieval.py
```

The unit tests use a deterministic mock encoder to test indexing, parent mapping, filter behavior, ranking, and validation without downloading a model. A successful command-line run uses actual LaBSE embeddings.

## Phase 3: audio similarity retrieval

`audio_retrieval.py` decodes recordings through SoundFile to mono 22.05 kHz audio and uses NumPy to calculate fixed-size MFCC-like mel cepstral and chroma frame statistics. This avoids Librosa's `scipy.signal` import stall in the current environment. It compares L2-normalized vectors using cosine similarity, then keeps the best recording score per composition. The score reflects acoustic feature similarity. It is not a probability and does not establish recognition of alternate performances.

Run a search with a query recording:

```powershell
python audio_retrieval.py "Recordings/audio001.mp3" --top-k 10
```

Paths may be absolute or relative to `annamayya_dataset_package`. The engine exposes `AudioRetrievalEngine(audio_records, compositions=...)`, `build_index()`, `search_audio(query_audio_path, top_k=10)`, and `get_index_stats()`. Index features are kept in memory and reused for later queries. Index statistics include encountered, indexed, skipped missing paths, and failed decodes/features.

Run audio and text unit tests:

```powershell
python -m unittest -v test_audio_retrieval.py test_text_retrieval.py
```

Audio unit tests use locally generated WAV tones and require no model downloads or internet connection. MP3 decoding uses the installed SoundFile/libsndfile MPEG decoder.

## Phase 4: multimodal ranking and optional Neo4j graph

`multimodal_retrieval.py` combines the existing branch rankings with weighted reciprocal rank fusion (RRF). RRF uses each branch's candidate order rather than averaging text/audio cosine values, whose ranges and distributions are not calibrated. Its contribution for rank `r` is `(rrf_constant + 1) / (rrf_constant + r)`; configured weights are normalized over successfully evaluated branches. Default weights are text `0.6` and audio `0.4`. They are starting settings, not tuned accuracy claims. Results include `fused_score`, original `text_score` and `audio_score`, branch ranks, modalities, text/audio evidence, and metadata. `fused_score` is a weighted rank score, not a probability.

Example searches:

```powershell
python multimodal_retrieval.py --text "sacred abode of Sri Hari" --top-k 5
python multimodal_retrieval.py --audio "Recordings/audio001.mp3" --top-k 5
python multimodal_retrieval.py --text "sacred abode of Sri Hari" --audio "Recordings/audio001.mp3" --top-k 5
```

For Neo4j, set `NEO4J_URI`, `NEO4J_USERNAME` (or `NEO4J_USER`), and `NEO4J_PASSWORD`. To enrich search results without importing data, add `--graph`. To create uniqueness constraints and upsert the normalized dataset first, use `--import-graph`. Neo4j nodes are `Composition`, `TextRecord`, `AudioRecording`, `Singer`, `Raga`, and `Tala`. Child records link to compositions with `BELONGS_TO` and `RECORDING_OF`; compositions link to supported metadata with `PERFORMED_BY`, `USES_RAGA`, and `USES_TALA`. Imports use parameterized `MERGE` queries and stable source IDs. Graph enrichment never changes fusion scores. If Neo4j or its driver is unavailable, the CLI reports that and continues without graph enrichment.

The graph API is `AnnamayyaKnowledgeGraph.connect()`, `initialize_schema()`, `import_dataset(compositions, text_records, audio_records)`, `get_composition_details(composition_id)`, `get_recordings(composition_id)`, `find_compositions("singer" | "raga" | "tala", value)`, and `close()`.

Run all current unit tests:

```powershell
python -m unittest -v test_text_retrieval.py test_audio_retrieval.py test_multimodal_retrieval.py test_knowledge_graph.py
```

Fusion and graph tests use mocks. They do not require LaBSE downloads or a live Neo4j server.

## Phase 5: Streamlit song explorer

Install the project dependencies, then launch the interface from this project folder:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

The app uses four sidebar destinations: Home, Discover, Ask, and Connections. Discover provides text-only, audio-only, and combined searches through `MultimodalRetrievalEngine`; rank-fusion weights use documented defaults and stay out of the ordinary search workflow. Ask preserves the selected search evidence and conversation context across navigation. Connections queries the configured Neo4j graph using stable composition IDs and renders the returned composition, singer, raga, tala, text, and recording relationships. A Discover result can open its corresponding composition directly in Connections. No graph data is fabricated when the service is unavailable or the selected node is missing.

The configurable two-word brand is `BRAND_NAME` in `app.py` (currently “Annamayya Darshan”). The light devotional theme is ivory and warm stone with restrained gold and maroon accents. Results show local lyric/transliteration/meaning excerpts, available metadata, source links, and local audio playback. Full content and technical ranking evidence are available in separate collapsed sections. Uploaded WAV/MP3 audio is written to a temporary file for synchronous retrieval and removed afterwards. The UI cache is keyed to packaged input-file metadata so changed data invalidates its indexes. Home and Connections load normalized records without initializing text/audio model indexes; Discover initializes retrieval when first opened.

Graph enrichment is optional. Set `NEO4J_URI`, `NEO4J_USERNAME` (or `NEO4J_USER`), and `NEO4J_PASSWORD` in the environment, then enable it under **Optional services**. Credentials are not entered in the ordinary interface. The app continues with local metadata if graph access fails. The first text-engine initialization may download LaBSE and requires model access. Streamlit is declared in `requirements.txt` as `streamlit>=1.32`; the palette is also configured in `.streamlit/config.toml`.

Run application helper and backend tests with:

```powershell
python -m unittest -v
```

## Phase 5.5: retrieval-grounded AI explanations

The **Ask** page becomes useful after a search. Select one returned composition or the bounded set of top candidates, ask a question, and explicitly submit to request an explanation. The app passes the real Phase 4 ranks/scores, matching text evidence, relevant local text variants and recording metadata, available source references, and returned graph fields to the isolated `gemini_explainer.py` service. Evidence is field-whitelisted, limited to 3 candidates and 18,000 serialized characters; long text is excerpted and marked as truncated. Missing fields are omitted rather than guessed. Audio evidence is labeled with its actual score type and whether the recording was processed by the audio branch.

Install the official Google Gen AI SDK with the project requirements. The default model is the currently documented stable `gemini-3.8-flash`; override it with `GEMINI_MODEL` if desired. The API key is read only from `GEMINI_API_KEY`; do not put it in source code or commit it. In PowerShell, set the variables in the same terminal before launching:

```powershell
$env:GEMINI_API_KEY = "your-Google-AI-Studio-API-key"
$env:GEMINI_MODEL = "gemini-3.8-flash"
python -m pip install -r requirements.txt
streamlit run app.py
```

On macOS/Linux, use `export GEMINI_API_KEY="your-key"` and `export GEMINI_MODEL="gemini-3.8-flash"`. Create an API key in [Google AI Studio](https://aistudio.google.com/app/apikey); a consumer Gemini subscription alone does not configure API access. The current key and model must have Gemini Developer API access. The app does not automatically load `.env` files.

The question and bounded selected evidence are transmitted to Google's Gemini API only after the user submits a request. Retrieval, ranking, and source data remain the authority for what was found; Gemini only explains them. Answers are generated interpretations, not verified translations, historical verification, or proof that a recording is authentic. The application shows the exact evidence snapshot and available record IDs/sources alongside each answer. Follow-up context is limited to the last three turns for that same evidence fingerprint; changing the search or evidence selection clears stale answers. Requests are not made on Streamlit reruns, and repeated identical question/evidence/model submissions reuse the saved answer. Missing `GEMINI_API_KEY` leaves search fully functional and disables generation.

Explanation tests use mocked Gemini responses and never call the live API:

```powershell
python -m unittest -v test_gemini_explainer.py
python -m unittest -v
```

The SDK/client uses a bounded request timeout and one attempt. No live Gemini request is part of the ordinary test suite.

## Phase 6: final validation and release artifacts

Run these commands from the project root in PowerShell:

```powershell
python -m pip install -r requirements.txt
python scripts/validate_data.py
python -m unittest -v
python scripts/evaluate_retrieval.py --prepare
python -m streamlit run app.py
```

`validate_data.py` checks the actual workbook and local package without editing source data. It saves `docs/data_integrity_report.json`. The current package has 25 compositions, 60 usable text records, and 14 audio references: 13 recordings exist locally and `Recordings/audio025.mp3` is missing. This missing file is reported as a warning, never indexed as an audio match. Tala is absent for 24 compositions; optional metadata is not filled in.

The evaluation runner writes `data/evaluation/evaluation_queries.csv`, `data/evaluation/evaluation_results.csv`, `data/evaluation/evaluation_summary.json`, and `docs/evaluation_report.md`. The generated cases use each existing recording as both query and indexed candidate, with the associated text from the workbook. This is an in-sample direct-recording smoke check only—not an independent or cross-performance accuracy benchmark. Every case is marked tuning/in-sample, none is held out, and the workbook's composition association is not independently verified. `python scripts/evaluate_retrieval.py --prepare --skip-text` runs without trying to initialize LaBSE; text and multimodal rows are then explicitly recorded as failed, not silently replaced. Without `--skip-text`, LaBSE requires the existing model to be cached or downloadable.

The Streamlit engine cache key fingerprints packaged file paths, sizes, and modification times, so a changed workbook or recording invalidates cached retrieval resources on the next rerun (without repeatedly hashing multi-megabyte audio payloads).

The metadata baseline uses only case-insensitive substring matching against title, singer, raga, and tala, with deterministic tie-breaking. It does not use embeddings, audio, graph results, or fusion. Failed queries and empty result lists count as zero in aggregate metrics, with all eligible cases as the denominator.

See [`docs/evaluation_report.md`](docs/evaluation_report.md), [`docs/data_integrity_report.json`](docs/data_integrity_report.json), and [`docs/demo_guide.md`](docs/demo_guide.md) for observed results and limitations. The full app/browser flow, live Neo4j, and live Gemini remain optional integration steps: the current environment does not have Streamlit, Neo4j's driver/configuration, the Google Gen AI SDK/API key, or cached LaBSE model weights. The backend tests use mocks for external services and do not prove live connectivity.
