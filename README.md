# Multimodal Annamayya Knowledge Explorer

An interactive, multimodal music information retrieval and exploration system for Annamayya compositions. The platform integrates cross-script text search (Telugu lyrics, Romanized transliterations, English translations), acoustic feature similarity audio matching, weighted Reciprocal Rank Fusion (RRF), cloud-hosted Neo4j Knowledge Graph integration, interactive PyVis network visualizations, and retrieval-grounded AI explanations via Google Gemini.

---

## Quick-Start & Environment Setup

Set your environment variables in your terminal prior to running graph imports, retrieval scripts, or launching Streamlit. Credentials are read from environment variables and are never hardcoded in source files.

### PowerShell (Windows)
```powershell
$env:NEO4J_URI="neo4j+s://your-auradb-instance-id.databases.neo4j.io"
$env:NEO4J_USERNAME="neo4j"
$env:NEO4J_PASSWORD="your-auradb-password"
$env:GEMINI_API_KEY="your-google-ai-studio-key"
$env:GEMINI_MODEL="gemini-3.8-flash"
```

### Bash (macOS / Linux)
```bash
export NEO4J_URI="neo4j+s://your-auradb-instance-id.databases.neo4j.io"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your-auradb-password"
export GEMINI_API_KEY="your-google-ai-studio-key"
export GEMINI_MODEL="gemini-3.8-flash"
```

---

## System Architecture & Implementation Phases

### Phase 1: Dataset Normalization

`data_loader.py` reads `annamayya_dataset_package/Annamayya_25_Song_Dataset.xlsx` and produces normalized DataFrames for `Compositions`, `TextRecords`, and `AudioRecords` with stable identifiers and source provenance.

Run data loading verification:

```powershell
python data_loader.py
```

---

### Phase 2: Semantic Text Retrieval

`text_retrieval.py` reuses `load_normalized_data()` from Phase 1. It embeds usable `TextRecords` with `sentence-transformers/LaBSE`, L2-normalizes the vectors, and calculates cosine similarity with the search query. Text-level matches are grouped by `composition_id`; the highest matching text score becomes the composition score. Results include matching text records as evidence.

Optional `--language` and `--text-type` filters apply before ranking. The engine keeps embeddings in memory for the active process.

Install dependencies (first run downloads LaBSE model weights):

```powershell
python -m pip install -r requirements.txt
```

Run text searches:

```powershell
python text_retrieval.py "sacred abode of Sri Hari" --top-k 5
python text_retrieval.py "అదివో అల్లదివో" --language Telugu --text-type lyrics
```

Run unit tests:

```powershell
python -m unittest -v test_text_retrieval.py
```

---

### Phase 3: Audio Similarity Retrieval

`audio_retrieval.py` decodes recordings through `SoundFile` to mono 22.05 kHz audio and uses `NumPy` matrix operations to calculate fixed-size MFCC and Chroma frame statistics. It compares L2-normalized vectors using cosine similarity and keeps the best recording score per composition. The score reflects acoustic feature similarity; it is a ranking signal, not a probability.

Run search with a query recording:

```powershell
python audio_retrieval.py "Recordings/audio001.mp3" --top-k 10
```

Paths may be absolute or relative to `annamayya_dataset_package`.

Run audio and text unit tests:

```powershell
python -m unittest -v test_audio_retrieval.py test_text_retrieval.py
```

---

### Phase 4: Multimodal Ranking & Neo4j Knowledge Graph

`multimodal_retrieval.py` combines text and audio branch rankings using weighted Reciprocal Rank Fusion (RRF). RRF uses each branch's candidate order rather than averaging raw cosine values. Default weights are text `0.6` and audio `0.4`. Results include `fused_score`, individual branch scores, ranks, modalities, text/audio evidence, and metadata.

#### 1. Standalone Graph Import (Neo4j AuraDB)
To create uniqueness constraints and populate your cloud database with `Composition`, `TextRecord`, `AudioRecording`, `Singer`, `Raga`, and `Tala` nodes:

```powershell
python multimodal_retrieval.py --import-graph
```

#### 2. Run Multimodal Fused Search
```powershell
python multimodal_retrieval.py --text "sacred abode of Sri Hari" --top-k 5
python multimodal_retrieval.py --audio "Recordings/audio001.mp3" --top-k 5
python multimodal_retrieval.py --text "sacred abode of Sri Hari" --audio "Recordings/audio001.mp3" --top-k 5 --graph
```

Graph enrichment is optional. If Neo4j credentials are missing or the database is offline, the CLI reports the status and continues with local retrieval seamlessly.

---

### Phase 5: Streamlit Knowledge Explorer & PyVis Graph View

Install dependencies and launch the interactive web application:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

#### Application Navigation:
* **Home:** Overview of the platform, collection statistics, and project overview.
* **Discover:** Supports text-only, audio-only, and combined RRF searches through `MultimodalRetrievalEngine`.
* **Interactive Knowledge Graph (PyVis Integration):** Under **Optional services** in the sidebar, toggle **"Add knowledge-graph details"** ON. When expanding *"Why this result?"* on any search result card, the app renders an interactive, physics-driven `pyvis` network diagram displaying connected nodes for the composition, singer, raga, tala, audio recording, lyrics, transliterations, and translations.
* **Connections:** Browse relationships across the entire database by querying compositions linked to specific singers, ragas, or talas.
* **Ask:** Preserves search evidence and conversation context for AI-grounded explanations.

---
The app provides Discover search and a Connections view. Discover provides text-only, audio-only, and combined searches through `MultimodalRetrievalEngine`; rank-fusion weights use documented defaults and stay out of the ordinary search workflow. Connections queries the configured Neo4j graph and offers a full graph view or a single composition neighborhood with a selectable depth. Both views display actual nodes and typed relationships in an interactive force-directed graph. Set `NEO4J_URI`, `NEO4J_USERNAME` (or `NEO4J_USER`), and `NEO4J_PASSWORD` to use the view. No graph data is fabricated when the service is unavailable.

The configurable two-word brand is `BRAND_NAME` in `app.py` (currently “Annamayya Darshan”). The light devotional theme is ivory and warm stone with restrained gold and maroon accents. Results show local lyric/transliteration/meaning excerpts, available metadata, source links, and local audio playback. Full content and technical ranking evidence are available in separate collapsed sections. Uploaded WAV/MP3 audio is written to a temporary file for synchronous retrieval and removed afterwards. The UI cache is keyed to packaged input-file metadata so changed data invalidates its indexes. Connections loads Neo4j neighborhoods without initializing text/audio model indexes; Discover initializes retrieval when first opened.

Graph enrichment is optional. Set `NEO4J_URI`, `NEO4J_USERNAME` (or `NEO4J_USER`), and `NEO4J_PASSWORD` in the environment, then enable it under **Optional services**. Credentials are not entered in the ordinary interface. The app continues with local metadata if graph access fails. The first text-engine initialization may download LaBSE and requires model access. Streamlit is declared in `requirements.txt` as `streamlit>=1.32`; the palette is also configured in `.streamlit/config.toml`.

Run application helper and backend tests with:

```powershell
python -m unittest -v
```

### Phase 5.5: Retrieval-Grounded AI Explanations

Select a returned composition or set of top candidates on the **Ask** page to request an explanation. The app passes real Phase 4 ranks, scores, matching text evidence, recording metadata, and returned graph fields to `gemini_explainer.py`. Evidence is field-whitelisted and bounded.

Launch Streamlit with Gemini credentials enabled:

```powershell
$env:GEMINI_API_KEY="your-api-key"
$env:GEMINI_MODEL="gemini-3.8-flash"
streamlit run app.py
```

Missing `GEMINI_API_KEY` leaves search and graph visualization fully functional while gracefully disabling explanation generation.

Explanation tests use mocked Gemini responses:

```powershell
python -m unittest -v test_gemini_explainer.py
```

---

### Phase 6: System Testing & Final Validation

#### Run the Complete Test Suite
Execute all backend unit tests (text retrieval, audio retrieval, multimodal fusion, Neo4j graph mocks, Streamlit helpers, and Gemini explainer mocks):

```powershell
python -m unittest discover -v
```

#### Run Data Validation & Evaluation Benchmark Scripts
```powershell
python scripts/validate_data.py
python scripts/evaluate_retrieval.py --prepare
```

* `validate_data.py` inspects the workbook and dataset package, saving `docs/data_integrity_report.json`.
* `evaluate_retrieval.py` outputs retrieval query evaluation summaries to `data/evaluation/` and `docs/evaluation_report.md`.

---

## Documentation References

* Architecture and design specification: [`docs/architecture.md`](docs/architecture.md)
* Evaluation benchmark results: [`docs/evaluation_report.md`](docs/evaluation_report.md)
* Data integrity report: [`docs/data_integrity_report.json`](docs/data_integrity_report.json)
* Application demo guide: [`docs/demo_guide.md`](docs/demo_guide.md)