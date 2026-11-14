# Annamayya Knowledge Explorer

A standalone web app for searching and exploring Annamacharya compositions. The frontend is plain HTML, CSS, and JavaScript served by FastAPI. SQLite stores the normalized dataset; the existing text and audio retrieval modules continue to rank search results.

## Run locally

Use Python 3.10 or later. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app:app --reload --port 1234
```

Open <http://127.0.0.1:1234>. On first startup, the app normalizes `annamayya_dataset_package/Annamayya_25_Song_Dataset.xlsx` into `data/annamayya.sqlite3`. The original workbook and recordings remain the source package. Set `ANNAMAYYA_DATABASE` to use a different SQLite file.

The first text search initializes the LaBSE sentence-transformer and may download its model weights. Audio indexing reads the packaged recordings when the backend starts initializing retrieval.

## Features

- Search Telugu lyrics, Romanized Telugu, and English meanings.
- Search by an uploaded WAV/MP3 recording or combine audio and text. Ranking uses the existing weighted reciprocal-rank fusion behavior.
- Read full lyrics, transliterations, and meanings; inspect available singer, raga, tala, sources, and play packaged recordings.
- Explore recordings in the browser-based Remix Studio, including Lo-fi Chill and Temple Hall effects.
The app does not make AI or language-model API calls. Search, song details, duet merging, and browser-based remixing run through the local application.

## SQL tables

SQLite contains three normalized tables: `Compositions`, `TextRecords`, and `AudioRecords`. Text/audio rows reference `Compositions.composition_id`; IDs and normalization rules come from `data_loader.py`. The app refreshes the SQLite database when it is missing or older than the packaged workbook.

## Backend API

- `GET /api/health` — service status and collection counts.
- `POST /api/search` — multipart text/audio search, with optional graph enrichment.
- `GET /api/compositions/{composition_id}` — local composition, text, and audio records.

The original retrieval modules and command line tools remain available for evaluation and offline use. The web app itself no longer uses Streamlit.
