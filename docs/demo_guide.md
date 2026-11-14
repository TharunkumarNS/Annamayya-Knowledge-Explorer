# Local Demo Guide

## Start the web app

From the project root, install dependencies and start the FastAPI server:

```powershell
python -m pip install -r requirements.txt
uvicorn app:app --reload --port 1234
```

Open <http://127.0.0.1:1234>. The frontend is plain HTML, CSS, and JavaScript. FastAPI serves the page and API from the same local origin. On the first request, the normalized tables are created in `data/annamayya.sqlite3` from the packaged workbook.

## Discover compositions

Enter Telugu, Romanized Telugu, or English text and select **Find compositions**. For acoustic similarity, attach a WAV or MP3 file; a text query and audio file can be submitted together. Each result shows its match type, available lyrics/meaning excerpts, metadata, sources, and local recording player. **Full lyrics & song details** opens all local text and recording rows. **Why this result?** shows branch ranks and scores; the scores are ranking signals, not probabilities.

The first text search initializes LaBSE and can take longer while model weights are fetched. Audio search builds its feature index from the packaged recordings.

## Optional services

Set `NEO4J_URI`, `NEO4J_USERNAME` (or `NEO4J_USER`), and `NEO4J_PASSWORD` before starting the server to enable search enrichment and the Connections graph. Set `GEMINI_API_KEY` and optionally `GEMINI_MODEL` to enable explanations for returned search results. The UI reports when a service is unavailable; local search does not require either service.

## Data and API

`GET /api/health` reports table counts and optional service configuration. `POST /api/search` accepts multipart form data. `GET /api/compositions/{composition_id}` returns source rows; `GET /api/connections` returns Neo4j links; and `POST /api/explain` generates an evidence-grounded answer. See the project README for request fields and SQL table details.
