"""FastAPI backend for the standalone Annamayya Knowledge Explorer frontend."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from audio_retrieval import AudioRetrievalEngine
from data_loader import DATASET_DIR
from gemini_explainer import GeminiExplainer, prepare_evidence
from multimodal_retrieval import MultimodalRetrievalEngine
from sql_store import DATABASE_PATH, database_stats, initialize_database, load_sql_data
from text_retrieval import build_engine as build_text_engine

LOGGER = logging.getLogger("annamayya_api")
PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "frontend"
_data: dict[str, Any] = {}


def _build_data() -> dict[str, Any]:
    compositions, texts, audio = load_sql_data()
    result: dict[str, Any] = {
        "compositions": compositions, "text_records": texts, "audio_records": audio,
        "text_engine": None, "audio_engine": None, "audio_index_stats": None,
        "branch_initialization_errors": {},
    }
    try:
        result["text_engine"] = build_text_engine(texts, compositions)
    except Exception as exc:
        result["branch_initialization_errors"]["text"] = str(exc)
        LOGGER.exception("Text retrieval initialization failed")
    try:
        engine = AudioRetrievalEngine(audio, compositions)
        result["audio_index_stats"] = engine.build_index()
        result["audio_engine"] = engine
    except Exception as exc:
        result["branch_initialization_errors"]["audio"] = str(exc)
        LOGGER.exception("Audio retrieval initialization failed")
    return result


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _data
    initialize_database()
    yield
    _data.clear()


app = FastAPI(title="Annamayya Knowledge Explorer API", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/vendor", StaticFiles(directory=PROJECT_ROOT / "lib" / "vis-9.1.2"), name="vendor")
app.mount("/media", StaticFiles(directory=DATASET_DIR / "Recordings"), name="recordings")


def get_data() -> dict[str, Any]:
    global _data
    if not _data:
        _data = _build_data()
    return _data


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "to_dict") and not isinstance(value, dict):
        try:
            return json_safe(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            return None
        return value
    return str(value)


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "database": str(DATABASE_PATH.name), "tables": database_stats(),
            "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
            "neo4j_configured": bool(os.getenv("NEO4J_URI") and (os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")) and os.getenv("NEO4J_PASSWORD"))}


@app.get("/api/recordings")
def recordings() -> dict[str, Any]:
    """Playable packaged recordings, used by the Remix Studio song picker."""
    compositions, _texts, audio = load_sql_data()
    titles = dict(zip(compositions["composition_id"].astype(str), compositions["canonical_title"]))
    rows = []
    for record in audio.to_dict("records"):
        filename = Path(str(record.get("audio_path", "")).replace("\\", "/")).name
        if not filename or not (DATASET_DIR / "Recordings" / filename).is_file():
            continue
        composition_id = str(record.get("composition_id", ""))
        rows.append({"audio_id": record.get("audio_id"), "composition_id": composition_id, "file": filename,
                     "title": record.get("recording_title") or titles.get(composition_id) or filename,
                     "singer": record.get("singer"), "raga": record.get("raga"), "tala": record.get("tala")})
    return json_safe({"recordings": rows})


@app.post("/api/search")
async def search(
    text_query: str = Form(""),
    audio: UploadFile | None = File(None),
    enrich_graph: bool = Form(False),
) -> dict[str, Any]:
    query = text_query.strip() or None
    if query is None and audio is None:
        raise HTTPException(400, "Enter a text query or choose a WAV/MP3 recording.")
    if audio and Path(audio.filename or "").suffix.lower() not in {".wav", ".mp3"}:
        raise HTTPException(400, "Please upload a WAV or MP3 audio file.")
    data = get_data()
    if data["text_engine"] is None and query:
        raise HTTPException(503, "Text search is unavailable. Check LaBSE model access and server logs.")
    if data["audio_engine"] is None and audio:
        raise HTTPException(503, "Audio search is unavailable. Check audio dependencies and server logs.")

    graph = None
    graph_warning = None
    if enrich_graph:
        try:
            from knowledge_graph import AnnamayyaKnowledgeGraph
            graph = AnnamayyaKnowledgeGraph()
            graph.connect()
        except Exception:
            graph_warning = "Neo4j is unavailable; showing local retrieval results."
            graph = None
    temp_path = None
    try:
        if audio:
            suffix = Path(audio.filename or "audio.wav").suffix.lower()
            payload = await audio.read()
            if not payload:
                raise HTTPException(400, "The uploaded audio file is empty.")
            with tempfile.NamedTemporaryFile(prefix="annamayya_query_", suffix=suffix, delete=False) as handle:
                handle.write(payload)
                temp_path = Path(handle.name)
        engine = MultimodalRetrievalEngine(
            text_engine=data["text_engine"], audio_engine=data["audio_engine"],
            compositions=data["compositions"], knowledge_graph=graph,
        )
        results = engine.search(query, temp_path, top_k=10, raise_on_branch_error=False)
        for item in results or []:
            composition_id = str(item.get("composition_id", ""))
            local_texts = data["text_records"].loc[data["text_records"]["composition_id"].astype(str) == composition_id]
            local_audio = data["audio_records"].loc[data["audio_records"]["composition_id"].astype(str) == composition_id]
            item["local_text_records"] = local_texts.to_dict("records")
            item["local_audio_records"] = local_audio.to_dict("records")
        return json_safe({"results": results or [], "branch_errors": engine.last_branch_errors,
                          "graph_warning": graph_warning, "query": query,
                          "audio_filename": audio.filename if audio else None})
    except HTTPException:
        raise
    except (ValueError, OSError) as exc:
        raise HTTPException(400, "The audio or text query could not be processed. Check the input and try again.") from exc
    except Exception as exc:
        LOGGER.exception("Search failed")
        raise HTTPException(500, f"Search failed ({type(exc).__name__}).") from exc
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)
        if graph:
            graph.close()


@app.get("/api/compositions/{composition_id}")
def composition_details(composition_id: str) -> dict[str, Any]:
    data = get_data()
    composition = data["compositions"].loc[data["compositions"]["composition_id"].astype(str) == composition_id]
    if composition.empty:
        raise HTTPException(404, "Composition not found.")
    texts = data["text_records"].loc[data["text_records"]["composition_id"].astype(str) == composition_id]
    audio = data["audio_records"].loc[data["audio_records"]["composition_id"].astype(str) == composition_id]
    return json_safe({"composition": composition.iloc[0].to_dict(), "text_records": texts.to_dict("records"),
                      "audio_records": audio.to_dict("records")})


@app.get("/api/connections")
def connections(composition_id: str | None = None, field: str | None = None, value: str | None = None) -> dict[str, Any]:
    if not (os.getenv("NEO4J_URI") and (os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")) and os.getenv("NEO4J_PASSWORD")):
        raise HTTPException(503, "Neo4j is not configured. Set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD.")
    from knowledge_graph import AnnamayyaKnowledgeGraph
    graph = AnnamayyaKnowledgeGraph()
    try:
        graph.connect()
        if field and value:
            compositions = graph.find_compositions(field, value)
            edges = []
            for item in compositions:
                edges.extend(graph.get_graph_network(item.get("composition_id")))
        else:
            compositions = []
            edges = graph.get_graph_network(composition_id)
        return json_safe({"compositions": compositions, "edges": edges})
    except Exception as exc:
        LOGGER.exception("Connections query failed")
        raise HTTPException(503, "Neo4j could not be reached or queried.") from exc
    finally:
        graph.close()


class ExplanationRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    composition_id: str
    results: list[dict[str, Any]]
    history: list[dict[str, str]] = Field(default_factory=list)


@app.post("/api/explain")
def explain(request: ExplanationRequest) -> dict[str, Any]:
    data = get_data()
    evidence = prepare_evidence(request.results, compositions=data["compositions"],
                                text_records=data["text_records"], audio_records=data["audio_records"],
                                selected_composition_id=request.composition_id)
    try:
        result = GeminiExplainer().explain(request.question, evidence, history=request.history[-3:])
        return json_safe(result)
    except Exception as exc:
        raise HTTPException(503, str(exc)) from exc