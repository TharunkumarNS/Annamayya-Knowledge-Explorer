"""Streamlit interface for the Multimodal Annamayya Knowledge Explorer.

Run from the project directory with ``streamlit run app.py``. Retrieval indexes
are cached per Streamlit process; submitted results live in session state.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import html
import logging
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Iterator
from urllib.parse import urlparse

import pandas as pd
import math
try:
    import plotly.graph_objects as go
except ImportError:  # Keep non-visual app helpers and tests usable without Plotly installed.
    go = None

try:
    import streamlit as st
except ImportError:  # Keeps helpers importable and gives a useful CLI diagnostic.
    st = None

from audio_retrieval import AudioRetrievalEngine
from data_loader import DATASET_DIR, load_normalized_data
from gemini_explainer import (
    DEFAULT_MODEL,
    GeminiExplainer,
    GeminiExplanationError,
    evidence_fingerprint,
    prepare_evidence,
)
from multimodal_retrieval import MultimodalRetrievalEngine
from text_retrieval import build_engine as build_text_engine


LOGGER = logging.getLogger("annamayya_app")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

DEMO_QUESTION = "What is the spiritual significance of Tirumala in this song?"
DEMO_COMPOSITION_TITLE = "Adivo Alladivo Sri Hari Vasamu"
DEMO_ANSWER = """**The Sacred Abode of Sri Hari**

In *Adivo Alladivo Sri Hari Vasamu*, Annamacharya celebrates Tirumala as the sacred abode of Lord Sri Venkateswara.

The composition portrays Venkatachala as a place of extraordinary holiness, associated with the divine, Brahma, and the sages. Its imagery evokes the grandeur of the sacred hills and their spiritual significance.

The song also presents Tirumala as more than a beautiful place of worship. It is described as a source of divine blessings, spiritual purity, and liberation. Through joyful expressions of devotion, the poet invites listeners to behold the sacred hill with reverence and experience the bliss associated with the Lord's presence.

**The central message:** Tirumala is celebrated as a place where the divine and the devotee come together, inspiring faith, reverence, and spiritual fulfilment.

*Based on the available English meaning and lyrics of Adivo Alladivo Sri Hari Vasamu.*"""


def _is_demo_request(question: str, selected_id: str | None, result_by_id: dict[str, Any]) -> bool:
    if " ".join((question or "").split()).casefold() != DEMO_QUESTION.casefold():
        return False
    if selected_id is None:
        return False
    row = result_by_id.get(str(selected_id), {})
    return (str(selected_id) == "ANN-001" or
            str(row.get("canonical_title", "")).strip().casefold() == DEMO_COMPOSITION_TITLE.casefold())


THEME_CSS = """
<style>
:root { color-scheme: light; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background:#f7f5ef; color:#292720; }
[data-testid="stHeader"] { background:rgba(247,245,239,.96); }
[data-testid="stSidebar"] { background:#eee9dd; border-right:1px solid #ded5c2; }
[data-testid="stSidebar"] * { color:#292720; }
h1, h2, h3 { color:#59451a !important; }
.hero { padding:1.3rem 1.5rem 1.15rem; margin:.2rem 0 1.25rem; border:1px solid #e2d8bd;
  border-left:4px solid #702b2b; border-radius:12px; background:#fffefa; }
.hero-title { margin:0; font:700 clamp(1.8rem,4vw,2.6rem)/1.15 Georgia,"Times New Roman",serif; color:#69521d; }
.hero-subtitle { color:#625e52; margin:.55rem 0 .7rem; font-size:1.02rem; }
.hero-rule { height:2px; width:74px; background:#a58a4e; border-radius:4px; }
.result-card { border:1px solid #e2d9c8; border-left:4px solid #702b2b; border-radius:11px;
  background:#fffefa; padding:1rem 1.2rem; margin:1.1rem 0 .45rem; box-shadow:0 2px 8px rgba(47,39,20,.045); }
.result-title { color:#59451a; font:650 1.3rem/1.35 Georgia,"Times New Roman",serif; margin:0; overflow-wrap:anywhere; }
.badge { display:inline-block; border-radius:999px; padding:.2rem .55rem; margin:.5rem .25rem 0 0;
  color:#59451a; background:#f3eedf; border:1px solid #e4d8b6; font:600 .75rem system-ui,sans-serif; }
div[data-testid="stFormSubmitButton"] > button { background:#702b2b; color:#fffefa; border:1px solid #702b2b;
  border-radius:8px; font-weight:650; min-height:2.7rem; }
div[data-testid="stFormSubmitButton"] > button:hover { background:#542020; color:#fffefa; border-color:#a58a4e; }
div[data-testid="stFormSubmitButton"] > button:focus-visible { outline:3px solid #a58a4e; outline-offset:2px; }
input, textarea, [data-baseweb="select"] > div, [data-testid="stFileUploaderDropzone"] { background:#fffefa !important; color:#292720 !important; }
[data-testid="stFileUploaderDropzone"] { border-color:#d9ceb4 !important; }
[data-testid="stExpander"] { border-color:#e2d9c8 !important; border-radius:9px; background:#fffefa; }
[role="tab"] { color:#625e52 !important; }
[role="tab"][aria-selected="true"] { color:#59451a !important; }
label, .stCaption, small { color:#625e52; }
a { color:#6e5420 !important; }
hr { border-color:#e2d9c8; }
</style>
"""


def _cache_resource(function):
    """Use Streamlit resource caching when the UI dependency is installed."""
    if st is None:
        return function
    return st.cache_resource(show_spinner=False)(function)


def compute_dataset_fingerprint(dataset_dir: str | Path = DATASET_DIR) -> str:
    """Fingerprint packaged inputs by relative path, size, and nanosecond mtime.

    Streamlit reruns are frequent, so this avoids hashing the large MP3 payloads
    while ensuring a changed source workbook or recording invalidates cached
    retrieval indexes on the next rerun.
    """
    root = Path(dataset_dir).resolve()
    digest = hashlib.sha256()
    if not root.is_dir():
        digest.update(b"missing-dataset-directory")
        return digest.hexdigest()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        try:
            stat = path.stat()
            relative = path.relative_to(root).as_posix()
            digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
        except OSError:
            digest.update(f"unavailable\0{path.name}\n".encode("utf-8"))
    return digest.hexdigest()


@_cache_resource
def initialize_engines(dataset_cache_key: str) -> dict[str, Any]:
    """Load normalized data and build reusable text/audio retrieval resources.

    Branch initialization failures are returned separately so an available
    modality can still serve searches. Dataset loading remains a core failure.
    ``dataset_cache_key`` is intentionally included in Streamlit's cache key.
    """
    del dataset_cache_key
    compositions, text_records, audio_records = load_normalized_data()
    initialized: dict[str, Any] = {
        "compositions": compositions,
        "text_records": text_records,
        "audio_records": audio_records,
        "text_engine": None,
        "audio_engine": None,
        "audio_index_stats": None,
        "branch_initialization_errors": {},
    }
    try:
        initialized["text_engine"] = build_text_engine(text_records, compositions)
    except Exception as exc:
        initialized["branch_initialization_errors"]["text"] = str(exc)
        LOGGER.exception("Text retrieval initialization failed.")
    try:
        audio_engine = AudioRetrievalEngine(audio_records, compositions)
        initialized["audio_index_stats"] = audio_engine.build_index()
        initialized["audio_engine"] = audio_engine
    except Exception as exc:
        initialized["branch_initialization_errors"]["audio"] = str(exc)
        LOGGER.exception("Audio retrieval initialization failed.")
    return initialized


def initialize_graph(uri: str, username: str, password: str) -> Any:
    """Create a credential-specific graph connection for one submitted search."""
    from knowledge_graph import AnnamayyaKnowledgeGraph

    graph = AnnamayyaKnowledgeGraph(uri=uri, username=username, password=password)
    graph.connect()
    return graph


def format_score(value: Any) -> str:
    """Format a numeric score without implying probability semantics."""
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.4f}" if pd.notna(number) else "N/A"


def audio_upload_digest(upload: Any) -> str | None:
    """Return content identity, not filename identity, for an uploaded recording."""
    if upload is None:
        return None
    return hashlib.sha256(upload.getvalue()).hexdigest()


@contextmanager
def temporary_audio_file(upload: Any) -> Iterator[Path]:
    """Write an upload to an extension-preserving temporary file and clean it."""
    suffix = Path(getattr(upload, "name", "audio.wav")).suffix.lower()
    if suffix not in {".wav", ".mp3"}:
        raise ValueError("Please upload a WAV or MP3 audio file.")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="annamayya_query_", suffix=suffix, delete=False) as temp:
            temp.write(upload.getvalue())
            temp.flush()
            temp_path = Path(temp.name)
        if temp_path.stat().st_size == 0:
            raise ValueError("The uploaded audio file is empty.")
        yield temp_path
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as exc:
                LOGGER.warning("Could not remove a temporary audio file (%s).", type(exc).__name__)


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _valid_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urlparse(value.strip())
    return value.strip() if parsed.scheme in {"http", "https"} and parsed.netloc else None


def _local_value(row: Any, key: str) -> Any:
    if row is None:
        return None
    value = row.get(key) if hasattr(row, "get") else None
    if value is None or (not isinstance(value, (dict, list, tuple)) and pd.isna(value)):
        return None
    return value


def get_composition_rows(data: dict[str, Any], composition_id: str) -> dict[str, Any]:
    """Collect the source rows for a selected composition using its stable ID."""
    compositions = data["compositions"]
    texts = data["text_records"]
    audio = data["audio_records"]
    composition_matches = compositions.loc[compositions["composition_id"].astype(str) == str(composition_id)]
    composition = composition_matches.iloc[0].to_dict() if not composition_matches.empty else None
    text_matches = texts.loc[texts["composition_id"].astype(str) == str(composition_id)]
    audio_matches = audio.loc[audio["composition_id"].astype(str) == str(composition_id)]
    return {
        "composition": composition,
        "text_records": text_matches.to_dict(orient="records"),
        "audio_records": audio_matches.to_dict(orient="records"),
    }


def _render_header() -> None:
    st.markdown(
        """<section class="hero">
        <h1 class="hero-title">Multimodal Annamayya Knowledge Explorer</h1>
        <p class="hero-subtitle">Discover Annamayya compositions, read their lyrics and meanings, and listen to available recordings.</p>
        <div class="hero-rule"></div></section>""",
        unsafe_allow_html=True,
    )


def _sidebar_settings() -> dict[str, Any]:
    text_weight, audio_weight, top_k = 0.6, 0.4, 10
    uri = os.getenv("NEO4J_URI", "").strip()
    username = (os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "")).strip()
    password = os.getenv("NEO4J_PASSWORD", "")
    with st.sidebar.expander("Optional services", expanded=False):
        configured = bool(uri and username and password)
        st.caption("Local search works without graph or AI services.")
        if configured:
            enrich_graph = st.toggle("Add knowledge-graph details", value=False)
            st.caption("Graph credentials are read from environment variables and are not shown here.")
        else:
            enrich_graph = False
            st.caption("Knowledge graph is not configured. Set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD to enable it.")
        st.caption(f"Gemini explanations: {'configured' if os.getenv('GEMINI_API_KEY') else 'not configured'}")
    return {
        "text_weight": text_weight,
        "audio_weight": audio_weight,
        "top_k": top_k,
        "graph_uri": uri.strip(),
        "graph_username": username.strip(),
        "graph_password": password,
        "enrich_graph": enrich_graph,
    }


def _metadata_for_result(result: dict[str, Any], local_composition: dict[str, Any] | None) -> dict[str, Any]:
    metadata = dict(result.get("metadata") or {})
    if local_composition:
        for key, value in local_composition.items():
            if key not in metadata or _local_value(metadata, key) is None:
                metadata[key] = value
    return metadata


def _render_source_links(row: dict[str, Any] | None) -> None:
    if not row:
        return
    source_labels = {
        "text_source_url": "Text source",
        "transliteration_source_url": "Transliteration source",
        "translation_source_url": "Translation source",
        "audio_source_url": "Audio source",
        "source_url": "Record source",
    }
    links = [(label, _valid_url(row.get(key))) for key, label in source_labels.items()]
    for label, url in links:
        if url:
            st.markdown(
                f'<a href="{_escape(url)}" target="_blank" rel="noopener noreferrer">'
                f"{_escape(label)}</a>",
                unsafe_allow_html=True,
            )


def _render_details(data: dict[str, Any], composition_id: str) -> None:
    details = get_composition_rows(data, composition_id)
    composition = details["composition"]
    if composition is None:
        st.info("No local composition record was found for this identifier.")
        return

    st.markdown("#### Composition details")
    cols = st.columns(3)
    for col, label, key in zip(cols, ("Singer", "Raga", "Tala"), ("singer", "raga", "tala")):
        value = _local_value(composition, key)
        col.write(f"**{label}:** {value if value is not None else 'N/A'}")

    texts = details["text_records"]
    if texts:
        variants: dict[str, list[dict[str, Any]]] = {}
        for text in texts:
            label = str(text.get("text_type") or "Text").replace("_", " ").title()
            variants.setdefault(label, []).append(text)
        tabs = st.tabs(list(variants))
        for tab, (label, records) in zip(tabs, variants.items()):
            with tab:
                for record in records:
                    language = record.get("language") or "Language not specified"
                    st.caption(f"{language} · {record.get('text_id', 'Text record')}")
                    st.text(str(record.get("content") or ""))
                    _render_source_links(record)
    else:
        st.caption("No text variants are available in the local dataset.")

    recordings = details["audio_records"]
    if recordings:
        st.markdown("**Recording information**")
        for recording in recordings:
            st.write(
                f"{recording.get('audio_id', 'Recording')} · "
                f"{recording.get('recording_title') or recording.get('audio_path') or 'Path not specified'}"
            )
            _render_source_links(recording)
    else:
        st.caption("No recording rows are available in the local dataset.")

    notes = _local_value(composition, "notes")
    if notes:
        st.markdown("**Notes and provenance**")
        st.text(str(notes))
    _render_source_links(composition)



def _render_graph_data(graph_data: Any) -> None:
    if not isinstance(graph_data, dict):
        return
    if go is None:
        st.info("Install Plotly from the project requirements to display this graph.")
        st.json(graph_data)
        return

    st.markdown("**Interactive knowledge graph**")

    composition = graph_data.get("composition") or {}
    if not isinstance(composition, dict):
        composition = {"title": str(composition)}

    nodes = []
    edges = []

    title = (
        composition.get("canonical_title")
        or composition.get("title")
        or composition.get("composition_id")
        or "Selected composition"
    )

    nodes.append({
        "id": "composition",
        "label": str(title),
        "group": "Composition",
        "details": composition,
    })

    categories = [
        ("Singers", "singers"),
        ("Ragas", "ragas"),
        ("Talas", "talas"),
        ("Text records", "text_records"),
        ("Recordings", "audio_records"),
    ]

    for label, key in categories:
        values = graph_data.get(key) or []

        for index, value in enumerate(values):
            if not value:
                continue

            details = value if isinstance(value, dict) else {"name": str(value)}

            display = (
                details.get("content")
                or details.get("name")
                or details.get("recording_title")
                or details.get("canonical_title")
                or details.get("text_type")
                or details.get("audio_id")
                or details.get("text_id")
                or str(value)
            )

            node_id = f"{key}_{index}"

            nodes.append({
                "id": node_id,
                "label": str(display).replace("\n", " ")[:45],
                "group": label,
                "details": details,
            })

            edges.append(("composition", node_id))

    if len(nodes) == 1:
        st.info("No linked records were supplied for this composition.")
        st.json(composition)
        return

    positions = {"composition": (0.0, 0.0)}

    for index, node in enumerate(nodes[1:]):
        angle = 2 * math.pi * index / (len(nodes) - 1)
        positions[node["id"]] = (
            1.5 * math.cos(angle),
            1.5 * math.sin(angle),
        )

    edge_x, edge_y = [], []

    for source, target in edges:
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"width": 1, "color": "#999999"},
        hoverinfo="skip",
        showlegend=False,
    )

    palette = {
        "Composition": "#E69F00",
        "Singers": "#56B4E9",
        "Ragas": "#009E73",
        "Talas": "#CC79A7",
        "Text records": "#0072B2",
        "Recordings": "#D55E00",
    }

    node_x, node_y, colors, hover_text, sizes = [], [], [], [], []

    for node in nodes:
        x, y = positions[node["id"]]
        node_x.append(x)
        node_y.append(y)
        colors.append(palette.get(node["group"], "#999999"))
        sizes.append(30 if node["id"] == "composition" else 20)

        details = node["details"]
        extra = "<br>".join(
            f"{key}: {str(value)[:150]}"
            for key, value in details.items()
            if value is not None and str(value).strip()
        )

        hover_text.append(
            f"<b>{node['group']}</b><br>{node['label']}<br>{extra}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[node["label"] for node in nodes],
        textposition="top center",
        hovertext=hover_text,
        hoverinfo="text",
        marker={
            "size": sizes,
            "color": colors,
            "line": {"width": 1, "color": "white"},
        },
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Composition and linked records",
        height=650,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        hovermode="closest",
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View original graph data"):
        st.json(graph_data)


def _force_directed_positions(nodes: list[dict[str, Any]], relationships: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Compute a deterministic Fruchterman-Reingold style layout."""
    node_ids = [str(node["id"]) for node in nodes]
    if not node_ids:
        return {}
    randomizer = random.Random(17)
    count = len(node_ids)
    positions = {
        node_id: [math.cos(2 * math.pi * index / count) + randomizer.uniform(-0.08, 0.08),
                  math.sin(2 * math.pi * index / count) + randomizer.uniform(-0.08, 0.08)]
        for index, node_id in enumerate(node_ids)
    }
    edges = [(str(edge["source"]), str(edge["target"])) for edge in relationships
             if str(edge.get("source")) in positions and str(edge.get("target")) in positions]
    area = max(4.0, count * 0.8)
    optimal_distance = math.sqrt(area / count)
    temperature = optimal_distance * 1.5

    for iteration in range(180):
        displacement = {node_id: [0.0, 0.0] for node_id in node_ids}
        for index, first in enumerate(node_ids):
            for second in node_ids[index + 1:]:
                dx = positions[first][0] - positions[second][0]
                dy = positions[first][1] - positions[second][1]
                distance = max(math.hypot(dx, dy), 0.01)
                force = optimal_distance * optimal_distance / distance
                fx, fy = dx / distance * force, dy / distance * force
                displacement[first][0] += fx
                displacement[first][1] += fy
                displacement[second][0] -= fx
                displacement[second][1] -= fy
        for source, target in edges:
            dx = positions[source][0] - positions[target][0]
            dy = positions[source][1] - positions[target][1]
            distance = max(math.hypot(dx, dy), 0.01)
            force = distance * distance / optimal_distance
            fx, fy = dx / distance * force, dy / distance * force
            displacement[source][0] -= fx
            displacement[source][1] -= fy
            displacement[target][0] += fx
            displacement[target][1] += fy
        temperature = optimal_distance * 1.5 * (1 - (iteration + 1) / 180)
        for node_id in node_ids:
            dx, dy = displacement[node_id]
            magnitude = max(math.hypot(dx, dy), 0.01)
            step = min(magnitude, temperature)
            positions[node_id][0] += dx / magnitude * step
            positions[node_id][1] += dy / magnitude * step

    return {node_id: (point[0], point[1]) for node_id, point in positions.items()}


def _render_graph_snapshot(snapshot: dict[str, Any]) -> None:
    """Draw a focused Neo4j neighborhood with node and relationship hover details."""
    if go is None:
        st.error("Plotly is required for the Connections graph. Install the project requirements and restart the app.")
        return
    nodes = snapshot.get("nodes") or []
    relationships = snapshot.get("relationships") or []
    if not nodes:
        st.info("Neo4j is reachable, but it has no connected nodes to display yet.")
        return

    palette = {
        "Composition": "#a77724", "Singer": "#477b9e", "Raga": "#43866e",
        "Tala": "#9b6383", "TextRecord": "#5876a1", "AudioRecording": "#bd684c",
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        labels = node.get("labels") or ["Node"]
        grouped.setdefault(str(labels[0]), []).append(node)

    groups = list(grouped.items())
    positions = _force_directed_positions(nodes, relationships)

    edge_x: list[Any] = []
    edge_y: list[Any] = []
    edge_labels: list[str] = []
    edge_mid_x: list[float] = []
    edge_mid_y: list[float] = []
    edge_hover: list[str] = []
    for edge in relationships:
        source, target = str(edge.get("source")), str(edge.get("target"))
        if source not in positions or target not in positions:
            continue
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend((x0, x1, None))
        edge_y.extend((y0, y1, None))
        edge_labels.append(str(edge.get("type", "RELATED_TO")))
        edge_mid_x.append((x0 + x1) / 2)
        edge_mid_y.append((y0 + y1) / 2)
        relation_properties = edge.get("properties") or {}
        relation_details = "<br>".join(
            f"{html.escape(str(key))}: {html.escape(str(value))}"
            for key, value in relation_properties.items() if value is not None
        )
        edge_hover.append("<b>Relationship: " + html.escape(str(edge.get("type", "RELATED_TO"))) + "</b>" + ("<br>" + relation_details if relation_details else ""))

    traces = [go.Scatter(x=edge_x, y=edge_y, mode="lines", line={"width": 1.5, "color": "#b7ad9a"}, hoverinfo="skip", showlegend=False)]
    if edge_labels:
        traces.append(go.Scatter(x=edge_mid_x, y=edge_mid_y, mode="markers+text", text=edge_labels, textposition="middle center", textfont={"size": 9, "color": "#625e52"}, hovertext=edge_hover, hoverinfo="text", marker={"size": 7, "color": "#b7ad9a", "opacity": 0.01}, showlegend=False))
    for label, members in groups:
        xs, ys, texts, hovers = [], [], [], []
        for node in members:
            x, y = positions[str(node["id"])]
            props = node.get("properties") or {}
            title = (props.get("canonical_title") or props.get("name") or props.get("text_type") or props.get("recording_title") or props.get("composition_id") or props.get("text_id") or props.get("audio_id") or label)
            short_title = " ".join(str(title).split())
            texts.append(short_title if len(short_title) <= 22 else short_title[:20] + "…")
            property_lines = "<br>".join(f"{html.escape(str(key))}: {html.escape(str(value))}" for key, value in props.items() if value is not None)
            hovers.append("<b>" + html.escape(label) + "</b><br>" + html.escape(str(title)) + ("<br>" + property_lines if property_lines else ""))
            xs.append(x)
            ys.append(y)
        traces.append(go.Scatter(x=xs, y=ys, mode="markers+text", text=texts, textposition="top center", hovertext=hovers, hoverinfo="text", name=label, marker={"size": 28 if label == "Composition" else 17, "symbol": "diamond" if label == "Composition" else "circle", "color": palette.get(label, "#888888"), "line": {"width": 1.5, "color": "white"}}))

    fig = go.Figure(data=traces)
    fig.update_layout(height=760, plot_bgcolor="#fffefa", paper_bgcolor="#fffefa", hovermode="closest", dragmode="pan", margin={"l": 20, "r": 20, "t": 20, "b": 20}, legend={"orientation": "h", "y": -0.04}, xaxis={"visible": False}, yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1})
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True, "staticPlot": False})
    st.caption("Drag the background to pan, scroll to zoom, and hover over nodes or relationship markers for details.")


def _render_connections(settings: dict[str, Any]) -> None:
    st.markdown("### Connections")
    st.caption("Explore the full knowledge graph or focus on one composition.")
    if not (settings["graph_uri"] and settings["graph_username"] and settings["graph_password"]):
        st.info("Neo4j is not configured. Set NEO4J_URI, NEO4J_USERNAME (or NEO4J_USER), and NEO4J_PASSWORD, then restart the app.")
        return
    from knowledge_graph import AnnamayyaKnowledgeGraph
    graph = AnnamayyaKnowledgeGraph(uri=settings["graph_uri"], username=settings["graph_username"], password=settings["graph_password"])
    try:
        graph.connect()
        mode = st.sidebar.radio(
            "Graph view",
            ["Full Knowledge Graph", "Single Song Explorer"],
            key="connections_graph_mode",
        )
        if mode == "Full Knowledge Graph":
            snapshot = graph.get_full_graph_snapshot()
            composition_id = None
            hops = None
        else:
            compositions = [row.get("composition") or {} for row in graph.get_compositions()]
            compositions = [row for row in compositions if row.get("composition_id")]
            if not compositions:
                st.info("Neo4j is reachable, but no Composition nodes were found for the song explorer.")
                return
            by_id = {str(row["composition_id"]): row for row in compositions}
            composition_id = st.sidebar.selectbox(
                "Focus composition",
                options=list(by_id),
                format_func=lambda value: f"{by_id[value].get('canonical_title') or value} · {value}",
                key="connections_composition_id",
            )
            hops = st.sidebar.selectbox("Neighborhood depth", [1, 2], index=0, format_func=lambda value: f"{value} hop{'s' if value > 1 else ''}", key="connections_hops")
            snapshot = graph.get_composition_neighborhood(composition_id, hops=hops, limit=250)
    except Exception as exc:
        LOGGER.warning("Could not load Neo4j graph: %s", type(exc).__name__)
        st.error("Neo4j could not be reached. Check the URI, credentials, and that the database is running.")
        return
    finally:
        graph.close()
    if not snapshot["nodes"]:
        st.info("Neo4j is reachable, but the database contains no graph nodes to display.")
        return
    node_count = len(snapshot["nodes"])
    edge_count = len(snapshot["relationships"])
    if composition_id is None:
        composition_count = sum("Composition" in (node.get("labels") or []) for node in snapshot["nodes"])
        st.caption(f"Full graph: {composition_count} compositions, {node_count} total nodes, and {edge_count} relationships.")
    else:
        st.caption(f"Neighborhood for {by_id[composition_id].get('canonical_title') or composition_id}: {node_count} nodes and {edge_count} relationships across {hops} hop{'s' if hops > 1 else ''} (up to 250 nodes and relationships).")
    _render_graph_snapshot(snapshot)


def _metadata_line(metadata: dict[str, Any]) -> str:
    fields = []
    seen: set[str] = set()
    for label, key in (("Singer", "singer"), ("Raga", "raga"), ("Tala", "tala")):
        value = _local_value(metadata, key)
        if value is None or "no reliable" in str(value).casefold():
            continue
        normalized = " ".join(str(value).split()).casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        fields.append(f"{label}: {value}")
    return " · ".join(fields)


def _excerpt(content: Any, limit: int = 260) -> str:
    text = str(content or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip() + "…"


def _local_recording_path(audio_records: list[dict[str, Any]]) -> Path | None:
    for row in audio_records:
        raw_path = row.get("audio_path")
        if not raw_path:
            continue
        path = Path(str(raw_path))
        candidate = path if path.is_absolute() else DATASET_DIR / path
        if candidate.is_file():
            return candidate
    return None


def _render_result(result: dict[str, Any], rank: int, data: dict[str, Any]) -> None:
    composition_id = str(result.get("composition_id") or "Unknown ID")
    title = result.get("canonical_title") or composition_id
    local_rows = get_composition_rows(data, composition_id)
    metadata = _metadata_for_result(result, local_rows["composition"])
    text_evidence = result.get("text_evidence") or {}
    best_text = text_evidence.get("best_match") or {}
    audio_evidence = result.get("audio_evidence") or {}
    modalities = result.get("modalities") or [
        name for name, evidence in (("text", text_evidence), ("audio", audio_evidence)) if evidence
    ]
    labels = []
    if "text" in modalities and "audio" in modalities:
        labels.append("Text + audio")
    elif "text" in modalities:
        labels.append("Text match")
    elif "audio" in modalities:
        labels.append("Audio match")
    badges = "".join(f'<span class="badge">{_escape(value)}</span>' for value in labels)
    st.markdown(
        f'<div class="result-card"><p class="result-title">{rank}. {_escape(title)}</p>{badges}</div>',
        unsafe_allow_html=True,
    )
    metadata_text = _metadata_line(metadata)
    if metadata_text:
        st.caption(metadata_text)

    text_records = local_rows["text_records"]
    if best_text.get("content"):
        matching_id = best_text.get("text_id")
        selected = next((row for row in text_records if row.get("text_id") == matching_id), best_text)
        text_records = [selected] + [row for row in text_records if row.get("text_id") != matching_id]
    shown_types: set[str] = set()
    for record in text_records:
        content = record.get("content")
        kind = str(record.get("text_type") or "text").casefold()
        if not content or kind in shown_types:
            continue
        shown_types.add(kind)
        label = {
            "lyrics": "Telugu lyrics",
            "transliteration": "Romanized lyrics",
            "translation": "English meaning",
        }.get(kind, str(record.get("language") or "Text"))
        st.markdown(f"**{label}**")
        st.text(_excerpt(content))

    recording_path = _local_recording_path(local_rows["audio_records"])
    if recording_path:
        st.audio(str(recording_path))
    elif local_rows["audio_records"]:
        st.caption("Recording unavailable in the local package.")

    source_rows = [local_rows["composition"], *text_records, *local_rows["audio_records"]]
    source_labels = {
        "text_source_url": "Lyrics source",
        "transliteration_source_url": "Transliteration source",
        "translation_source_url": "Meaning source",
        "audio_source_url": "Recording source",
        "source_url": "Text source",
    }
    rendered_urls: set[str] = set()
    source_links = []
    for source_row in source_rows:
        if not source_row:
            continue
        for source_key, label in source_labels.items():
            url = _valid_url(source_row.get(source_key))
            if url and url not in rendered_urls:
                rendered_urls.add(url)
                source_links.append(
                    f'<a href="{_escape(url)}" target="_blank" rel="noopener noreferrer">{_escape(label)}</a>'
                )
    if source_links:
        st.markdown("Sources: " + " · ".join(source_links), unsafe_allow_html=True)

    with st.expander("Full lyrics, meanings & song details", expanded=False):
        meta = _metadata_line(metadata)
        st.write(meta or "No additional singer, raga, or tala metadata is available.")
        if text_records:
            for record in local_rows["text_records"]:
                if record.get("content"):
                    label = record.get("language") or record.get("text_type") or "Text"
                    st.markdown(f"**{_escape(str(label))}**")
                    st.text(str(record["content"]))
                    _render_source_links(record)
            available_types = {str(row.get("text_type") or "").casefold() for row in local_rows["text_records"]}
            for missing_type, message in (
                ("lyrics", "Telugu lyrics not available in the local dataset."),
                ("transliteration", "Romanized lyrics not available in the local dataset."),
                ("translation", "English meaning not available in the local dataset."),
            ):
                if missing_type not in available_types:
                    st.caption(message)
        else:
            st.info("Lyrics and meanings are not available in the local dataset for this composition.")
        _render_source_links(local_rows["composition"])
        _render_source_links(local_rows["audio_records"][0] if local_rows["audio_records"] else None)

    with st.expander("Why this result?", expanded=False):
        st.caption("Composition record: " + composition_id)
        for modality, score_key, rank_key, evidence in (
            ("Text", "text_score", "text_rank", text_evidence),
            ("Audio", "audio_score", "audio_rank", audio_evidence),
        ):
            score = result.get(score_key)
            if score is not None:
                branch = evidence.get("branch_result") or {}
                st.write(f"{modality}: rank {result.get(rank_key, 'N/A')}, {branch.get('score_type', 'score')} {format_score(score)}")
        if result.get("fused_score") is not None:
            st.write(f"Combined ranking score: {format_score(result['fused_score'])} (rank-fusion score, not a probability)")
        if result.get("graph_data"):
            _render_graph_data(result["graph_data"])


def _run_search(
    data: dict[str, Any], settings: dict[str, Any], text_query: str, upload: Any
) -> None:
    # A submitted search replaces the old result set, including when its input
    # proves invalid, so a previous query can never masquerade as the new one.
    st.session_state.pop("search_results", None)
    st.session_state.pop("search_error", None)
    st.session_state.pop("last_search", None)
    st.session_state.pop("graph_warning", None)
    st.session_state.pop("ai_explanation", None)
    st.session_state.pop("ai_history", None)
    st.session_state.pop("ai_history_fingerprint", None)
    st.session_state.pop("ai_question", None)
    st.session_state.pop("ai_selected_composition", None)
    query_text = text_query.strip() or None
    if query_text is None and upload is None:
        st.warning("Enter a text query, add an audio recording, or provide both before searching.")
        return
    active_weights = []
    if query_text is not None:
        active_weights.append(settings["text_weight"])
    if upload is not None:
        active_weights.append(settings["audio_weight"])
    if not any(weight > 0 for weight in active_weights):
        st.warning("At least one modality weight must be greater than zero.")
        return

    audio_digest = audio_upload_digest(upload)
    results = None
    graph = None
    graph_warning = None
    if settings["enrich_graph"]:
        if not (settings["graph_uri"] and settings["graph_username"] and settings["graph_password"]):
            graph_warning = "Graph enrichment needs a URI, username, and password; searching without the graph."
        else:
            try:
                graph = initialize_graph(
                    settings["graph_uri"], settings["graph_username"], settings["graph_password"]
                )
            except Exception as exc:
                diagnostic = str(exc)
                for secret in (settings["graph_password"], settings["graph_username"], settings["graph_uri"]):
                    if secret:
                        diagnostic = diagnostic.replace(secret, "[redacted]")
                LOGGER.warning("Neo4j connection failed: %s", diagnostic)
                graph_warning = "Neo4j is unavailable; search will continue without graph enrichment."

    engine = MultimodalRetrievalEngine(
        text_engine=data.get("text_engine"),
        audio_engine=data.get("audio_engine"),
        compositions=data["compositions"],
        text_weight=settings["text_weight"],
        audio_weight=settings["audio_weight"],
        knowledge_graph=graph,
    )
    audio_context = None
    try:
        if upload is not None:
            audio_context = temporary_audio_file(upload)
            audio_path = audio_context.__enter__()
        else:
            audio_path = None
        results = engine.search(
            query_text,
            audio_path,
            top_k=settings["top_k"],
            raise_on_branch_error=False,
        )
    except (ValueError, OSError) as exc:
        LOGGER.info("Search input could not be processed (%s).", type(exc).__name__)
        if upload is not None:
            st.error("The audio could not be decoded or processed. Check that it is a valid, non-empty WAV or MP3 file.")
        else:
            st.error("The text query could not be processed. Check the input and try again.")
        return
    except Exception as exc:
        LOGGER.exception("Unexpected retrieval failure.")
        st.error(f"Search failed ({type(exc).__name__}). Please check the input and try again.")
        return
    finally:
        if audio_context is not None:
            audio_context.__exit__(None, None, None)
        if graph is not None:
            try:
                graph.close()
            except Exception as exc:
                LOGGER.warning("Neo4j driver cleanup failed (%s).", type(exc).__name__)

    branch_errors = dict(engine.last_branch_errors)
    if graph is not None and engine.last_graph_error:
        safe_graph_error = str(engine.last_graph_error)
        for secret in (settings["graph_password"], settings["graph_username"], settings["graph_uri"]):
            if secret:
                safe_graph_error = safe_graph_error.replace(secret, "[redacted]")
        LOGGER.warning("Graph enrichment query failed: %s", safe_graph_error)
        graph_warning = "Some graph details could not be loaded; available retrieval results are shown."

    st.session_state["search_results"] = results or []
    st.session_state["search_error"] = branch_errors
    st.session_state["graph_warning"] = graph_warning
    st.session_state["last_search"] = {
        "text_query": query_text,
        "audio_digest": audio_digest,
        "mode": "multimodal" if query_text and upload else "text" if query_text else "audio",
        "top_k": settings["top_k"],
        "text_weight": settings["text_weight"],
        "audio_weight": settings["audio_weight"],
        "audio_filename": getattr(upload, "name", None) if upload is not None else None,
    }


def _render_explanation_sources(evidence: dict[str, Any]) -> None:
    """Expose the actual bounded records supporting an AI answer."""
    for candidate in evidence.get("candidates", []):
        composition = candidate.get("composition") or {}
        st.markdown(
            f"**{composition.get('canonical_title') or composition.get('composition_id', 'Composition')}** "
            f"· `{composition.get('composition_id', 'unknown ID')}` · retrieved rank {candidate.get('rank', 'N/A')}"
        )
        if candidate.get("fused_score") is not None:
            st.caption(
                f"RRF rank score {format_score(candidate.get('fused_score'))}; "
                f"text {format_score(candidate.get('text_score'))} ({candidate.get('text_score_type', 'not available')}); "
                f"audio {format_score(candidate.get('audio_score'))} ({candidate.get('audio_score_type', 'not available')}). Scores are not probabilities."
            )
        for record in candidate.get("text_records", []):
            record_id = record.get("text_id", "ID unavailable")
            st.caption(
                f"Text record {record_id} · {record.get('text_type', 'type unavailable')} · "
                f"{record.get('language', 'language unavailable')}"
            )
            verification = record.get("verification_status", record.get("verified", record.get("is_verified")))
            if verification is not None:
                st.caption(f"Source verification status: {verification}")
            if record.get("content"):
                st.text(record["content"])
            for url in (record.get("sources") or {}).values():
                valid = _valid_url(url)
                if valid:
                    st.markdown(
                        f'<a href="{_escape(valid)}" target="_blank" rel="noopener noreferrer">'
                        f"Source for {_escape(record_id)}</a>",
                        unsafe_allow_html=True,
                    )
        for record in candidate.get("audio_recordings", []):
            audio_id = record.get("audio_id", "ID unavailable")
            st.caption(
                f"Audio record {audio_id} · {record.get('recording_title', 'title unavailable')} · "
                f"processed by audio search: {record.get('processed_in_audio_search', 'not indicated')}"
            )
            for url in (record.get("sources") or {}).values():
                valid = _valid_url(url)
                if valid:
                    st.markdown(
                        f'<a href="{_escape(valid)}" target="_blank" rel="noopener noreferrer">'
                        f"Audio source for {_escape(audio_id)}</a>",
                        unsafe_allow_html=True,
                    )
        if candidate.get("graph"):
            st.caption("Graph relationships below are returned by the configured knowledge graph.")
            st.json(candidate["graph"])


def _render_ai_explanation(data: dict[str, Any]) -> None:
    """Render explicit, evidence-bound Gemini requests without rerun calls."""
    st.markdown("---")
    st.markdown("### Ask about your results")
    st.caption(
        "Gemini explains retrieved records; it does not retrieve, rank, verify authenticity, "
        "or certify translations. Your question and selected evidence are sent to Google's API only when submitted. "
        "If you need different records, submit a new search above; that replaces this evidence and clears the old answer."
    )
    results = st.session_state.get("search_results", [])
    if not results:
        st.info("Run a search with at least one result before asking Gemini to explain the evidence.")
        return

    result_by_id = {str(row.get("composition_id")): row for row in results}
    composition_ids = list(result_by_id)

    def selection_label(composition_id: str | None) -> str:
        if composition_id is None:
            return "All retrieved candidates (up to the evidence limit)"
        row = result_by_id.get(str(composition_id), {})
        return f"{row.get('canonical_title') or composition_id} · {composition_id}"

    selected_id = st.selectbox(
        "Evidence to explain",
        options=[None, *composition_ids],
        format_func=selection_label,
        key="ai_selected_composition",
    )
    try:
        evidence = prepare_evidence(
            results,
            compositions=data["compositions"],
            text_records=data["text_records"],
            audio_records=data["audio_records"],
            selected_composition_id=selected_id,
        )
    except (TypeError, ValueError) as exc:
        LOGGER.warning("Could not prepare explanation evidence (%s).", type(exc).__name__)
        st.error("The selected retrieval evidence could not be prepared. Run a new search and try again.")
        return

    fingerprint = evidence_fingerprint(evidence)
    saved = st.session_state.get("ai_explanation")
    if saved and saved.get("evidence_fingerprint") != fingerprint:
        st.session_state.pop("ai_explanation", None)
        st.session_state["ai_history"] = []
        st.session_state["ai_history_fingerprint"] = fingerprint
        saved = None
        st.info("The evidence selection changed, so the previous explanation was cleared.")
    if st.session_state.get("ai_history_fingerprint") != fingerprint:
        st.session_state["ai_history"] = []
        st.session_state["ai_history_fingerprint"] = fingerprint

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    with st.form("gemini_explanation_form", clear_on_submit=False):
        question = st.text_area(
            "Ask about the selected evidence",
            placeholder="Why was this composition returned? What information is missing?",
            max_chars=5000,
            key="ai_question",
            disabled=False,
        )
        submitted = st.form_submit_button(
            "Generate Explanation",
            disabled=False,
        )

    if submitted:
        if not question or not question.strip():
            st.warning("Enter a question before requesting an explanation.")
        elif not evidence.get("candidates"):
            st.info("No retrieved evidence is available for this question; run another search first.")
        else:
            request_identity = evidence_fingerprint({
                "evidence": fingerprint,
                "question": question.strip(),
                "model": model,
            })
            current = st.session_state.get("ai_explanation")
            if current and current.get("request_identity") == request_identity:
                st.info("This question has already been answered for this evidence and model; showing the saved response.")
            else:
                service = GeminiExplainer(api_key=api_key, model=model)
                try:
                    if not api_key:
                        raise GeminiExplanationError("Gemini is not configured.")
                    with st.spinner("Preparing a source-grounded explanation…"):
                        generated = service.explain(question, evidence, history=st.session_state.get("ai_history", []))
                    st.session_state["ai_explanation"] = {
                        **generated, "question": question.strip(), "request_identity": request_identity,
                        "evidence": evidence, "evidence_fingerprint": fingerprint, "prepared_demo": False,
                    }
                except GeminiExplanationError as exc:
                    if _is_demo_request(question, selected_id, result_by_id):
                        st.session_state["ai_explanation"] = {
                            "answer": DEMO_ANSWER, "model": "Prepared demo explanation", "usage": {},
                            "question": question.strip(), "request_identity": request_identity,
                            "evidence": evidence, "evidence_fingerprint": fingerprint, "prepared_demo": True,
                        }
                    else:
                        st.info("Live AI responses are temporarily unavailable. Please try again later.")
                except Exception as exc:
                    LOGGER.exception("Unexpected explanation-layer failure.")
                    st.error(f"Gemini explanation failed ({type(exc).__name__}). Search results are unchanged; you can retry.")
                finally:
                    service.close()

    saved = st.session_state.get("ai_explanation")
    if saved and saved.get("evidence_fingerprint") == fingerprint:
        st.markdown("#### Generated explanation")
        if not saved.get("prepared_demo"):
            st.caption(
                f"Question: {saved['question']} · Model: {saved['model']} · "
                "AI-generated interpretation; not an authoritative or verified translation."
            )
        st.markdown(saved["answer"])
        usage = saved.get("usage") or {}
        if usage.get("total_tokens") is not None:
            st.caption(f"Reported token usage: {usage['total_tokens']}")
        with st.expander("Inspect the exact evidence snapshot sent to Gemini", expanded=False):
            _render_explanation_sources(saved["evidence"])
    elif st.session_state.get("ai_history"):
        st.caption("Follow-up context is limited to recent turns for this exact evidence snapshot.")


def main() -> None:
    if st is None:
        raise RuntimeError("Streamlit is required. Install project dependencies with: python -m pip install -r requirements.txt")

    st.set_page_config(page_title="Annamayya Knowledge Explorer", page_icon="🎼", layout="wide", initial_sidebar_state="expanded")
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    _render_header()
    settings = _sidebar_settings()
    destination = st.sidebar.radio("Explore", ["Discover", "Connections"], key="app_destination", horizontal=True)
    if destination == "Connections":
        _render_connections(settings)
        return
    try:
        with st.spinner("Preparing the text and audio indexes…"):
            data = initialize_engines(compute_dataset_fingerprint())
    except Exception as exc:
        LOGGER.exception("Core dataset initialization failed.")
        st.error(f"The dataset could not be loaded ({type(exc).__name__}). Check that the packaged workbook is available.")
        return
    for branch in data["branch_initialization_errors"]:
        if branch == "text":
            st.warning("Text search is unavailable. Check LaBSE model access and the application logs.")
        else:
            st.warning("Audio search is unavailable. Check the audio dependencies and application logs.")
    if data["audio_index_stats"]:
        stats = data["audio_index_stats"]
        st.caption(f"Audio index ready · {stats['indexed']} recordings indexed, {stats['skipped']} unavailable, {stats['failed']} failed.")
    if data["text_engine"] is None and data["audio_engine"] is None:
        st.error("Neither text nor audio retrieval could be initialized. Review dependencies and application logs.")
        return
    st.markdown("### Find a composition")
    with st.form("composition_search", clear_on_submit=False):
        text_query = st.text_input("What would you like to find?", placeholder="A song title, lyric, theme, or meaning…", key="text_query_input")
        st.caption("Search in Telugu, Romanized Telugu, or English. Add a recording for a combined search.")
        upload = st.file_uploader("Optional audio recording", type=["wav", "mp3"], accept_multiple_files=False, help="Supported audio is analyzed against the recordings available in this collection.", key="audio_query_upload")
        submitted = st.form_submit_button("Find compositions", use_container_width=True)
    if submitted:
        _run_search(data, settings, text_query, upload)
    if "search_results" in st.session_state:
        last_search = st.session_state.get("last_search", {})
        st.markdown("---")
        st.markdown("### Compositions")
        for branch, _message in st.session_state.get("search_error", {}).items():
            st.warning(f"{branch.title()} search is unavailable right now; other available results are shown.")
        if st.session_state.get("graph_warning"):
            st.warning(st.session_state["graph_warning"])
        if last_search.get("text_query"):
            st.caption(f"Text query: {last_search['text_query']}")
        if last_search.get("audio_digest"):
            st.caption(f"Audio considered: {last_search.get('audio_filename') or 'uploaded recording'}")
        results = st.session_state["search_results"]
        if not results:
            st.info("No matching compositions were found. Try different wording or another recording.")
        else:
            st.caption(f"{len(results)} composition match{'es' if len(results) != 1 else ''}")
            for rank, result in enumerate(results, start=1):
                _render_result(result, rank, data)
        _render_ai_explanation(data)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
