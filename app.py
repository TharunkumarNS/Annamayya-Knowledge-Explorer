"""Compatibility entry point for the FastAPI application.

Run with ``uvicorn app:app --reload``. The browser UI is served independently
from ``frontend/``; Streamlit is no longer part of the application runtime.
"""

from api import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
