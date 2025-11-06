from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from modern_wisdom_rag_pipeline import paths

app = FastAPI(title="Modern Wisdom RAG API", version="0.1.0")


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/info")
def info():
    """Get API information."""
    return {
        "name": "modern_wisdom_rag_pipeline",
        "status": "ok",
        "index": paths.INDEX_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}
