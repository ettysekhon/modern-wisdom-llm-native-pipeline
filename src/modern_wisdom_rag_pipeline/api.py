from __future__ import annotations

from fastapi import FastAPI
from spike8_rag_contract import paths

app = FastAPI(title="Modern Wisdom RAG API", version="0.1.0")


@app.get("/")
def root():
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
