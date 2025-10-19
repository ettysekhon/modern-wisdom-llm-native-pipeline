from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

SPECS_DIR: Final = Path("data/agents/specs")
SPECS_DIR.mkdir(parents=True, exist_ok=True)


def _write(name: str, obj: dict[str, Any]) -> Path:
    p = SPECS_DIR / f"{name}.json"
    p.write_text(json.dumps(obj, indent=2))
    return p


# ---- Tool input/output JSON Schemas ----

RAG_SEARCH_IN = {
    "type": "object",
    "required": ["question", "top_k", "scope"],  # episode_id is optional (can be null)
    "properties": {
        "question": {"type": "string"},
        "episode_id": {"type": ["string", "null"]},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
        "scope": {"type": "string", "enum": ["episode", "corpus", "auto"]},
    },
    "additionalProperties": False,
}

RAG_SEARCH_OUT = {
    "type": "object",
    "required": ["retrieved", "retrieve_ms", "fallback_used", "scope"],
    "properties": {
        "retrieved": {"type": "array"},
        "retrieve_ms": {"type": "number"},
        "fallback_used": {"type": "boolean"},
        "scope": {"type": "string", "enum": ["episode", "corpus"]},
    },
    "additionalProperties": True,
}

EPISODE_LOCATOR_IN = {
    "type": "object",
    "required": ["query", "limit"],
    "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
    },
    "additionalProperties": False,
}

EPISODE_LOCATOR_OUT = {
    "type": "object",
    "required": ["episodes"],
    "properties": {
        "episodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["episode_id", "title", "guest", "headline", "description"],
                "properties": {
                    "episode_id": {"type": "string"},
                    "title": {"type": "string"},
                    "guest": {"type": "string"},
                    "headline": {"type": "string"},
                    "description": {"type": "string"},
                },
                "additionalProperties": True,
            },
        }
    },
    "additionalProperties": False,
}

TIMELINE_BUILDER_IN = {
    "type": "object",
    "required": ["segments"],
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["episode_id", "start_ts", "end_ts", "text"],
                "properties": {
                    "episode_id": {"type": "string"},
                    "start_ts": {"type": "number"},
                    "end_ts": {"type": "number"},
                    "text": {"type": "string"},
                },
                "additionalProperties": True,
            },
        }
    },
    "additionalProperties": False,
}

TIMELINE_BUILDER_OUT = {
    "type": "object",
    "required": ["timeline"],
    "properties": {
        "timeline": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["episode_id", "t0", "t1", "summary"],
                "properties": {
                    "episode_id": {"type": "string"},
                    "t0": {"type": "number"},
                    "t1": {"type": "number"},
                    "summary": {"type": "string"},
                },
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}

SQL_DUCKDB_IN = {
    "type": "object",
    "required": ["sql"],
    "properties": {"sql": {"type": "string"}},
    "additionalProperties": False,
}

SQL_DUCKDB_OUT = {
    "type": "object",
    "required": ["rows"],
    "properties": {"rows": {"type": "array"}},
    "additionalProperties": False,
}

CLIP_LINKER_IN = {
    "type": "object",
    "required": ["episode_id", "timestamp"],
    "properties": {"episode_id": {"type": "string"}, "timestamp": {"type": "number"}},
    "additionalProperties": False,
}

CLIP_LINKER_OUT = {
    "type": "object",
    "required": ["url"],
    "properties": {"url": {"type": "string"}},
    "additionalProperties": False,
}


def write_specs() -> list[Path]:
    out_paths = []
    out_paths.append(_write("rag_search.in", RAG_SEARCH_IN))
    out_paths.append(_write("rag_search.out", RAG_SEARCH_OUT))
    out_paths.append(_write("episode_locator.in", EPISODE_LOCATOR_IN))
    out_paths.append(_write("episode_locator.out", EPISODE_LOCATOR_OUT))
    out_paths.append(_write("timeline_builder.in", TIMELINE_BUILDER_IN))
    out_paths.append(_write("timeline_builder.out", TIMELINE_BUILDER_OUT))
    out_paths.append(_write("sql_duckdb.in", SQL_DUCKDB_IN))
    out_paths.append(_write("sql_duckdb.out", SQL_DUCKDB_OUT))
    out_paths.append(_write("clip_linker.in", CLIP_LINKER_IN))
    out_paths.append(_write("clip_linker.out", CLIP_LINKER_OUT))
    return out_paths
