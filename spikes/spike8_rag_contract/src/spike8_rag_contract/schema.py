from __future__ import annotations

from pathlib import Path
from typing import Final

from . import paths

# ---- Envelope (shared) ----
ENVELOPE_BASE: Final = {
    "type": "object",
    "required": [
        "answer",
        "citations",
        "trace_id",
        "timings",
        "cost",
        "model_id",
        "index_version",
        "retrieved",
    ],
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "trace_id": {"type": "string"},
        "timings": {
            "type": "object",
            "properties": {
                "retrieve_ms": {"type": "number"},
                "generate_ms": {"type": "number"},
            },
            "required": ["retrieve_ms", "generate_ms"],
            "additionalProperties": True,
        },
        "cost": {"type": "object", "additionalProperties": True},
        "model_id": {"type": "string"},
        "index_version": {"type": "string"},
        "retrieved": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["chunk_id", "score"],
                "properties": {
                    "chunk_id": {"type": "string"},
                    "start_ts": {"type": "number"},
                    "end_ts": {"type": "number"},
                    "score": {"type": "number"},
                    "text": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
    },
    "additionalProperties": True,
}

# ---- Specific output bodies (answer/summary/digest) ----
ANSWER_BODY: Final = {"type": "string"}
GUEST_SUMMARY_BODY: Final = {
    "type": "object",
    "required": ["guest", "key_points"],
    "properties": {
        "guest": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}
WEEKLY_DIGEST_BODY: Final = {
    "type": "object",
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title", "summary"],
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "episode_id": {"type": "string"},
                    "timestamp": {"type": "number"},
                },
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}

WEEKLY_DIGEST_SCHEMA: Final = {
    "type": "object",
    "required": ["items"],
    "properties": {
        "week": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title", "summary", "episode_id", "timestamp"],
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "episode_id": {"type": "string"},
                    "timestamp": {"type": "number"},
                },
            },
        },
    },
    "additionalProperties": False,
}


# ---- Envelopes (compose base + specific bodies) ----
def _envelope_with_body(body_schema: dict) -> dict:
    schema = dict(ENVELOPE_BASE)  # shallow copy is OK (no nested mutations here)
    schema = {**schema}
    # override 'answer' property to accept object/string depending on body
    schema["properties"] = {**schema["properties"], "answer": body_schema}
    return schema


ANSWER_ENVELOPE_SCHEMA: Final = _envelope_with_body(ANSWER_BODY)
GUEST_SUMMARY_ENVELOPE_SCHEMA: Final = _envelope_with_body(GUEST_SUMMARY_BODY)
WEEKLY_DIGEST_ENVELOPE_SCHEMA: Final = _envelope_with_body(WEEKLY_DIGEST_BODY)

# Back-compat: keep ANSWER_SCHEMA name in imports
ANSWER_SCHEMA: Final = ANSWER_ENVELOPE_SCHEMA


# ---- Persist schemas to disk (idempotent) ----
def write_schemas() -> list[Path]:
    paths.SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    to_write = {
        "answer_envelope.schema.json": ANSWER_ENVELOPE_SCHEMA,
        "guest_summary_envelope.schema.json": GUEST_SUMMARY_ENVELOPE_SCHEMA,
        "weekly_digest_envelope.schema.json": WEEKLY_DIGEST_ENVELOPE_SCHEMA,
    }
    import json

    for fname, schema in to_write.items():
        p = paths.SCHEMAS_DIR / fname
        p.write_text(json.dumps(schema, indent=2))
        out_paths.append(p)
    return out_paths
