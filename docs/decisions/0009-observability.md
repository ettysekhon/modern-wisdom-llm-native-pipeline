# Decision 0009 — Observability & Tracing (Phoenix)

**Date:** YYYY-MM-DD  
**Owner:** your-name

## Context

- We need end-to-end visibility across retrieval → generation → validation for RAG CLI.

## Decision

- Use Arize Phoenix + OpenInference + OTEL.
- Instrument retriever (Qdrant), LLM (OpenAI or mock), and JSON Schema validation as spans.
- Emit attributes: k, index_version, model_id, tokens, p95 latency.
- Collect optional user feedback as a span.

## Consequences

- Phoenix UI shows full traces and metrics; baseline dashboards created.
- Enables later per-run evals and SLO monitoring in API/agents.

## Artifacts

- `data/monitoring/phoenix_dashboard.json` (export later from UI)
