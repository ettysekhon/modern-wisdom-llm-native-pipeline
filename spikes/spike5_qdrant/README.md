# Spike 5 — Qdrant Collection & Blue-Green Alias

**Date:** 2025-10-09
**Owner:** ettyekhon

---

## Qdrant Objective

Integrate **Qdrant vector search** as the production index for all chunked + embedded podcast transcripts (from Spikes 3 & 4), ensuring:

- Deterministic collection naming per embedding version (`emb_v`)
- Idempotent upserts (no duplication)
- Blue/green deployment pattern via live alias
- Local persistence (via Docker volume)
- Simple CLI tooling for upsert, query, and health check

---

## Qdrant Inputs

| Source                 | Description                                                                               | Example                   |
| ---------------------- | ----------------------------------------------------------------------------------------- | ------------------------- |
| **Embeddings Parquet** | Output from Spike 4 (`/data/embeddings/<emb_v>/episode_id=.../part-00000.snappy.parquet`) | `fe_bge_small_en_v1_5_v1` |
| **Chunk Metadata**     | From Spike 3 chunking results                                                             | `sentence_bound`          |
| **Qdrant Instance**    | Local Docker container                                                                    | `http://localhost:6333`   |

---

## Qdrant Output

| Type             | Location / Name                       | Notes                                     |
| ---------------- | ------------------------------------- | ----------------------------------------- |
| **Collection**   | `mw_chunks_<emb_v>`                   | e.g., `mw_chunks_fe_bge_small_en_v1_5_v1` |
| **Alias**        | `mw_chunks_live`                      | Always points to latest “live” collection |
| **Qdrant Data**  | persisted in `./data/qdrant`          | Mounted as Docker volume                  |
| **CLI Commands** | `run-spike5 upsert`, `check`, `query` | Fully idempotent                          |

---

## Qdrant Setup (Qdrant local)

```bash
docker compose -f infra/qdrant/docker-compose.yml up -d
```

Qdrant Web UI:
[http://localhost:6333/dashboard](http://localhost:6333/dashboard)

---

## How to Run (End-to-End)

### Upsert episode embeddings

```bash
uv run run-spike5 upsert \
  --episode-id 0a4fa77e-bc0f-11ef-bab6-3f37b4906b43 \
  --method sentence_bound \
  --emb-v fe_bge_small_en_v1_5_v1 \
  --set-live
```

Expected output:

```text
Alias set: mw_chunks_live → mw_chunks_fe_bge_small_en_v1_5_v1
```

###  Check collection + alias

```bash
uv run run-spike5 check --emb-v fe_bge_small_en_v1_5_v1
```

Expected:

```text
mw_chunks_fe_bge_small_en_v1_5_v1: status=green, vector_size=384, distance=Cosine, points=66
Aliases for mw_chunks_fe_bge_small_en_v1_5_v1: ['mw_chunks_live']
```

#### Query (sanity search)

```bash
uv run run-spike5 query \
  --emb-v fe_bge_small_en_v1_5_v1 \
  --episode-id 0a4fa77e-bc0f-11ef-bab6-3f37b4906b43 \
  --top-k 5
```

Expected:

```text
id=uuid-123 score=0.91 episode=0a4fa77e-bc0f-11ef-bab6-3f37b4906b43
...
```

---

### Decision

**Chosen approach:**
Qdrant + FastEmbed + DuckDB
(using `qdrant-client` v1.15.x API with `get_collection_aliases`)

**Rationale:**

- Local, fast, embeddable vector DB
- Deterministic collection per `emb_v`
- Explicit blue/green alias (`mw_chunks_live`)
- Supports both HTTP & gRPC APIs
- Integrates cleanly with FastEmbed (no external OpenAI dependency)
