# Modern Wisdom – LLM Native Pipeline

## Motivation and Context

This project applies retrieval-augmented generation (RAG) techniques to *Modern Wisdom*, a long-form podcast hosted by Chris Williamson.
The motivation came from wanting to make multi-hour episodes **searchable, comparable, and summarised** through natural language questions.

Modern Wisdom covers deep topics – philosophy, self-improvement, science, and culture – but most insights are locked inside audio.
The aim was to build an **LLM-native pipeline** that transforms raw audio into a structured, queryable knowledge base.
From ingestion of podcast metadata, through transcription, chunking, embedding, vector storage, retrieval, and evaluation – each spike incrementally builds towards an interactive, explainable RAG system.

The final solution supports:

* **Natural language search** across episodes and years.
* **Comparisons over time** (e.g. a guest’s views in 2021 vs 2024).
* **Clip linking** directly to transcript timestamps.
* **Evaluation and monitoring** with Phoenix.
* **Reproducible, containerised deployment**.

---

## Project Overview

| Area                 | Description                                                                                                              |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Dataset**          | Modern Wisdom podcast RSS + audio                                                                                        |
| **Objective**        | Build an end-to-end RAG system: ingestion → transcription → chunking → embedding → vector store → retrieval → agentic QA |
| **Architecture**     | Python + DuckDB + Parquet + Qdrant + FastEmbed + OpenAI (optional)                                                       |
| **Interface**        | FastAPI (programmatic API) and Chainlit 2.8.3 (chat UI)                                                                  |
| **Evaluation**       | Retrieval metrics (Hit@k, MRR, p95 latency), LLM output validation, Phoenix tracing                                      |
| **Containerisation** | Docker Compose (Qdrant, Phoenix, API, Chainlit)                                                                          |
| **Reproducibility**  | `uv.lock` pinned dependencies and documented setup                                                                       |

---

## Repository Structure

```bash
modern-wisdom-llm-native-pipeline/
├── data/
│   ├── duckdb/modern_wisdom.duckdb      # Local DB (episodes, transcripts, chunks, embeddings)
│   ├── transcripts/                     # Parquet ASR output per episode
│   ├── chunks/sentence_bound/           # Chunks ready for embedding
│   ├── embeddings/BAAI_bge_small_en_v1_5/  # Embedding vectors per episode
│   ├── qdrant/                          # Local vector DB storage (Docker volume)
│   ├── qa/labels.csv                    # Ground truth Q/A pairs for evaluation
│   ├── evals/                           # Retrieval and LLM evaluation outputs
│   └── tmp/                             # Temporary lists for backfills
│
├── docs/
│   └── decisions/                       # Design and evaluation decisions per spike
│
├── spikes/
│   ├── spike1_rss_to_duckdb/            # RSS ingestion
│   ├── spike2_asr_timestamps/           # ASR transcription
│   ├── spike3_chunking_and_metadata/    # Chunking experiments
│   ├── spike4_embeddings/               # Embedding generation
│   ├── spike5_qdrant_collection/        # Vector store creation
│   ├── spike6_qdrant_retrieval/         # Retrieval baseline
│   ├── spike7_hybrid_search/            # Hybrid RRF search
│   ├── spike8_rag_contract/             # RAG generation contract
│   ├── spike9_sql_introspection/        # SQL and metadata tools
│   ├── spike10_tracing_monitoring/      # Phoenix tracing
│   └── spike11_agent/                   # Constrained agent with reasoning chain
│
├── src/modern_wisdom_rag_pipeline/
│   ├── api.py                           # FastAPI app for programmatic access
│   ├── chainlit_app.py                  # Chainlit conversational UI
│   ├── main.py                          # Entry point
│   └── ...                              # Utilities and tools
│
├── infra/                               # Optional standalone docker-compose files
├── Dockerfile                           # Multi-stage build
├── docker-compose.yml                   # Full local stack (Qdrant, Phoenix, API, Chainlit)
├── pyproject.toml                       # Dependency and build configuration
└── uv.lock                              # Locked dependency versions
```

---

## Summary of Spikes 1–7

Each spike explores one layer of the pipeline. Full details are in the individual README files within each spike folder.

### Spike 1 – RSS to DuckDB

**Purpose:** Incremental ingestion of the Modern Wisdom RSS feed into a structured DuckDB database using dlt.
**Outcome:** 991 episodes loaded, incremental updates confirmed idempotent.
**Rationale:** Local DuckDB offers analytical speed and SQL ergonomics without requiring a remote DB.

### Spike 2 – ASR with timestamps

**Purpose:** Convert audio into timestamped transcripts using AssemblyAI or local Faster-Whisper.
**Outcome:** Complete corpus transcribed to Parquet. Average confidence ≈ 0.92.
**Rationale:** Timestamps enable search, clipping, and alignment with video/audio.

### Spike 3 – Chunking and metadata

**Purpose:** Split transcripts into semantically meaningful windows.
**Methods tested:** fixed-size, sentence-bound, time-window.
**Decision:** *Sentence-bound* performed best (Hit@20 = 0.72, MRR = 0.38) balancing recall and readability.
**Rationale:** Sentence boundaries maintain context and minimise mid-sentence cuts.

### Spike 4 – Embeddings

**Purpose:** Generate vector embeddings from chunks.
**Comparison:**

* OpenAI `t3-small` (1536 d) – accurate, slower.
* FastEmbed `BAAI/bge-small-en-v1.5` (384 d) – fast, open, cost-free.
  **Decision:** FastEmbed chosen for local reproducibility and good recall–latency trade-off.

### Spike 5 – Qdrant Collection & Alias

**Purpose:** Persist embeddings into a local vector store with live aliasing.
**Outcome:** Deterministic collection per `emb_v`, blue/green alias `mw_chunks_live`.
**Rationale:** Qdrant provides a simple REST + gRPC API, strong local performance, and alias support.

![Qdrant Collection](docs/screenshots/qdrant_collections.png)
![Qdrant Collection Details](docs/screenshots/qdrant_collection_details.png)

### Spike 6 – Retrieval Baseline

**Purpose:** Evaluate pure-vector retrieval using the labelled QA set.
**Metrics:** Hit@10 = 0.60, MRR = 0.25, p95 latency ≈ 39 ms.
**Rationale:** Establish baseline for comparison with hybrid methods.

### Spike 7 – Hybrid Search

**Purpose:** Combine lexical (BM25) and vector retrieval using Reciprocal Rank Fusion.
**Improvement:** Hit@10 → 0.62 → 1.00 with BGE query prefix; latency ≈ 46 ms.
**Decision:** Keep **query prefix ON**, continue with hybrid for production.

---

## Later Spikes (8–11) Overview

### Spike 8 – RAG Contract

Defines a lightweight schema and contract for RAG generation, decoupled from provider.
Provides deterministic JSON output validated by `jsonschema`.

### Spike 9 – SQL Introspection

Adds SQL inspection and local DuckDB utilities for debugging and metadata queries.

### Spike 10 – Tracing & Monitoring

Integrates OpenTelemetry and Arize Phoenix.
Each step (embedding, retrieval, generation) emits spans.
Phoenix dashboard accessible at `http://localhost:6006`.

![Phoenix Project](docs/screenshots/phoenix_project.png)
![Phoenix Trace](docs/screenshots/phoenix_trace.png)

### Spike 11 – Agentic Reasoning

Implements a **constrained agent** that plans tool usage (RAG search, timeline builder, clip linker, etc.).
Safely executes multi-step reasoning capped at 6 steps.
CLI:

```bash
uv run run-spike11 deep-agent --question "Compare Chris’s views on discipline in 2021 vs 2024"
```

Result: Coherent, evidence-linked comparison with timestamps and clip URLs.

---

## Evaluation Summary

| Criterion                | Approach                                          | Result                                  |
| ------------------------ | ------------------------------------------------- | --------------------------------------- |
| **Problem description**  | Long-form audio locked in podcast format          | Addressed with ASR + RAG pipeline       |
| **Retrieval flow**       | Hybrid (BM25 + vector) over Qdrant                | Hit@10 = 1.00 with BGE query prefix     |
| **Retrieval evaluation** | Vector vs Hybrid compared                         | Hybrid chosen, p95 ≈ 46 ms              |
| **LLM evaluation**       | Agent answers vs reference QA                     | JSON-validated correctness              |
| **Interface**            | FastAPI + Chainlit UI                             | API on :8000 / UI on :8001              |
| **Ingestion pipeline**   | Automated Python scripts using dlt + ASR + DuckDB | End-to-end reproducible                 |
| **Monitoring**           | Phoenix dashboard + trace spans                   | 5+ charts, latency breakdown            |
| **Containerisation**     | Docker Compose (Qdrant, Phoenix, API, Chainlit)   | Single-command deployment               |
| **Reproducibility**      | `uv sync`, `uv lock`, bind mounts                 | Fully self-contained and version-pinned |

---

## Getting Started

### Prerequisites

* Docker ≥ 25
* uv ≥ 0.4
* Python ≥ 3.11 (if running locally)

### Quick Start

```bash
uv sync
export OPENAI_API_KEY=sk-your-real-openai-key
```

```bash
docker compose up
```

Services:

* Qdrant: [http://localhost:6333](http://localhost:6333)
* Phoenix: [http://localhost:6006](http://localhost:6006)
* API (FastAPI docs): [http://localhost:8000/docs](http://localhost:8000/docs)
* Chainlit UI: [http://localhost:8001](http://localhost:8001)

### Summary of steps to backfill (index initialisation - create collection + alias) almost 1000 episodes - chunk, embed and upsert into Qdrant

As all the chunks and embeddings have already been created (located in the `data` directory). You only need to upsert the embeddings to Qdrant. Note this can take over 30 minutes.

```bash
EMB_V="BAAI/bge-small-en-v1.5"
INDEX_VERSION="mw_chunks_live"

cat data/tmp/epids_2018_2025.txt | while read -r EID; do
  echo "Upserting vectors for $EID"
  uv run run-spike5 upsert \
    --episode-id "$EID" \
    --method sentence_bound \
    --emb-v "$EMB_V" \
    --set-live \
    --live-alias "$INDEX_VERSION"
done
```

### 4) Use the UI

![Chainlit](docs/screenshots/chainlit_sample.png)

Open **Chainlit** at [http://localhost:8001](http://localhost:8001) and ask similar questions.

Open **API docs** at [http://localhost:8000/docs](http://localhost:8000/docs) for programmatic calls.

Open **Phoenix** at [http://localhost:6006](http://localhost:6006) to see traces and metrics.

Open **Qdrant** at [http://localhost:6333/dashboard](http://localhost:6333/dashboard) to view vector embeddings, collections, and their details.

---

## Tool and Service Rationale

| Component                   | Motivation              | Role                         |
| --------------------------- | ----------------------- | ---------------------------- |
| DuckDB                      | Fast local analytics    | Episode + transcript lineage |
| dlt                         | Declarative ingestion   | RSS → DB                     |
| AssemblyAI / Faster-Whisper | ASR                     | Audio → text                 |
| FastEmbed (BGE)             | Open, deterministic     | Embeddings                   |
| Qdrant                      | Local vector DB + alias | Similarity search            |
| BM25                        | Lexical grounding       | Hybrid retrieval             |
| RRF                         | Rank fusion             | Better recall                |
| FastAPI                     | Programmatic API        | Integration surface          |
| Chainlit 2.8.3              | Chat UI                 | Human-in-the-loop            |
| Phoenix (Arize)             | OTel visualisation      | Monitoring                   |
| uv                          | Fast resolver           | Reproducibility              |
| Docker Compose              | One-command stack       | Local demo                   |

---

## Reproducibility and Deployment

* **All dependencies pinned** in `uv.lock`.
* **Data persisted** under `./data` (bind-mounted volumes).
* **Docker Compose** launches Qdrant, Phoenix, FastAPI, and Chainlit with one command.
* **Local evaluation notebooks** validate each stage.
* **Agent CLI** provides end-to-end testing and demo questions.

To rebuild everything cleanly:

```bash
docker compose down -v
uv sync --frozen
docker compose up --build
```

---

## Conclusion

This project demonstrates a full end-to-end retrieval-augmented generation system using a real-world podcast dataset.
Starting from raw audio, it delivers a searchable, explainable knowledge base capable of producing timestamped, evidence-linked answers.
Each design decision was guided by empirical evaluation, simplicity, and reproducibility — making it straightforward for others to run, extend, and mark.
