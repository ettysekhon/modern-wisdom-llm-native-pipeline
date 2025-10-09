# LLM Native Pipeline Spikes

A series of spikes exploring how to build an LLM-native data pipeline, using [Chris Williamson's *Modern Wisdom* podcast](https://chriswillx.com/podcast/) as the dataset.

## Getting Started

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Repo Layout

```bash
modern-wisdom-llm-native-pipeline/
├─ data/duckdb/modern_wisdom.duckdb   # Local DuckDB file created by Spike 1
├─ spikes/
│  └─ spike1_rss_to_duckdb/
│     ├─ src/spike1_rss_to_duckdb/main.py
│     └─ notebooks/spike1_validation.ipynb
└─ pyproject.toml
```

## Troubleshooting

- If you need to query `duckdb` run the following to execute queries `duckdb ./data/duckdb/modern_wisdom.duckdb`

##  Spikes

### Spike 1 - RSS -> dlt -> DuckDB (incremental ingestion)

Goal

- Ingest the Modern Wisdom podcast RSS feed into a local database
- Build an incremental pipeline so only new or updated episodes are added.

Run the pipeline

```bash
uv run run-spike1
```

Expected output (example):

```text
Loaded (new): 0, Updated (est): 0
Total episodes: 991
DuckDB at /data/duckdb/modern_wisdom.duckdb
```

- First run: all episodes (≈991) are ingested.
- Immediate re-run: Loaded (new): 0, Updated (est): 0 (idempotency check).
- After a new episode is published or a record changes: counts reflect insert/update.

Validate results

Open the Jupyter notebook:

```bash
uv run --with jupyter jupyter lab
```

Then navigate to [spike1_validation.ipynb](spikes/spike1_rss_to_duckdb/notebooks/spike1_validation.ipynb) and run the cells to:

- Preview recent episodes.
- Check yearly distribution of episodes.
- Verify ingest run logs (mw.ingest_runs table).

### Spike 2 – ASR pipeline with timestamps

Automatic Speech Recognition (ASR) is the process of converting spoken language from audio into written text.  
This spike builds an ASR pipeline that transcribes podcast episodes into **timestamped segments**, enabling search, analysis, and downstream NLP tasks.

- Turn Modern Wisdom episode audio into **timestamped transcripts**.
- Support **multiple ASR backends**:
  - **AssemblyAI** (default; recommended for speed/scale, supports diarization).
  - **Local faster-whisper** (for offline runs / cost-saving).
- Store transcripts in **parquet format** for downstream analysis.

#### Goals

- Populate table `transcripts(episode_id, start_ts, end_ts, text, asr_model, confidence[, speaker])`.
- Validate with spot-checks (e.g., Word Error Rate on 3 clips).
- Produce:
  - Small QA sheet of transcript samples
  - A clear re-transcription policy
  - Defined storage layout (parquet/csv per episode)

#### Getting Started on ASR pipeline

1. **Set environment variables:**

   ```bash
   export ASSEMBLYAI_API_KEY="your-api-key"
   export ASR_BACKEND=assemblyai   # or "local" for faster-whisper
   export ASR_LIMIT=5              # 0 = all episodes, or limit to first 
   export ASR_DIARIZATION=1        # optional, enable speaker labels

2. **Run the pipline**

    ```bash
    uv run run-spike2
    ```

3. **Sample output***

    ```bash
    ASR start | backend=assemblyai | mode=all episodes | available=991 | planned=991
    [1/991] a1ccef1f... | OK | took 32.7s | avg 32.7s/ep | ETA 00:08:30 | segs 42
    ...
    ASR done | backend=assemblyai | processed=991 | ok=987 | fail=4 | elapsed=09:01:22
    ```

    - Parquet transcripts: `data/transcripts/episode_id=<id>/part-00000.snappy.parquet`
    - Index CSV: `data/transcripts/index.csv` (tracks run stats, duration, status, errors)

Validate results - Navigate to [spike2_validation.ipynb](spikes/spike2_asr_timestamps/notebooks/spike2_validation.ipynb) to view output stored in parquet files.

## Spike 3

The purpose of spike 3 is to decide how to split transcripts into chunks and attach the right metadata so retrieval later is accurate and efficient. We will then validate this choice with a tiny labeled Q/A set before moving on.

###  Inputs

- Transcripts from Spike 2 (per-episode parquet: segment_idx, start_ts, end_ts, text, asr_model, confidence).
- Episode metadata from DuckDB (mw.episodes: title, guest, publish_date, episode_number, headline, duration…).
- Q/A set (5–10 rows) with:
  - question
  - episode_id
  - answer_start_ts, answer_end_ts
  - optional keywords.

### Approach

```python
from jsonschema import validate
import json, pathlib

schema = json.loads(pathlib.Path("data/evals/chunking/chunk_eval_report.schema.json").read_text())
report = json.loads(pathlib.Path("data/evals/chunking/chunk_eval_report.json").read_text())
validate(instance=report, schema=schema)
```

### Logic

The logic is split into a number of files, the key logic resides in `eval.py`:

`paths.py` — knows where everything lives (data/…, DuckDB path), and initialises the tokenizer.

`utils.py` — small helpers: token counting, overlap math, UUIDs, BM25 tokenization, etc.

`schema.py` — defines the required chunk columns and a validator so chunks are consistent.

`transcript.py` — loads one episode’s transcript parquet and validates it (shape, timestamps, text).

`chunkers.py` — builds chunks (3 methods): fixed, sentence_bound, time_window. Returns rows with full schema.

`persist.py` — writes chunk rows to parquet and (optionally) upserts into DuckDB (table chunks).

`eval.py` — runs BM25 over chunk text vs a tiny Q/A set and produces the method comparison + winner.

`cli.py` — the thin command-line glue (check, chunk, eval).

`main.py` — just calls the CLI.

```bash
# step 1
uv run run-spike3 check --episode-id <EPISODE_ID>
# step 2
uv run run-spike3 chunk \
  --episode-id <EPISODE_ID> \
  --methods sentence_bound \
  --size-tokens 700 --overlap-tokens 100

# step 3
uv run run-spike3 chunk \
  --episode-id <EPISODE_ID> \
  --methods fixed,sentence_bound,time_window \
  --size-tokens 700 --overlap-tokens 100 \
  --window-seconds 190 --overlap-seconds 30 \
  --duckdb

# step 4
uv run run-spike3 eval \
  --episode-id <EPISODE_ID> \
  --methods fixed,sentence_bound,time_window \
  --qa-csv data/qa/labels.csv \
  --k 20 --tolerance-s 7

```

### Step 1 - Preflight one episode (input sanity)

```bash
uv run run-spike3 check --episode-id 0a4fa77e-bc0f-11ef-bab6-3f37b4906b43
              Transcript check — 0a4fa77e-bc0f-11ef-bab6-3f37b4906b43
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric ┃ Value                                                                  ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Rows   │ 2657                                                                   │
│ Cols   │ segment_idx, start_ts, end_ts, text, asr_model, confidence, episode_id │
```

### Step 2 - Produce chunks (no metadata yet) for 1 method

```bash
uv run run-spike3 chunk --episode-id 0a4fa77e-bc0f-11ef-bab6-3f37b4906b43 \
  --methods sentence_bound --size-tokens 700 --overlap-tokens 100
Wrote 66 rows for sentence_bound → 
/modern-wisdom-llm-native-pipeline/data/chunks/sentence_bound/episode_id=0a4fa77e-bc0f-11ef-bab6-3f37b4906b43/part-00000.snappy.parquet
```

### Step 3 - Add episode metadata enrichment (missing piece) and then produce for other methods

```bash
uv run run-spike3 chunk \
  --episode-id 0a4fa77e-bc0f-11ef-bab6-3f37b4906b43 \
  --methods fixed,sentence_bound,time_window \
  --size-tokens 700 --overlap-tokens 100 \
  --window-seconds 190 --overlap-seconds 30 \
  --duckdb
Wrote 49 rows for fixed → 
/Users/ettyekhon/code/src/petroineos/petroineos-related/modern-wisdom-llm-native-pipeline/data/chunks/fixed/episode_id=0a4fa77e-bc0f-11ef-bab6-3f37b4906b43/part-00000.snappy.parquet
Wrote 66 rows for sentence_bound → 
/Users/ettyekhon/code/src/petroineos/petroineos-related/modern-wisdom-llm-native-pipeline/data/chunks/sentence_bound/episode_id=0a4fa77e-bc0f-11ef-bab6-3f37b4906b43/part-00000.snappy.p
arquet
Wrote 50 rows for time_window → 
/Users/ettyekhon/code/src/petroineos/petroineos-related/modern-wisdom-llm-native-pipeline/data/chunks/time_window/episode_id=0a4fa77e-bc0f-11ef-bab6-3f37b4906b43/part-00000.snappy.parq
uet

### Step 4 — Evaluate (BM25 "findability")

```bash
uv run run-spike3 eval \
  --episode-id 0a4fa77e-bc0f-11ef-bab6-3f37b4906b43 \
  --methods fixed,sentence_bound,time_window \
  --qa-csv data/qa/labels.csv \
  --k 20 --tolerance-s 7 \
  --prefer-efficient
Eval report complete → /modern-wisdom-llm-native-pipeline/data/evals/chunking/chunk_eval_report.json, 
/Users/ettyekhon/code/src/petroineos/petroineos-related/modern-wisdom-llm-native-pipeline/docs/decisions/0003-chunking.md
Eval decision complete → /modern-wisdom-llm-native-pipeline/data/evals/chunking/chunk_eval_report.json
Decision updated → /modern-wisdom-llm-native-pipeline/docs/decisions/0003-chunking.md
Config updated → configs/chunking.toml
```

The [decision report](/docs/decisions/03-chunking.md) summarises the results and the winner. Visualisations summarising the results have also been created in the [spike3_validation.ipynb](/spikes/spike3_chunking_and_metadata/notebooks/spike3_validation.ipynb) notebook.

###  Unit Tests for Spike 3

```bash
uv sync --all-extras
uv run pytest -q
```

5 passed in 0.17s

## Spike 4 — Embeddings

### Objective

Take transcript chunks from Spike 3 and generate embeddings for them.
Ensure the process is:

- Deterministic (idempotent: don’t re-embed what’s already done)
- Configurable (support OpenAI, OSS, FastEmbed)
- Persisted (Parquet + DuckDB for lineage)
- Ready for Qdrant ingestion (Spike 5)

### Inputs

- Chunk parquet files from Spike 3
  - Located under data/chunks/{method}/episode_id=.../part-*.parquet
  - Each row includes:
    chunk_id, episode_id, method, text, n_tokens, duration_s, metadata...
- Embedding provider + model ID
  - OpenAI: e.g. text-embedding-3-small (dim=1536)
  - FastEmbed: e.g. BAAI/bge-small-en-v1.5 (dim=384)
  - Optional: HuggingFace / sentence-transformers later

Parameters via CLI:
--episode-id
--method
--emb-v (embedding version tag)
--provider (openai | fastembed | oss)
--model-id
--batch-size, --retries, --sleep-base-ms
--duckdb (flag to upsert into DuckDB)

### Outputs

- Parquet file with embeddings:
  data/embeddings/{emb_v}/episode_id={id}/part-*.parquet
- Each row contains:
  - chunk_id (PK)
  - episode_id
  - method
  - emb_v (embedding version)
  - provider, model_id
  - vector (float[])
  - tokens, dim, attempts, status
  - created_at, text_hash

DuckDB table: embeddings

- Same schema as above
- Upsert by chunk_id + emb_v
- Enables traceability across transcripts → chunks → embeddings

### Details

- Idempotency: before embedding, check DuckDB/parquet for existing chunk_id+emb_v; skip if    found.
- Retries: exponential backoff with jitter on provider errors.
- Dimension inference:
  - OpenAI via lookup (text-embedding-3-small = 1536, etc.)
  - FastEmbed via model introspection.
- Provider abstraction: pluggable select_provider() returns a function that maps List[str] → List[vector].

### Acceptance

- Able to embed one episode (--episode-id) end-to-end with OpenAI and FastEmbed.
- Re-running same command is a no-op (idempotent).
- DuckDB shows correct count of embeddings.
- CLI shows rich batch summary.

### Decision Record (Outcome)

- Chosen provider for default experiments: OpenAI (t3-small, 1536d).
- OSS alternative tested: FastEmbed BGE-small-en-v1.5 (384d).
- Result: Both pipelines work; FastEmbed cheaper/faster but lower-dim.
- Next: Use Spike 4 outputs as input to Spike 5 (Qdrant).

---

## Spike 5 — Qdrant Collection & Blue-Green Alias

**Date:** 2025-10-09
**Owner:** ettyekhon

---

### Qdrant Objective

Integrate **Qdrant vector search** as the production index for all chunked + embedded podcast transcripts (from Spikes 3 & 4), ensuring:

- Deterministic collection naming per embedding version (`emb_v`)
- Idempotent upserts (no duplication)
- Blue/green deployment pattern via live alias
- Local persistence (via Docker volume)
- Simple CLI tooling for upsert, query, and health check

---

### Qdrant Inputs

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

### Qdrant Setup (Qdrant local)

```bash
docker compose -f infra/qdrant/docker-compose.yml up -d
```

Qdrant Web UI:
[http://localhost:6333/dashboard](http://localhost:6333/dashboard)

---

### How to Run (End-to-End)

#### Upsert episode embeddings

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

####  Check collection + alias

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

#### Decision

**Chosen approach:**
Qdrant + FastEmbed + DuckDB
(using `qdrant-client` v1.15.x API with `get_collection_aliases`)

**Rationale:**

- Local, fast, embeddable vector DB
- Deterministic collection per `emb_v`
- Explicit blue/green alias (`mw_chunks_live`)
- Supports both HTTP & gRPC APIs
- Integrates cleanly with FastEmbed (no external OpenAI dependency)

---

## Spike 6 — Qdrant Retrieval

| Step                     | What it did                                              | Output                                      |
| ------------------------ | -------------------------------------------------------- | ------------------------------------------- |
| **1. Loaded QA**         | Pulled `labels.csv` with ground-truth start/end times.   | `data/qa/labels.csv`                        |
| **2. Loaded Embeddings** | Read from `/data/embeddings/fe_bge_small_en_v1_5_v1/...` | Verified shape = 384 dim                    |
| **3. Queried Qdrant**    | For each question, searched `mw_chunks_live`             | Measured per-query latency                  |
| **4. Scored Retrieval**  | Computed Hit@k, MRR & p95 latency                        | `retrieval_baseline.json`                   |
| **5. Wrote Docs**        | Created Markdown decision file                           | `docs/decisions/0006-retrieval-baseline.md` |

- FastEmbed model (`BAAI/bge-small-en-v1.5`) was automatically downloaded and cached
- Qdrant search succeeded via the `mw_chunks_live` alias
- Results persisted deterministically

---

### Output

```text
data/evals/retrieval/retrieval_baseline.json
docs/decisions/0006-retrieval-baseline.md
```

These files contain metrics like:

```json
{
  "Hit@5": 0.40,
  "Hit@10": 0.60,
  "Hit@20": 0.70,
  "MRR": 0.25,
  "p95_latency_ms": 38.7
}
```

Can be used later for charts or regression comparisons when we test hybrid retrieval.

---

### Spike 6 Completion Checklist

| Goal                                          |
| --------------------------------------------- |
| Vector-only retrieval baseline                |
| Deterministic I/O + reports                   |
| Uses FastEmbed + Qdrant (no OpenAI)           |
| Ready for Phoenix tracing or hybrid extension |

---
