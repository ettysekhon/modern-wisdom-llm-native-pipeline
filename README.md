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
