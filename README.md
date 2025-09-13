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
