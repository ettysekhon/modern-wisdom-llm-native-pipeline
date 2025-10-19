# Spike 3

The purpose of spike 3 is to decide how to split transcripts into chunks and attach the right metadata so retrieval later is accurate and efficient. We will then validate this choice with a tiny labeled Q/A set before moving on.

##  Inputs

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
