# Decision Record — Chunking & Metadata (Spike 3)

Date: 2025-09-30
Owner: ettysekhon

## Context

- Transcript segments: avg duration = —, avg tokens/segment = —
- Episode metadata available in DuckDB: title, guest, publish_date, episode_number, headline, duration
- Q/A set: 2 questions labeled with start/end timestamps
- Constraint: downstream LLM context window = 8k–16k tokens (typical)

## Options Considered

- V0 — Fixed tokens (700 size, 100 overlap)
- V1 — Sentence-bounded (700 size, 100 overlap)
- V2 — Time-windowed (≈180–200s window, 30s overlap)

## Evaluation Setup

- Scorer: BM25 (rank_bm25)
- Top-K: 20
- Hit definition: retrieved chunk overlaps labeled answer by ≥1s (±7s tolerance)
- Metrics: Hit@5/10/20, MRR, coverage (via overlap), avg time distance

## Results

```json
{
  "fixed": {
    "Hit@10": 0.5,
    "MRR": 0.0625,
    "AvgTimeDistanceSec": 75.08000000000001,
    "AvgTokens": 681.1224489795918,
    "AvgDurationSec": 162.7561224489795
  },
  "sentence_bound": {
    "Hit@10": 0.5,
    "MRR": 0.0625,
    "AvgTimeDistanceSec": 75.08000000000001,
    "AvgTokens": 611.7575757575758,
    "AvgDurationSec": 147.0887727272727
  },
  "time_window": {
    "Hit@10": 0.5,
    "MRR": 0.05,
    "AvgTimeDistanceSec": 91.05950000000001,
    "AvgTokens": 790.92,
    "AvgDurationSec": 189.08038
  }
}
```

## Decision

Chosen: **sentence_bound, 700 tokens, 100 overlap**  
Rationale: Winner selected on Hit@10 (primary), then MRR (tie-breaker). If equal, prefer smaller AvgTokens and AvgDuration to minimize context cost.

## Consequences

- Retrieval will use **sentence_bound** chunks.
- Metadata enrichment includes guest, publish_date, episode_number, title, headline, duration.
- Index size and retrieval latency expected to remain within budget; shorter chunks reduce prompt cost.
- Re-chunking may be required if ASR quality or the embedding model/context window changes significantly.
