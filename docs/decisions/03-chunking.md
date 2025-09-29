# Decision Record — Chunking & Metadata (Spike 3)

Date: 2025-09-29  
Owner: ettyekhon

## Context

- Transcript segments: avg duration ≈ 3–4 sec, avg tokens/segment ≈ 8–10  
- Episode metadata available in DuckDB: title, guest, publish_date, episode_number, headline, duration  
- Q/A set: 2 questions labeled with start/end timestamps  
- Constraint: downstream LLM context window = 8k tokens (OpenAI / OSS embedding models)

## Options Considered

- V0 — Fixed tokens (700 size, 100 overlap)  
- V1 — Sentence-bounded (700 size, 100 overlap)  
- V2 — Time-windowed (≈180s window, 30s overlap)

## Evaluation Setup

- Scorer: BM25 (rank_bm25)  
- Top-K: 20  
- Hit definition: retrieved chunk overlaps labeled answer by ≥1s (±7s tolerance)  
- Metrics: Hit@5/10/20, MRR, coverage, avg time distance

## Results

```json
{
  "fixed": {
    "Hit@10": 0.5,
    "MRR": 0.0625,
    "AvgTimeDistanceSec": 75.08
  },
  "sentence_bound": {
    "Hit@10": 0.5,
    "MRR": 0.0625,
    "AvgTimeDistanceSec": 75.08
  },
  "time_window": {
    "Hit@10": 0.5,
    "MRR": 0.05,
    "AvgTimeDistanceSec": 91.06
  }
}

Winner (Hit@10 → tie MRR): **sentence_bound**
