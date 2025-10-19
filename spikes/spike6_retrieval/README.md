# Spike 6 — Qdrant Retrieval

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

## Output

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
