# Spike 7 Hybrid Search

| Step                                       | What Happened                                                      | Output                                           |
| ------------------------------------------ | ------------------------------------------------------------------ | ------------------------------------------------ |
| **1. Load QA + embeddings + chunks**       | Read your labeled Q/A CSV, episode embeddings, and chunk metadata. | from `data/qa`, `data/embeddings`, `data/chunks` |
| **2. BM25 lexical retrieval**              | Scored all chunks textually using `rank_bm25`.                     | lexical rank list                                |
| **3. Vector retrieval**                    | Queried Qdrant for dense similarity (Cosine).                      | vector rank list                                 |
| **4. RRF Fusion (Reciprocal Rank Fusion)** | Combined both scores deterministically.                            | fused ranked list                                |
| **5. Metrics computation**                 | Calculated Hit@5/10/20, MRR, and p95 latency.                      | `retrieval_hybrid.json`                          |
| **6. Documentation output**                | Created an auditable decision record.                              | `0007-retrieval-hybrid.md`                       |

---

## Spike 7 Output

```text
data/evals/retrieval/retrieval_hybrid.json
```

Contains something like:

```json
{
  "summary": {
    "episode_id": "0a4fa77e-bc0f-11ef-bab6-3f37b4906b43",
    "emb_v": "fe_bge_small_en_v1_5_v1",
    "method": "sentence_bound",
    "metrics": {
      "Hit@5": 0.54,
      "Hit@10": 0.62,
      "Hit@20": 0.72,
      "MRR": 0.38
    },
    "p95_latency_ms": 45.8
  }
}
```

```text
docs/decisions/0007-retrieval-hybrid.md
```

Contains:

### Retrieval hybrid — fe_bge_small_en_v1_5_v1 (episode 0a4fa77e-bc0f-11ef-bab6-3f37b4906b43)

- collection: `mw_chunks_live`
- method: `sentence_bound`
- ks: [5, 10, 20] tol_s=7 rrf_k=60.0 vec_k=20 lex_k=200
- filters: {"guest": null, "date_from": null, "date_to": null}

### Metrics

```json
{
  "Hit@5": 0.54,
  "Hit@10": 0.62,
  "Hit@20": 0.72,
  "MRR": 0.38
}
```

- p95_latency_ms: 45.8

---

## Summary of results

- **No prefix ("False")**

  - **Vector-only** cratered (Hit@10 = 0.0, MRR = 0.0) → classic mismatch.
  - **Hybrid** still salvaged some signal (Hit@10 = 0.5), but not great.

- **With BGE query prefix ("True")**

  - **Hybrid pops**: **Hit@10 = 1.0**, **Hit@20 = 1.0**, p95 ≈ **3.75 ms**.
  - MRR improved vs "False" (0.1875 vs 0.0625). For tiny QA sets, MRR is sensitive; Hit@k is your main gate.

## What this means

- For **BAAI/bge-small-en-v1.5**, using **`query: …`** on queries materially improves retrieval when your chunks were embedded **without** `passage:`.
- **Keep the query prefix ON** for Spike 7+. Document this as part of the contract: *“BGE queries use `query:` prefix; docs currently un-prefixed.”*

## Completed

- Combine vector + BM25 retrieval via RRF
- Add filters (guest/date range)
- Generate metrics + decision record
