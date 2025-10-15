# Retrieval hybrid — fe_bge_small_en_v1_5_v1 (episode 0a4fa77e-bc0f-11ef-bab6-3f37b4906b43)

- collection: `mw_chunks_live`
- method: `sentence_bound`
- ks: [5, 10, 20] tol_s=7 rrf_k=60.0 vec_k=20 lex_k=200
- filters: {'guest': None, 'date_from': None, 'date_to': None}

## Metrics
```json
{
  "Hit@5": 0.5,
  "Hit@10": 1.0,
  "Hit@20": 1.0,
  "MRR": 0.1875
}
```
- p95_latency_ms: 4.83