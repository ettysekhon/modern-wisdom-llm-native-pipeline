# Decision — Embeddings (Spike 4)

Date: 2025-10-03
Owner: ettysekhon

Chosen provider: FastEmbed
Model: BAAI/bge-small-en-v1.5
emb_v: fe_bge_small_en_v1_5_v1
dim: 384
normalized: true
distance_metric (Qdrant): Cosine

Rationale:

- Local, fast, free; well integrated with Qdrant.
- Deterministic outputs; good tradeoff of quality/speed for retrieval baseline.
- Idempotent runs via chunk_id + emb_v; retries bounded.

Throughput/cost:

- Tokens: N (from CLI summary)
- Runtime: ~X rows/s (CPU)
- Cost: $0 (local)
