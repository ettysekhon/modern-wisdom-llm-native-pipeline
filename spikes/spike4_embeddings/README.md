# Spike 4 — Embeddings

## Objective

Take transcript chunks from Spike 3 and generate embeddings for them.
Ensure the process is:

- Deterministic (idempotent: don’t re-embed what’s already done)
- Configurable (support OpenAI, OSS, FastEmbed)
- Persisted (Parquet + DuckDB for lineage)
- Ready for Qdrant ingestion (Spike 5)

## Inputs

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

## Outputs

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

## Details

- Idempotency: before embedding, check DuckDB/parquet for existing chunk_id+emb_v; skip if    found.
- Retries: exponential backoff with jitter on provider errors.
- Dimension inference:
  - OpenAI via lookup (text-embedding-3-small = 1536, etc.)
  - FastEmbed via model introspection.
- Provider abstraction: pluggable select_provider() returns a function that maps List[str] → List[vector].

## Acceptance

- Able to embed one episode (--episode-id) end-to-end with OpenAI and FastEmbed.
- Re-running same command is a no-op (idempotent).
- DuckDB shows correct count of embeddings.
- CLI shows rich batch summary.

## Decision Record (Outcome)

- Chosen provider for default experiments: OpenAI (t3-small, 1536d).
- OSS alternative tested: FastEmbed BGE-small-en-v1.5 (384d).
- Result: Both pipelines work; FastEmbed cheaper/faster but lower-dim.
- Next: Use Spike 4 outputs as input to Spike 5 (Qdrant).
