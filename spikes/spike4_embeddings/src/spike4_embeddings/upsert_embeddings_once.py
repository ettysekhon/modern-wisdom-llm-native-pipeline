from __future__ import annotations

import duckdb

from spike4_embeddings import paths

PARQUET = (
    paths.EMB_DIR
    / "openai_t3small_v1"
    / "episode_id=0a4fa77e-bc0f-11ef-bab6-3f37b4906b43"
    / "part-00000.snappy.parquet"
)

DDL_WITH_PK = """
CREATE TABLE embeddings (
    chunk_id TEXT PRIMARY KEY,
    episode_id TEXT,
    method TEXT,
    emb_v TEXT,
    dim INTEGER,
    model_id TEXT,
    provider TEXT,
    created_at BIGINT,
    text_hash TEXT,
    tokens INTEGER,
    vector FLOAT[],
    attempts INTEGER,
    status TEXT
);
"""


def ensure_embeddings_with_pk(con: duckdb.DuckDBPyConnection):
    # Does table exist?
    row = con.execute("""
        SELECT COUNT(*)>0 AS exists
        FROM information_schema.tables
        WHERE table_name = 'embeddings'
    """).fetchone()
    exists = bool(row[0]) if row is not None else False

    if not exists:
        con.execute(DDL_WITH_PK)
        return

    # Has PK?
    row = con.execute("""
        SELECT COUNT(*)>0 AS has_pk
        FROM pragma_table_info('embeddings')
        WHERE pk > 0
    """).fetchone()
    has_pk = bool(row[0]) if row is not None else False

    if has_pk:
        return

    # No PK: if empty, drop & recreate; else migrate rows
    row = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()
    rowcount = int(row[0]) if row is not None else 0
    if rowcount == 0:
        con.execute("DROP TABLE embeddings;")
        con.execute(DDL_WITH_PK)
        return

    # Migrate
    con.execute("CREATE TABLE embeddings_new AS SELECT * FROM embeddings LIMIT 0;")
    # Recreate embeddings_new with the right schema (CTAS has no PK), so drop & create with PK, then insert
    con.execute("DROP TABLE embeddings_new;")
    con.execute(DDL_WITH_PK.replace("embeddings", "embeddings_new"))
    con.execute("INSERT INTO embeddings_new BY NAME SELECT * FROM embeddings;")
    con.execute("DROP TABLE embeddings;")
    con.execute("ALTER TABLE embeddings_new RENAME TO embeddings;")


def main():
    if not PARQUET.exists():
        raise SystemExit(f"Parquet not found: {PARQUET}")

    con = duckdb.connect(paths.DUCKDB_PATH.as_posix())

    ensure_embeddings_with_pk(con)

    con.execute(
        """
        INSERT OR REPLACE INTO embeddings BY NAME
        SELECT * FROM read_parquet(?);
    """,
        [PARQUET.as_posix()],
    )

    df = con.execute("""
        SELECT COUNT(*) AS n, MIN(created_at) AS min_t, MAX(created_at) AS max_t
        FROM embeddings
        WHERE emb_v='openai_t3small_v1' AND episode_id='0a4fa77e-bc0f-11ef-bab6-3f37b4906b43';
    """).df()
    print(df)

    con.close()


if __name__ == "__main__":
    main()
