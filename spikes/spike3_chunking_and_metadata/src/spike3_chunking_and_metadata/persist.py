import time
from pathlib import Path

import duckdb
import pandas as pd

from .schema import validate_chunk_df_columns

LOCK_MSG = "Could not set lock on file"


def write_chunks_parquet(rows: list[dict], method: str, episode_id: str, out_dir: Path) -> Path:
    if not rows:
        raise RuntimeError(f"No chunks produced for method={method}")
    df = pd.DataFrame(rows)
    missing = validate_chunk_df_columns(df)
    if missing:
        raise ValueError(f"Chunk schema missing columns: {missing}")
    dest = out_dir / method / f"episode_id={episode_id}"
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / "part-00000.snappy.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def upsert_duckdb(
    parquet_path: Path, db_path: Path, max_retries: int = 5, base_sleep_s: float = 0.3
):
    attempt = 0
    while True:
        try:
            con = duckdb.connect(str(db_path))  # read-write (we need to insert)
            try:
                con.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chunks AS
                    SELECT * FROM read_parquet(?) WHERE 0=1
                """,
                    [str(parquet_path)],
                )

                con.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS chunks_chunk_id_idx ON chunks(chunk_id)"
                )

                con.execute(
                    "INSERT OR REPLACE INTO chunks SELECT * FROM read_parquet(?)",
                    [str(parquet_path)],
                )
            finally:
                con.close()
            break
        except duckdb.IOException as e:
            msg = str(e)
            attempt += 1
            if LOCK_MSG in msg and attempt <= max_retries:
                time.sleep(base_sleep_s * (2 ** (attempt - 1)))
                continue
            raise
