# Spike 11 Agents

## Summary of steps to backfill almost 1000 episodes - chunk, embed and upsert into Qdrant

Save a file called tmp_2018_2025.sql

```bash
COPY (
  SELECT id
  FROM mw.episodes
  WHERE try_cast(publish_date AS DATE) IS NOT NULL
    AND extract(year FROM try_cast(publish_date AS DATE)) IN (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)
  ORDER BY publish_date
) TO 'data/tmp/epids_2028_2025.txt' (HEADER false, DELIMITER '\n');
```

Run the script in DuckDB

```bash
duckdb data/duckdb/modern_wisdom.duckdb -readonly -c ".read tmp_2018_2025.sql"
```

Check if chunkable!

```bash
while read -r EPID; do
  echo "Checking if $EPID is chunkable"
  uv run run-spike3 check --episode-id "$EPID"
  if [ $? -ne 0 ]; then
    echo "Error processing $EPID"
  fi
done < data/tmp/tmp_2018_2025.txt
```

Chunk all target episodes

```bash
cat data/tmp/epids_2018_2025.txt | while read -r EID; do
  echo "Chunking $EID"
  uv run run-spike3 chunk \
    --episode-id "$EID" \
    --methods sentence_bound \
    --size-tokens 700 --overlap-tokens 100
  
  if [ $? -ne 0 ]; then
    echo "Error processing $EPID"
  fi
done
```

verify

```bash
find data/chunks/sentence_bound -maxdepth 1 -type d -name "episode_id=*" | wc -l
```

Embed the chunks (per episode)

```bash
EMB_V="BAAI/bge-small-en-v1.5"

cat data/tmp/epids_2018_2025.txt | while read -r EID; do
  echo "Embedding $EID"
  uv run run-spike4 embed \
    --episode-id "$EID" \
    --emb-v "$EMB_V" \
    --method sentence_bound
done
```

```bash
EMB_V="BAAI/bge-small-en-v1.5"
INDEX_VERSION="mw_chunks_live"

cat data/tmp/epids_2018_2025.txt | while read -r EID; do
  echo "Upserting vectors for $EID"
  uv run run-spike5 upsert \
    --episode-id "$EID" \
    --method sentence_bound \
    --emb-v "$EMB_V" \
    --set-live \
    --live-alias "$INDEX_VERSION"
done
```
