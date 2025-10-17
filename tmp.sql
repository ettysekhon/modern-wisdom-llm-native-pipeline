-- Run with `duckdb data/duckdb/modern_wisdom.duckdb -readonly -c ".read tmp.sql"`
.copy (
  select id
  from mw.episodes
  where (title ilike '%discipline%'
     or headline ilike '%discipline%'
     or description ilike '%discipline%')
    and extract(year from publish_date) in (2021, 2024)
) to 'data/tmp/discipline_ids.txt' (HEADER false, DELIMITER '\n');
