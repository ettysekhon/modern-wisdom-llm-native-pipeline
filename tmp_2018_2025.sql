COPY (
  SELECT id
  FROM mw.episodes
  WHERE try_cast(publish_date AS DATE) IS NOT NULL
    AND extract(year FROM try_cast(publish_date AS DATE)) IN (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)
  ORDER BY publish_date
) TO 'data/tmp/epids_2018_2025.txt' (HEADER false, DELIMITER '\n');