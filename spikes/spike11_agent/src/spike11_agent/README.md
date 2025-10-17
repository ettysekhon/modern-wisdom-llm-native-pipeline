# Spike 11 Agent Tooling

First ensure the following docker services are running (from root)

- Qdrant is running `docker compose -f infra/qdrant/docker-compose.yml up -d`
- Phoenix Arize is running for tracing and observability `docker compose -f infra/phoenix/docker-compose.yml up -d`

The run

```bash
uv run run-spike11 run \
  --episode-id 0a4fa77e-bc0f-11ef-bab6-3f37b4906b43 \
  --question "What did the guest argue about walkability?" \
  --top-k 8 --llm-provider openai --llm-model-id gpt-4o-mini
```

```bash
uv run run-spike11 deep-agent \
  --episode-id 0a4fa77e-bc0f-11ef-bab6-3f37b4906b43 \
  --question "Compare Chris’s views on discipline in 2021 vs 2024 and link the most relevant clips." \
  --scope auto \
  --top-k 8 \
  --llm-provider openai \
  --llm-model-id gpt-4o-mini
```
