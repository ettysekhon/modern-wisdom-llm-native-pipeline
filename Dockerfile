# Production build for Modern Wisdom RAG Pipeline
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    CHAINLIT_TELEMETRY=False

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* ./

COPY src ./src
COPY .chainlit ./.chainlit
COPY public ./public
COPY chainlit.md ./chainlit.md
COPY data/duckdb ./data/duckdb

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8001/ || exit 1

CMD ["chainlit", "run", "src/modern_wisdom_rag_pipeline/chainlit_app.py", "--host", "0.0.0.0", "--port", "8001"]
