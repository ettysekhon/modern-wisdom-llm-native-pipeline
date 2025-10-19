# Simple, reliable build for API + Chainlit + spikes CLIs
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.4.20

COPY pyproject.toml uv.lock* ./

COPY src ./src
COPY spikes ./spikes

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    QDRANT_URL="http://qdrant:6333" \
    INDEX_VERSION="mw_chunks_live" \
    MW_DATA_DIR="/app/data" \
    CHAINLIT_TELEMETRY=False

EXPOSE 8000 8001

CMD ["uvicorn", "modern_wisdom_rag_pipeline.api:app", "--host", "0.0.0.0", "--port", "8000"]
