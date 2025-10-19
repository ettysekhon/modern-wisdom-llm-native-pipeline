from __future__ import annotations

import os

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

MW_DATA_DIR = os.environ.get("MW_DATA_DIR", "data")
