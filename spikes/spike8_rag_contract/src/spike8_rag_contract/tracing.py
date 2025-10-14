from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Timings:
    retrieve_ms: float = 0.0
    generate_ms: float = 0.0
