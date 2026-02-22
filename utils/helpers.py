from __future__ import annotations

"""Helper utilities for the civic complaint system.

This module provides:

* Unique complaint ID generation.
* Priority score computation for queue ordering.
* Human-readable ETA estimation based on urgency and queue position.
"""

import math
import time
from typing import Dict


def generate_complaint_id() -> str:
    """Generate a unique complaint ID.

    Uses the current UNIX timestamp with millisecond precision. In a
    production system you might use UUIDs instead.
    """
    millis = int(time.time() * 1000)
    return f"C{millis}"


URGENCY_WEIGHTS: Dict[str, int] = {
    "Critical": 3,
    "High": 2,
    "Medium": 1,
    "Low": 0,
}


def compute_priority_score(
    urgency: str,
    severity_score: float,
    repeat_complaint_count: float,
    affected_population: float,
) -> float:
    """Compute a numeric priority score for the queue.

    Higher scores indicate higher priority. This combines the urgency
    level with severity, repeat count, and affected population.
    """
    base = URGENCY_WEIGHTS.get(urgency, 0)
    score = (
        base * 10.0
        + float(severity_score) * 5.0
        + float(repeat_complaint_count) * 1.5
        + float(affected_population) * 1.0
    )
    return float(score)


def estimate_eta_text(urgency: str, queue_position: int) -> str:
    """Estimate a human-readable response time based on urgency and queue.

    This is a heuristic intended for user-facing feedback rather than a
    strict SLA.
    """
    queue_position = max(queue_position, 1)

    if urgency == "Critical":
        base_hours = 6
    elif urgency == "High":
        base_hours = 12
    elif urgency == "Medium":
        base_hours = 36
    else:  # Low
        base_hours = 72

    total_hours = base_hours * math.log2(queue_position + 1)
    days = total_hours / 24.0

    if days <= 1:
        return "0–1 day"
    if days <= 2:
        return "1–2 days"
    if days <= 3:
        return "2–3 days"
    return "3–5 days"


__all__ = [
    "generate_complaint_id",
    "compute_priority_score",
    "estimate_eta_text",
]