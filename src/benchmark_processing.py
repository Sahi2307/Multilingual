from __future__ import annotations

"""Simple performance benchmark for complaint processing.

This script measures the average end-to-end processing time for a
handful of sample complaints using `ComplaintProcessor`. It is intended
as a lightweight sanity check against the project target of processing a
complaint in under ~4 seconds (including model inference and SHAP
explanations), on your local hardware.

Run from the project root:

    python -m src.benchmark_processing

The script prints per-complaint timings and an overall average.
"""

import statistics
import time
from pathlib import Path

from src.complaint_processor import ComplaintProcessor


SAMPLE_COMPLAINTS = [
    {
        "name": "Bench User",
        "email": "bench_user_1@example.org",
        "phone": "9999990001",
        "location": "Ward 10, Sample Colony",
        "complaint_text": (
            "Hamare area ki road bahut kharab ho gayi hai, "
            "potholes ki wajah se accident ka risk hai"
        ),
        "language": "Hinglish",
        "affected_label": "Neighborhood / locality",
    },
    {
        "name": "Bench User",
        "email": "bench_user_2@example.org",
        "phone": "9999990002",
        "location": "Ward 5, River Side",
        "complaint_text": "There has been no water supply in our area for the past 3 days, residents are suffering.",
        "language": "English",
        "affected_label": "One street / lane",
    },
    {
        "name": "Bench User",
        "email": "bench_user_3@example.org",
        "phone": "9999990003",
        "location": "Ward 18, Market Area",
        "complaint_text": "गली से पिछले कई दिनों से कूड़ा नहीं उठाया गया है, बदबू से रहना मुश्किल हो गया है।",
        "language": "Hindi",
        "affected_label": "Large area / crowd",
    },
]


def main() -> None:
    """Run a simple timing benchmark over a few sample complaints."""

    project_root = Path(__file__).resolve().parents[1]
    processor = ComplaintProcessor(project_root=project_root)

    durations = []

    for idx, payload in enumerate(SAMPLE_COMPLAINTS, start=1):
        start = time.perf_counter()
        _ = processor.process_complaint(
            name=payload["name"],
            email=payload["email"],
            phone=payload["phone"],
            location=payload["location"],
            complaint_text=payload["complaint_text"],
            language=payload["language"],
            affected_label=payload["affected_label"],
        )
        end = time.perf_counter()
        duration = end - start
        durations.append(duration)
        print(f"Complaint {idx}: {duration:.3f} seconds")

    avg = statistics.mean(durations) if durations else 0.0
    print(f"Average processing time over {len(durations)} complaints: {avg:.3f} seconds")


if __name__ == "__main__":  # pragma: no cover - manual benchmark
    main()
