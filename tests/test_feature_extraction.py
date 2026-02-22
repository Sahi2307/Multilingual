from __future__ import annotations

"""Unit tests for Phase 1 feature extraction.

These tests assume that ``src.feature_extraction.create_and_save_urgency_features``
has been run at least once (or will be run by the test itself if needed),
and that it produces a 776-dimensional feature vector per complaint.
"""

from pathlib import Path

import numpy as np

from src.feature_extraction import STRUCTURED_FEATURE_COLUMNS, create_and_save_urgency_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NPZ_PATH = PROJECT_ROOT / "data" / "processed" / "urgency_features.npz"


def _ensure_features_exist() -> None:
    """Ensure the NPZ feature file exists by running the pipeline if needed."""

    if NPZ_PATH.exists():
        return

    # This will load MuRIL, compute embeddings, and save the NPZ file.
    # It may take a little while the first time (model download).
    create_and_save_urgency_features()


def test_urgency_feature_shapes() -> None:
    """Urgency feature matrices should have shape (n, 776)."""

    _ensure_features_exist()

    data = np.load(NPZ_PATH, allow_pickle=True)

    for split_name in ("train", "val", "test"):
        X = data[f"X_{split_name}"]
        y = data[f"y_{split_name}"]
        ids = data[f"id_{split_name}"]

        assert X.ndim == 2, f"X_{split_name} must be 2D, got {X.ndim}D"
        assert X.shape[1] == 776, f"X_{split_name} must have 776 features, got {X.shape[1]}"
        assert X.shape[0] == y.shape[0] == ids.shape[0], (
            f"Split {split_name}: inconsistent lengths - X={X.shape[0]}, y={y.shape[0]}, ids={ids.shape[0]}"
        )


def test_structured_feature_count() -> None:
    """Sanity check that we always use exactly 8 structured features."""

    assert len(STRUCTURED_FEATURE_COLUMNS) == 8, "Expected 8 structured features."
