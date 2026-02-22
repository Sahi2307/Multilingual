from __future__ import annotations

"""Integration tests for Phase 2 explainability.

These tests validate that the SHAP-based explainability layer (Phase 2)
correctly builds on top of the trained models and feature pipeline from
Phase 1.

They require that you have already run:

- ``python -m src.data_preparation``
- ``python -m src.feature_extraction``
- ``python -m src.train_category_model``
- ``python -m src.train_urgency_model``

If the necessary model artifacts are missing, the tests will be skipped
with an informative message instead of failing.
"""

from pathlib import Path

import numpy as np
import pytest

from src.data_preparation import compute_severity_score, extract_emergency_keywords
from src.explainability import (
    CATEGORY_LABELS,
    URGENCY_LEVELS,
    ExplainabilityEngine,
)
from src.feature_extraction import STRUCTURED_FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def _models_available() -> bool:
    """Return ``True`` if all required Phase 1 model artifacts exist."""

    cat_dir = MODELS_DIR / "muril_category_classifier"
    xgb_path = MODELS_DIR / "xgboost_urgency_predictor.pkl"
    scaler_path = MODELS_DIR / "feature_scaler.pkl"
    return cat_dir.exists() and xgb_path.exists() and scaler_path.exists()


@pytest.mark.skipif(not _models_available(), reason="Phase 1 models are not trained yet")
def test_explainability_end_to_end() -> None:
    """End-to-end test for category and urgency explanations on a sample text.

    This mirrors the example from the project specification and the
    ``src.test_explainability`` script, but adds assertions about the
    structure of SHAP outputs.
    """

    text = (
        "Hamare area ki road bahut kharab ho gayi hai, "
        "potholes ki wajah se accident ka risk hai"
    )

    # Build structured features in a way consistent with Phase 1
    emergency_keywords = extract_emergency_keywords(text)
    emergency_keyword_score = 1.0 if emergency_keywords else 0.0
    severity_score = compute_severity_score("Medium", emergency_keywords, text)

    structured_features = {
        "emergency_keyword_score": emergency_keyword_score,
        "severity_score": float(severity_score),
        "text_length": float(len(text.split())),
        "affected_population": 2.0,  # neighbourhood-level impact
        "repeat_complaint_count": 1.0,
        "hour_of_day": 14.0,
        "is_weekend": 0.0,
        "is_monsoon_season": 1.0,
    }

    # Sanity: we must provide exactly the 8 structured features used in Phase 1
    assert set(structured_features.keys()) == set(STRUCTURED_FEATURE_COLUMNS)

    engine = ExplainabilityEngine(project_root=PROJECT_ROOT)

    # Category explanation
    cat_exp = engine.explain_category(text, top_k=5)

    assert cat_exp.predicted_label in CATEGORY_LABELS
    assert 0.0 <= cat_exp.confidence <= 1.0
    assert len(cat_exp.tokens) == len(cat_exp.token_importances) > 0
    assert len(cat_exp.top_keywords) <= 5
    assert cat_exp.shap_values.shape[0] == len(cat_exp.tokens)
    assert isinstance(cat_exp.nl_explanation, str) and cat_exp.nl_explanation

    # Urgency explanation
    urg_exp = engine.explain_urgency(text, structured_features)

    assert urg_exp.predicted_label in URGENCY_LEVELS
    assert 0.0 <= urg_exp.confidence <= 1.0

    # We expect at least text_embedding + all structured features in the maps
    expected_keys = {"text_embedding", *STRUCTURED_FEATURE_COLUMNS}
    assert expected_keys.issubset(urg_exp.feature_contributions.keys())
    assert expected_keys.issubset(urg_exp.factor_importance.keys())

    # SHAP vector length must be 776 and should align with the 768+8 split
    assert urg_exp.shap_values_vector.shape == (776,)

    # Factor importances should sum to approximately 100% (allowing small
    # numerical differences)
    total_importance = float(sum(urg_exp.factor_importance.values()))
    assert 95.0 <= total_importance <= 105.0

    # Natural-language explanation should be non-empty and mention the
    # predicted urgency label.
    assert isinstance(urg_exp.nl_explanation, str) and urg_exp.nl_explanation
    assert urg_exp.predicted_label in urg_exp.nl_explanation
