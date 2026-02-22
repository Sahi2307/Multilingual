from __future__ import annotations

"""Integration tests for the thin API layer and backend (Phase 4).

These tests exercise the `src.api` functions on top of the full Phase 1
(model training) and Phase 2 (explainability) stack. They verify that:

* Complaints can be registered via `api_register_complaint`.
* The complaint is persisted in the SQLite database.
* Complaint status can be retrieved via `api_get_complaint_status`.
* Notifications are stored for the associated user and can be listed via
  `api_list_notifications`.

The tests assume that you have already run the Phase 1 scripts:

- `python -m src.data_preparation`
- `python -m src.feature_extraction`
- `python -m src.train_category_model`
- `python -m src.train_urgency_model`

As with the explainability integration tests, they are skipped if the
required model artifacts are not present.
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from src import api as civic_api
from utils.database import get_connection, init_db


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def _models_available() -> bool:
    """Return True if Phase 1 model artifacts exist on disk."""

    cat_dir = MODELS_DIR / "muril_category_classifier"
    xgb_path = MODELS_DIR / "xgboost_urgency_predictor.pkl"
    scaler_path = MODELS_DIR / "feature_scaler.pkl"
    return cat_dir.exists() and xgb_path.exists() and scaler_path.exists()


@pytest.mark.skipif(not _models_available(), reason="Phase 1 models are not trained yet")
def test_register_complaint_and_fetch_status(tmp_path: Path) -> None:
    """End-to-end API test: register complaint, then fetch status + notifications.

    This uses a dedicated SQLite file under `tmp_path` so that the test
    does not interfere with any existing project database. The same
    backend logic (`ComplaintProcessor`, DB helpers, and SHAP
    explanations) is exercised as in the Streamlit UI.
    """

    # Use a temporary DB path for isolation
    test_db_path = tmp_path / "civic_system_test.db"
    os.environ["CIVIC_DB_PATH"] = str(test_db_path)

    # Initialise schema in the temporary DB
    init_db(PROJECT_ROOT)

    sample_email = "test_user_phase4@example.org"
    sample_phone = "9999999999"

    payload = {
        "name": "Test User",
        "email": sample_email,
        "phone": sample_phone,
        "location": "Ward 42, Test Nagar",
        "complaint_text": (
            "Hamare area ki road bahut kharab ho gayi hai, "
            "potholes ki wajah se accident ka risk hai"
        ),
        "language": "Hinglish",
        "affected_label": "Neighborhood / locality",
        "category_hint": None,
    }

    # 1) Register the complaint via the API facade
    resp = civic_api.api_register_complaint(payload)

    # Basic response structure and types
    assert "complaint_id" in resp and isinstance(resp["complaint_id"], str)
    assert resp["category"] in {"Sanitation", "Water Supply", "Transportation"}
    assert resp["urgency"] in {"Critical", "High", "Medium", "Low"}
    assert 0.0 <= resp["category_confidence"] <= 1.0
    assert 0.0 <= resp["urgency_confidence"] <= 1.0
    assert isinstance(resp["queue_position"], int) and resp["queue_position"] >= 1
    assert isinstance(resp["eta_text"], str) and resp["eta_text"]

    complaint_id = resp["complaint_id"]

    # 2) Verify complaint is present in the DB and fetch status via API
    status_payload = civic_api.api_get_complaint_status(complaint_id)
    assert status_payload is not None
    assert "complaint" in status_payload and "status_updates" in status_payload

    comp_rec = status_payload["complaint"]
    assert comp_rec["id"] == complaint_id
    assert comp_rec["status"] == "Registered"
    assert comp_rec["category"] == resp["category"]
    assert comp_rec["urgency"] == resp["urgency"]

    updates = status_payload["status_updates"]
    assert isinstance(updates, list) and len(updates) >= 1
    assert updates[0]["status"] == "Registered"

    # 3) Verify that a notification was stored for the user and can be listed
    with get_connection(PROJECT_ROOT) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE email = ?", (sample_email,))
        row = cur.fetchone()
        assert row is not None, "User should exist in users table after registration"
        user_id = int(row["id"])

    notifications = civic_api.api_list_notifications(user_id, include_read=True)
    assert isinstance(notifications, list) and len(notifications) >= 1
    # At least one notification should reference the complaint ID
    assert any(complaint_id in n.get("message", "") for n in notifications)
