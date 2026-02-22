from __future__ import annotations

"""Thin API layer over the complaint processor and database.

This module does not start an HTTP server but provides clear functional
APIs that could easily be exposed via FastAPI or another framework if
needed.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.complaint_processor import ComplaintProcessor
from utils.database import (
    get_complaint,
    list_notifications,
    list_status_updates_for_complaint,
)


_project_root = Path(__file__).resolve().parents[1]
_processor = ComplaintProcessor(project_root=_project_root)


def api_register_complaint(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Register a new complaint and return a JSON-serializable summary.

    Expected payload keys (all strings):
      - name, email, phone, location, complaint_text, language,
        affected_label, category_hint (optional).
    """
    result = _processor.process_complaint(
        name=str(payload.get("name", "")),
        email=str(payload.get("email", "")),
        phone=str(payload.get("phone", "")),
        location=str(payload.get("location", "")),
        complaint_text=str(payload.get("complaint_text", "")),
        language=str(payload.get("language", "English")),
        affected_label=str(payload.get("affected_label", "One street / lane")),
        category_hint=payload.get("category_hint"),
    )

    return {
        "complaint_id": result.complaint_id,
        "category": result.category,
        "category_confidence": result.category_confidence,
        "urgency": result.urgency,
        "urgency_confidence": result.urgency_confidence,
        "department": result.department_name,
        "queue_position": result.queue_position,
        "eta_text": result.eta_text,
        "category_top_keywords": result.category_explanation.top_keywords,
        "urgency_factor_importance": result.urgency_explanation.factor_importance,
    }


def api_get_complaint_status(complaint_id: str) -> Optional[Dict[str, Any]]:
    """Return complaint details and status updates for a given ID."""
    rec = get_complaint(complaint_id, project_root=_project_root)
    if rec is None:
        return None

    updates = list_status_updates_for_complaint(complaint_id, project_root=_project_root)

    return {
        "complaint": rec,
        "status_updates": updates,
    }


def api_list_notifications(user_id: int, include_read: bool = False) -> List[Dict[str, Any]]:
    """Return notifications for a given user ID."""
    return list_notifications(user_id, include_read=include_read, project_root=_project_root)


__all__ = [
    "api_register_complaint",
    "api_get_complaint_status",
    "api_list_notifications",
]
