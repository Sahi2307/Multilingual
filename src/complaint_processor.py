from __future__ import annotations

"""Complaint processing pipeline (Phase 4 backend logic).

This module defines :class:`ComplaintProcessor`, which encapsulates the
end-to-end logic for:

* Loading the trained category and urgency models.
* Predicting category and urgency for new complaints.
* Generating SHAP-based explanations for both predictions.
* Persisting complaints, status updates, notifications and model
  predictions into the SQLite database.
* Computing queue position and estimated response time based on a
  simple priority scoring algorithm.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import xgboost as xgb

from src.data_preparation import compute_severity_score, extract_emergency_keywords
from src.explainability import (
    CATEGORY_LABELS,
    URGENCY_LEVELS,
    CategoryExplanation,
    ExplainabilityEngine,
    UrgencyExplanation,
)
from utils.database import (
    get_complaint,
    get_department_id_for_category,
    get_or_create_user,
    init_db,
    insert_complaint,
    insert_model_prediction,
    insert_status_update,
    insert_notification,
    list_open_complaints_for_department,
)
from utils.helpers import compute_priority_score, estimate_eta_text, generate_complaint_id
from utils.notifications import create_in_app_notification, send_email_notification, send_sms_notification

logger = logging.getLogger(__name__)


@dataclass
class ProcessedComplaintResult:
    """Result of end-to-end complaint processing."""

    complaint_id: str
    category: str
    category_confidence: float
    urgency: str
    urgency_confidence: float
    department_name: str
    queue_position: int
    eta_text: str
    category_explanation: CategoryExplanation
    urgency_explanation: UrgencyExplanation


class ComplaintProcessor:
    """High-level complaint processing pipeline.

    Instances of this class are intended to be long-lived (e.g., cached
    via :func:`streamlit.cache_resource`) and reused across requests.
    """

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self.project_root = project_root or Path(__file__).resolve().parents[1]

        # Initialise DB tables on startup
        init_db(self.project_root)

        # Lazy-loaded explainability engine and models
        self.engine = ExplainabilityEngine(project_root=self.project_root)

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------

    def load_category_model(self) -> None:
        """Ensure the category model is loaded into memory."""
        _ = self.engine.category_explainer  # triggers lazy load

    def load_urgency_model(self) -> None:
        """Ensure the urgency model is loaded into memory."""
        _ = self.engine.urgency_explainer  # triggers lazy load

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def predict_category(
        self, text: str
    ) -> Tuple[str, float, Dict[str, float], CategoryExplanation]:
        """Predict category for a complaint text and return SHAP explanation.

        Args:
            text: Complaint text.

        Returns:
            Tuple of (label, confidence, probability mapping, explanation).
        """
        explanation = self.engine.explain_category(text, top_k=5)
        
        # In XGBoost multi:softprob, the probabilities are in the explanation's confidence
        # but we need the full prob_map.
        # For simplicity, we'll re-run a quick predict here or adapt the explainer.
        explainer = self.engine.category_explainer
        X_scaled = explainer._build_feature_vector(text)
        dmatrix = xgb.DMatrix(X_scaled)
        probs = np.asarray(explainer.model.predict(dmatrix)).reshape(-1, len(CATEGORY_LABELS))[0]
        
        prob_map = {label: float(probs[i]) for i, label in enumerate(CATEGORY_LABELS)}
        label = explanation.predicted_label
        confidence = explanation.confidence

        return label, confidence, prob_map, explanation

    def _build_structured_features(
        self,
        text: str,
        affected_label: str,
    ) -> Dict[str, float]:
        """Rebuild structured features for a new complaint.

        This mirrors the logic used during training so that the urgency
        model receives compatible features.
        """
        emergency_keywords = extract_emergency_keywords(text)
        emergency_keyword_score = 1.0 if emergency_keywords else 0.0

        text_length = float(len(text.split()))
        affected_map = {
            "Few individuals": 0.0,
            "One street / lane": 1.0,
            "Neighborhood / locality": 2.0,
            "Large area / crowd": 3.0,
        }
        affected_population = affected_map.get(affected_label, 1.0)
        
        # Map population to scope_id
        scope_id = affected_population # They are directly equivalent in our current logic (0-3)

        # Compute severity score (baseline for queue priority)
        # Note: We use a placeholder urgency for now as predicting it requires these features.
        # But for the priority score later, we'll have the actual predicted urgency.
        severity_score = compute_severity_score("Medium", emergency_keywords, text)

        # For now, use simple heuristics for repeat count and time-based
        # features. In a production system these would come from
        # historical complaint data and timestamps.
        repeat_complaint_count = 0.0

        from datetime import datetime

        now = datetime.now()
        hour_of_day = float(now.hour)
        is_weekend = float(1 if now.weekday() >= 5 else 0)
        is_monsoon = float(1 if now.month in {6, 7, 8, 9} else 0)

        return {
            "emergency_keyword_score": emergency_keyword_score,
            "severity_score": severity_score,
            "text_length": text_length,
            "affected_population": affected_population,
            "scope_id": scope_id,
            "repeat_complaint_count": repeat_complaint_count,
            "hour_of_day": hour_of_day,
            "is_weekend": is_weekend,
            "is_monsoon_season": is_monsoon,
        }

    def predict_urgency(
        self,
        text: str,
        structured_features: Dict[str, float],
    ) -> Tuple[str, float, Dict[str, float], UrgencyExplanation]:
        """Predict urgency for a complaint and return SHAP explanation."""
        explainer = self.engine.urgency_explainer

        # Build the same 776-dim feature vector used during training
        X_raw, _ = explainer._build_feature_vector(text, structured_features)  # noqa: SLF001
        X_scaled = explainer.scaler.transform(X_raw)

        # XGBoost Booster requires DMatrix input; model is trained with multi:softprob
        dmatrix = xgb.DMatrix(X_scaled)
        probs = explainer.model.predict(dmatrix)
        probs = np.asarray(probs).reshape(-1, len(URGENCY_LEVELS))[0]
        pred_idx = int(np.argmax(probs))
        label = URGENCY_LEVELS[pred_idx]
        confidence = float(probs[pred_idx])
        prob_map = {URGENCY_LEVELS[i]: float(probs[i]) for i in range(len(URGENCY_LEVELS))}

        explanation = explainer.explain(text, structured_features)
        return label, confidence, prob_map, explanation

    # ------------------------------------------------------------------
    # Clustering helpers
    # ------------------------------------------------------------------

    def find_similar_complaints(
        self,
        category: str,
        location: str,
        threshold: float = 0.6,
    ) -> Optional[Dict[str, Any]]:
        """Find an existing active complaint that matches the category and location.
        
        Uses difflib.SequenceMatcher for fuzzy location matching.
        Returns the 'Parent' complaint if a match is found.
        """
        from difflib import SequenceMatcher
        from utils.database import list_active_complaints_by_category

        candidates = list_active_complaints_by_category(category, project_root=self.project_root)
        
        best_match = None
        best_ratio = 0.0

        for comp in candidates:
            # Compare locations
            # We normalize by lowercasing
            loc1 = location.lower()
            loc2 = (comp.get("location") or "").lower()
            
            ratio = SequenceMatcher(None, loc1, loc2).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = comp

        if best_ratio >= threshold:
            return best_match
        return None

    # ------------------------------------------------------------------
    # End-to-end processing
    # ------------------------------------------------------------------

    def _compute_queue_position_and_eta(
        self,
        department_id: int,
        urgency: str,
        severity_score: float,
        repeat_complaint_count: float,
        affected_population: float,
        complaint_id: str,
    ) -> Tuple[int, str]:
        """Compute queue position and ETA for a newly created complaint."""
        open_complaints = list_open_complaints_for_department(
            department_id=department_id,
            project_root=self.project_root,
        )

        # Add the new complaint into the candidate list
        open_complaints.append(
            {
                "id": complaint_id,
                "urgency": urgency,
                "created_at": "9999-12-31T23:59:59",  # ensured to be last if same priority
            }
        )

        scores = {}
        for row in open_complaints:
            row_id = str(row["id"])
            row_urgency = str(row.get("urgency", urgency))
            # For existing complaints we lack severity/repeat/affected
            # in the DB, so approximate with neutral values.
            if row_id == complaint_id:
                sev = severity_score
                rep = repeat_complaint_count
                aff = affected_population
            else:
                sev = 0.5
                rep = 0.0
                aff = 1.0
            scores[row_id] = compute_priority_score(row_urgency, sev, rep, aff)

        # Sort by score desc, then by created_at asc for fairness.
        open_complaints.sort(
            key=lambda r: (-scores[str(r["id"])], str(r.get("created_at", ""))),
        )

        queue_position = 1
        for idx, row in enumerate(open_complaints, start=1):
            if str(row["id"]) == complaint_id:
                queue_position = idx
                break

        eta_text = estimate_eta_text(urgency, queue_position)
        return queue_position, eta_text

    def process_complaint(
        self,
        name: str,
        email: str,
        phone: str,
        location: str,
        complaint_text: str,
        language: str,
        affected_label: str,
        category_hint: Optional[str] = None,
    ) -> ProcessedComplaintResult:
        """Full complaint intake pipeline.

        Args:
            name: Citizen's name.
            email: Citizen's email (used as primary user key).
            phone: Citizen's phone number.
            location: Location description.
            complaint_text: Complaint body.
            language: Declared language ("English", "Hindi", or "Hinglish").
            affected_label: Human-readable impacted population label.
            category_hint: Optional pre-selected category; if provided it
                is used as a hint but prediction and explanations still
                come from the model.

        Returns:
            :class:`ProcessedComplaintResult` with IDs, predictions,
            SHAP explanations, queue position, and ETA.
        """
        # 1. Ensure models are available
        self.load_category_model()
        self.load_urgency_model()

        # 2. Get or create user
        user = get_or_create_user(
            name=name or "Citizen",
            email=email or f"anonymous_{language.lower()}@example.org",
            phone=phone or "",
            project_root=self.project_root,
        )

        # 3. Generate predictions and SHAP explanations
        cat_label, cat_conf, cat_probs, cat_exp = self.predict_category(complaint_text)
        # If the user provided a category hint, use it for routing/storage while
        # still keeping the model prediction for explanations/metrics.
        effective_category = category_hint if category_hint in CATEGORY_LABELS else cat_label

        structured_features = self._build_structured_features(complaint_text, affected_label)
        urg_label, urg_conf, urg_probs, urg_exp = self.predict_urgency(
            complaint_text,
            structured_features,
        )

        # 4. Department routing
        department_id = get_department_id_for_category(effective_category, project_root=self.project_root)
        department_name = f"Municipal Department - {effective_category}"

        # --- CLUSTERING LOGIC START ---
        from utils.database import update_record
        
        # Check for similar complaints
        parent_complaint = self.find_similar_complaints(effective_category, location)
        
        is_clustered = False
        parent_id = None
        cluster_size = 1
        
        if parent_complaint:
            # We found a match! This new complaint will be a child.
            is_clustered = True
            parent_id = parent_complaint["complaint_id"]
            
            # Update parent cluster size
            current_size = parent_complaint.get("cluster_size", 1) or 1
            new_size = current_size + 1
            
            # Auto-escalation Logic
            parent_urgency = parent_complaint.get("urgency", "Low")
            new_urgency = parent_urgency
            
            upgraded = False
            if new_size > 3 and parent_urgency not in ["Critical", "High"]:
                 new_urgency = "High"
                 upgraded = True
            elif new_size > 5 and parent_urgency != "Critical":
                 new_urgency = "Critical"
                 upgraded = True
            
            update_data = {"cluster_size": new_size}
            if upgraded:
                update_data["urgency"] = new_urgency
                # Insert system status update for escalation
                insert_status_update(
                    complaint_id=parent_id,
                    status=parent_complaint.get("status", "Registered"),
                    remarks=f"System: Priority escalated to {new_urgency} due to multiple reports ({new_size}).",
                    project_root=self.project_root
                )
            
            update_record("complaints", parent_complaint["id"], update_data) # use row_id or check logic
            
            # For the current user result, we might want to show the PARENT's info or own info?
            # We will show mix: Their own ID, but maybe the Urgency of the cluster?
            # For now, let's keep their own urgency prediction but routing is same.
            
        # --- CLUSTERING LOGIC END ---

        # 5. Compute queue position and ETA
        complaint_id = generate_complaint_id()
        queue_position, eta_text = self._compute_queue_position_and_eta(
            department_id=department_id,
            urgency=urg_label,
            severity_score=structured_features["severity_score"],
            repeat_complaint_count=structured_features["repeat_complaint_count"],
            affected_population=structured_features["affected_population"],
            complaint_id=complaint_id,
        )

        # 6. Persist complaint and initial status
        insert_complaint(
            complaint_id=complaint_id,
            user_id=user.id,
            text=complaint_text,
            category=effective_category,
            urgency=urg_label,
            language=language,
            location=location,
            status="Registered",
            department_id=department_id,
            parent_complaint_id=parent_id,
            is_clustered=is_clustered,
            cluster_size=1, # New complaint starts as size 1 (if parent, its contributing to parent, itself is size 1)
            project_root=self.project_root,
        )

        insert_status_update(
            complaint_id=complaint_id,
            status="Registered",
            remarks="Complaint registered via web form." + (" (Linked to cluster)" if is_clustered else ""),
            official_id=user.id,
            project_root=self.project_root,
        )

        # 7. Persist model prediction metadata
        shap_summary = {
            "category_top_keywords": cat_exp.top_keywords,
            "urgency_factor_importance": urg_exp.factor_importance,
        }
        insert_model_prediction(
            complaint_id=complaint_id,
            category_prob=cat_probs,
            urgency_prob=urg_probs,
            shap_summary=shap_summary,
            project_root=self.project_root,
        )

        # 8. Notifications
        message = (
            f"Your complaint {complaint_id} has been registered with category "
            f"'{effective_category}' (model suggested '{cat_label}') and urgency '{urg_label}'. "
            f"Estimated response time: {eta_text}."
        )
        if is_clustered:
            message += " We noticed similar reports in your area and have linked them to prioritize resolution."

        insert_notification(user_id=user.id, message=message, complaint_id=complaint_id, project_root=self.project_root)
        if email:
            send_email_notification(email, "Complaint registered", message)
        if phone:
            send_sms_notification(phone, message)

        return ProcessedComplaintResult(
            complaint_id=complaint_id,
            category=effective_category,
            category_confidence=cat_conf,
            urgency=urg_label,
            urgency_confidence=urg_conf,
            department_name=department_name,
            queue_position=queue_position,
            eta_text=eta_text,
            category_explanation=cat_exp,
            urgency_explanation=urg_exp,
        )


__all__ = ["ComplaintProcessor", "ProcessedComplaintResult"]
