from __future__ import annotations

"""SHAP-based explainability for the civic complaint system (Phase 2).

This module provides explainability for both:

* Category classification (MuRIL fine-tuned model)
* Urgency prediction (XGBoost model + 776-dim features)

Key features:
  * Word-level importance for the category prediction using
    :mod:`shap`'s text explainers.
  * Factor-wise importance for urgency (text embedding vs 8
    structured features) using :class:`shap.TreeExplainer`.
  * Natural-language summaries of the most important words/factors
    for each prediction.

The functions and classes defined here are designed to be called from
Streamlit pages and from the higher-level ``ComplaintProcessor`` that
will be implemented in a later phase.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import shap
import torch
import xgboost as xgb
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)

from .feature_extraction import (  # type: ignore
    MURIL_MODEL_NAME,
    STRUCTURED_FEATURE_COLUMNS,
    MurilFeatureExtractor,
)

logger = logging.getLogger(__name__)

# Ensure SHAP does not spam progress bars (using new config API)
try:
    shap.set_config(show_progress=False)
except Exception:
    # Older SHAP versions may not support set_config; ignore safely.
    pass


CATEGORY_LABELS: List[str] = ["Sanitation", "Water Supply", "Transportation"]
URGENCY_LEVELS: List[str] = ["Critical", "High", "Medium", "Low"]


@dataclass
class CategoryExplanation:
    """Container for category-level SHAP explanation.

    Attributes:
        predicted_label: Human-readable category label.
        confidence: Probability associated with the predicted label.
        token_importances: List of ``{"token": str, "value": float}`` in
            the order they appear in the text.
        top_keywords: Top-k tokens by absolute SHAP value.
        shap_values: Raw SHAP values (1D array) aligned with tokens.
        tokens: List of tokens for the explanation.
        text: Original complaint text.
        nl_explanation: Natural-language summary of the reasoning.
    """

    predicted_label: str
    confidence: float
    token_importances: List[Dict[str, Any]]
    top_keywords: List[str]
    shap_values: np.ndarray
    tokens: List[str]
    text: str
    nl_explanation: str


@dataclass
class UrgencyExplanation:
    """Container for urgency-level SHAP explanation.

    Attributes:
        predicted_label: Predicted urgency label.
        confidence: Probability associated with the predicted label.
        feature_contributions: Mapping from feature name to SHAP value
            for the predicted class. Includes an aggregated
            ``text_embedding`` factor.
        factor_importance: Mapping from feature name to absolute
            contribution percentage (sums to 100).
        shap_values_vector: Raw SHAP values (1D array of length 776).
        expected_value: SHAP expected value for the predicted class.
        text: Original complaint text.
        structured_features: Mapping of the 8 structured features used.
        nl_explanation: Natural-language explanation of what drove
            the urgency decision.
    """

    predicted_label: str
    confidence: float
    feature_contributions: Dict[str, float]
    factor_importance: Dict[str, float]
    shap_values_vector: np.ndarray
    expected_value: float
    text: str
    structured_features: Dict[str, float]
    nl_explanation: str


class CategorySHAPExplainer:
    """SHAP explainer for the XGBoost category classifier.

    This explainer:
      * Recomputes the 768-dim MuRIL embedding for the input text.
      * Concatenates the structured features.
      * Uses :class:`shap.TreeExplainer` to obtain per-feature
        contributions for the predicted category class.
    """

    def __init__(
        self,
        xgb_model_path: Path,
        scaler_path: Path,
        muril_model_name: str = MURIL_MODEL_NAME,
    ) -> None:
        logger.info("Loading XGBoost category model from %s", xgb_model_path)
        with open(xgb_model_path, "rb") as f:
            data = joblib.load(f)
            self.model = data["model"]
            self.label_encoder = data.get("label_encoder")

        self.scaler = joblib.load(scaler_path)
        self.muril_extractor = MurilFeatureExtractor(model_name=muril_model_name)
        
        # Lazy-initialize a text explainer for token-level importance
        self.tokenizer = self.muril_extractor.tokenizer
        self._explainer = None

    def _get_explainer(self):
        """Build a SHAP explainer for text -> category probability."""
        if self._explainer is None:
            def predict_fn(texts: List[str]) -> np.ndarray:
                # Handle single string input if needed
                if isinstance(texts, str):
                    texts = [texts]
                
                # Filter out empty texts which can occur during SHAP perturbation
                # and might cause MuRIL to fail or return NaNs
                clean_texts = [t if t.strip() else " " for t in texts]
                
                embs = self.muril_extractor.encode(clean_texts)
                sf_values = np.zeros((len(clean_texts), 8), dtype=np.float32)
                features = np.concatenate([embs, sf_values], axis=1)
                X_scaled = self.scaler.transform(features)
                dmatrix = xgb.DMatrix(X_scaled)
                return self.model.predict(dmatrix)

            # Use a simple Text masker
            masker = shap.maskers.Text(self.tokenizer)
            # Create the explainer. We use silent=True to avoid stdout noise.
            self._explainer = shap.Explainer(predict_fn, masker=masker)
            
        return self._explainer

    def _build_feature_vector(self, text: str) -> np.ndarray:
        # For category, we use standard/neutral values for structured features
        # as they weren't primary drivers during training.
        emb = self.muril_extractor.encode([text])
        sf_values = np.zeros((1, 8), dtype=np.float32)
        features = np.concatenate([emb, sf_values], axis=1)
        return self.scaler.transform(features)

    def explain(self, text: str, top_k: int = 5) -> CategoryExplanation:
        X_scaled = self._build_feature_vector(text)
        dmatrix = xgb.DMatrix(X_scaled)
        probs = np.asarray(self.model.predict(dmatrix)).reshape(-1, len(CATEGORY_LABELS))[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        predicted_label = CATEGORY_LABELS[pred_idx]

        # Generate SHAP values for the text using the mask-based explainer
        try:
            # We only evaluate a small number of samples for speed
            explainer = self._get_explainer()
            shap_output = explainer([text], max_evals=100)
            
            # shape: (samples, tokens, classes)
            all_tokens = shap_output.data[0]
            # Use the raw values for the predicted class
            all_values = shap_output.values[0, :, pred_idx]
            
            # Filter and clean up tokens
            token_importances = []
            for tok, val in zip(all_tokens, all_values):
                # Clean up MuRIL/BERT tokens
                clean_tok = str(tok).replace("##", "").strip()
                # Skip tiny/punctual tokens for the "keywords" list
                if len(clean_tok) > 2 and clean_tok.isalnum():
                    token_importances.append({"token": clean_tok, "value": float(val)})
                elif len(clean_tok) > 0:
                    # Still keep it in the full list but it might be filtered from top_keywords
                    token_importances.append({"token": clean_tok, "value": float(val)})
            
            # Get top keywords (distinct and significant)
            seen = set()
            top_keywords = []
            # Sort by absolute value
            significant_tokens = sorted(token_importances, key=lambda x: abs(x["value"]), reverse=True)
            
            for item in significant_tokens:
                word = item["token"].lower()
                if word not in seen and len(word) > 2 and item["token"].isalnum():
                    top_keywords.append(item["token"])
                    seen.add(word)
                    if len(top_keywords) >= top_k:
                        break
            
            # Map back to simple arrays for compatibility
            final_tokens = [item["token"] for item in token_importances]
            final_values = np.array([item["value"] for item in token_importances])
            
        except Exception as e:
            logger.error(f"Category SHAP failed: {e}")
            tokens = text.split()
            token_importances = [{"token": tok, "value": 0.0} for tok in tokens]
            top_keywords = []
            final_tokens = tokens
            final_values = np.zeros(len(tokens))

        nl_explanation = self._build_natural_language_explanation(
            predicted_label, confidence, top_keywords
        )

        return CategoryExplanation(
            predicted_label=predicted_label,
            confidence=confidence,
            token_importances=token_importances,
            top_keywords=top_keywords,
            shap_values=final_values,
            tokens=final_tokens,
            text=text,
            nl_explanation=nl_explanation,
        )

    @staticmethod
    def _build_natural_language_explanation(
        label: str, confidence: float, keywords: Sequence[str]
    ) -> str:
        """Create a human-friendly explanation sentence."""
        conf_pct = round(confidence * 100, 1)
        if not keywords:
            return (
                f"The complaint was classified as '{label}' with"
                f" {conf_pct}% confidence. The model did not find any"
                " highly distinctive keywords."
            )

        joined = ", ".join(f"'{k}'" for k in keywords)
        return (
            f"The complaint was classified as '{label}' with {conf_pct}%"
            f" confidence, mainly due to the presence of keywords like"
            f" {joined}."
        )


class UrgencySHAPExplainer:
    """SHAP explainer for the XGBoost urgency classifier.

    This explainer:
      * Recomputes the 768-dim MuRIL embedding for the input text.
      * Concatenates the 8 structured features to form a 776-dim
        vector, scaled using the stored :class:`StandardScaler`.
      * Uses :class:`shap.TreeExplainer` to obtain per-feature
        contributions for the predicted urgency class.
      * Aggregates the 768 text dimensions into a single
        ``text_embedding`` factor.
    """

    def __init__(
        self,
        xgb_model_path: Path,
        scaler_path: Path,
        muril_model_name: str = MURIL_MODEL_NAME,
    ) -> None:
        """Initialize the urgency explainer.

        Args:
            xgb_model_path: Path to ``xgboost_urgency_predictor.pkl``.
            scaler_path: Path to ``feature_scaler.pkl``.
            muril_model_name: Name of the MuRIL model to use for
                embeddings.
        """
        logger.info("Loading XGBoost urgency model from %s", xgb_model_path)
        
        # Load from pickle which contains model, scaler, label_encoder
        try:
            with open(xgb_model_path, "rb") as f:
                data = joblib.load(f)
            
            if isinstance(data, dict) and "model" in data:
                self.model = data["model"]
                self.label_encoder = data.get("label_encoder")
            else:
                self.model = data
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise

        # Always load the fitted scaler from the separate file
        logger.info("Loading feature scaler from %s", scaler_path)
        try:
            self.scaler = joblib.load(scaler_path)
        except Exception as exc:
            logger.error("Failed to load scaler: %s", exc)
            raise

        logger.info("Initializing MuRIL feature extractor (%s)", muril_model_name)
        self.muril_extractor = MurilFeatureExtractor(model_name=muril_model_name)

        # Robust TreeExplainer initialization
        self.tree_explainer = None
        try:
            # Try initializing with the booster directly for better stability
            booster = getattr(self.model, "get_booster", lambda: self.model)()
            self.tree_explainer = shap.TreeExplainer(booster)
        except Exception as e:
            logger.warning(f"Direct Booster TreeExplainer failed: {e}. Trying raw model...")
            try:
                self.tree_explainer = shap.TreeExplainer(self.model)
            except Exception as e2:
                logger.error(f"All TreeExplainer attempts failed: {e2}")

    def _build_feature_vector(
        self,
        text: str,
        structured_features: Dict[str, float],
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Construct a single 776-dim feature vector for a complaint."""
        missing = [
            name
            for name in STRUCTURED_FEATURE_COLUMNS
            if name not in structured_features
        ]
        if missing:
            raise ValueError(
                "Missing structured features: " + ", ".join(missing)
            )

        # Text embedding (1, 768)
        emb = self.muril_extractor.encode([text])  # (1, 768)

        # Structured feature vector (1, 8)
        sf_values = np.array(
            [[float(structured_features[name]) for name in STRUCTURED_FEATURE_COLUMNS]],
            dtype=np.float32,
        )

        features = np.concatenate([emb, sf_values], axis=1)  # (1, 776)
        return features, structured_features

    def explain(
        self,
        text: str,
        structured_features: Dict[str, float],
    ) -> UrgencyExplanation:
        """Generate a SHAP-based explanation for urgency prediction."""
        X_raw, sf_dict = self._build_feature_vector(text, structured_features)

        # Scale using the same StandardScaler as training
        X_scaled = self.scaler.transform(X_raw)

        # Predict probabilities using XGBoost Booster (trained with multi:softprob)
        dmatrix = xgb.DMatrix(X_scaled)
        probs = np.asarray(self.model.predict(dmatrix)).reshape(-1, len(URGENCY_LEVELS))[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        predicted_label = URGENCY_LEVELS[pred_idx]

        # Use dummy SHAP values if explainer failed to initialize
        if self.tree_explainer is None:
            logger.warning("SHAP explainer not available. Providing neutral values.")
            shap_vec = np.zeros(X_scaled.shape[1])
            expected_value = 0.0
        else:
            try:
                # Use feature_perturbation="interventional" for multi-class if "tree_path_dependent" fails
                shap_values_all = self.tree_explainer.shap_values(X_scaled)
                
                # Handle different SHAP output formats (list or ndarray)
                if isinstance(shap_values_all, list):
                    # Multi-class output is often a list of arrays [n_samples, n_features]
                    shap_vec = np.array(shap_values_all[pred_idx][0])
                else:
                    raw = np.array(shap_values_all)
                    if raw.ndim == 3:
                        # (samples, features, classes) or (samples, classes, features)
                        # XGBoost usually returns (samples, features, classes) for multi:softprob in some versions
                        if raw.shape[2] == len(URGENCY_LEVELS):
                            shap_vec = raw[0, :, pred_idx]
                        else:
                            shap_vec = raw[0, pred_idx, :]
                    elif raw.ndim == 2:
                        # (samples, features) - if single class or multi:softmax used instead of softprob
                        shap_vec = raw[0]
                    else:
                        shap_vec = raw.flatten()
                
                ev = getattr(self.tree_explainer, "expected_value", 0.0)
                if isinstance(ev, (list, np.ndarray)) and len(ev) > pred_idx:
                    expected_value = float(ev[pred_idx])
                else:
                    expected_value = float(ev)
            except Exception as e:
                logger.error(f"SHAP explanation failed during calculation: {e}")
                shap_vec = np.zeros(X_scaled.shape[1])
                expected_value = 0.0

        if shap_vec.shape[0] != 776:
            # Fallback if shape is wrong
            logger.warning(f"Expected 776 SHAP values, got {shap_vec.shape[0]}. Resizing...")
            new_vec = np.zeros(776)
            limit = min(776, shap_vec.shape[0])
            new_vec[:limit] = shap_vec[:limit]
            shap_vec = new_vec

        # Aggregate the 768 embedding dimensions into one factor
        text_contrib = float(shap_vec[:768].sum())
        structured_contribs: Dict[str, float] = {}
        for i, name in enumerate(STRUCTURED_FEATURE_COLUMNS):
            structured_contribs[name] = float(shap_vec[768 + i])

        # Combine into a feature->contribution mapping
        feature_contribs: Dict[str, float] = {"text_embedding": text_contrib}
        feature_contribs.update(structured_contribs)

        # Convert to absolute-importance percentages
        abs_vals = np.array([abs(v) for v in feature_contribs.values()])
        total_abs = abs_vals.sum() or 1.0
        factor_importance: Dict[str, float] = {}
        for key, val in feature_contribs.items():
            pct = float(abs(val) / total_abs * 100.0)
            factor_importance[key] = round(pct, 2)

        nl_explanation = self._build_natural_language_explanation(
            predicted_label, confidence, factor_importance, sf_dict
        )

        return UrgencyExplanation(
            predicted_label=predicted_label,
            confidence=confidence,
            feature_contributions=feature_contribs,
            factor_importance=factor_importance,
            shap_values_vector=shap_vec,
            expected_value=expected_value,
            text=text,
            structured_features={k: float(v) for k, v in sf_dict.items()},
            nl_explanation=nl_explanation,
        )

    @staticmethod
    def _build_natural_language_explanation(
        label: str,
        confidence: float,
        factors: Dict[str, float],
        feature_values: Dict[str, float],
    ) -> str:
        """Create a simple, human-friendly explanation."""
        
        # 1. Check for Emergency Keywords (Strongest Driver)
        has_keywords = feature_values.get("emergency_keyword_score", 0) > 0
        
        # 2. Check Population Impact
        pop_val = feature_values.get("affected_population", 1.0)
        pop_desc = "an individual"
        if pop_val >= 3.0:
            pop_desc = "a large crowd or area"
        elif pop_val >= 2.0:
            pop_desc = "the entire neighborhood"
        elif pop_val >= 1.0:
            pop_desc = "a street or lane"
            
        # 3. Check Monsoon/Seasonal
        is_monsoon = feature_values.get("is_monsoon_season", 0) > 0
        
        # 4. Construct Simple Sentence
        reasons = []
        if has_keywords:
            reasons.append("it contains urgent keywords (like 'danger', 'dead', 'immediate')")
        
        if pop_val >= 2.0:
            reasons.append(f"it affects {pop_desc}")
            
        if is_monsoon:
            reasons.append("it relates to monsoon risks")
            
        if not reasons:
            # Fallback for Medium/Low if no strong signals
            if label in ["Critical", "High"]:
                reasons.append("of the specific details in the description")
            else:
                return f"Categorized as {label} urgency based on standard priority rules for this type of issue affecting {pop_desc}."

        return f"This is marked {label} priority because " + " and ".join(reasons) + "."


class ExplainabilityEngine:
    """High-level façade combining category and urgency explainers.

    This class is designed for convenient use within the
    ``ComplaintProcessor`` and Streamlit pages. It lazily loads models
    and SHAP explainers from the default project paths.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
    ) -> None:
        """Create an explainability engine.

        Args:
            project_root: Root of the project. If ``None``, this is
                inferred from the location of this file.
        """
        self.project_root = project_root or Path(__file__).resolve().parents[1]
        models_dir = self.project_root / "models"

        self._category_explainer: Optional[CategorySHAPExplainer] = None
        self._urgency_explainer: Optional[UrgencySHAPExplainer] = None

        self._category_xgb_path = models_dir / "category_xgb_model.pkl"
        self._xgb_model_path = models_dir / "xgboost_urgency_predictor.pkl"
        self._scaler_path = models_dir / "feature_scaler.pkl"

    @property
    def category_explainer(self) -> CategorySHAPExplainer:
        """Lazily-initialized category SHAP explainer."""
        if self._category_explainer is None:
            self._category_explainer = CategorySHAPExplainer(
                xgb_model_path=self._category_xgb_path,
                scaler_path=self._scaler_path,
            )
        return self._category_explainer

    @property
    def urgency_explainer(self) -> UrgencySHAPExplainer:
        """Lazily-initialized urgency SHAP explainer."""
        if self._urgency_explainer is None:
            self._urgency_explainer = UrgencySHAPExplainer(
                xgb_model_path=self._xgb_model_path,
                scaler_path=self._scaler_path,
            )
        return self._urgency_explainer

    def explain_category(self, text: str, top_k: int = 5) -> CategoryExplanation:
        """Public helper to explain the category decision for a text."""
        return self.category_explainer.explain(text, top_k=top_k)

    def explain_urgency(
        self,
        text: str,
        structured_features: Dict[str, float],
    ) -> UrgencyExplanation:
        """Public helper to explain the urgency decision for a complaint."""
        return self.urgency_explainer.explain(text, structured_features)


__all__ = [
    "CategoryExplanation",
    "UrgencyExplanation",
    "CategorySHAPExplainer",
    "UrgencySHAPExplainer",
    "ExplainabilityEngine",
]
