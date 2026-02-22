from __future__ import annotations

"""Quick manual test for the explainability engine.

Run:
    python -m src.test_explainability

This will:
  * Load the trained category and urgency models.
  * Run them on the example Hinglish complaint from the spec.
  * Print category + urgency predictions and natural-language
    SHAP-based explanations to the console.
"""

from pathlib import Path

from src.data_preparation import compute_severity_score, extract_emergency_keywords
from src.explainability import ExplainabilityEngine


def main() -> None:
    """Run a simple explainability test on a fixed complaint text."""
    project_root = Path(__file__).resolve().parents[1]

    text = (
        "Hamare area ki road bahut kharab ho gayi hai, "
        "potholes ki wajah se accident ka risk hai"
    )

    # Derive structured features similar to those used in training.
    emergency_keywords = extract_emergency_keywords(text)
    emergency_keyword_score = 1.0 if emergency_keywords else 0.0

    # Use a neutral base urgency ("Medium") just to compute a
    # comparable severity score in [0, 1].
    severity_score = compute_severity_score("Medium", emergency_keywords, text)

    structured_features = {
        "emergency_keyword_score": emergency_keyword_score,
        "severity_score": float(severity_score),
        "text_length": float(len(text.split())),
        "affected_population": 2.0,  # neighborhood-level impact
        "repeat_complaint_count": 1.0,
        "hour_of_day": 14.0,
        "is_weekend": 0.0,
        "is_monsoon_season": 1.0,
    }

    engine = ExplainabilityEngine(project_root=project_root)

    print("==== CATEGORY EXPLANATION ====")
    cat_exp = engine.explain_category(text, top_k=5)
    print(f"Predicted category: {cat_exp.predicted_label} (confidence={cat_exp.confidence:.3f})")
    print(f"Top keywords: {cat_exp.top_keywords}")
    print("NL explanation:")
    print(cat_exp.nl_explanation)

    print("\n==== URGENCY EXPLANATION ====")
    urg_exp = engine.explain_urgency(text, structured_features)
    print(f"Predicted urgency: {urg_exp.predicted_label} (confidence={urg_exp.confidence:.3f})")
    print("Factor importance (%):")
    for name, pct in sorted(urg_exp.factor_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {pct:.2f}%")
    print("NL explanation:")
    print(urg_exp.nl_explanation)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
