from __future__ import annotations

"""Train the XGBoost urgency prediction model.

This script trains an XGBoost classifier on 776-dimensional feature
vectors (768-dim MuRIL embeddings + 8 structured features) produced by
``feature_extraction.py``.

Urgency levels:
  * Critical
  * High
  * Medium
  * Low

Run as a script:
    python -m src.train_urgency_model
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

RANDOM_SEED: int = 42

URGENCY_LEVELS: List[str] = ["Critical", "High", "Medium", "Low"]
ID_TO_URGENCY: Dict[int, str] = {i: u for i, u in enumerate(URGENCY_LEVELS)}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class UrgencyTrainer:
    """Trainer for XGBoost urgency prediction model."""

    def __init__(self):
        """Initialize label encoder and scaler."""
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(URGENCY_LEVELS)
        self.scaler = StandardScaler()
        self.model = None

    def load_data(self, data_dir: Path) -> tuple:
        """Load pre-extracted features and labels from urgency_features.npz."""
        logger.info("Loading features and labels...")

        npz_path = data_dir / "urgency_features.npz"
        data = np.load(npz_path)

        X_train = data["X_train"]
        X_val = data["X_val"]
        X_test = data["X_test"]

        # Labels are already integer-encoded (0-3) in the NPZ
        y_train = data["y_train"].astype(int)
        y_val = data["y_val"].astype(int)
        y_test = data["y_test"].astype(int)
        
        # Debug: Print Class Distributions
        logger.info(f"Class distribution Train: {np.bincount(y_train)}")
        logger.info(f"Class distribution Val: {np.bincount(y_val)}")
        logger.info(f"Class distribution Test: {np.bincount(y_test)}")

        logger.info(
            "Train: %s, Val: %s, Test: %s", X_train.shape, X_val.shape, X_test.shape
        )
        logger.info("Feature dimension: %d", X_train.shape[1])

        # Features are already scaled in feature_extraction; keep as-is
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def train(self, X_train, y_train, X_val, y_val) -> dict:
        """Train XGBoost model with early stopping and class weights."""
        logger.info("Training XGBoost model...")

        # Calculate class weights to handle imbalance
        class_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        logger.info(f"Computed class weights (unique): {np.unique(class_weights)}")
        logger.info("Computed class weights for training data")

        # Create DMatrix with weights
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=class_weights)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Parameters: Tuned for better generalization on small, imbalanced data
        params = {
            "objective": "multi:softprob",
            "num_class": 4,
            "max_depth": 3,  # Reduced from 4 -> 3
            "learning_rate": 0.1,  # Increased back to 0.1 to learn faster
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1, # Reduced to 1 to allow catching small classes
            "gamma": 0.1, 
            "eval_metric": ["merror", "mlogloss"], # Monitor both
            "seed": RANDOM_SEED,
            "tree_method": "hist",
        }

        # Train with early stopping
        evals = [(dtrain, "train"), (dval, "val")]
        evals_result = {}

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000, # Increased from 500
            evals=evals,
            early_stopping_rounds=50, # Increased patience
            evals_result=evals_result,
            verbose_eval=50,
        )

        logger.info("Best iteration: %d", self.model.best_iteration)
        logger.info("Best score: %f", self.model.best_score)

        return evals_result

    def evaluate(self, X, y, split_name: str = "Test") -> dict:
        """Evaluate model and return metrics."""
        logger.info(f"\nEvaluating on {split_name} set...")

        dtest = xgb.DMatrix(X)
        y_prob = self.model.predict(dtest)
        y_pred = np.asarray(y_prob).reshape(-1, len(URGENCY_LEVELS)).argmax(axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average="weighted"
        )

        # Per-class metrics
        per_class_metrics = {}
        for idx, urgency in enumerate(URGENCY_LEVELS):
            mask = y == idx
            if mask.sum() > 0:
                class_acc = accuracy_score(y[mask], y_pred[mask])
                per_class_metrics[urgency] = class_acc

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)

        # Classification report
        y_true_labels = self.label_encoder.inverse_transform(y)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        report = classification_report(
            y_true_labels, y_pred_labels, target_names=URGENCY_LEVELS
        )

        logger.info(f"{split_name} Accuracy: {accuracy:.4f}")
        logger.info(f"{split_name} Precision: {precision:.4f}")
        logger.info(f"{split_name} Recall: {recall:.4f}")
        logger.info(f"{split_name} F1: {f1:.4f}")

        logger.info(f"\nPer-class accuracy:")
        for urgency, acc in per_class_metrics.items():
            logger.info(f"  {urgency}: {acc:.4f}")

        logger.info(f"\nConfusion Matrix:\n{cm}")
        logger.info(f"\nClassification Report:\n{report}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "per_class_accuracy": per_class_metrics,
            "confusion_matrix": cm,
            "classification_report": report,
        }

    def save_model(self, output_path: Path):
        """Save model, scaler, and label encoder."""
        logger.info(f"Saving model to {output_path}...")

        # Save XGBoost model
        self.model.save_model(str(output_path.with_suffix(".json")))

        # Save scaler and label encoder
        with open(output_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "label_encoder": self.label_encoder,
                },
                f,
            )

        logger.info("Model saved successfully")


def train_and_save_urgency_model() -> None:
    """Train XGBoost urgency model and save it to disk."""
    root_dir = Path(__file__).resolve().parents[1]
    data_dir = root_dir / "data" / "processed"
    model_path = root_dir / "models" / "xgboost_urgency_predictor.pkl"

    # Create models directory
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize trainer
    trainer = UrgencyTrainer()

    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = trainer.load_data(data_dir)

    # Train model
    training_history = trainer.train(X_train, y_train, X_val, y_val)

    # Evaluate on validation set
    val_metrics = trainer.evaluate(X_val, y_val, "Validation")

    # Evaluate on test set
    logger.info("\n" + "=" * 50)
    logger.info("FINAL TEST SET EVALUATION")
    logger.info("=" * 50)
    test_metrics = trainer.evaluate(X_test, y_test, "Test")

    # Save model
    trainer.save_model(model_path)

    # Save test results
    test_results = {
        "accuracy": float(test_metrics["accuracy"]),
        "precision": float(test_metrics["precision"]),
        "recall": float(test_metrics["recall"]),
        "f1": float(test_metrics["f1"]),
        "per_class_accuracy": {
            k: float(v) for k, v in test_metrics["per_class_accuracy"].items()
        },
        "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
    }
    results_path = model_path.parent / "urgency_test_results.json"
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)

    logger.info(f"\nTest results saved to {results_path}")
    logger.info("Training complete!")

def main() -> None:
    """CLI entry-point for urgency model training."""
    try:
        train_and_save_urgency_model()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to train urgency model: %s", exc)
        raise SystemExit(1) from exc

if __name__ == "__main__":  # pragma: no cover - CLI path
    main()