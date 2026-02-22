from __future__ import annotations
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
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

RANDOM_SEED: int = 42
CATEGORIES: List[str] = ["Sanitation", "Water Supply", "Transportation"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)

class CategoryXGBTrainer:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(CATEGORIES)
        self.model = None

    def load_data(self, data_dir: Path):
        npz_path = data_dir / "urgency_features.npz"
        data = np.load(npz_path)
        
        # We use the same embeddings but y_cat instead of y
        return (data["X_train"], data["y_cat_train"]), (data["X_val"], data["y_cat_val"]), (data["X_test"], data["y_cat_test"])

    def train(self, X_train, y_train, X_val, y_val):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": 6,
            "learning_rate": 0.1,
            "eval_metric": "mlogloss",
            "seed": RANDOM_SEED,
            "tree_method": "hist",
        }

        evals = [(dtrain, "train"), (dval, "val")]
        self.model = xgb.train(params, dtrain, num_boost_round=500, evals=evals, early_stopping_rounds=20, verbose_eval=50)

    def evaluate(self, X, y, split_name="Test"):
        dtest = xgb.DMatrix(X)
        y_prob = self.model.predict(dtest)
        y_pred = y_prob.argmax(axis=1)

        acc = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, target_names=CATEGORIES)
        logger.info(f"{split_name} Accuracy: {acc:.4f}")
        logger.info(f"\nReport:\n{report}")
        return acc

    def save_model(self, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump({"model": self.model, "label_encoder": self.label_encoder}, f)

def main():
    root_dir = Path(__file__).resolve().parents[1]
    data_dir = root_dir / "data" / "processed"
    model_path = root_dir / "models" / "category_xgb_model.pkl"

    trainer = CategoryXGBTrainer()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = trainer.load_data(data_dir)
    trainer.train(X_train, y_train, X_val, y_val)
    trainer.evaluate(X_test, y_test)
    trainer.save_model(model_path)
    logger.info(f"Category model saved to {model_path}")

if __name__ == "__main__":
    main()
