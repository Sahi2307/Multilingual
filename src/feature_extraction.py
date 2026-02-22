from __future__ import annotations

"""Feature extraction module using MuRIL embeddings and structured features.
Generates 776-dimensional feature vectors (768 + 8).

This module:
  * Loads the processed complaints dataset (``data/civic_complaints.csv``).
  * Computes 768-dimensional MuRIL embeddings (mean-pooled) for each text.
  * Concatenates 8 structured features to form a 776-dimensional vector.
  * Fits a StandardScaler on training features and applies it to all splits.
  * Saves:
      - ``data/processed/urgency_features.npz`` with X/y and IDs per split.
      - ``models/feature_scaler.pkl`` for later use during inference.

Run as a script:
    python -m src.feature_extraction
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

RANDOM_SEED: int = 42
MURIL_MODEL_NAME: str = "google/muril-base-cased"
MAX_LENGTH: int = 128
BATCH_SIZE: int = 8

URGENCY_LEVELS: List[str] = ["Critical", "High", "Medium", "Low"]
URGENCY_TO_ID: Dict[str, int] = {u: i for i, u in enumerate(URGENCY_LEVELS)}

CATEGORIES: List[str] = ["Sanitation", "Water Supply", "Transportation"]
CATEGORY_TO_ID: Dict[str, int] = {c: i for i, c in enumerate(CATEGORIES)}

STRUCTURED_FEATURE_COLUMNS: List[str] = [
    "emergency_keyword_score",
    "text_length",
    "affected_population",
    "scope_id",
    "repeat_complaint_count",
    "hour_of_day",
    "is_weekend",
    "is_monsoon_season",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


@dataclass
class MurilFeatureExtractor:
    """Wrapper around MuRIL to produce sentence embeddings.

    Attributes:
        model_name: Hugging Face model name to load.
        device: Torch device string, "cuda" or "cpu". If ``None``,
            the best available device is chosen automatically.
    """

    model_name: str = MURIL_MODEL_NAME
    device: Optional[str] = None

    def __post_init__(self) -> None:
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading MuRIL model '%s' on device '%s'...", self.model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = BATCH_SIZE, max_length: int = MAX_LENGTH) -> np.ndarray:
        """Compute mean-pooled MuRIL embeddings for a list of texts.

        Args:
            texts: List of complaint texts.
            batch_size: Batch size for inference.
            max_length: Maximum token length.

        Returns:
            Array of shape (len(texts), 768) with float32 embeddings.
        """
        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = self.model(**encoded)
            last_hidden = outputs.last_hidden_state  # (batch, seq, hidden)

            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            masked_hidden = last_hidden * attention_mask
            sum_embeddings = masked_hidden.sum(dim=1)
            lengths = attention_mask.sum(dim=1).clamp(min=1)
            mean_embeddings = sum_embeddings / lengths

            # Use list conversion to bypass PyTorch/Numpy 2.x version check error
            all_embeddings.append(np.array(mean_embeddings.cpu().detach().tolist(), dtype=np.float32))

        embeddings = np.vstack(all_embeddings).astype(np.float32)
        logger.info("Computed embeddings with shape %s", embeddings.shape)
        return embeddings

    def extract_muril_embedding(self, text: str) -> np.ndarray:
        """
        Extract 768-dimensional MuRIL embedding for text.

        Args:
            text (str): Input text

        Returns:
            np.ndarray: 768-dimensional embedding
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
        ).to(self.device)

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            # Use list conversion to bypass PyTorch/Numpy 2.x version check error
            embedding = np.array(outputs.last_hidden_state[:, 0, :].cpu().detach().tolist(), dtype=np.float32)

        return embedding.squeeze()


def encode_urgency_labels(labels: List[str]) -> np.ndarray:
    """Encode urgency labels as integer IDs.

    Args:
        labels: List of urgency strings.

    Returns:
        NumPy array of shape (n,) with integer-encoded labels.
    """
    return np.array([URGENCY_TO_ID[l] for l in labels], dtype=np.int64)


def build_structured_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build structured feature matrix from the DataFrame.

    Args:
        df: DataFrame containing all structured feature columns.

    Returns:
        NumPy array of shape (n, 8) with float32 features.
    """
    missing = [c for c in STRUCTURED_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing structured feature columns: {missing}")
    mat = df[STRUCTURED_FEATURE_COLUMNS].astype(np.float32).values
    return mat


def build_features_for_split(
    df: pd.DataFrame,
    extractor: MurilFeatureExtractor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 776-dim features and labels for a given split DataFrame.

    Args:
        df: Subset of the complaints DataFrame for a single split.
        extractor: Initialized MuRIL feature extractor.

    Returns:
        Tuple of (X, y, ids):
          * X: ndarray of shape (n, 776)
          * y: ndarray of shape (n,) with urgency label IDs
          * ids: ndarray of complaint IDs (dtype=str)
    """
    texts = df["text"].astype(str).tolist()
    logger.info("Encoding %d texts with MuRIL...", len(texts))
    text_embeddings = extractor.encode(texts)

    structured = build_structured_feature_matrix(df)
    if text_embeddings.shape[0] != structured.shape[0]:
        raise RuntimeError("Mismatch between text embeddings and structured features.")

    features = np.concatenate([text_embeddings, structured], axis=1)
    if features.shape[1] != 776:
        raise RuntimeError(f"Expected 776-dim features, got {features.shape[1]}-dim.")

    y = encode_urgency_labels(df["urgency"].tolist())
    ids = df["complaint_id"].astype(str).to_numpy()
    return features, y, ids


def extract_and_save_features(data_dir: Path, output_dir: Path) -> None:
    """
    Extract features from train, val, test sets and save.

    Args:
        data_dir (Path): Directory containing CSV files
        output_dir (Path): Directory to save feature arrays
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = MurilFeatureExtractor()

    full_df = pd.read_csv(data_dir / "civic_complaints.csv")

    for split in ["train", "val", "test"]:
        logger.info("\nProcessing %s set...", split)

        # Load data
        df = full_df[full_df["split"] == split].reset_index(drop=True)

        # Extract features
        features = extractor.encode(df["cleaned_text"].astype(str).tolist())

        # Extract labels
        category_labels = df["category"].values
        urgency_labels = df["urgency"].values

        # Save
        np.save(output_dir / f"{split}_features.npy", features)
        np.save(output_dir / f"{split}_category_labels.npy", category_labels)
        np.save(output_dir / f"{split}_urgency_labels.npy", urgency_labels)

        logger.info("Saved %s features: %s", split, features.shape)


def create_and_save_urgency_features() -> None:
    """Generate and persist features for the urgency prediction model.

    Side effects:
        * Reads ``data/civic_complaints.csv``.
        * Writes ``data/processed/urgency_features.npz``.
        * Writes ``models/feature_scaler.pkl``.
    """
    root_dir = Path(__file__).resolve().parents[1]
    data_path = root_dir / "data" / "civic_complaints.csv"
    processed_dir = root_dir / "data" / "processed"
    models_dir = root_dir / "models"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {data_path}. Run data_preparation.py first."
        )

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    logger.info("Loaded dataset with %d rows from %s", len(df), data_path)

    extractor = MurilFeatureExtractor()

    split_dfs = {
        split: df[df["split"] == split].reset_index(drop=True)
        for split in ("train", "val", "test")
    }

    features: Dict[str, np.ndarray] = {}
    urgency_labels: Dict[str, np.ndarray] = {}
    category_labels: Dict[str, np.ndarray] = {}
    ids: Dict[str, np.ndarray] = {}

    for split_name, split_df in split_dfs.items():
        logger.info("Building features for split '%s' with %d examples...", split_name, len(split_df))
        X, _, id_arr = build_features_for_split(split_df, extractor)
        features[split_name] = X
        ids[split_name] = id_arr
        
        # Standardize category labels to IDs
        cat_ids = [CATEGORY_TO_ID[c] for c in split_df["category"]]
        category_labels[split_name] = np.array(cat_ids)
        
        # Use urgency labels
        urg_ids = encode_urgency_labels(split_df["urgency"].tolist())
        urgency_labels[split_name] = urg_ids

    # Fit scaler on training features and apply to all splits
    scaler = StandardScaler()
    scaler.fit(features["train"])
    logger.info("Fitted StandardScaler on training features with shape %s", features["train"].shape)

    X_train_scaled = scaler.transform(features["train"])
    X_val_scaled = scaler.transform(features["val"])
    X_test_scaled = scaler.transform(features["test"])

    out_path = processed_dir / "urgency_features.npz"
    np.savez_compressed(
        out_path,
        X_train=X_train_scaled,
        y_train=urgency_labels["train"],
        y_cat_train=category_labels["train"],
        id_train=ids["train"],
        X_val=X_val_scaled,
        y_val=urgency_labels["val"],
        y_cat_val=category_labels["val"],
        id_val=ids["val"],
        X_test=X_test_scaled,
        y_test=urgency_labels["test"],
        y_cat_test=category_labels["test"],
        id_test=ids["test"],
    )
    logger.info("Saved scaled urgency features to %s", out_path)

    scaler_path = models_dir / "feature_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info("Saved StandardScaler to %s", scaler_path)


def main() -> None:
    """CLI entry-point for feature extraction."""
    try:
        create_and_save_urgency_features()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to create features: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover - CLI path
    main()
