from __future__ import annotations

"""Train MuRIL model for complaint category classification.

Fine-tunes ``google/muril-base-cased`` for 3-class categorization.
"""

import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, Trainer, TrainingArguments,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW
from tqdm import tqdm

RANDOM_SEED: int = 42
MURIL_MODEL_NAME: str = "google/muril-base-cased"
MAX_LENGTH: int = 64
BATCH_SIZE: int = 4  # modest batch size to balance stability/speed
NUM_EPOCHS: int = 3  # Reduced for stability
LEARNING_RATE: float = 2e-5
WEIGHT_DECAY: float = 0.01

CATEGORIES: List[str] = ["Sanitation", "Water Supply", "Transportation"]
CATEGORY_TO_ID: Dict[str, int] = {c: i for i, c in enumerate(CATEGORIES)}
ID_TO_CATEGORY: Dict[int, str] = {i: c for c, i in CATEGORY_TO_ID.items()}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ComplaintDataset(Dataset):
    """PyTorch dataset for complaint category classification.

    Attributes:
        texts: List of complaint texts.
        labels: List of integer-encoded labels.
        tokenizer: Hugging Face tokenizer.
    """

    texts: List[str]
    labels: List[int]
    tokenizer: AutoTokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = int(self.labels[idx])
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def encode_labels(labels: List[str]) -> List[int]:
    """Map string category labels to integer IDs.

    Args:
        labels: List of category names.

    Returns:
        List of integer labels.
    """
    return [CATEGORY_TO_ID[l] for l in labels]


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Compute evaluation metrics for the Trainer.

    Args:
        eval_pred: Tuple of (logits, labels).

    Returns:
        Dictionary containing accuracy, precision, recall, and weighted F1.
    """
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
    }


class CategoryTrainer:
    """Trainer for MuRIL category classification model."""

    def __init__(self, model_name: str = MURIL_MODEL_NAME, num_labels: int = 3):
        """Initialize model and tokenizer."""
        logger.info(f"Initializing model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )

        self.best_val_accuracy = 0.0
        self.class_weights: torch.Tensor | None = None

    def prepare_data(self, data_dir: Path) -> tuple:
        """Load and prepare train, val, test datasets."""
        logger.info("Loading datasets...")

        full_df = pd.read_csv(data_dir / "civic_complaints.csv")
        train_df = full_df[full_df["split"] == "train"].reset_index(drop=True)
        val_df = full_df[full_df["split"] == "val"].reset_index(drop=True)
        test_df = full_df[full_df["split"] == "test"].reset_index(drop=True)

        # Convert categories to IDs
        train_labels = [CATEGORY_TO_ID[cat] for cat in train_df["category"]]
        val_labels = [CATEGORY_TO_ID[cat] for cat in val_df["category"]]
        test_labels = [CATEGORY_TO_ID[cat] for cat in test_df["category"]]

        # Compute class weights to counter imbalance
        counts = np.bincount(train_labels, minlength=len(CATEGORIES)).astype(float)
        inv_freq = 1.0 / np.clip(counts, a_min=1.0, a_max=None)
        weights = inv_freq / inv_freq.sum() * len(CATEGORIES)
        self.class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
        logger.info("Class weights: %s", weights)

        # Create datasets
        train_dataset = ComplaintDataset(
            texts=train_df["cleaned_text"].tolist(),
            labels=train_labels,
            tokenizer=self.tokenizer,
        )

        val_dataset = ComplaintDataset(
            texts=val_df["cleaned_text"].tolist(),
            labels=val_labels,
            tokenizer=self.tokenizer,
        )

        test_dataset = ComplaintDataset(
            texts=test_df["cleaned_text"].tolist(),
            labels=test_labels,
            tokenizer=self.tokenizer,
        )

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")

        return train_dataset, val_dataset, test_dataset

    def train_epoch(self, dataloader: DataLoader, optimizer, scheduler) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)

        progress_bar = tqdm(dataloader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
            )

            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({"loss": loss.item()})

        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> Dict:
        """Evaluate model on validation/test set."""
        self.model.eval()
        predictions = []
        true_labels = []
        total_loss = 0
        criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,
                )

                loss = criterion(outputs.logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(np.array(preds.cpu().detach().tolist()))
                true_labels.extend(np.array(labels.cpu().detach().tolist()))

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average="weighted"
        )

        # Per-class metrics
        per_class_metrics = {}
        for idx, category in ID_TO_CATEGORY.items():
            mask = np.array(true_labels) == idx
            if mask.sum() > 0:
                class_acc = accuracy_score(
                    np.array(true_labels)[mask],
                    np.array(predictions)[mask],
                )
                per_class_metrics[category] = class_acc

        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)

        return {
            "loss": total_loss / len(dataloader),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "per_class_accuracy": per_class_metrics,
            "confusion_matrix": cm,
            "true_labels": true_labels,
            "predictions": predictions,
        }

    def train(
        self, train_dataset, val_dataset, output_dir: Path, num_epochs: int = NUM_EPOCHS
    ):
        """Full training loop with early stopping."""
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        logger.info(f"Starting training for {num_epochs} epochs...")

        training_history = []
        patience = 4
        patience_counter = 0

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            logger.info(f"Train loss: {train_loss:.4f}")

            # Validate
            val_metrics = self.evaluate(val_loader)
            logger.info(f"Val loss: {val_metrics['loss']:.4f}")
            logger.info(f"Val accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Val F1: {val_metrics['f1']:.4f}")

            # Per-class accuracy
            for cat, acc in val_metrics["per_class_accuracy"].items():
                logger.info(f"  {cat}: {acc:.4f}")

            # Save history
            training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1": val_metrics["f1"],
                }
            )

            # Save history immediately (safety checkpoint)
            with open(output_dir / "training_history.json", "w") as f:
                json.dump(training_history, f, indent=2)

            # Early stopping
            if val_metrics["accuracy"] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics["accuracy"]
                patience_counter = 0

                # Save best model
                logger.info(
                    f"New best accuracy: {self.best_val_accuracy:.4f}. Saving model..."
                )
                self.save_model(output_dir)
            else:
                patience_counter += 1
                logger.info(
                    f"No improvement. Patience: {patience_counter}/{patience}"
                )

                if patience_counter >= patience:
                    logger.info("Early stopping triggered!")
                    break

        # Save training history
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(training_history, f, indent=2)

        return training_history

    def save_model(self, output_dir: Path | str):
        """Save model and tokenizer."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")


def train_and_save_category_model() -> None:
    """Fine-tune MuRIL for category classification and save the model."""
    root_dir = Path(__file__).resolve().parents[1]
    data_path = root_dir / "data" / "civic_complaints.csv"
    data_dir = root_dir / "data"
    models_dir = root_dir / "models" / "muril_category_classifier"
    
    if (data_dir / "civic_complaints.csv").exists():
        full_df = pd.read_csv(data_dir / "civic_complaints.csv")
        test_df = full_df[full_df["split"] == "test"].reset_index(drop=True)
    else:
        test_df = None

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run data_preparation.py first."
        )

    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    logger.info("Loaded dataset with %d rows from %s", len(df), data_path)

    trainer = CategoryTrainer()

    # Prepare data
    train_dataset, val_dataset, test_dataset = trainer.prepare_data(data_dir)

    # Train model
    training_history = trainer.train(train_dataset, val_dataset, models_dir)

    logger.info("Evaluating on validation set...")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    val_metrics = trainer.evaluate(val_loader)
    logger.info("Validation metrics: %s", val_metrics)

    logger.info("Evaluating on test set...")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    test_metrics = trainer.evaluate(test_loader)

    test_labels = np.array(test_metrics["true_labels"])
    test_preds = np.array(test_metrics["predictions"])

    acc = test_metrics["accuracy"]
    precision = test_metrics["precision"]
    recall = test_metrics["recall"]
    f1 = test_metrics["f1"]
    cm = test_metrics["confusion_matrix"]

    logger.info(
        "Test accuracy: %.4f, precision: %.4f, recall: %.4f, f1_weighted: %.4f",
        acc,
        precision,
        recall,
        f1,
    )

    logger.info("Test confusion matrix (rows=true, cols=pred):\n%s", cm)

    report = classification_report(
        test_labels,
        test_preds,
        target_names=CATEGORIES,
        digits=4,
        zero_division=0,
    )
    logger.info("Per-class classification report (test):\n%s", report)

    # Language-specific performance on test set
    if test_df is None:
        logger.warning("Test split CSV not found; skipping language-specific metrics.")
    else:
        test_langs = test_df["language"].tolist()

        for lang in sorted(set(test_langs)):
            mask = np.array([l == lang for l in test_langs])
            if mask.sum() == 0:
                continue
            lang_true = test_labels[mask]
            lang_pred = test_preds[mask]
            lang_acc = accuracy_score(lang_true, lang_pred)
            _, _, lang_f1, _ = precision_recall_fscore_support(
                lang_true, lang_pred, average="weighted", zero_division=0
            )
            logger.info(
                "Test language '%s': n=%d, accuracy=%.4f, f1_weighted=%.4f",
                lang,
                mask.sum(),
                lang_acc,
                lang_f1,
            )

    # Save the best model
    trainer.save_model(str(models_dir))
    logger.info("Saved fine-tuned MuRIL model to %s", models_dir)

    # SANITY CHECK: Predict on a few training examples
    logger.info("Running post-training sanity check...")
    trainer.model.eval()
    sample_texts = [
        "There is garbage all over the street.",
        "Water pipe is leaking heavily.",
        "Bus tracking is not accurate.",
    ]
    sample_labels = ["Sanitation", "Water Supply", "Transportation"]
    
    with torch.no_grad():
        for text, label in zip(sample_texts, sample_labels):
            inputs = trainer.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
            logits = trainer.model(**inputs).logits
            probs = np.array(torch.softmax(logits, dim=-1).cpu().detach().tolist())[0]
            pred_idx = np.argmax(probs)
            # CATEGORIES list order matches the ID encoding [0,1,2] from CATEGORY_TO_ID
            pred_label = CATEGORIES[pred_idx] 
            conf = probs[pred_idx]
            logger.info(f"Input: '{text}' | True: {label} | Pred: {pred_label} (Conf: {conf:.4f})")



def main() -> None:
    """CLI entry-point for category model training."""
    try:
        train_and_save_category_model()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to train category model: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover - CLI path
    main()
