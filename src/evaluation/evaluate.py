import os
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
from tensorflow.keras.models import load_model

from src.config import config
from src.preprocessing.prepare_data import prepare_data

LABELS = ["negative", "neutral", "positive"]
DISPLAY_LABELS = ["Negative", "Neutral", "Positive"]


def evaluate(
    model_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    metrics_dir: Optional[Path] = None
) -> Dict[str, Any]:
    model_file = Path(model_path) if model_path else config.MODEL_PATH
    out_dir = Path(output_dir) if output_dir else config.FIGURES_DIR
    met_dir = Path(metrics_dir) if metrics_dir else config.METRICS_DIR

    out_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at: {model_file}")

    print(f"Evaluating BiLSTM model from: {model_file}")
    model = load_model(model_file)

    _, X_test, _, y_test, _, _, _ = prepare_data()

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=DISPLAY_LABELS,
        yticklabels=DISPLAY_LABELS
    )
    plt.title("BiLSTM Confusion Matrix", fontsize=13, pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()

    cm_path = out_dir / "bilstm_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    report_str = classification_report(
        y_true,
        y_pred,
        target_names=DISPLAY_LABELS,
        digits=4
    )
    print(f"\nClassification Report:\n\n{report_str}")

    report_path = met_dir / "bilstm_report.txt"
    with open(report_path, "w") as f:
        f.write(report_str)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

    return {
        "model": "BiLSTM",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


if __name__ == "__main__":
    evaluate()
