import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)

from src.preprocessing.prepare_data import prepare_data

LABELS = ["negative", "neutral", "positive"]
DISPLAY_LABELS = ["Negative", "Neutral", "Positive"]


def evaluate_model(
    model_path,
    output_dir="results/figures",
    metrics_dir="results/metrics",
    model_name="LSTM",
    data_tuple=None
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    if data_tuple is None:
        (
            X_train,
            X_test,
            y_train,
            y_test,
            tokenizer,
            label_encoder,
            df
        ) = prepare_data()
    else:
        X_train, X_test, y_train, y_test, tokenizer, label_encoder = data_tuple

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    print(f"\nEvaluating {model_name} from: {model_path}")
    model = load_model(model_path)

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    if len(y_test.shape) > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test

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
    plt.title(f"Confusion Matrix: {model_name}", fontsize=14, pad=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, f"{model_name.lower()}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    report_str = classification_report(
        y_true,
        y_pred,
        target_names=DISPLAY_LABELS,
        digits=4
    )
    print(f"\n{model_name} Classification Report:\n")
    print(report_str)

    report_path = os.path.join(metrics_dir, f"{model_name.lower()}_report.txt")
    with open(report_path, "w") as file:
        file.write(report_str)
    print(f"Report saved to {report_path}")

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

    return {
        "model": model_name,
        "accuracy": acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1
    }


def compare_models(models_to_eval=None, metrics_dir="results/metrics"):
    if models_to_eval is None:
        models_to_eval = [
            ("models/lstm/lstm_model.keras", "LSTM"),
            ("models/bilstm/bilstm_model.keras", "BiLSTM")
        ]

    (
        X_train,
        X_test,
        y_train,
        y_test,
        tokenizer,
        label_encoder,
        _
    ) = prepare_data()
    shared_data = (X_train, X_test, y_train, y_test, tokenizer, label_encoder)

    results = []
    for model_path, model_name in models_to_eval:
        if os.path.exists(model_path):
            res = evaluate_model(
                model_path=model_path,
                model_name=model_name,
                data_tuple=shared_data
            )
            results.append(res)
        else:
            print(f"Skipping {model_name}: {model_path} not found.")

    if results:
        comp_df = pd.DataFrame(results)
        comp_path = os.path.join(metrics_dir, "comparison.csv")
        comp_df.to_csv(comp_path, index=False)
        print(f"\nModel Comparison saved to {comp_path}:")
        print(comp_df.to_string(index=False))


if __name__ == "__main__":
    compare_models()
