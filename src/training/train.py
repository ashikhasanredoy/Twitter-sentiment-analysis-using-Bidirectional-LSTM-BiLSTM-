import os
import argparse
import pickle
import logging
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    SpatialDropout1D,
    Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from src.config import config
from src.preprocessing.prepare_data import prepare_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_bilstm_model(
    vocab_size: int = config.MAX_FEATURES,
    embedding_dim: int = config.EMBEDDING_DIM,
    lstm_units: int = config.LSTM_UNITS,
    spatial_dropout: float = config.SPATIAL_DROPOUT,
    recurrent_dropout: float = config.RECURRENT_DROPOUT,
    dropout: float = config.DROPOUT,
    num_classes: int = 3
) -> Sequential:
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim
        ),
        SpatialDropout1D(spatial_dropout),
        Bidirectional(
            LSTM(
                lstm_units,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout
            )
        ),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    return model


def train(epochs: int = config.EPOCHS, batch_size: int = config.BATCH_SIZE) -> Tuple[Sequential, dict]:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing data...")
    X_train, X_test, y_train, y_test, tokenizer, label_encoder, _ = prepare_data()

    logger.info("Building BiLSTM model...")
    model = build_bilstm_model()
    model.summary(print_fn=logger.info)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(config.MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=1,
            min_lr=1e-5,
            verbose=1
        )
    ]

    logger.info(f"Starting BiLSTM training (Epochs: {epochs}, Batch size: {batch_size})...")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    with open(config.TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    logger.info(f"Tokenizer saved to {config.TOKENIZER_PATH}")

    with open(config.LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)
    logger.info(f"Label encoder saved to {config.LABEL_ENCODER_PATH}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history["accuracy"], label="Train")
    ax1.plot(history.history["val_accuracy"], label="Validation")
    ax1.set_title("BiLSTM Model Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Train")
    ax2.plot(history.history["val_loss"], label="Validation")
    ax2.set_title("BiLSTM Model Loss")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    chart_path = config.FIGURES_DIR / "bilstm_training_history.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    logger.info(f"Training curves saved to {chart_path}")

    return model, history.history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BiLSTM Sentiment Analysis Model")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size)
