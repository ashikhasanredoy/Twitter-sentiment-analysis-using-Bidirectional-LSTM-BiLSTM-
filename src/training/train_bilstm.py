import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    SpatialDropout1D,
    Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping

from src.preprocessing.prepare_data import prepare_data
from src.preprocessing.tokenize import MAX_FEATURES, MAX_LENGTH


EMBEDDING_DIM = 128
LSTM_UNITS = 128


def build_bilstm_model(input_length=MAX_LENGTH):
    model = Sequential([
        Embedding(
            input_dim=MAX_FEATURES,
            output_dim=EMBEDDING_DIM
        ),
        SpatialDropout1D(0.4),
        Bidirectional(
            LSTM(
                LSTM_UNITS,
                dropout=0.2,
                recurrent_dropout=0.2
            )
        ),
        Dense(
            3,
            activation="softmax"
        )
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    return model


def train_bilstm(epochs=10, batch_size=32):
    (
        X_train,
        X_test,
        y_train,
        y_test,
        tokenizer,
        label_encoder,
        df
    ) = prepare_data()

    model = build_bilstm_model(X_train.shape[1])
    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1
    )

    print("Starting BiLSTM training...")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    os.makedirs("models/bilstm", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    model_save_path = "models/bilstm/bilstm_model.keras"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    with open("models/bilstm/tokenizer.pkl", "wb") as file:
        pickle.dump(tokenizer, file)
    print("Tokenizer saved to models/bilstm/tokenizer.pkl")

    with open("models/bilstm/label_encoder.pkl", "wb") as file:
        pickle.dump(label_encoder, file)
    print("Label encoder saved to models/bilstm/label_encoder.pkl")

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("BiLSTM Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/bilstm_accuracy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("BiLSTM Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/bilstm_loss.png", dpi=300)
    plt.close()

    return model, history


if __name__ == "__main__":
    train_bilstm()
