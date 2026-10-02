import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.preprocessing.clean_text import clean_tweet
from src.preprocessing.tokenize import (
    create_tokenizer,
    tokenize_and_pad,
    MAX_FEATURES,
    MAX_LENGTH
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data/raw/training.csv")


def load_dataset(data_path=None):
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    if not os.path.exists(data_path):
        data_path = "data/raw/training.csv"

    df = pd.read_csv(data_path, encoding="latin1")
    return df


def prepare_data(data_path=None, test_size=0.2, val_size=None, random_state=42):
    df = load_dataset(data_path)
    df = df.dropna(subset=["text", "sentiment"]).reset_index(drop=True)

    df["clean_text"] = df["text"].apply(clean_tweet)

    label_encoder = LabelEncoder()
    df["target"] = label_encoder.fit_transform(df["sentiment"])

    tokenizer = create_tokenizer(df["clean_text"].values, max_features=MAX_FEATURES)
    X = tokenize_and_pad(df["clean_text"].values, tokenizer, max_length=MAX_LENGTH)

    y = pd.get_dummies(df["target"]).values

    if val_size is not None and val_size > 0:
        combined_test_size = test_size + val_size
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=combined_test_size,
            random_state=random_state,
            stratify=df["target"]
        )

        relative_test_size = test_size / combined_test_size
        temp_targets = np.argmax(y_temp, axis=1)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_test_size,
            random_state=random_state,
            stratify=temp_targets
        )

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            tokenizer,
            label_encoder,
            df
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=df["target"]
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        tokenizer,
        label_encoder,
        df
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, tokenizer, label_encoder, df = prepare_data()
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
