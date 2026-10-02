import os
from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer

from src.config import config
from src.preprocessing.clean_text import clean_tweet
from src.preprocessing.tokenize import create_tokenizer, tokenize_and_pad


def load_dataset(data_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    path = Path(data_path) if data_path else config.DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path, encoding="latin1")


def prepare_data(
    data_path: Optional[Union[str, Path]] = None,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tokenizer, LabelEncoder, pd.DataFrame]:
    df = load_dataset(data_path)
    df = df.dropna(subset=["text", "sentiment"]).reset_index(drop=True)
    df["clean_text"] = df["text"].apply(clean_tweet)

    label_encoder = LabelEncoder()
    df["target"] = label_encoder.fit_transform(df["sentiment"])

    tokenizer = create_tokenizer(df["clean_text"].values, max_features=config.MAX_FEATURES)
    X = tokenize_and_pad(df["clean_text"].values, tokenizer, max_length=config.MAX_LENGTH)
    y = pd.get_dummies(df["target"]).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=df["target"]
    )

    return X_train, X_test, y_train, y_test, tokenizer, label_encoder, df
