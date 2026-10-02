import pickle
from pathlib import Path
from typing import List, Union
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.config import config


def create_tokenizer(texts: Union[List[str], np.ndarray], max_features: int = config.MAX_FEATURES) -> Tokenizer:
    tokenizer = Tokenizer(num_words=max_features, split=" ")
    tokenizer.fit_on_texts(texts)
    return tokenizer


def tokenize_and_pad(
    texts: Union[List[str], np.ndarray],
    tokenizer: Tokenizer,
    max_length: int = config.MAX_LENGTH
) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_length)


def save_tokenizer(tokenizer: Tokenizer, path: Union[str, Path]) -> None:
    with open(path, "wb") as file:
        pickle.dump(tokenizer, file)


def load_tokenizer(path: Union[str, Path]) -> Tokenizer:
    with open(path, "rb") as file:
        return pickle.load(file)
