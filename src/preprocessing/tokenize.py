import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


MAX_FEATURES = 5000
MAX_LENGTH = 50


def create_tokenizer(texts, max_features=MAX_FEATURES):
    tokenizer = Tokenizer(
        num_words=max_features,
        split=" "
    )
    tokenizer.fit_on_texts(texts)
    return tokenizer


def tokenize_and_pad(texts, tokenizer, max_length=MAX_LENGTH):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(
        sequences,
        maxlen=max_length
    )
    return padded_sequences


def save_tokenizer(tokenizer, path):
    with open(path, "wb") as file:
        pickle.dump(tokenizer, file)


def load_tokenizer(path):
    with open(path, "rb") as file:
        tokenizer = pickle.load(file)
    return tokenizer
