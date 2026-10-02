import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.preprocessing.clean_text import clean_tweet
from src.preprocessing.tokenize import MAX_LENGTH


class SentimentPredictor:
    def __init__(
        self,
        model_path: Union[str, Path] = "models/lstm/lstm_model.keras",
        tokenizer_path: Union[str, Path] = "models/lstm/tokenizer.pkl",
        label_encoder_path: Union[str, Path] = "models/lstm/label_encoder.pkl"
    ):
        for path in (model_path, tokenizer_path, label_encoder_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Artifact not found: {path}")

        self.model = load_model(model_path)

        with open(tokenizer_path, "rb") as file:
            self.tokenizer = pickle.load(file)

        with open(label_encoder_path, "rb") as file:
            self.label_encoder = pickle.load(file)

        self.classes = list(self.label_encoder.classes_)

    def _format_result(self, text: str, cleaned_text: str, probs: np.ndarray) -> Dict[str, Any]:
        pred_idx = int(np.argmax(probs))
        return {
            "text": text,
            "cleaned_text": cleaned_text,
            "sentiment": str(self.label_encoder.inverse_transform([pred_idx])[0]),
            "confidence": round(float(probs[pred_idx]), 4),
            "probabilities": {
                cls: round(float(prob), 4)
                for cls, prob in zip(self.classes, probs)
            }
        }

    def predict(self, text: str) -> Dict[str, Any]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        cleaned_texts = [clean_tweet(t) for t in texts]
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        padded = pad_sequences(sequences, maxlen=MAX_LENGTH)
        all_probs = self.model.predict(padded, verbose=0)

        return [
            self._format_result(text, cleaned, probs)
            for text, cleaned, probs in zip(texts, cleaned_texts, all_probs)
        ]
