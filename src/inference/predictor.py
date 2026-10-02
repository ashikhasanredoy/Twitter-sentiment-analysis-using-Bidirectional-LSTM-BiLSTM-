import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import config
from src.preprocessing.clean_text import clean_tweet
from src.preprocessing.tokenize import tokenize_and_pad


class SentimentPredictor:
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        tokenizer_path: Optional[Union[str, Path]] = None,
        label_encoder_path: Optional[Union[str, Path]] = None
    ):
        self.model_path = Path(model_path) if model_path else config.MODEL_PATH
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else config.TOKENIZER_PATH
        self.label_encoder_path = Path(label_encoder_path) if label_encoder_path else config.LABEL_ENCODER_PATH

        for path in (self.model_path, self.tokenizer_path, self.label_encoder_path):
            if not path.exists():
                raise FileNotFoundError(f"Model artifact not found: {path}. Run: python -m src.training.train")

        self.model = load_model(self.model_path)
        with open(self.tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)
        with open(self.label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

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
        padded = tokenize_and_pad(cleaned_texts, self.tokenizer, max_length=config.MAX_LENGTH)
        all_probs = self.model.predict(padded, verbose=0)

        return [
            self._format_result(text, cleaned, probs)
            for text, cleaned, probs in zip(texts, cleaned_texts, all_probs)
        ]
