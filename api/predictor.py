import os
from typing import Dict
from src.inference.predictor import SentimentPredictor


class ModelService:
    MODEL_CONFIGS = {
        "lstm": ("lstm", "models/lstm/lstm_model.keras"),
        "bilstm": ("bilstm", "models/bilstm/bilstm_model.keras"),
        "bi-lstm": ("bilstm", "models/bilstm/bilstm_model.keras"),
    }

    def __init__(self):
        self._cache: Dict[str, SentimentPredictor] = {}

    def get_predictor(self, model_type: str = "lstm") -> SentimentPredictor:
        normalized_type = model_type.lower()
        if normalized_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'lstm' or 'bilstm'.")

        name, model_path = self.MODEL_CONFIGS[normalized_type]
        if name not in self._cache:
            tok_path = f"models/{name}/tokenizer.pkl"
            le_path = f"models/{name}/label_encoder.pkl"
            if not all(os.path.exists(p) for p in (model_path, tok_path, le_path)):
                raise FileNotFoundError(
                    f"{name.upper()} model artifacts not found. Please run: python -m src.training.train_{name}"
                )
            self._cache[name] = SentimentPredictor(model_path, tok_path, le_path)

        return self._cache[name]

    def get_status(self) -> Dict[str, bool]:
        return {
            "lstm_ready": os.path.exists("models/lstm/lstm_model.keras"),
            "bilstm_ready": os.path.exists("models/bilstm/bilstm_model.keras"),
        }


model_service = ModelService()
