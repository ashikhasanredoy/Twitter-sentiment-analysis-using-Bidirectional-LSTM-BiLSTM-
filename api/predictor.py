import os
from typing import Optional
from src.inference.predictor import SentimentPredictor


class ModelService:
    def __init__(self):
        self._lstm_predictor: Optional[SentimentPredictor] = None
        self._bilstm_predictor: Optional[SentimentPredictor] = None

    def get_predictor(self, model_type: str = "lstm") -> SentimentPredictor:
        model_type = model_type.lower()
        if model_type == "lstm":
            if self._lstm_predictor is None:
                path = "models/lstm/lstm_model.keras"
                tok = "models/lstm/tokenizer.pkl"
                le = "models/lstm/label_encoder.pkl"
                if not (os.path.exists(path) and os.path.exists(tok) and os.path.exists(le)):
                    raise FileNotFoundError(
                        "LSTM model artifacts not found. Please run: python -m src.training.train_lstm"
                    )
                self._lstm_predictor = SentimentPredictor(path, tok, le)
            return self._lstm_predictor

        elif model_type in ["bilstm", "bi-lstm"]:
            if self._bilstm_predictor is None:
                path = "models/bilstm/bilstm_model.keras"
                tok = "models/bilstm/tokenizer.pkl"
                le = "models/bilstm/label_encoder.pkl"
                if not (os.path.exists(path) and os.path.exists(tok) and os.path.exists(le)):
                    raise FileNotFoundError(
                        "BiLSTM model artifacts not found. Please run: python -m src.training.train_bilstm"
                    )
                self._bilstm_predictor = SentimentPredictor(path, tok, le)
            return self._bilstm_predictor

        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'lstm' or 'bilstm'.")

    def get_status(self):
        return {
            "lstm_ready": os.path.exists("models/lstm/lstm_model.keras"),
            "bilstm_ready": os.path.exists("models/bilstm/bilstm_model.keras"),
        }


model_service = ModelService()
