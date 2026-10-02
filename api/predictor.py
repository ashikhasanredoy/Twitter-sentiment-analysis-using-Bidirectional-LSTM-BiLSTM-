from typing import Optional
from src.config import config
from src.inference.predictor import SentimentPredictor


class ModelService:
    def __init__(self):
        self._predictor: Optional[SentimentPredictor] = None

    def get_predictor(self) -> SentimentPredictor:
        if self._predictor is None:
            if not self.is_ready():
                raise FileNotFoundError(
                    "BiLSTM model artifacts not found. Please run: python -m src.training.train"
                )
            self._predictor = SentimentPredictor()
        return self._predictor

    def is_ready(self) -> bool:
        return (
            config.MODEL_PATH.exists()
            and config.TOKENIZER_PATH.exists()
            and config.LABEL_ENCODER_PATH.exists()
        )


model_service = ModelService()
