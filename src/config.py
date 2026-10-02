import os
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class AppConfig:
    BASE_DIR: Path = PROJECT_ROOT
    DATA_PATH: Path = PROJECT_ROOT / "data" / "raw" / "training.csv"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    MODEL_PATH: Path = MODELS_DIR / "bilstm_model.keras"
    TOKENIZER_PATH: Path = MODELS_DIR / "tokenizer.pkl"
    LABEL_ENCODER_PATH: Path = MODELS_DIR / "label_encoder.pkl"
    FRONTEND_DIR: Path = PROJECT_ROOT / "frontend"
    
    FIGURES_DIR: Path = PROJECT_ROOT / "results" / "figures"
    METRICS_DIR: Path = PROJECT_ROOT / "results" / "metrics"

    MAX_FEATURES: int = 5000
    MAX_LENGTH: int = 50
    EMBEDDING_DIM: int = 128
    LSTM_UNITS: int = 128
    SPATIAL_DROPOUT: float = 0.4
    RECURRENT_DROPOUT: float = 0.2
    DROPOUT: float = 0.2

    BATCH_SIZE: int = 32
    EPOCHS: int = 10
    PATIENCE: int = 2
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

config = AppConfig()
