from pydantic import BaseModel, Field
from typing import List, Dict


class PredictionRequest(BaseModel):
    text: str = Field(..., example="I really love using this platform, it is super helpful!")


class PredictionResponse(BaseModel):
    text: str
    cleaned_text: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    model_name: str = "BiLSTM"


class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., example=["Best day ever!", "Terrible experience, very disappointed.", "Flight leaves at 10 AM."])


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
    model_name: str = "BiLSTM"


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_name: str = "BiLSTM"
