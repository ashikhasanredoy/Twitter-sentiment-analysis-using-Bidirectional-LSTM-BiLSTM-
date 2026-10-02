from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class PredictionRequest(BaseModel):
    text: str = Field(..., example="I really love using this platform, it is super helpful!")
    model_type: Optional[str] = Field("lstm", description="Choose 'lstm' or 'bilstm'")


class PredictionResponse(BaseModel):
    text: str
    cleaned_text: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    model_type: str


class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., example=["Best day ever!", "Terrible experience, very disappointed.", "Flight leaves at 10 AM."])
    model_type: Optional[str] = Field("lstm", description="Choose 'lstm' or 'bilstm'")


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    model_type: str
    count: int


class ModelInfo(BaseModel):
    name: str
    type: str
    status: str
    path: str
