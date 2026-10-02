import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.config import config
from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse
)
from api.predictor import model_service

app = FastAPI(
    title="Twitter Sentiment Analysis API",
    description="Production BiLSTM Deep Learning API for Sentiment Classification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if config.FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(config.FIGURES_DIR)), name="figures")

if config.FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(config.FRONTEND_DIR)), name="static")


@app.get("/", response_class=FileResponse)
def root():
    index_file = config.FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Twitter Sentiment Analysis BiLSTM API is running. Visit /docs for OpenAPI specs."}


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        model_ready=model_service.is_ready(),
        model_name="BiLSTM"
    )


@app.post("/api/predict", response_model=PredictionResponse)
def predict_sentiment(request: PredictionRequest):
    try:
        predictor = model_service.get_predictor()
        res = predictor.predict(request.text)
        return PredictionResponse(
            text=res["text"],
            cleaned_text=res["cleaned_text"],
            sentiment=res["sentiment"],
            confidence=res["confidence"],
            probabilities=res["probabilities"],
            model_name="BiLSTM"
        )
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=503, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    try:
        predictor = model_service.get_predictor()
        results = predictor.predict_batch(request.texts)
        responses = [
            PredictionResponse(
                text=r["text"],
                cleaned_text=r["cleaned_text"],
                sentiment=r["sentiment"],
                confidence=r["confidence"],
                probabilities=r["probabilities"],
                model_name="BiLSTM"
            )
            for r in results
        ]
        return BatchPredictionResponse(
            predictions=responses,
            count=len(responses),
            model_name="BiLSTM"
        )
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=503, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
