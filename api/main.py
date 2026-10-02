import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo
)
from api.predictor import model_service

app = FastAPI(
    title="Twitter Sentiment Analysis API",
    description="Dual-Model (LSTM vs BiLSTM) Sentiment Classification API for Tweets",
    version="1.0.0"
)

# Enable CORS for local testing and frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount figures directory if it exists
if os.path.exists("results/figures"):
    app.mount("/figures", StaticFiles(directory="results/figures"), name="figures")

# Mount frontend directory for static assets
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/", response_class=FileResponse)
def root():
    frontend_index = os.path.join("frontend", "index.html")
    if os.path.exists(frontend_index):
        return FileResponse(frontend_index)
    return {"message": "Twitter Sentiment Analysis API is running. Visit /docs for OpenAPI specs."}


@app.get("/api/health")
def health_check():
    status = model_service.get_status()
    return {
        "status": "healthy",
        "models": status
    }


@app.get("/api/models")
def list_models():
    status = model_service.get_status()
    return [
        ModelInfo(
            name="Long Short-Term Memory (LSTM)",
            type="lstm",
            status="available" if status["lstm_ready"] else "not_trained",
            path="models/lstm/lstm_model.keras"
        ),
        ModelInfo(
            name="Bidirectional LSTM (BiLSTM)",
            type="bilstm",
            status="available" if status["bilstm_ready"] else "not_trained",
            path="models/bilstm/bilstm_model.keras"
        )
    ]


@app.post("/api/predict", response_model=PredictionResponse)
def predict_sentiment(request: PredictionRequest):
    model_type = (request.model_type or "lstm").lower()
    try:
        predictor = model_service.get_predictor(model_type)
        res = predictor.predict(request.text)
        return PredictionResponse(
            text=res["text"],
            cleaned_text=res["cleaned_text"],
            sentiment=res["sentiment"],
            confidence=res["confidence"],
            probabilities=res["probabilities"],
            model_type=model_type
        )
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=503, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    model_type = (request.model_type or "lstm").lower()
    try:
        predictor = model_service.get_predictor(model_type)
        results = predictor.predict_batch(request.texts)
        responses = [
            PredictionResponse(
                text=r["text"],
                cleaned_text=r["cleaned_text"],
                sentiment=r["sentiment"],
                confidence=r["confidence"],
                probabilities=r["probabilities"],
                model_type=model_type
            )
            for r in results
        ]
        return BatchPredictionResponse(
            predictions=responses,
            model_type=model_type,
            count=len(responses)
        )
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=503, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
