# Twitter Sentiment Analysis: LSTM vs. BiLSTM

A production-ready Deep Learning framework for 3-class Twitter Sentiment Analysis (Positive, Neutral, Negative) comparing unidirectional **LSTM** and **Bidirectional LSTM (BiLSTM)** architectures with a shared, reproducible preprocessing pipeline, FastAPI inference backend, and interactive modern frontend.

---

## 📌 Project Overview

- **Core Task**: 3-Class Sentiment Classification (`positive`, `neutral`, `negative`) on real-world Twitter data.
- **Architectures**:
  - **LSTM**: Captures forward sequential context with `SpatialDropout1D` regularization.
  - **BiLSTM**: Captures both forward and backward contextual nuances simultaneously for enhanced feature extraction.
- **Shared Pipeline**: Guaranteed fair comparison by sharing identical cleaning, vocabulary indexing, sequence padding, and stratified data splitting.
- **Serving Layer**: High-performance FastAPI server with batch & single prediction endpoints, and an interactive dark-mode glassmorphic web dashboard.

---

## 📂 Project Structure

```text
sentiment-analysis/
│
├── data/
│   └── raw/
│       └── training.csv               <- Raw tweets dataset
│
├── models/
│   ├── lstm/
│   │   ├── lstm_model.keras           <- Trained LSTM weights
│   │   ├── tokenizer.pkl              <- Tokenizer vocabulary
│   │   └── label_encoder.pkl          <- Target class mappings
│   │
│   └── bilstm/
│       ├── bilstm_model.keras         <- Trained BiLSTM weights
│       ├── tokenizer.pkl              <- Tokenizer vocabulary
│       └── label_encoder.pkl          <- Target class mappings
│
├── notebooks/
│   ├── twitter-sentiment-analysis.ipynb <- Original exploratory notebook
│   └── eda.py                         <- Standalone EDA generation script
│
├── src/
│   ├── __init__.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── clean_text.py              <- Tweet text regex cleaner
│   │   ├── tokenize.py                <- Tokenization, padding & persistence
│   │   └── prepare_data.py            <- End-to-end dataset pipeline & splitting
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_lstm.py              <- Unidirectional LSTM training & curves
│   │   └── train_bilstm.py            <- Bidirectional LSTM training & curves
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py                <- Confusion matrices & comparative report
│   │
│   └── inference/
│       ├── __init__.py
│       └── predictor.py               <- Production SentimentPredictor class
│
├── results/
│   ├── figures/                       <- Generated charts, curves & wordclouds
│   └── metrics/                       <- Classification reports & comparison.csv
│
├── api/
│   ├── __init__.py
│   ├── main.py                        <- FastAPI application & static routing
│   ├── schemas.py                     <- Pydantic validation schemas
│   └── predictor.py                   <- Model lifecycle manager & cache
│
├── frontend/
│   ├── index.html                     <- Modern web UI dashboard
│   ├── style.css                      <- Glassmorphic dark styling
│   └── script.js                      <- Interactive JS client
│
├── requirements.txt                   <- Python dependencies
├── README.md                          <- Project documentation
└── .gitignore                         <- Excluded files & checkpoints
```

---

## 🚀 Getting Started

### 1. Installation

Create a virtual environment (Python 3.9+) and install dependencies:

```bash
# Clone or navigate to the repository
cd "sent ana"

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 1. Exploratory Data Analysis (EDA)

Run the automated EDA script to generate distribution plots and wordclouds:

```bash
python notebooks/eda.py
```

This outputs the following publication-quality figures into `results/figures/`:
- `sentiment_distribution.png`: Class balance across tweets.
- `sentiment_time.png`: Tweet volume across morning, noon, and night.
- `sentiment_age.png`: Stacked distribution of user age brackets against sentiment.
- `country_distribution.png`: Top 10 origin countries in dataset.
- `positive_wordcloud.png`: High-frequency positive lexicon.
- `negative_wordcloud.png`: High-frequency negative lexicon.

---

## 🧠 2. Training the Models

### Train LSTM

```bash
python -m src.training.train_lstm
```

- Trains the unidirectional LSTM architecture.
- Saves model weights to `models/lstm/lstm_model.keras`.
- Saves tokenizer and label encoder to `models/lstm/`.
- Generates `results/figures/lstm_accuracy.png` and `results/figures/lstm_loss.png`.

### Train BiLSTM

```bash
python -m src.training.train_bilstm
```

- Trains the Bidirectional LSTM architecture.
- Saves model weights to `models/bilstm/bilstm_model.keras`.
- Saves tokenizer and label encoder to `models/bilstm/`.
- Generates `results/figures/bilstm_accuracy.png` and `results/figures/bilstm_loss.png`.

---

## 📈 3. Evaluation and Model Comparison

Compare both models on the held-out test set:

```bash
python -m src.evaluation.evaluate
```

This evaluates both models on identical test data and produces:
- `results/figures/lstm_confusion_matrix.png`
- `results/figures/bilstm_confusion_matrix.png`
- `results/metrics/lstm_report.txt`
- `results/metrics/bilstm_report.txt`
- `results/metrics/comparison.csv`

---

## 🌐 4. Running the Web Application & API

Launch the FastAPI application:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- **Interactive Web UI**: Open your browser at [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🔌 API Endpoints Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web dashboard |
| `GET` | `/api/health` | Healthcheck & model readiness flags |
| `GET` | `/api/models` | List supported architectures (`lstm`, `bilstm`) |
| `POST` | `/api/predict` | Single tweet sentiment prediction |
| `POST` | `/api/predict/batch` | Batch predictions for multiple tweets |
| `GET` | `/figures/*` | Direct access to generated EDA and training charts |

### Example Request (`POST /api/predict`):

```json
{
  "text": "Great customer service and super fast response, thank you!",
  "model_type": "bilstm"
}
```

### Example Response:

```json
{
  "text": "Great customer service and super fast response, thank you!",
  "cleaned_text": "great customer service and super fast response thank you",
  "sentiment": "positive",
  "confidence": 0.9421,
  "probabilities": {
    "negative": 0.0215,
    "neutral": 0.0364,
    "positive": 0.9421
  },
  "model_type": "bilstm"
}
```

---

## 🛡️ License & Acknowledgements

- Built with TensorFlow 2.16+, Scikit-Learn, Pandas, and FastAPI.
- Dataset: Sentiment140 / Twitter Sentiment Dataset.
