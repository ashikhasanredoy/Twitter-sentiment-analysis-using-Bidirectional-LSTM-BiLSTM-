# Twitter Sentiment Analysis (BiLSTM Deep Learning Pipeline)

Production-grade Twitter Sentiment Analysis system powered by a Bidirectional LSTM (BiLSTM) neural network, FastAPI inference backend, and pure flat-white UI dashboard.

---

## 📂 Project Architecture

```text
sentiment-analysis/
│
├── data/
│   └── raw/
│       └── training.csv               <- Raw tweets dataset
│
├── models/
│   ├── bilstm_model.keras             <- Best checkpointed model weights
│   ├── tokenizer.pkl                  <- Vocabulary mapping
│   └── label_encoder.pkl              <- Class mapping (negative, neutral, positive)
│
├── notebooks/
│   ├── twitter-sentiment-analysis.ipynb <- Original exploratory notebook
│   └── eda.py                         <- Automated dataset analysis script
│
├── src/
│   ├── __init__.py
│   ├── config.py                      <- Centralized paths & hyperparameters
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── clean_text.py              <- Pre-compiled regex text normalizer
│   │   ├── tokenize.py                <- Tokenizer fitting and sequence padding
│   │   └── prepare_data.py            <- Stratified dataset loader & split
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py                   <- BiLSTM training with EarlyStopping & Checkpoints
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py                <- Confusion matrix & classification report
│   │
│   └── inference/
│       ├── __init__.py
│       └── predictor.py               <- Thread-safe SentimentPredictor
│
├── results/
│   ├── figures/                       <- Training curves & EDA visualizations
│   └── metrics/                       <- Evaluation metrics report
│
├── api/
│   ├── __init__.py
│   ├── main.py                        <- FastAPI app with CORS & static routes
│   ├── schemas.py                     <- Pydantic validation models
│   └── predictor.py                   <- Lifecycle model cache
│
├── frontend/
│   ├── index.html                     <- Pure flat white UI
│   ├── style.css                      <- Flat border styling (zero shadows)
│   └── script.js                      <- Interactive API client
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Quickstart

### 1. Train the BiLSTM Model

```bash
python -m src.training.train --epochs 10 --batch-size 32
```

- Trains BiLSTM with `EarlyStopping(patience=2, restore_best_weights=True)`.
- Automatically checkpoints the best weights to `models/bilstm_model.keras`.
- Dynamically reduces learning rate with `ReduceLROnPlateau`.
- Generates `results/figures/bilstm_training_history.png`.

### 2. Evaluate the Model

```bash
python -m src.evaluation.evaluate
```

- Computes test accuracy, precision, recall, and macro F1 score.
- Generates confusion matrix heatmap at `results/figures/bilstm_confusion_matrix.png`.
- Saves report to `results/metrics/bilstm_report.txt`.

### 3. Launch Web Dashboard & API

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8009
```

- Web UI: [http://127.0.0.1:8009](http://127.0.0.1:8009)
- Interactive API Docs: [http://127.0.0.1:8009/docs](http://127.0.0.1:8009/docs)
