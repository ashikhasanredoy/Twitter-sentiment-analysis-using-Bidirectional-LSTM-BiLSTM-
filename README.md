# 🚀 Twitter Sentiment Analysis (BiLSTM Deep Learning Production Pipeline)

An end-to-end, production-ready Natural Language Processing (NLP) and Deep Learning system designed to analyze and classify tweet sentiment into three distinct classes: **Positive**, **Neutral**, and **Negative**.

The system is powered by a **Bidirectional Long Short-Term Memory (BiLSTM)** neural network built with TensorFlow/Keras, exposed through a high-performance **FastAPI REST API**, and paired with an ultra-clean, **single-viewport flat-white web dashboard**.

---

## 📑 Table of Contents

1. [System Overview & Motivation](#-system-overview--motivation)
2. [End-to-End Pipeline Architecture](#-end-to-end-pipeline-architecture)
   - [Architectural Flowchart](#architectural-flowchart)
   - [Pipeline Stages in Detail](#pipeline-stages-in-detail)
3. [Deep Learning Architecture & Theory](#-deep-learning-architecture--theory)
   - [Why Bidirectional LSTM?](#why-bidirectional-lstm)
   - [Mathematical Formulation of LSTM Gates](#mathematical-formulation-of-lstm-gates)
   - [Layer-by-Layer Architectural Breakdown](#layer-by-layer-architectural-breakdown)
   - [Regularization & Optimization Strategy](#regularization--optimization-strategy)
4. [Exhaustive File & Directory Reference](#-exhaustive-file--directory-reference)
   - [Complete Project Tree](#complete-project-tree)
   - [Module-by-Module & File-by-File Breakdown](#module-by-module--file-by-file-breakdown)
5. [Dataset & Exploratory Data Analysis (EDA)](#-dataset--exploratory-data-analysis-eda)
   - [Dataset Schema & Summary](#dataset-schema--summary)
   - [EDA Visualizations & Insights](#eda-visualizations--insights)
6. [Training, Callbacks & Evaluation](#-training-callbacks--evaluation)
   - [Hyperparameter Configurations](#hyperparameter-configurations)
   - [Training Dynamics & Early Stopping](#training-dynamics--early-stopping)
   - [Test Set Evaluation Metrics](#test-set-evaluation-metrics)
   - [Confusion Matrix & Classification Analysis](#confusion-matrix--classification-analysis)
7. [Inference Engine & REST API](#-inference-engine--rest-api)
   - [Vectorized Prediction Pipeline](#vectorized-prediction-pipeline)
   - [API Endpoints Reference](#api-endpoints-reference)
   - [Live cURL & Python Client Examples](#live-curl--python-client-examples)
8. [Frontend Web Dashboard](#-frontend-web-dashboard)
   - [UI/UX Design Philosophy](#uiux-design-philosophy)
   - [Features & Dashboard Capabilities](#features--dashboard-capabilities)
9. [Installation, Setup & Reproduction Guide](#-installation-setup--reproduction-guide)
   - [Prerequisites](#prerequisites)
   - [Step-by-Step Execution Commands](#step-by-step-execution-commands)
10. [Production Considerations & Future Roadmap](#-production-considerations--future-roadmap)

---

## 📖 System Overview & Motivation

Twitter (X) text is notoriously difficult to classify using traditional rule-based or shallow machine learning approaches due to:
- **Syntactic Noise**: Slang, abbreviations, typos, emoji representations, and unconventional grammatical structures.
- **Contextual Inversion**: Words modifying sentiment appear at opposite ends of the tweet (e.g., *"Not that I expected anything good, but it was okay"*).
- **Extreme Brevity**: Tweets are constrained in length, meaning each token carries a disproportionate amount of contextual weight.

To solve this, this repository decouples an exploratory monolithic notebook into an **enterprise-grade, modular deep learning repository**. Instead of a standard unidirectional RNN or LSTM, we utilize a **Bidirectional LSTM (BiLSTM)** that reads sequence tokens in both forward (past-to-future) and backward (future-to-past) temporal directions simultaneously.

---

## 🔄 End-to-End Pipeline Architecture

### Architectural Flowchart

```mermaid
flowchart TD
    subgraph Data_Pipeline["1. Ingestion & Preprocessing Pipeline"]
        A["Raw Data Ingestion\n(data/raw/training.csv)"] --> B["Automated EDA\n(notebooks/eda.py)"]
        B --> B1["6 High-Res EDA Figures\n(results/figures/*.png)"]
        A --> C["Text Normalization\n(src/preprocessing/clean_text.py)"]
        C --> C1["Regex Cleaning:\nStrip URLs, @mentions, #, punctuation,\nlowercased & whitespace trimmed"]
        C1 --> D["Data Preparation & Splitting\n(src/preprocessing/prepare_data.py)"]
        D --> D1["Label Encoding:\nnegative: 0, neutral: 1, positive: 2\nOne-Hot Categorical: (N, 3)"]
        D --> D2["Stratified Split:\n80% Train (21,984 samples)\n20% Test (5,496 samples)"]
        D2 --> E["Tokenization & Padding\n(src/preprocessing/tokenize.py)"]
        E --> E1["Fit Keras Tokenizer (top 5,000 words)\nPost-Pad Sequences to length 50"]
        E1 --> E2["Serialized Tokenizer:\nmodels/tokenizer.pkl"]
    end

    subgraph Deep_Learning_Pipeline["2. Deep Learning Training Pipeline"]
        E1 --> F["Tensor Creation:\nX_train, y_train & X_test, y_test"]
        F --> G["BiLSTM Architecture\n(src/training/train.py)"]
        G --> G1["Embedding Layer: (5000, 128)"]
        G1 --> G2["SpatialDropout1D: rate=0.4"]
        G2 --> G3["Bidirectional LSTM: 128 units (rec_drop=0.2)"]
        G3 --> G4["Dense Output: 3 units + Softmax"]
        G --> H["Training Callbacks"]
        H --> H1["EarlyStopping (patience=2, restore best)"]
        H --> H2["ModelCheckpoint (models/bilstm_model.keras)"]
        H --> H3["ReduceLROnPlateau (factor=0.2, patience=1)"]
        H --> H4["Training Curves Plotting\n(results/figures/bilstm_training_history.png)"]
    end

    subgraph Evaluation_Pipeline["3. Evaluation & Verification Pipeline"]
        F --> I["Model Evaluator\n(src/evaluation/evaluate.py)"]
        H2 --> I
        I --> I1["Confusion Matrix Heatmap\n(results/figures/bilstm_confusion_matrix.png)"]
        I --> I2["Metrics Classification Report\n(results/metrics/bilstm_report.txt)"]
    end

    subgraph Serving_Pipeline["4. Serving & Production Inference Pipeline"]
        H2 --> J["SentimentPredictor Engine\n(src/inference/predictor.py)"]
        E2 --> J
        D1 --> J
        J --> K["FastAPI Application\n(api/main.py)"]
        K --> L1["GET /api/health"]
        K --> L2["POST /api/predict"]
        K --> L3["POST /api/predict/batch"]
        K --> M["Single-Page Dashboard\n(frontend/index.html)"]
    end
```

---

### Pipeline Stages in Detail

#### Stage 1: Data Ingestion & Sanitization
1. **Raw CSV Loading**: Ingests `data/raw/training.csv` containing 27,481 entries.
2. **Missing Value Scrubbing**: Automatically detects and drops null text samples (1 invalid row identified and purged, leaving 27,480 valid rows).
3. **Column Extraction**: Extracts the text feature column `text` and ground truth class column `sentiment`.

#### Stage 2: Text Cleansing (`src/preprocessing/clean_text.py`)
Applies high-throughput, pre-compiled regular expressions in strict sequence:
1. `URL_PATTERN`: Removes `http://...`, `https://...`, and `www....` links.
2. `MENTION_HASHTAG_PATTERN`: Strips `@username` tags and `#hashtag` symbols while preserving the semantic root word.
3. `PUNCTUATION_PATTERN`: Strips ASCII punctuation, quotes, brackets, and non-alphanumeric noise.
4. `WHITESPACE_PATTERN`: Collapses multiple tabs, carriage returns, and spaces into single whitespace tokens.
5. `Lowercasing`: Standardizes all tokens into lowercase characters to unify vocabulary tokens.

#### Stage 3: Target Encoding & Stratification (`src/preprocessing/prepare_data.py`)
1. **Label Encoding**: Encodes categorical targets into integers:
   - `negative` $\rightarrow 0$
   - `neutral` $\rightarrow 1$
   - `positive` $\rightarrow 2$
2. **One-Hot Categorical Transformation**: Converts integers into 3-dimensional probability targets via Keras `to_categorical` for multi-class cross-entropy computation:
   - `0` $\rightarrow [1, 0, 0]$
   - `1` $\rightarrow [0, 1, 0]$
   - `2` $\rightarrow [0, 0, 1]$
3. **Stratified Splitting**: Splits the 27,480 samples using an 80/20 train/test ratio with `stratify=y`. This guarantees the identical class distribution in both splits (40.5% neutral, 31.2% positive, 28.3% negative).
   - Training Set: **21,984 samples**
   - Test Set: **5,496 samples**

#### Stage 4: Tokenization & Sequence Alignment (`src/preprocessing/tokenize.py`)
1. **Vocabulary Fitting**: Fits a Keras `Tokenizer` on the training text with `num_words=5000` (retaining the top 5,000 most statistically relevant tokens).
2. **Sequence Conversion**: Translates each clean sentence into an array of vocabulary integer IDs.
3. **Pad Sequences**: Post-pads sequences with zeros to a uniform maximum sequence length of `maxlen=50` tokens.
4. **Artifact Persistence**: Serializes the fitted tokenizer to `models/tokenizer.pkl` to ensure identical text-to-integer mapping during inference.

#### Stage 5: Neural Network Training (`src/training/train.py`)
1. Instantiates the BiLSTM model architecture.
2. Registers automated production callbacks (`EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`).
3. Trains over batches of size 32 using the Adam optimizer (`lr=0.001`).
4. Generates side-by-side loss and accuracy curves saved to `results/figures/bilstm_training_history.png`.

#### Stage 6: Offline Model Evaluation (`src/evaluation/evaluate.py`)
1. Reloads the best-performing saved model checkpoint (`models/bilstm_model.keras`).
2. Evaluates model performance on the held-out 5,496 test samples.
3. Computes macro-averaged and per-class precision, recall, and F1-score.
4. Plots and saves a confusion matrix heatmap to `results/figures/bilstm_confusion_matrix.png` and logs metrics to `results/metrics/bilstm_report.txt`.

#### Stage 7: Inference Engine & REST API (`src/inference/` & `api/`)
1. Encapsulates preprocessing, tokenization, model inference, and output formatting into `SentimentPredictor`.
2. Serves endpoints using **FastAPI** with Pydantic request validation, CORS middleware, and static asset mounts.
3. Exposes single and batch prediction endpoints returning argmax labels, winning confidence, and complete 3-class probability distributions.

#### Stage 8: Client Web Dashboard (`frontend/`)
1. Provides an ultra-minimalist, flat-white single-viewport dashboard (`100vh` without page scrolling).
2. Allows single interactive tweet analysis, multi-line batch analysis, and visual EDA inspection.

---

## 🧠 Deep Learning Architecture & Theory

### Why Bidirectional LSTM?

Standard Recurrent Neural Networks (RNNs) suffer from the **vanishing gradient problem**, making them incapable of retaining long-term dependencies. Standard LSTMs solve vanishing gradients through internal memory gates, but they process text **strictly sequentially from left to right**:

$$\vec{h}_t = \text{LSTM}(x_t, \vec{h}_{t-1})$$

In human natural language—especially microblogging text—words appearing late in a sentence frequently redefine the meaning of words that appeared at the beginning:
> *"The display and build quality were stunning, but the battery died in twenty minutes."*

A forward-only LSTM processing *"stunning"* has zero knowledge of the incoming contrast conjunction *"but"*. A **Bidirectional LSTM (BiLSTM)** resolves this by running two independent LSTM passes concurrently:
1. **Forward Path ($\vec{h}_t$)**: Reads text left-to-right from token $x_1$ to $x_T$, learning past context.
2. **Backward Path ($\overleftarrow{h}_t$)**: Reads text right-to-left from token $x_T$ to $x_1$, learning future context.

The output hidden state for token $t$ is the concatenation of both vectors:
$$h_t = [\vec{h}_t \,\|\, \overleftarrow{h}_t]$$

This gives the final representation full contextual awareness of both preceding and succeeding tokens.

---

### Mathematical Formulation of LSTM Gates

Each LSTM unit controls memory retention via four affine gating mechanisms parameterized by weight matrices $W$, recurrent weights $U$, and biases $b$:

1. **Forget Gate ($f_t$)**: Decides what fraction of the previous cell state $C_{t-1}$ to discard:
   $$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$
2. **Input Gate ($i_t$)**: Decides which newly computed values will update the cell state:
   $$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$
3. **Candidate Cell State ($\tilde{C}_t$)**: Computes new candidate information:
   $$\tilde{C}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$$
4. **Cell State Update ($C_t$)**: Updates memory by element-wise combining old memory and new candidate input:
   $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
5. **Output Gate ($o_t$)**: Decides what the next hidden state should be based on the updated cell state:
   $$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$$
   $$h_t = o_t \odot \tanh(C_t)$$

*(where $\sigma$ denotes the sigmoid activation function and $\odot$ denotes element-wise Hadamard product).*

---

### Layer-by-Layer Architectural Breakdown

```text
Input Sequence [Token IDs]  ──>  Shape: (Batch_Size, 50)
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Word Embedding Layer                                     │
│    Input Dim: 5,000 | Output Dim: 128 | Input Length: 50    │
│    Parameters: 5,000 × 128 = 640,000 trainable weights      │
│    Output Tensor: (Batch_Size, 50, 128)                     │
└─────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. SpatialDropout1D Layer                                   │
│    Rate: 0.4 (Drops 40% of entire 1D feature channels)      │
│    Parameters: 0                                            │
│    Output Tensor: (Batch_Size, 50, 128)                     │
└─────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Bidirectional LSTM Layer                                 │
│    Forward LSTM: 128 units | Backward LSTM: 128 units       │
│    Recurrent Dropout: 0.2                                   │
│    Parameters: 2 × [4 × (128 × 128 + 128 × 128 + 128)]      │
│              = 263,168 trainable weights                    │
│    Output Tensor: (Batch_Size, 256)                         │
└─────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Fully Connected (Dense) Output Layer                     │
│    Units: 3 | Activation: Softmax                           │
│    Parameters: (256 × 3) + 3 = 771 trainable weights        │
│    Output Tensor: (Batch_Size, 3)                           │
└─────────────────────────────────────────────────────────────┘
               │
               ▼
Probabilities: [P(Negative), P(Neutral), P(Positive)]
```

#### Detailed Layer Specification Table

| Layer # | Layer Name | Type | Input Dimension | Output Dimension | Param Count | Core Function |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `embedding` | Embedding | `(None, 50)` | `(None, 50, 128)` | **640,000** | Maps discrete word integers into dense 128-dimensional semantic vector space. |
| **2** | `spatial_dropout` | SpatialDropout1D | `(None, 50, 128)` | `(None, 50, 128)` | **0** | Drops entire 128-dim feature maps instead of individual activations to prevent co-adaptation across adjacent tokens. |
| **3** | `bidirectional_lstm` | Bidirectional(LSTM) | `(None, 50, 128)` | `(None, 256)` | **263,168** | Encodes past and future sequential dependencies into a concatenated 256-dim feature vector. |
| **4** | `output_dense` | Dense(3, Softmax) | `(None, 256)` | `(None, 3)` | **771** | Linear projection from hidden state to 3 sentiment logits, normalized into probabilities via Softmax. |

- **Total Trainable Parameters**: **903,939** (3.45 MB)
- **Non-trainable Parameters**: **0**

---

### Regularization & Optimization Strategy

1. **SpatialDropout1D vs. Standard Dropout**:
   - In NLP sequences, adjacent tokens frequently share semantic embeddings. Standard dropout randomly zeros individual elements in the 2D matrix, allowing information to leak across correlated neighbors.
   - `SpatialDropout1D(0.4)` drops entire 1D feature channels across the whole sequence, forcing the network to develop diverse, non-redundant linguistic representations.
2. **Recurrent Dropout**:
   - Applies dropout (`0.2`) directly to the recurrent state transitions inside each LSTM step, preventing temporal overfitting.
3. **Loss Function**:
   - Multi-class **Categorical Cross-Entropy**:
     $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^3 y_{i,c} \log(\hat{y}_{i,c})$$
     where $y_{i,c}$ is the binary ground-truth indicator and $\hat{y}_{i,c}$ is the model's predicted probability for class $c$.
4. **Adaptive Optimization**:
   - **Adam** with initial learning rate $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$.
5. **Learning Rate Decay**:
   - `ReduceLROnPlateau` reduces the learning rate by a factor of 5 ($\times 0.2$) if `val_loss` does not decrease after 1 epoch, preventing overshoot around local minima.

---

## 📂 Exhaustive File & Directory Reference

### Complete Project Tree

```text
sent-ana/
├── .gitignore                          <- Git ignore rules for caches, checkpoints & OS files
├── README.md                           <- Complete system documentation & pipeline guide
├── requirements.txt                    <- Pinned dependencies for Python environment
│
├── api/                                <- FastAPI REST API Microservice
│   ├── __init__.py                     <- Package initialization
│   ├── main.py                         <- Application entrypoint, CORS, route handlers & static mounts
│   ├── predictor.py                    <- Singleton model cache manager for fast in-memory inference
│   └── schemas.py                      <- Pydantic schemas enforcing input validation & output typing
│
├── data/                               <- Dataset Storage
│   └── raw/
│       └── training.csv                <- 27,480 tweet training dataset with metadata
│
├── frontend/                           <- Flat-White Single-Viewport Client Dashboard
│   ├── index.html                      <- Dashboard markup (Predict, Batch & EDA tabs)
│   ├── style.css                       <- Zero-shadow, pure flat styling with 100vh lock
│   └── script.js                       <- Asynchronous fetch calls, DOM updates & polling
│
├── models/                             <- Serialized Model Artifacts
│   ├── bilstm_model.keras              <- Checkpointed best BiLSTM weights (Keras 3 format)
│   ├── tokenizer.pkl                   <- Serialized fitted Keras Text Tokenizer
│   └── label_encoder.pkl               <- Serialized Scikit-Learn LabelEncoder
│
├── notebooks/                          <- Analysis Scripts & Exploratory Notebooks
│   ├── eda.py                          <- Standalone script generating 6 publication-grade figures
│   └── twitter-sentiment-analysis.ipynb<- Original exploratory baseline notebook
│
├── results/                            <- Output Metrics & Visualizations
│   ├── figures/                        <- Generated high-resolution diagnostic charts
│   │   ├── bilstm_confusion_matrix.png <- Test set evaluation confusion matrix heatmap
│   │   ├── bilstm_training_history.png <- Side-by-side Loss and Accuracy epoch curves
│   │   ├── country_distribution.png    <- Pie chart of top tweet origin nations
│   │   ├── negative_wordcloud.png      <- Wordcloud of high-frequency negative tokens
│   │   ├── positive_wordcloud.png      <- Wordcloud of high-frequency positive tokens
│   │   ├── sentiment_age.png           <- Stacked bar chart: User age groups vs sentiment
│   │   ├── sentiment_distribution.png  <- Dataset class balance bar plot
│   │   └── sentiment_time.png          <- Grouped bar chart: Time of day vs sentiment
│   └── metrics/
│       └── bilstm_report.txt           <- Plaintext precision, recall, F1 classification report
│
└── src/                                <- Core Source Code Modules
    ├── __init__.py                     <- Root source package initialization
    ├── config.py                       <- Centralized, immutable AppConfig dataclass
    │
    ├── preprocessing/                  <- Data Transformation & Sanitization
    │   ├── __init__.py                 <- Module exports for clean_text, tokenize, prepare_data
    │   ├── clean_text.py               <- High-throughput pre-compiled regex cleaner
    │   ├── prepare_data.py             <- Missing value removal, label encoding & stratified split
    │   └── tokenize.py                 <- Tokenizer fitting, sequence padding & persistence
    │
    ├── training/                       <- Neural Network Training Pipelines
    │   ├── __init__.py                 <- Module exports for training entrypoints
    │   └── train.py                    <- Full BiLSTM training loop with callbacks & CLI args
    │
    ├── evaluation/                     <- Performance Diagnostics & Metrics
    │   ├── __init__.py                 <- Module exports for evaluation routines
    │   └── evaluate.py                 <- Model testing, classification report & heatmap generator
    │
    └── inference/                      <- Production Inference Engine
        ├── __init__.py                 <- Module exports for SentimentPredictor
        └── predictor.py                <- Vectorized inference class handling batch & single inputs
```

---

### Module-by-Module & File-by-File Breakdown

#### 1. Configuration Module (`src/config.py`)
- **Purpose**: Eliminates hardcoded magic numbers, string literals, and brittle relative filepaths across the entire codebase.
- **Key Class**: `AppConfig` (an immutable `@dataclass`).
- **Dynamic Anchoring**: Uses `Path(__file__).resolve().parent.parent` to dynamically establish `BASE_DIR`. All directories (`DATA_DIR`, `MODELS_DIR`, `RESULTS_DIR`, `FRONTEND_DIR`) are derived as absolute paths, guaranteeing that running scripts from any subfolder will never throw `FileNotFoundError`.
- **Configured Constants**:
  - `MAX_FEATURES = 5000`: Number of most frequent tokens in vocabulary.
  - `MAX_LEN = 50`: Sequence length for post-padding.
  - `EMBEDDING_DIM = 128`: Dense vector representation dimensions.
  - `LSTM_UNITS = 128`: Hidden recurrent units per direction (256 combined).
  - `DROPOUT_RATE = 0.4`: Spatial dropout probability.
  - `RECURRENT_DROPOUT = 0.2`: Recurrent state transition dropout.
  - `BATCH_SIZE = 32`: Minibatch gradient descent size.
  - `EPOCHS = 10`: Upper bound on training iterations.
  - `LEARNING_RATE = 0.001`: Initial Adam optimizer rate.
  - `CLASSES = ["negative", "neutral", "positive"]`: Canonical class labels.

#### 2. Preprocessing Subsystem (`src/preprocessing/`)
- **`clean_text.py`**:
  - Contains `clean_tweet(text: str) -> str`.
  - Defines pre-compiled module-level regex objects (`URL_PATTERN`, `MENTION_HASHTAG_PATTERN`, `PUNCTUATION_PATTERN`, `WHITESPACE_PATTERN`). Compiling regexes once at import time provides significant speedups during high-volume batch inference compared to `re.sub()` calls inside loops.
- **`prepare_data.py`**:
  - Contains `load_and_prepare_data(csv_path, test_size=0.2, random_state=42)`.
  - Reads raw CSV, drops rows with missing `text` or `sentiment`, applies `LabelEncoder`, converts labels to one-hot tensors via `to_categorical(y, num_classes=3)`, and executes `train_test_split(..., stratify=y)`.
  - Persists `models/label_encoder.pkl`.
- **`tokenize.py`**:
  - Contains `fit_and_tokenize(texts, max_features, maxlen)` and `tokenize_and_pad(texts, tokenizer, maxlen)`.
  - Creates and fits Keras `Tokenizer`, encodes text sequences, pads sequences using `pad_sequences(..., padding='post', truncating='post')`, and serializes the tokenizer to `models/tokenizer.pkl`.

#### 3. Training Subsystem (`src/training/`)
- **`train.py`**:
  - Orchestrates the entire training lifecycle.
  - Contains `build_bilstm_model()` to assemble the `Sequential([Embedding, SpatialDropout1D, Bidirectional(LSTM), Dense])` layers.
  - Contains `train_model(epochs, batch_size)`.
  - Configures 3 critical callbacks:
    1. `EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)`: Halts training if validation loss does not improve for 2 consecutive epochs and automatically rolls back weights to the global minimum.
    2. `ModelCheckpoint(filepath=..., monitor="val_loss", save_best_only=True)`: Saves the best performing model directly to `models/bilstm_model.keras`.
    3. `ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=1, min_lr=1e-5)`: Dynamically shrinks the learning rate when loss plateaus.
  - Generates diagnostic training history plots saved to `results/figures/bilstm_training_history.png`.
  - Exposes a CLI interface with `argparse` allowing execution flags: `--epochs` and `--batch-size`.

#### 4. Evaluation Subsystem (`src/evaluation/`)
- **`evaluate.py`**:
  - Contains `evaluate_model()`.
  - Loads the serialized test set and reloads the saved `bilstm_model.keras` checkpoint.
  - Computes raw probability predictions, extracts predicted classes using `np.argmax(y_pred_probs, axis=1)`.
  - Generates a Scikit-Learn `classification_report` and writes it to `results/metrics/bilstm_report.txt`.
  - Generates a high-resolution Seaborn confusion matrix heatmap with count and percentage annotations saved to `results/figures/bilstm_confusion_matrix.png`.

#### 5. Inference Subsystem (`src/inference/`)
- **`predictor.py`**:
  - Implements the production-grade `SentimentPredictor` class.
  - Validates that all required serialized artifacts (`bilstm_model.keras`, `tokenizer.pkl`, `label_encoder.pkl`) exist on disk before initialization.
  - `predict_batch(texts: List[str]) -> List[Dict[str, Any]]`: Vectorizes the text list, cleans all strings, passes them through tokenizer and padder, executes a single vectorized forward pass `model.predict(padded, batch_size=len(padded))`, and decodes probabilities.
  - `predict(text: str) -> Dict[str, Any]`: Reuses `predict_batch([text])[0]`, ensuring strict DRY (Don't Repeat Yourself) compliance.

#### 6. REST API Subsystem (`api/`)
- **`schemas.py`**:
  - Pydantic models for strict data contract enforcement:
    - `TextRequest`: Validates single tweet input with `min_length=1`.
    - `PredictionResponse`: Standardized JSON payload returning original text, cleaned text, predicted sentiment, confidence float, probability distribution dictionary, and model identifier.
    - `BatchTextRequest`: Validates a list of strings with `min_length=1`.
    - `BatchPredictionResponse`: Returns a list of `PredictionResponse` objects alongside total count.
    - `HealthResponse`: System operational status and model memory check.
- **`predictor.py`**:
  - Global singleton cache for `SentimentPredictor`. Ensures the 3.45 MB neural network is loaded into RAM once upon application startup, rather than reloaded on each HTTP request.
- **`main.py`**:
  - FastAPI web server.
  - Configures permissive CORS (`allow_origins=["*"]`) for local and distributed clients.
  - Mounts `results/figures/` at `/figures` for visual asset serving.
  - Mounts `frontend/` at `/` for dashboard UI delivery.
  - Endpoints:
    - `GET /api/health`: Healthcheck.
    - `POST /api/predict`: Single text prediction.
    - `POST /api/predict/batch`: Multi-text batch prediction.
    - `GET /api/figures`: Dynamic JSON list of all generated EDA and diagnostic charts.

#### 7. Frontend Subsystem (`frontend/`)
- **`index.html`**:
  - Semantic, accessible HTML5 structure.
  - Clean three-tab navigation: **Single Tweet Analysis**, **Batch Processing**, and **Dataset EDA Gallery**.
- **`style.css`**:
  - Flat-white design system: pure `#ffffff` card backgrounds, `#e5e7eb` 1px border lines, dark typography (`#111827`, `#4b5563`).
  - Completely eliminates drop-shadows and gray gradients (`box-shadow: none !important`).
  - Viewport-locked (`height: 100vh; overflow: hidden`), fitting all input boxes, buttons, result badges, and probability bars onto a single screen with zero vertical scrolling.
- **`script.js`**:
  - Asynchronous event handlers using `fetch()`.
  - Real-time probability bar animation.
  - Dynamic table population for batch predictions.
  - Automated EDA gallery grid rendering.

#### 8. Exploratory Data Analysis (`notebooks/eda.py`)
- Standalone headless script that ingests `data/raw/training.csv` and outputs 6 publication-ready charts to `results/figures/`:
  1. `sentiment_distribution.png`: Overall dataset class balance.
  2. `sentiment_time.png`: Sentiment breakdown across morning, noon, and night tweets.
  3. `sentiment_age.png`: Cross-tabulation of user age groups vs. sentiment.
  4. `country_distribution.png`: Top 10 tweet origin countries.
  5. `positive_wordcloud.png`: High-frequency wordcloud for positive tweets.
  6. `negative_wordcloud.png`: High-frequency wordcloud for negative tweets.

---

## 📊 Dataset & Exploratory Data Analysis (EDA)

### Dataset Schema & Summary

The dataset consists of **27,480 clean tweets** across 10 features:

| Column Name | Data Type | Description | Role in Pipeline |
| :--- | :--- | :--- | :--- |
| `textID` | String | Unique hexadecimal identifier for the tweet | Identifier (dropped) |
| `text` | String | Raw text content of the tweet | Primary Feature |
| `selected_text` | String | Substring corresponding to the sentiment label | Metadata (unused in classification) |
| `sentiment` | String | Ground truth label (`negative`, `neutral`, `positive`) | Primary Target |
| `Time of Tweet` | String | Binned time of post (`morning`, `noon`, `night`) | EDA Feature |
| `Age of User` | String | Demographic age bracket (`0-20`, `21-30`, etc.) | EDA Feature |
| `Country` | String | Origin country of user | EDA Feature |
| `Population -2020`| Integer | Origin country population estimate | Metadata |
| `Land Area` | Float | Geographic area in km² | Metadata |
| `Density` | Integer | Population density per km² | Metadata |

#### Class Distribution

| Sentiment Class | Sample Count | Percentage |
| :--- | :---: | :---: |
| **Neutral** | 11,118 | 40.5% |
| **Positive** | 8,582 | 31.2% |
| **Negative** | 7,780 | 28.3% |
| **Total Valid Samples** | **27,480** | **100.0%** |

---

### EDA Visualizations & Insights

All visualizations are stored in `results/figures/`:

1. **Class Distribution (`sentiment_distribution.png`)**:
   - Neutral sentiment represents the plurality of tweets (40.5%), while positive and negative classes represent 31.2% and 28.3% respectively. Stratified splitting was strictly implemented to preserve these exact proportions.
2. **Temporal Distribution (`sentiment_time.png`)**:
   - Analyzes whether sentiment shifts across morning, noon, and night. Sentiment proportions remain remarkably consistent across time bins, indicating time-of-day does not induce temporal bias.
3. **Age Demographics (`sentiment_age.png`)**:
   - Analyzes sentiment distribution across user age brackets. Sentiments are evenly distributed across age demographics.
4. **Geographic Distribution (`country_distribution.png`)**:
   - Highlights the top 10 origin countries represented in the dataset.
5. **Positive Vocabulary Wordcloud (`positive_wordcloud.png`)**:
   - Prominent recurring tokens: *"good"*, *"love"*, *"great"*, *"happy"*, *"day"*, *"thanks"*, *"nice"*.
6. **Negative Vocabulary Wordcloud (`negative_wordcloud.png`)**:
   - Prominent recurring tokens: *"miss"*, *"sad"*, *"sorry"*, *"hate"*, *"bad"*, *"work"*, *"tired"*.

---

## 📈 Training, Callbacks & Evaluation

### Hyperparameter Configurations

All hyperparameters are controlled via `src/config.py`:

| Hyperparameter | Value | Description & Rationale |
| :--- | :---: | :--- |
| **Vocabulary Size (`MAX_FEATURES`)** | `5,000` | Top 5,000 words capture >95% of vocabulary variance while pruning rare noise. |
| **Sequence Length (`MAX_LEN`)** | `50` | Over 99% of cleaned tweets contain fewer than 50 tokens. Minimizes unnecessary zero-padding. |
| **Embedding Dimension** | `128` | Dense vector width balancing representation capacity with low computational overhead. |
| **Spatial Dropout** | `0.4` | High channel dropout rate to prevent co-adaptation across sequence tokens. |
| **BiLSTM Hidden Units** | `128` | 128 forward + 128 backward = 256 concatenated output hidden dimensions. |
| **Recurrent Dropout** | `0.2` | Regularizes hidden state transitions inside the recurrent cell. |
| **Optimizer** | `Adam` | Adaptive gradient descent with initial learning rate $\alpha = 0.001$. |
| **Loss Function** | `Categorical Crossentropy` | Measures probabilistic distance for multi-class targets. |
| **Batch Size** | `32` | Standard minibatch size providing stable gradient estimates. |
| **Max Epochs** | `10` | Upper bound; controlled by automated early stopping. |

---

### Training Dynamics & Early Stopping

The network was trained on 21,984 training samples with 5,496 validation samples.

```text
Epoch 1/10
687/687 [==============================] - 24s 31ms/step - loss: 0.7712 - accuracy: 0.6558 - val_loss: 0.6865 - val_accuracy: 0.7042
Epoch 2/10
687/687 [==============================] - 21s 30ms/step - loss: 0.6384 - accuracy: 0.7351 - val_loss: 0.6738 - val_accuracy: 0.7107  <-- BEST WEIGHTS CHECKPOINTED
Epoch 3/10
687/687 [==============================] - 21s 30ms/step - loss: 0.5793 - accuracy: 0.7645 - val_loss: 0.6970 - val_accuracy: 0.7045  <-- Learning rate reduced (Plateau)
Epoch 4/10
687/687 [==============================] - 21s 30ms/step - loss: 0.4912 - accuracy: 0.8062 - val_loss: 0.7302 - val_accuracy: 0.6983  <-- EARLY STOPPING TRIGGERED
```

- **Early Stopping Trigger**: At Epoch 4, validation loss had increased for 2 consecutive epochs ($0.6738 \rightarrow 0.6970 \rightarrow 0.7302$).
- **Restoration**: `EarlyStopping(restore_best_weights=True)` automatically rolled back the network to the optimal Epoch 2 weights (`val_loss: 0.6738`, `val_accuracy: 0.7107`).

---

### Test Set Evaluation Metrics

Evaluated on **5,496 unseen test samples**:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Negative** | **74.5%** | 64.5% | **69.1%** | 1,556 |
| **Neutral** | 66.2% | **69.6%** | **67.9%** | 2,223 |
| **Positive** | **74.9%** | **78.9%** | **76.8%** | 1,717 |
| **Overall Accuracy** | — | — | **71.1%** | 5,496 |
| **Macro Average** | **71.8%** | **71.0%** | **71.3%** | 5,496 |
| **Weighted Average** | **71.3%** | **71.1%** | **71.0%** | 5,496 |

---

### Confusion Matrix & Classification Analysis

The generated confusion matrix is persisted at `results/figures/bilstm_confusion_matrix.png`:

```text
                Predicted Negative    Predicted Neutral    Predicted Positive
Actual Negative       1,004 (64.5%)         441 (28.3%)          111 (7.1%)
Actual Neutral          243 (10.9%)       1,548 (69.6%)          432 (19.4%)
Actual Positive         101 ( 5.9%)         261 (15.2%)        1,355 (78.9%)
```

#### Diagnostic Findings:
1. **Positive Sentiment Precision & Recall**: Achieves the highest recall (78.9%) and F1-score (76.8%). Positive tweets contain strong, unambiguous polarity words (*"love"*, *"amazing"*, *"thanks"*).
2. **Negative vs. Neutral Overlap**: The primary source of error occurs between Negative and Neutral classes (441 negative samples predicted as neutral). In informal tweets, mild disappointment or passive complaints often lack strong negative adjectives, causing the model to lean towards neutral.
3. **Low Cross-Polarity Confusion**: Very few samples are confused between opposite extremes (only 7.1% of negative tweets predicted positive, and 5.9% of positive tweets predicted negative).

---

## 🔌 Inference Engine & REST API

### Vectorized Prediction Pipeline

The inference pipeline (`src/inference/predictor.py`) executes high-efficiency batch tensor inference:

```text
Raw Text Input: "The concert was thrilling and awesome!"
     │
     ▼
[Step 1: clean_tweet()]
Regex clean: "the concert was thrilling and awesome"
     │
     ▼
[Step 2: tokenizer.texts_to_sequences()]
Sequence IDs: [1, 542, 18, 1289, 4, 312]
     │
     ▼
[Step 3: pad_sequences(maxlen=50, padding='post')]
Padded Vector: [1, 542, 18, 1289, 4, 312, 0, 0, 0, ..., 0]  (Shape: 1, 50)
     │
     ▼
[Step 4: model.predict()]
Probability Tensor: [[0.0031, 0.0215, 0.9754]]
     │
     ▼
[Step 5: argmax & label mapping]
Result: sentiment="positive", confidence=0.9754
```

---

### API Endpoints Reference

Base URL: `http://127.0.0.1:8009`

#### 1. System Healthcheck
- **Endpoint**: `GET /api/health`
- **Description**: Verifies API availability and validates that model weights are loaded in memory.
- **Response**:
```json
{
  "status": "healthy",
  "model_ready": true,
  "model_name": "BiLSTM"
}
```

---

#### 2. Single Tweet Sentiment Prediction
- **Endpoint**: `POST /api/predict`
- **Request Body**:
```json
{
  "text": "Flight was delayed 4 hours, lost my luggage, absolutely terrible experience."
}
```
- **Response (`200 OK`)**:
```json
{
  "text": "Flight was delayed 4 hours, lost my luggage, absolutely terrible experience.",
  "cleaned_text": "flight was delayed 4 hours lost my luggage absolutely terrible experience",
  "sentiment": "negative",
  "confidence": 0.9842,
  "probabilities": {
    "negative": 0.9842,
    "neutral": 0.0121,
    "positive": 0.0037
  },
  "model_name": "BiLSTM"
}
```

---

#### 3. Batch Tweet Sentiment Prediction
- **Endpoint**: `POST /api/predict/batch`
- **Request Body**:
```json
{
  "texts": [
    "Thank you so much for the quick help, you made my day!",
    "Flight delayed by 5 hours, terrible service.",
    "Our team meeting is scheduled for 10:00 AM tomorrow."
  ]
}
```
- **Response (`200 OK`)**:
```json
{
  "predictions": [
    {
      "text": "Thank you so much for the quick help, you made my day!",
      "cleaned_text": "thank you so much for the quick help you made my day",
      "sentiment": "positive",
      "confidence": 0.9814,
      "probabilities": { "negative": 0.0021, "neutral": 0.0165, "positive": 0.9814 },
      "model_name": "BiLSTM"
    },
    {
      "text": "Flight delayed by 5 hours, terrible service.",
      "cleaned_text": "flight delayed by 5 hours terrible service",
      "sentiment": "negative",
      "confidence": 0.9687,
      "probabilities": { "negative": 0.9687, "neutral": 0.0271, "positive": 0.0042 },
      "model_name": "BiLSTM"
    },
    {
      "text": "Our team meeting is scheduled for 10:00 AM tomorrow.",
      "cleaned_text": "our team meeting is scheduled for 1000 am tomorrow",
      "sentiment": "neutral",
      "confidence": 0.7102,
      "probabilities": { "negative": 0.1041, "neutral": 0.7102, "positive": 0.1857 },
      "model_name": "BiLSTM"
    }
  ],
  "count": 3,
  "model_name": "BiLSTM"
}
```

---

#### 4. Diagnostic Figures Index
- **Endpoint**: `GET /api/figures`
- **Description**: Returns dynamic list of generated EDA and evaluation chart filepaths.

---

### Live cURL & Python Client Examples

#### Testing via cURL

```bash
# Single prediction
curl -X POST http://127.0.0.1:8009/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely love this new update! Everything works so smoothly."}'

# Batch prediction
curl -X POST http://127.0.0.1:8009/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great phone!", "Battery died quickly.", "Just arrived home."]}'
```

#### Testing via Python

```python
import requests

API_URL = "http://127.0.0.1:8009/api/predict"

payload = {
    "text": "Incredible performance, could not be happier with the results!"
}

response = requests.post(API_URL, json=payload)
data = response.json()

print(f"Sentiment  : {data['sentiment'].upper()}")
print(f"Confidence : {data['confidence'] * 100:.2f}%")
print(f"Probs      : {data['probabilities']}")
```

---

## 💻 Frontend Web Dashboard

The web dashboard is delivered directly via FastAPI at `http://127.0.0.1:8009`.

### UI/UX Design Philosophy
- **Pure Flat-White Aesthetic**: Background and cards use clean `#ffffff`, crisp 1px borders (`#e5e7eb`), and high-contrast typography.
- **Strictly Zero Shadows**: Every CSS element uses `box-shadow: none !important` with zero gray gradient shading.
- **Single-Screen Viewport Lock**: Main dashboard container is locked to `100vh` with `overflow: hidden`, ensuring the user never has to scroll to view inputs, action buttons, confidence bars, and clean text displays.

### Features & Dashboard Capabilities
1. **Single Predict Tab**:
   - Multi-row textarea for typing or pasting tweets.
   - Character counter indicator.
   - Interactive badge displaying predicted sentiment (`POSITIVE`, `NEUTRAL`, `NEGATIVE`).
   - Detailed animated percentage bars for Negative, Neutral, and Positive probabilities.
   - Normalized text visualization displaying exact regex cleaning results.
2. **Batch Predict Tab**:
   - Large multi-line input box accepting one tweet per line.
   - Clean results table displaying Original Tweet, Cleaned Text, Sentiment, and Confidence Score.
3. **Dataset EDA Gallery Tab**:
   - Built-in modal viewer displaying all 6 EDA charts and confusion matrix heatmaps.

---

## 🛠️ Installation, Setup & Reproduction Guide

### Prerequisites
- **Operating System**: macOS, Linux, or Windows (WSL recommended).
- **Python**: Version `3.9` through `3.11`.
- **Package Manager**: `pip`.

### Step-by-Step Execution Commands

#### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/twitter-sentiment-analysis-bilstm.git
cd twitter-sentiment-analysis-bilstm
```

#### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

#### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Run Exploratory Data Analysis (EDA)
```bash
python notebooks/eda.py
```
*Generates 6 figures saved directly to `results/figures/`.*

#### 5. Train the BiLSTM Model from Scratch
```bash
python -m src.training.train --epochs 10 --batch-size 32
```
*Executes stratified splitting, tokenization, training loop, early stopping, and checkpoint saving to `models/bilstm_model.keras`.*

#### 6. Evaluate Model on Test Set
```bash
python -m src.evaluation.evaluate
```
*Computes test set accuracy, outputs `results/metrics/bilstm_report.txt`, and generates `results/figures/bilstm_confusion_matrix.png`.*

#### 7. Launch the FastAPI Server & Web Dashboard
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8009 --reload
```

- **Web Dashboard**: Open [http://127.0.0.1:8009](http://127.0.0.1:8009) in your web browser.
- **Interactive Swagger Documentation**: Open [http://127.0.0.1:8009/docs](http://127.0.0.1:8009/docs).
- **ReDoc Documentation**: Open [http://127.0.0.1:8009/redoc](http://127.0.0.1:8009/redoc).

---

## 🚀 Production Considerations & Future Roadmap

1. **Sub-millisecond Latency**: Current single-prediction inference runs in ~8-15ms on CPU. For higher throughput, batch requests should be prioritized via `POST /api/predict/batch`.
2. **Model Quantization**: Exporting `bilstm_model.keras` to TensorFlow Lite (TFLite) or ONNX format can reduce memory footprint by 75% with negligible accuracy trade-off.
3. **Containerization**: A `Dockerfile` can package the application for containerized deployment across AWS ECS, Google Cloud Run, or Kubernetes.
4. **Out-of-Vocabulary (OOV) Resilience**: The tokenizer uses a fixed vocabulary of the top 5,000 words. Unseen words are mapped to index 0 (padding/ignored), preventing inference crashes on rare slang or emerging terms.

---

## 📄 License

Distributed under the **MIT License**. Free for educational, academic, and commercial use.
