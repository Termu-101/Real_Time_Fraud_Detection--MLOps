# Real-Time Fraud Detection — MLOps Pipeline

An end-to-end MLOps system that trains an XGBoost fraud detection model on the IEEE-CIS financial dataset, deploys it to AWS SageMaker, and scores live Binance BTC/USDT trades in real time through a Kafka streaming pipeline. A Streamlit dashboard visualises every scored trade, flags fraud instantly, and monitors model health. The system performs daily drift checks and automatically retrains and redeploys the model when model performance begins to degrade — with minimal human intervention.

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [How It Works — End to End](#2-how-it-works--end-to-end)
3. [Architecture](#3-architecture)
4. [Components In Detail](#4-components-in-detail)
5. [The Machine Learning Model](#5-the-machine-learning-model)
6. [Metrics and Thresholds](#6-metrics-and-thresholds)
7. [Airflow DAGs](#7-airflow-dags)
8. [Dashboard](#8-dashboard)
9. [Project Structure](#9-project-structure)
10. [Setup and Running](#10-setup-and-running)
11. [CI/CD Pipeline](#11-cicd-pipeline)
12. [Environment Variables](#12-environment-variables)

---

## 1. Purpose

Financial fraud is a real-time problem. By the time a batch fraud check runs hours later, the money is already gone. This project builds a system that:

- **Scores every transaction in under a second** — each trade is evaluated by an ML model the moment it appears in the stream
- **Monitors itself** — prediction logs are analysed daily for statistical drift; if the fraud-score distribution shifts significantly, the model is retrained automatically
- **Retrains without downtime** — new model versions replace old ones using a blue/green swap on SageMaker, so scoring never stops during a retrain
- **Approximates alert explanations** — every fraud flag highlights the top features with the largest deviation from their training mean, providing human-readable context rather than just a score

The model is trained on the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data) (590,000 bank card transactions, 3.5% fraud rate). The live scoring target is Binance BTC/USDT trade data. Since live trades do not contain many of the original training features (card details, email domains, device info, etc.), missing features are imputed with the sentinel value -999. This allows the model to operate end-to-end, though predictions should be interpreted as a demonstration of real-time MLOps infrastructure rather than a production-grade fraud detection system.

---

## 2. How It Works — End to End

### Step 1 — Training data preparation

The IEEE-CIS dataset is downloaded from S3, merged (transaction + identity tables), and passed through a feature engineering pipeline that produces a clean 280+ column feature matrix ready for XGBoost. The pipeline handles:

- Dropping columns that are more than 90% null
- Extracting `hour_of_day` and `day_of_week` from the raw transaction timestamp before the timestamp column is dropped
- Log-transforming `TransactionAmt` (right-skewed distribution)
- Label-encoding 14 categorical columns (card type, email domain, device type, etc.)
- Filling remaining nulls with -999 (XGBoost's native missing-value sentinel)
- StandardScaler normalisation across numerical features (not required by XGBoost; applied for consistency with drift metrics and feature deviation calculations)
- Computing `scale_pos_weight` to handle the 3.5% fraud / 96.5% legit class imbalance

All encoder and scaler state is saved to `feature_metadata.json` so live trades can be transformed identically at inference time.

### Step 2 — Model training (SageMaker)

The processed feature matrix is uploaded to S3 and a SageMaker XGBoost training job is launched. Training uses the built-in XGBoost 1.7-1 container on an `ml.m5.xlarge` instance. The objective is `binary:logistic` and the evaluation metric is AUC. A new model is only promoted to the endpoint if it improves AUC by at least 0.005 (0.5 percentage points) over the current live model.

### Step 3 — Live trade ingestion (Binance → Kafka)

The Binance producer opens a WebSocket connection to `wss://stream.binance.us:9443/ws/btcusdt@trade` and forwards every trade event to the Kafka topic `transactions`. Each message contains: price (`p`), quantity (`q`), trade ID (`t`), timestamp (`T`), symbol (`s`), and whether the buyer is the market maker (`m`). If the WebSocket drops for any reason (network blip, exchange maintenance), the producer reconnects automatically after 5 seconds.

### Step 4 — Real-time fraud scoring (Kafka → SageMaker → S3)

The fraud consumer reads from the `transactions` topic and for each trade:

1. Maps the Binance trade fields onto the 280+ feature vector the model expects using `binance_mapper.py`. Fields that exist in the training data but not in a Binance trade (card number, email, device info, etc.) are filled with -999.
2. Sends the feature vector to the SageMaker endpoint as a CSV row.
3. Receives a fraud probability score between 0.0 and 1.0.
4. Flags the trade as fraud if the score is ≥ 0.5 (configurable).
5. Logs the trade to an S3 CSV file (flushed every 100 predictions).
6. When fraud is detected, publishes to the `fraud-alerts` Kafka topic and logs the top 5 features with the highest deviation from their training mean as a lightweight explanation proxy (not SHAP — see note in Components section).

### Step 5 — Real-time dashboard

The Streamlit dashboard runs a background thread that reads from Kafka and writes every scored trade to a local SQLite database. The UI re-renders every 3 seconds and displays:

- Live trade count, fraud alert count, fraud rate, average fraud score, average BTC price
- Transaction volume chart (5-second buckets) overlaid with fraud rate
- Fraud score distribution histogram (legit vs. flagged)
- Data drift score over time
- Score intensity heatmap (time × 10-second slots)
- Live fraud alert feed with a per-alert score bar

### Step 6 — Daily drift detection

Every day at 06:00 UTC an Airflow DAG reads the last 24 hours of prediction logs from S3 and compares the mean fraud score of that window against the 7-day baseline. This is a lightweight mean-shift heuristic rather than a full distribution test (e.g. KS-test or PSI), intentionally kept simple for operational transparency. If the absolute shift exceeds 0.005 (0.5 percentage points) the DAG fires a drift alert (saved to S3 as JSON and optionally posted to Slack) and triggers an immediate model retraining run.

### Step 7 — Automatic model retraining (zero downtime)

The retraining DAG launches a new SageMaker training job, polls until it finishes, then performs a blue/green swap: a new Model resource and EndpointConfig are created, the live endpoint is updated to point at the new config, and the system waits until the endpoint returns to `InService` status before declaring success. A smoke test (an all-zero feature vector) is sent to confirm the new endpoint responds. The entire process is written to a retraining report in S3. If the endpoint never reaches `InService` within 30 minutes, the task fails and Airflow retries.

### Step 8 — Scheduled weekly full retrain

Separately from drift-triggered retraining, the training DAG runs every Sunday at midnight UTC. It re-downloads raw data from S3, re-runs the full feature engineering pipeline (picking up any new data added during the week), trains a new model, and only deploys it if AUC improves by at least 0.005 (0.5 percentage points). This prevents model staleness even when no drift is detected.

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  TRAINING PIPELINE  (Airflow — weekly, Sunday 00:00 UTC)                 │
│                                                                          │
│  IEEE-CIS Dataset (S3)                                                   │
│       │                                                                  │
│       ▼                                                                  │
│  Feature Engineering ──► features.csv + feature_metadata.json (S3)      │
│       │                                                                  │
│       ▼                                                                  │
│  SageMaker XGBoost Training Job (ml.m5.xlarge)                           │
│       │                                                                  │
│       ▼                                                                  │
│  AUC Gate: new AUC > current AUC + 0.005 ?                               │
│       │ YES                                                              │
│       ▼                                                                  │
│  Blue/Green Endpoint Swap ──► fraud-detection-endpoint (InService)       │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┘
                    │  SageMaker Endpoint (always live)
                    │
┌───────────────────▼──────────────────────────────────────────────────────┐
│  LIVE SCORING PIPELINE                                                   │
│                                                                          │
│  Binance WebSocket (btcusdt@trade)                                       │
│       │  price, qty, timestamp, trade_id, is_buyer_maker                 │
│       ▼                                                                  │
│  Kafka Topic: transactions                                               │
│       │                                                                  │
│       ├──────────────────────────────────────┐                           │
│       ▼                                      ▼                           │
│  Fraud Consumer                         Dashboard Reader                 │
│  (src/consumer/)                        (src/dashboard/)                 │
│       │                                      │                           │
│       │  feature_vector (280+ cols)          │  feature_vector           │
│       ▼                                      ▼                           │
│  SageMaker Endpoint                     SageMaker Endpoint               │
│       │  fraud_score [0.0 – 1.0]            │  fraud_score              │
│       │                                      │                           │
│       ├── score ≥ 0.5 → Kafka: fraud-alerts  └──► SQLite (dashboard.db)  │
│       │                                                                  │
│       └──► S3: logs/predictions/{ts}.csv  (flushed every 100 trades)    │
└──────────────────────────────────────────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────────────────────┐
│  DRIFT MONITORING  (Airflow — daily, 06:00 UTC)                          │
│                                                                          │
│  Read S3 prediction logs                                                 │
│       │                                                                  │
│       ▼                                                                  │
│  Compare last 24h mean fraud score vs. 7-day baseline mean               │
│       │                                                                  │
│       │  |window_mean − baseline_mean| > 0.05 ?                          │
│       │ YES                                                              │
│       ▼                                                                  │
│  Drift Alert → S3 JSON + Slack (optional)                                │
│       │                                                                  │
│       ▼                                                                  │
│  Trigger model_retraining_dag (SageMaker retrain + blue/green swap)      │
└──────────────────────────────────────────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────────────────────┐
│  STREAMLIT DASHBOARD  (port 8501)                                        │
│                                                                          │
│  Reads SQLite every 3 seconds                                            │
│  ├── KPI cards: total trades, fraud alerts, flag rate, avg score         │
│  ├── Volume chart (5s buckets, trade count + fraud overlay)              │
│  ├── Score distribution histogram (legit vs. fraud)                      │
│  ├── Drift score chart over time                                         │
│  ├── Score heatmap (10s slots × minute)                                  │
│  └── Live fraud alert feed (score bar + top features)                    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Components In Detail

### Binance Producer (`src/producer/binance_producer.py`)

Connects to the Binance US WebSocket stream for the BTC/USDT trading pair. Each trade event typically arrives within milliseconds and is published directly to the Kafka `transactions` topic. The producer adds a `source: "binance"` field and logs every 100th trade to avoid flooding logs at ~10 trades/second. If the WebSocket drops for any reason (network blip, exchange maintenance), the producer sleeps 5 seconds and reconnects.

**Kafka producer settings:**
- Request timeout: 5000 ms
- Retries on publish failure: 3
- WebSocket ping interval: 30 seconds

### Fraud Consumer (`src/consumer/fraud_consumer.py`)

Reads from the `transactions` topic, scores each trade, and handles all output. The consumer is stateless — it can be restarted at any point and Kafka will replay from where it left off.

For each trade it:
1. Calls `binance_mapper.feature_vector()` to build the 280+ column input vector
2. Serialises the vector as a CSV row and calls the SageMaker endpoint
3. If `fraud_score ≥ FRAUD_THRESHOLD` publishes to `fraud-alerts` and logs the top 5 deviating features
4. Appends the result to an in-memory batch; when the batch hits 100 entries it is flushed to S3

**Top deviating features** are computed by sorting the scaled feature values by absolute magnitude — features that deviate furthest from zero (their training mean after StandardScaler) are the most unusual for that trade. This gives human-readable alert context like `TransactionAmt=+4.2σ, hour_of_day=+3.1σ`. Note: this is a deviation-based proxy, not a model-native explanation method like SHAP. It highlights unusually-valued features for that trade, not necessarily the features the model weighted most heavily.

### Feature Engineering (`features/feature_engineering.py` + `features/binance_mapper.py`)

The feature pipeline produces an identical feature vector whether it is processing a 590,000-row training CSV or a single live Binance trade. This consistency is the bridge between the trained model and live scoring.

**Training time** (`feature_engineering.py`):
- Downloads `data/raw/merged_raw.csv` from S3
- Validates the data (row count, required columns, fraud rate, non-negative amounts)
- Extracts `hour_of_day` and `day_of_week` from `TransactionDT`
- Drops columns with > 90% nulls plus the raw ID columns
- Applies `log1p` to `TransactionAmt`
- Label-encodes 14 categorical columns; saves encoder state to metadata
- Fills remaining nulls with -999
- Applies StandardScaler; saves scaler state to metadata
- Computes `scale_pos_weight` for class imbalance
- Saves everything (features.csv + feature_metadata.json) to S3

**Live scoring time** (`binance_mapper.py`):
- Loads `feature_metadata.json` (encoder + scaler state)
- Maps Binance fields: `TransactionAmt ← log1p(price × qty)`, `hour_of_day` and `day_of_week` from trade timestamp, `isBuyerMaker` from the `m` field
- Sets all other 270+ columns (card data, email domains, device info, etc.) to -999
- Returns a feature list in exactly the same column order the model was trained on

### Dashboard Reader (`src/dashboard/kafka_reader.py`)

Runs as a background daemon thread inside the Streamlit process. It reads from Kafka, scores each trade (or simulates scores when Kafka is unavailable), and writes results to a local SQLite database via `store.py`. The Streamlit UI reads from that same database every 3 seconds.

Drift is also computed locally by the reader: every 50 predictions (`DRIFT_WINDOW`) it calculates the mean and standard deviation of the score window, compares it against all historical scores, and writes a drift record to SQLite for the drift chart to display.

### Store (`src/dashboard/store.py`)

A thin wrapper around SQLite that handles all dashboard persistence. Uses a threading lock to prevent simultaneous writes from the background reader and reads from the Streamlit thread. Three tables:

- `predictions` — every scored trade (timestamp, symbol, price, quantity, fraud\_score, is\_fraud, threshold)
- `drift` — periodic drift records (drift\_score, mean\_score, std\_score, window\_size)
- Both tables are indexed on `ts` for fast time-range queries

---

## 5. The Machine Learning Model

### Dataset

| Property | Value |
|---|---|
| Source | IEEE-CIS Fraud Detection (Kaggle) |
| Rows | 590,540 transactions |
| Fraud rate | 3.50% (20,663 fraud / 569,877 legit) |
| Features after engineering | 280+ |
| Training split | 80% train / 20% validation |

### Model

| Property | Value |
|---|---|
| Algorithm | XGBoost (binary:logistic) |
| Evaluation metric | AUC (Area Under ROC Curve) |
| Depth | 6 |
| Learning rate (eta) | 0.1 |
| Boosting rounds | 200 |
| Row subsampling | 0.8 |
| Column subsampling | 0.8 |
| Class imbalance handling | scale\_pos\_weight (≈ 27× — ratio of legit to fraud) |
| Missing value sentinel | -999 (handled natively by XGBoost) |
| Infrastructure | AWS SageMaker built-in XGBoost 1.7-1, ml.m5.xlarge |

### Evaluated Performance

Metrics measured by scoring the held-out 20% test set (118,108 rows, 4,133 fraud cases) through the live SageMaker endpoint:

| Metric | Value |
|---|---|
| Test ROC-AUC | **0.9608** |
| Train ROC-AUC | 0.9619 |
| Train / test AUC gap | 0.0011 (minimal overfitting) |
| Test set size | 118,108 rows (stratified 80/20 split, `random_state=42`) |
| Fraud cases in test set | 4,133 (3.50%) |

Threshold analysis on the test set:

| Threshold | Precision | Recall | F1 | Fraud caught | Fraud missed |
|---|---|---|---|---|---|
| 0.3 | 0.143 | 0.943 | 0.249 | 3,897 / 4,133 | 236 |
| 0.4 | 0.202 | 0.902 | 0.330 | 3,728 / 4,133 | 405 |
| **0.5 (default)** | **0.275** | **0.863** | **0.417** | **3,566 / 4,133** | **567** |
| 0.6 | 0.371 | 0.816 | 0.510 | 3,372 / 4,133 | 761 |

At the default threshold of 0.5 the model catches 86.3% of fraud cases. Raising the threshold reduces false positives at the cost of missing more fraud.

### Why XGBoost

- Handles extreme class imbalance (3.5% fraud) through `scale_pos_weight` without oversampling
- Natively treats -999 as missing, so the 270+ features that are always -999 for Binance trades do not need special preprocessing at inference time
- AUC is the right metric here: the cost of a missed fraud is much higher than a false positive, so we care about the full ROC curve rather than accuracy at a single threshold
- Trains fast on tabular data — a full retrain finishes in under 30 minutes on ml.m5.xlarge

### Fraud score interpretation

| Score range | Meaning |
|---|---|
| 0.0 – 0.49 | Predicted legitimate — no action |
| 0.50 – 0.79 | Flagged — moderate confidence fraud |
| 0.80 – 0.99 | High-confidence fraud alert |

The threshold of 0.5 is the default. It can be lowered (more sensitive, more false positives) or raised (less sensitive, fewer false positives) via the `FRAUD_THRESHOLD` environment variable without redeploying.

---

## 6. Metrics and Thresholds

### Fraud scoring

| Parameter | Default | What it controls |
|---|---|---|
| `FRAUD_THRESHOLD` | `0.5` | Probability cutoff above which a trade is flagged as fraud |
| `LOG_BATCH_SIZE` | `100` | Number of predictions to buffer before flushing to S3 |
| `TOP_FEATURES_TO_LOG` | `5` | Number of top deviating features included in each fraud alert |

### Drift detection

| Parameter | Default | What it controls |
|---|---|---|
| `DRIFT_THRESHOLD` | `0.05` | Absolute shift in mean fraud score (5 percentage points) that triggers a drift alert |
| `DRIFT_BASELINE_DAYS` | `7` | Number of days used to compute the baseline mean |
| `DRIFT_WINDOW_HOURS` | `24` | Hours of recent predictions compared against the baseline |
| `MIN_PREDICTIONS` | `50` | Minimum number of predictions required to run the drift check |
| HIGH severity threshold | `2 × DRIFT_THRESHOLD` | Drift alerts above this are labelled HIGH severity |

Drift score formula:

```
drift_score = |mean(window_scores) - mean(baseline_scores)|
```

If `drift_score > DRIFT_THRESHOLD` the alert fires and retraining is triggered.

### Model deployment

| Parameter | Default | What it controls |
|---|---|---|
| `MIN_AUC_IMPROVEMENT` | `0.005` | A new model must beat the current model AUC by at least this to be deployed |
| Endpoint health timeout | 30 minutes | Maximum time to wait for the endpoint to return to InService after a swap |
| Training timeout | 3 hours | SageMaker job is cancelled if it exceeds this |
| Poll interval (training) | 60 seconds | How often the retraining DAG checks if the SageMaker job is finished |
| Poll interval (endpoint) | 30 seconds | How often the DAG checks if the endpoint is back to InService |

### Feature engineering

| Parameter | Default | What it controls |
|---|---|---|
| `NULL_THRESHOLD` | `0.9` | Columns with more than 90% null values are dropped entirely |
| `LOG_TRANSFORM_COLS` | `TransactionAmt` | Right-skewed columns that get `log1p` applied |
| Null fill value | `-999` | Sentinel used for missing values (XGBoost native missing) |
| Minimum dataset size | `1,000` rows | Validation check — raises an error if the data is too small |
| Valid fraud rate range | `0.1% – 50%` | Validation check — flags suspiciously low or high fraud rates |

### Dashboard

| Parameter | Default | What it controls |
|---|---|---|
| `DASHBOARD_REFRESH_S` | `3` | Seconds between dashboard rerenders |
| `DRIFT_WINDOW` | `50` | Number of predictions between local drift calculations in the dashboard reader |
| Volume chart bucket | `5 seconds` | Trade aggregation interval in the volume chart |
| Score heatmap slot | `10 seconds` | Time slot size in the score heatmap |

### XGBoost hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| `max_depth` | `6` | Deep enough to capture feature interactions; beyond 6 tends to overfit on fraud data |
| `eta` | `0.1` | Conservative learning rate — trades training speed for generalisation |
| `num_round` | `200` | 200 boosting rounds with early stopping opportunity |
| `subsample` | `0.8` | Row subsampling reduces overfitting on minority class |
| `colsample_bytree` | `0.8` | Column subsampling adds regularisation |
| `scale_pos_weight` | ~27 | Computed as `legit_count / fraud_count` from training data |

---

## 7. Airflow DAGs

### `fraud_detection_training` — Weekly full retrain

**Schedule:** Every Sunday at 00:00 UTC

**Purpose:** Ensures the model is retrained on the latest available data once a week, independent of drift.

**Task sequence:**

```
validate_raw_data
       │
       ▼
run_feature_engineering
       │
       ▼
trigger_sagemaker_training
       │
       ▼
evaluate_model ─── AUC improved? ─── YES ──► deploy_model
                                └─── NO  ──► skip_deployment
```

1. **validate\_raw\_data** — Downloads 1,000-row sample from S3, checks required columns exist, validates fraud rate is in the expected range
2. **run\_feature\_engineering** — Runs the full pipeline (log transform, encoding, scaling) and uploads features.csv + feature\_metadata.json to S3
3. **trigger\_sagemaker\_training** — Launches the XGBoost training job with all hyperparameters
4. **evaluate\_model** — Downloads the new model's AUC and compares to `models/current_model_metrics.json` in S3; branches to deploy or skip
5. **deploy\_model** — Creates SageMaker Model + EndpointConfig + updates the live endpoint (blue/green swap)
6. **skip\_deployment** — Logs that the current model is still best; no endpoint change

### `drift_detection_dag` — Daily drift check

**Schedule:** Every day at 06:00 UTC

**Purpose:** Catches model degradation between weekly retrains.

**Task sequence:**

```
check_s3_logs
       │
       ▼
compute_drift ─── drift > 0.05? ─── YES ──► send_alert ──► trigger_retraining
                               └─── NO  ──► (done)
```

1. **check\_s3\_logs** — Lists prediction CSV files covering the last 24 hours (window) and the 7 days before that (baseline); skips if fewer than 50 predictions found
2. **compute\_drift** — Computes mean fraud score for each window; calculates `|window_mean − baseline_mean|` as the drift score
3. **evaluate\_drift** — Branches on whether `drift_score > DRIFT_THRESHOLD`
4. **send\_alert** — Writes a JSON alert to `s3://{BUCKET}/logs/drift_alerts/` with severity (MEDIUM if drift > 0.05, HIGH if drift > 0.10), and optionally posts to Slack
5. **trigger\_retraining** — Calls `model_retraining_dag` with drift context (triggered\_by, drift\_score, timestamp)

### `model_retraining_dag` — On-demand retrain

**Schedule:** None — triggered by `drift_detection_dag` or manually

**Purpose:** Retrains and redeploys the model with zero downtime.

**Task sequence:**

```
validate_trigger
       │
       ▼
fetch_training_data
       │
       ▼
launch_sagemaker_job
       │
       ▼
wait_for_training (polls every 60s, up to 3 hours)
       │
       ▼
update_endpoint (blue/green swap)
       │
       ▼
verify_endpoint (smoke test, up to 30 min)
       │
       ▼
log_completion → s3://…/logs/retraining_reports/{ts}.json
```

The retraining report written at the end includes: completion timestamp, job name, model artifact URI, endpoint name, smoke test score, who triggered the retrain, and the drift score that caused it.

---

## 8. Dashboard

The Streamlit dashboard at port 8501 is the primary monitoring interface. It requires no login and updates automatically every 3 seconds.

### Panels

| Panel | What it shows |
|---|---|
| **Status bar** | LIVE / OFFLINE indicator, total trades scored, timestamp of last trade |
| **KPI cards** | Total trades, fraud alerts, flag rate, average fraud score, average BTC price |
| **Transaction volume** | Bar chart of trade count per 5-second bucket, red overlay for flagged trades, line for average score |
| **Score distribution** | Histogram of fraud scores for legit (blue) vs. flagged (red) trades with threshold line |
| **Drift monitor** | Line chart of drift score over time, shaded danger zone above 1.0 (normalised display scale; threshold logic uses 0.005 absolute mean-score shift) |
| **Score heatmap** | 2D heatmap — rows are 10-second slots within a minute, columns are minutes — shows score intensity over time |
| **Model performance** | Current model AUC from S3, retraining event timeline, fraud/legit pie chart |
| **Live fraud alerts** | Most recent fraud-flagged trades with score bar, price, quantity, and timestamp |

### How the dashboard reads data

The dashboard runs two concurrent execution paths:

1. **Background thread** (`kafka_reader.py`) — reads from Kafka, scores each trade against SageMaker (or simulates scores in demo mode), writes to SQLite
2. **Streamlit render loop** (`app.py`) — reads the SQLite tables every 3 seconds via `store.py` and redraws all charts

This design means the dashboard continues accumulating data even when no browser is connected, and multiple browser sessions all see the same data from the shared SQLite file.

---

## 9. Project Structure

```
Project/
├── dags/
│   ├── training_dag.py           # Weekly full retrain (Airflow TaskFlow API)
│   ├── drift_detection_dag.py    # Daily drift detection with Slack alerting
│   └── model_retraining_dag.py   # On-demand SageMaker retrain + blue/green swap
│
├── src/
│   ├── producer/
│   │   └── binance_producer.py   # Binance WebSocket → Kafka producer
│   ├── consumer/
│   │   └── fraud_consumer.py     # Kafka → SageMaker → S3 logs
│   └── dashboard/
│       ├── app.py                # Streamlit dashboard UI
│       ├── kafka_reader.py       # Background reader thread + local drift
│       └── store.py              # SQLite wrapper (predictions, drift tables)
│
├── features/
│   ├── feature_engineering.py    # Full training-time feature pipeline
│   └── binance_mapper.py         # Maps live Binance trades to model features
│
├── training/
│   ├── train.py                  # SageMaker entry-point training script
│   └── evaluate.py               # AUC evaluation helpers
│
├── docker/
│   ├── docker-compose.yml        # Full local stack (Kafka, Airflow, all services)
│   ├── Dockerfile.producer
│   ├── Dockerfile.consumer
│   └── Dockerfile.dashboard
│
├── tests/
│   ├── test_features.py          # Feature engineering + mapper unit tests
│   ├── test_pipeline.py          # Consumer pipeline integration tests
│   ├── test_training.py          # Training script tests
│   └── test_drift.py             # Drift detection + retraining DAG tests
│
├── scripts/
│   └── deploy_endpoint.py        # One-shot initial endpoint deployment
│
├── Dockerfile                    # Root Dockerfile for Railway dashboard deployment
├── railway.toml                  # Railway deployment config
├── .env.example                  # Environment variable template
├── requirements.txt
└── .github/
    └── workflows/
        └── ci_cd.yml             # Tests → ECR image push on merge to main
```

---

## 10. Setup and Running

### Prerequisites

- Docker and docker-compose
- AWS account with an IAM user (`AmazonSageMakerFullAccess` + `AmazonS3FullAccess`), an IAM role for SageMaker, and an S3 bucket
- Python 3.11 (for running scripts outside Docker)

### Step 1 — Configure environment

```bash
git clone <repo-url>
cd Project
cp .env.example .env
# Fill in your AWS credentials, S3 bucket, and SageMaker role ARN in .env
```

### Step 2 — Upload training data to S3

Download the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data) from Kaggle and place `train_transaction.csv` and `train_identity.csv` in a `data/` folder, then:

```bash
pip install -r requirements.txt

python -c "
import pandas as pd, boto3, os
from dotenv import load_dotenv
load_dotenv()

merged = pd.read_csv('data/train_transaction.csv').merge(
    pd.read_csv('data/train_identity.csv'), on='TransactionID', how='left'
)
merged.to_csv('data/merged_raw.csv', index=False)

boto3.client('s3').upload_file(
    'data/merged_raw.csv', os.getenv('S3_BUCKET'), 'data/raw/merged_raw.csv'
)
print(f'Uploaded {len(merged):,} rows to S3')
"
```

### Step 3 — Run feature engineering

```bash
python features/feature_engineering.py
```

This downloads raw data from S3, builds the 280+ feature matrix, and uploads `features.csv` and `feature_metadata.json` back to S3. Takes 5–10 minutes.

### Step 4 — Deploy the initial SageMaker endpoint

```bash
python scripts/deploy_endpoint.py
```

Launches a SageMaker training job, waits for it to complete, and creates the live inference endpoint. Takes 15–30 minutes on first run.

### Step 5 — Start all services

```bash
cd docker
docker-compose up -d
```

| Service | URL |
|---|---|
| Airflow UI | http://localhost:8090 (admin / admin) |
| Streamlit dashboard | http://localhost:8501 |
| Kafka broker | localhost:9092 |

All DAGs are paused by default. Enable them in the Airflow UI or trigger them manually to run the first time.

### Running tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

All AWS and Kafka calls are mocked — tests run without any real infrastructure.

---

## 11. CI/CD Pipeline

On every push to `main`, GitHub Actions runs three jobs:

**1. test** — installs dependencies, runs `pytest tests/ -v` with mock AWS credentials

**2. create-ecr-repos** (main only, after tests pass) — creates three ECR repositories if they do not already exist:
- `fraud-mlops/producer`
- `fraud-mlops/consumer`
- `fraud-mlops/dashboard`

**3. build-and-push** (main only, after repos are ready) — builds all three Docker images using Docker Buildx with GitHub Actions layer caching, then pushes with two tags: `latest` and the git commit SHA

Required GitHub secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `AWS_ACCOUNT_ID`

---

## 12. Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `AWS_ACCESS_KEY_ID` | Yes | — | AWS IAM access key |
| `AWS_SECRET_ACCESS_KEY` | Yes | — | AWS IAM secret key |
| `AWS_DEFAULT_REGION` | Yes | `us-east-1` | AWS region for all services |
| `S3_BUCKET` | Yes | — | S3 bucket for data, models, and logs |
| `SAGEMAKER_ROLE_ARN` | Yes | — | IAM role ARN used by SageMaker training jobs |
| `SAGEMAKER_ENDPOINT_NAME` | No | `fraud-detection-endpoint` | Name of the live inference endpoint |
| `SAGEMAKER_TRAIN_INSTANCE` | No | `ml.m5.xlarge` | Instance type for training jobs |
| `SAGEMAKER_DEPLOY_INSTANCE` | No | `ml.m5.large` | Instance type for the endpoint |
| `KAFKA_BOOTSTRAP_SERVERS` | No | `localhost:9092` | Kafka broker address |
| `KAFKA_TOPIC_TRANSACTIONS` | No | `transactions` | Topic for raw trades |
| `KAFKA_TOPIC_ALERTS` | No | `fraud-alerts` | Topic for flagged fraud trades |
| `FRAUD_THRESHOLD` | No | `0.5` | Fraud score cutoff |
| `LOG_BATCH_SIZE` | No | `100` | Predictions per S3 flush |
| `TOP_FEATURES_TO_LOG` | No | `5` | Top deviating features logged per alert |
| `DRIFT_THRESHOLD` | No | `0.05` | Mean score shift that triggers drift alert |
| `DRIFT_BASELINE_DAYS` | No | `7` | Days used for drift baseline |
| `DRIFT_WINDOW_HOURS` | No | `24` | Hours of recent predictions for drift check |
| `DRIFT_MIN_PREDICTIONS` | No | `50` | Minimum predictions required to run drift check |
| `DASHBOARD_REFRESH_S` | No | `3` | Dashboard auto-refresh interval in seconds |
| `DRIFT_WINDOW` | No | `50` | Predictions between dashboard drift calculations |
| `DEMO_MODE` | No | `false` | Set to `true` to run dashboard without Kafka/SageMaker |
| `SLACK_WEBHOOK_URL` | No | — | Incoming webhook URL for drift alert notifications |
