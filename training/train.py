"""
train.py — SageMaker training script

SageMaker runs this script inside a managed ml.m4.xlarge instance.
When SageMaker starts a training job it:
  1. Spins up a machine with the XGBoost container
  2. Downloads your features.csv from S3 to /opt/ml/input/data/train/
  3. Runs this script
  4. Saves whatever is in /opt/ml/model/ back to S3 as model.tar.gz
  5. Shuts the machine down

You pay only for the minutes the machine runs.
Everything SageMaker needs to know comes through environment variables
and the fixed directory paths below — that is the SageMaker convention.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

# ── SageMaker fixed paths ─────────────────────────────────────────────────────
# SageMaker always puts training data here regardless of what you named
# your S3 channel. The channel name becomes the folder name.
# We named our channel "train" in the training job config so the data
# lands at /opt/ml/input/data/train/
INPUT_DIR  = Path(os.getenv("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))

# SageMaker reads model artifacts from here after training finishes.
# Everything you save here gets packaged into model.tar.gz and uploaded to S3.
MODEL_DIR  = Path(os.getenv("SM_MODEL_DIR", "/opt/ml/model"))

# SageMaker writes training metrics and logs here.
OUTPUT_DIR = Path(os.getenv("SM_OUTPUT_DATA_DIR", "/opt/ml/output"))

# ── hyperparameters ───────────────────────────────────────────────────────────
# SageMaker passes hyperparameters as environment variables prefixed with
# SM_HP_. We read them here with fallback defaults so the script also
# works when run locally for testing.
MAX_DEPTH         = int(os.getenv("SM_HP_MAX_DEPTH", "6"))
ETA               = float(os.getenv("SM_HP_ETA", "0.1"))
NUM_ROUND         = int(os.getenv("SM_HP_NUM_ROUND", "200"))
SUBSAMPLE         = float(os.getenv("SM_HP_SUBSAMPLE", "0.8"))
COLSAMPLE_BYTREE  = float(os.getenv("SM_HP_COLSAMPLE_BYTREE", "0.8"))
SCALE_POS_WEIGHT  = float(os.getenv("SM_HP_SCALE_POS_WEIGHT", "28"))

# The target column name — must match what feature_engineering.py used
TARGET_COL = "isFraud"


def load_data() -> tuple:
    """
    Loads features.csv from the SageMaker input directory.

    SageMaker downloads your S3 data to INPUT_DIR before running this script.
    We find the CSV file, load it, and split into features and target.

    We use an 80/20 train/validation split so we can evaluate the model
    on held-out data during training and log the AUC metric.
    SageMaker reads the logged AUC and displays it in the console
    and uses it for the evaluation gate in the Airflow DAG.

    Returns:
        X_train, X_val, y_train, y_val as numpy arrays
    """
    print("Loading training data...")

    # find the CSV file in the input directory
    csv_files = list(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {INPUT_DIR}. "
            "Check that feature_engineering.py ran successfully "
            "and features.csv was uploaded to S3."
        )

    data_path = csv_files[0]
    print(f"  Found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"  Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"  Fraud rate: {df[TARGET_COL].mean():.2%}")

    # separate features from target
    # isFraud is always the last column (we put it there in feature_engineering.py)
    X = df.drop(columns=[TARGET_COL]).values
    y = df[TARGET_COL].values

    # 80/20 split — stratify=y ensures both splits have the same fraud rate
    # without stratify, random chance could put most fraud cases in one split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"  Train: {X_train.shape[0]:,} rows")
    print(f"  Val  : {X_val.shape[0]:,} rows")
    print(f"  Features: {X_train.shape[1]}")

    return X_train, X_val, y_train, y_val


def train_model(X_train, X_val, y_train, y_val) -> xgb.Booster:
    """
    Trains an XGBoost model on the training data.

    XGBoost is the right choice here because:
    - It handles mixed feature types (numerical + encoded categoricals) well
    - It is robust to the remaining null values we filled with -999
    - scale_pos_weight handles class imbalance directly
    - It trains fast on tabular data compared to neural networks
    - It produces probability scores (needed for fraud threshold tuning)

    We use early stopping — if the validation AUC does not improve for
    20 rounds, training stops early to prevent overfitting.
    This is more reliable than a fixed num_round.

    SageMaker captures any line printed in the format:
    [metric_name]: value
    and logs it as a training metric. We use this to log AUC so
    the Airflow evaluation task can read it later.
    """
    print("\nTraining XGBoost model...")
    print(f"  max_depth       : {MAX_DEPTH}")
    print(f"  eta             : {ETA}")
    print(f"  num_round       : {NUM_ROUND}")
    print(f"  scale_pos_weight: {SCALE_POS_WEIGHT}")

    # convert to XGBoost's internal DMatrix format
    # DMatrix is more memory-efficient than numpy arrays for large datasets
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    params = {
        "objective":        "binary:logistic",
        # binary:logistic outputs fraud probability (0-1)
        # which is exactly what the consumer needs for threshold comparison

        "eval_metric":      "auc",
        # AUC (Area Under ROC Curve) is the right metric for imbalanced
        # classification. Accuracy would be misleading because a model
        # that always predicts "not fraud" gets 96.5% accuracy.
        # AUC measures how well the model separates fraud from non-fraud.

        "max_depth":        MAX_DEPTH,
        "eta":              ETA,
        # eta is the learning rate — how much each tree contributes.
        # Lower eta + more rounds = better generalization but slower training.

        "subsample":        SUBSAMPLE,
        # use 80% of rows per tree — reduces overfitting

        "colsample_bytree": COLSAMPLE_BYTREE,
        # use 80% of features per tree — reduces overfitting

        "scale_pos_weight": SCALE_POS_WEIGHT,
        # tells XGBoost to treat each fraud case as 28x more important
        # this compensates for the 96.5/3.5 class imbalance

        "seed":             42,
        "verbosity":        1,
    }

    # evals list tells XGBoost to evaluate on both train and val each round
    # eval_results captures these metrics so we can log them
    eval_results = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_ROUND,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=20,
        # early stopping stops if val AUC doesn't improve for 20 rounds
        evals_result=eval_results,
        verbose_eval=10,
        # print metrics every 10 rounds to avoid flooding the logs
    )

    # get the best validation AUC achieved during training
    best_auc = max(eval_results["validation"]["auc"])

    # SageMaker captures this exact format as a training metric
    # the Airflow DAG reads this value for the evaluation gate
    print(f"\nvalidation:auc: {best_auc:.4f}")
    print(f"Best round: {model.best_iteration}")

    return model, best_auc, eval_results


def evaluate_model(model: xgb.Booster, X_val, y_val, best_auc: float):
    """
    Runs final evaluation on the validation set and prints a full report.

    We use threshold 0.5 for the classification report but note that
    in production the Kafka consumer uses a configurable threshold
    (FRAUD_THRESHOLD env var) which can be tuned based on how many
    false positives are acceptable.
    """
    print("\nFinal model evaluation...")

    dval = xgb.DMatrix(X_val)
    y_pred_proba = model.predict(dval)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_val, y_pred_proba)
    print(f"  Validation AUC  : {auc:.4f}")
    print(f"  Best training AUC: {best_auc:.4f}")
    print(f"\nClassification Report (threshold=0.5):")
    print(classification_report(y_val, y_pred, target_names=["Not Fraud", "Fraud"]))

    # get feature importance — top 10 most important features
    importance = model.get_score(importance_type="gain")
    if importance:
        sorted_importance = sorted(
            importance.items(), key=lambda x: x[1], reverse=True
        )
        print("\nTop 10 most important features:")
        for feat, score in sorted_importance[:10]:
            print(f"  {feat}: {score:.2f}")

    return auc


def save_model(model: xgb.Booster, auc: float, feature_cols: list):
    """
    Saves the trained model and metadata to the SageMaker model directory.

    SageMaker packages everything in MODEL_DIR into model.tar.gz
    and uploads it to S3 automatically after training finishes.

    We save:
    - model.xgb: the XGBoost model in binary format
    - model_metadata.json: AUC score and feature column order

    The feature column order is critical — the model was trained on
    features in a specific order. When the consumer calls the endpoint,
    it must send features in the exact same order or predictions are wrong.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # save the XGBoost model
    model_path = MODEL_DIR / "model.xgb"
    model.save_model(str(model_path))
    print(f"\nModel saved to: {model_path}")

    # save metadata alongside the model
    metadata = {
        "auc": auc,
        "feature_cols": feature_cols,
        "threshold": 0.5,
        "model_type": "xgboost",
        "scale_pos_weight": SCALE_POS_WEIGHT,
    }
    metadata_path = MODEL_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")


def run():
    """
    Main entry point — called by SageMaker when the training job starts.
    """
    print("=" * 60)
    print("  Fraud Detection Model Training")
    print("=" * 60)

    X_train, X_val, y_train, y_val = load_data()

    model, best_auc, eval_results = train_model(
        X_train, X_val, y_train, y_val
    )

    auc = evaluate_model(model, X_val, y_val, best_auc)

    # load feature column names from the input data
    csv_files = list(INPUT_DIR.glob("*.csv"))
    df_cols = pd.read_csv(csv_files[0], nrows=1).columns.tolist()
    feature_cols = [c for c in df_cols if c != "isFraud"]

    save_model(model, auc, feature_cols)

    print("\n" + "=" * 60)
    print("  Training complete.")
    print(f"  Final AUC: {auc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run()