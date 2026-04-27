"""
evaluate.py — Standalone model evaluation script.

Run this anytime to get a full evaluation report on the model.
Unlike the Airflow evaluation gate which just compares AUC scores,
this script gives you the complete picture:
  - AUC at multiple thresholds
  - Precision, recall, F1 for each class
  - Confusion matrix
  - False positive and false negative rates
  - Threshold recommendation based on your tolerance for false positives

Run it with:
  python src/training/evaluate.py

It downloads the features.csv from S3, splits off a held-out test set,
loads the trained model from S3, and evaluates on the test set.
"""

import os
import json
import boto3
import joblib
import numpy as np
import pandas as pd
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split

load_dotenv()

# ── configuration ─────────────────────────────────────────────────────────────
BUCKET        = os.getenv("S3_BUCKET")
FEATURES_KEY  = "data/features/features.csv"
MODELS_PREFIX = "models/"
REGION        = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
TARGET_COL    = "isFraud"

# thresholds to evaluate — we test multiple values so you can
# pick the one that best fits your acceptable false positive rate
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def load_features_from_s3() -> pd.DataFrame:
    """
    Downloads features.csv from S3 and loads it into a dataframe.

    We use a fixed random seed to create a consistent 80/20 split.
    The same seed is used in train.py so the test set here is genuinely
    held-out data that the model never saw during training.
    """
    print("Loading features from S3...")
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET, Key=FEATURES_KEY)
    df = pd.read_csv(obj["Body"])
    print(f"  Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def load_model_from_s3():
    """
    Downloads and loads the trained XGBoost model from S3.

    The model artifact is stored as model.tar.gz in S3.
    We find the most recent training job's artifact by reading
    the current_model_metrics.json file that deploy_endpoint.py saved.
    """
    import xgboost as xgb
    import tarfile
    import tempfile

    print("Loading model from S3...")
    s3 = boto3.client("s3")

    # read the metrics file to find which model is currently deployed
    try:
        obj = s3.get_object(
            Bucket=BUCKET,
            Key=f"{MODELS_PREFIX}current_model_metrics.json"
        )
        metrics = json.loads(obj["Body"].read())
        job_name = metrics["job_name"]
        print(f"  Current model from job: {job_name}")
        print(f"  Deployed AUC: {metrics['auc']:.4f}")
    except Exception as e:
        raise FileNotFoundError(
            f"Could not find current model metrics in S3.\n"
            f"Run scripts/deploy_endpoint.py first.\n"
            f"Error: {e}"
        )

    # find the model artifact for this job
    model_artifact_key = (
        f"{MODELS_PREFIX}{job_name}/output/model.tar.gz"
    )

    # download to a temp directory and extract
    with tempfile.TemporaryDirectory() as tmpdir:
        local_tar = Path(tmpdir) / "model.tar.gz"

        try:
            s3.download_file(BUCKET, model_artifact_key, str(local_tar))
        except Exception as e:
            raise FileNotFoundError(
                f"Could not download model artifact from S3.\n"
                f"Expected at: s3://{BUCKET}/{model_artifact_key}\n"
                f"Error: {e}"
            )

        # extract model.xgb from the tar archive
        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(tmpdir)

        model_file = Path(tmpdir) / "model.xgb"
        if not model_file.exists():
            # try alternative name
            model_file = list(Path(tmpdir).glob("*.xgb"))[0]

        model = xgb.Booster()
        model.load_model(str(model_file))
        print(f"  Model loaded successfully")
        return model, metrics


def evaluate_at_threshold(y_true, y_pred_proba, threshold: float) -> dict:
    """
    Evaluates model performance at a specific fraud probability threshold.

    threshold is the cutoff above which a trade is flagged as fraud.
    Lower threshold = catches more fraud but also more false positives.
    Higher threshold = fewer false positives but misses some fraud.

    Returns a dict of metrics for this threshold.
    """
    import xgboost as xgb

    y_pred = (y_pred_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # precision: of trades flagged as fraud, how many actually were?
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # recall: of all actual fraud cases, how many did we catch?
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # f1: harmonic mean of precision and recall
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)

    # false positive rate: legitimate trades incorrectly flagged
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "threshold":   threshold,
        "precision":   round(precision, 4),
        "recall":      round(recall, 4),
        "f1":          round(f1, 4),
        "fpr":         round(fpr, 4),
        "tp":          int(tp),
        "fp":          int(fp),
        "fn":          int(fn),
        "tn":          int(tn),
        "flagged":     int(tp + fp),
        "missed":      int(fn),
    }


def print_confusion_matrix(y_true, y_pred_proba, threshold: float):
    """
    Prints a formatted confusion matrix for a given threshold.

    The confusion matrix shows:
    - True Positives  (TP): fraud correctly flagged
    - False Positives (FP): legitimate trades wrongly flagged
    - False Negatives (FN): fraud missed entirely
    - True Negatives  (TN): legitimate trades correctly cleared
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n  Confusion Matrix (threshold={threshold}):")
    print(f"  {'':20} Predicted NOT Fraud  Predicted Fraud")
    print(f"  {'Actual NOT Fraud':20} {cm[0][0]:>20,} {cm[0][1]:>16,}")
    print(f"  {'Actual Fraud':20} {cm[1][0]:>20,} {cm[1][1]:>16,}")


def recommend_threshold(results: list) -> float:
    """
    Recommends the best threshold based on a balance of precision and recall.

    For fraud detection the typical priority is:
    - High recall (catch as much fraud as possible)
    - Acceptable precision (don't flag too many legitimate trades)

    We pick the threshold where recall >= 0.70 and precision is maximised.
    This means we catch at least 70% of fraud while minimising false flags.
    You can adjust this logic based on your business requirements.
    """
    candidates = [r for r in results if r["recall"] >= 0.70]

    if not candidates:
        # if no threshold achieves 70% recall, pick highest recall
        return max(results, key=lambda x: x["recall"])["threshold"]

    # among candidates, pick highest precision
    best = max(candidates, key=lambda x: x["precision"])
    return best["threshold"]


def run():
    print("=" * 60)
    print("  Fraud Detection Model Evaluation")
    print("=" * 60 + "\n")

    import xgboost as xgb

    # load data
    df = load_features_from_s3()

    X = df.drop(columns=[TARGET_COL]).values
    y = df[TARGET_COL].values

    # use same split as training — test set is held-out data model never saw
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTest set: {len(X_test):,} rows")
    print(f"Fraud cases in test set: {y_test.sum():,} ({y_test.mean():.2%})")

    # load model
    model, deployed_metrics = load_model_from_s3()

    # get predictions
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)

    # overall AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nOverall AUC: {auc:.4f}")

    # evaluate at multiple thresholds
    print("\n" + "=" * 60)
    print("  Threshold Analysis")
    print("=" * 60)
    print(f"\n  {'Threshold':>10} {'Precision':>10} {'Recall':>8} "
          f"{'F1':>8} {'FPR':>8} {'Flagged':>10} {'Missed':>8}")
    print("  " + "-" * 66)

    results = []
    for threshold in THRESHOLDS:
        r = evaluate_at_threshold(y_test, y_pred_proba, threshold)
        results.append(r)
        print(f"  {r['threshold']:>10.1f} "
              f"{r['precision']:>10.4f} "
              f"{r['recall']:>8.4f} "
              f"{r['f1']:>8.4f} "
              f"{r['fpr']:>8.4f} "
              f"{r['flagged']:>10,} "
              f"{r['missed']:>8,}")

    # recommended threshold
    recommended = recommend_threshold(results)
    print(f"\n  Recommended threshold: {recommended}")
    print(f"  (catches >=70% of fraud with maximum precision)")

    # confusion matrix at recommended threshold
    print_confusion_matrix(y_test, y_pred_proba, recommended)

    # full classification report at recommended threshold
    y_pred = (y_pred_proba >= recommended).astype(int)
    print(f"\n  Classification Report (threshold={recommended}):")
    print(classification_report(
        y_test, y_pred,
        target_names=["Not Fraud", "Fraud"],
        digits=4
    ))

    # save evaluation report to S3
    report = {
        "auc": float(auc),
        "recommended_threshold": recommended,
        "threshold_analysis": results,
        "test_set_size": len(y_test),
        "fraud_cases_in_test": int(y_test.sum()),
        "deployed_model_job": deployed_metrics.get("job_name"),
    }

    s3 = boto3.client("s3")
    from datetime import datetime
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    s3.put_object(
        Bucket=BUCKET,
        Key=f"logs/evaluation_reports/{timestamp}.json",
        Body=json.dumps(report, indent=2).encode("utf-8"),
    )

    print(f"\nEvaluation report saved to S3:")
    print(f"  s3://{BUCKET}/logs/evaluation_reports/{timestamp}.json")
    print("\n" + "=" * 60)
    print("  Evaluation complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run()