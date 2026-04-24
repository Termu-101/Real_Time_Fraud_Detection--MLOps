"""
drift_dag.py — Daily drift detection DAG

This DAG runs every day and checks whether the live Binance trade
patterns have shifted significantly from the patterns in the training data.

If drift is detected it triggers the training DAG automatically —
no human intervention needed.

What is drift?
  The model was trained on data with certain statistical properties.
  Over time the real world changes — trade volumes shift, price ranges
  change, trading patterns evolve. When the live data looks too different
  from training data, the model's predictions become unreliable.
  We call this "data drift" and detecting it early is critical.

How we detect it:
  We compare the mean and standard deviation of key features in the
  last 24 hours of Binance prediction logs against the same statistics
  from the training data. If the difference is large enough (measured
  by Population Stability Index), we flag it as drift.
"""

import os
import json
import boto3
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from airflow.api.client.local_client import Client

log = logging.getLogger(__name__)

# ── configuration ──────────────────────────────────────────────────────────────

BUCKET          = os.getenv("S3_BUCKET")
LOGS_PREFIX     = os.getenv("S3_LOGS_PREFIX", "logs/predictions/")
METADATA_KEY    = "data/features/feature_metadata.json"
FEATURES_KEY    = "data/features/features.csv"

# Population Stability Index threshold for drift detection.
# PSI < 0.1  → no significant drift, model is stable
# PSI 0.1–0.2 → minor drift, worth monitoring
# PSI > 0.2  → significant drift, retrain the model
# We use 0.2 as the trigger threshold.
PSI_THRESHOLD = 0.2

# Features we monitor for drift.
# We pick the most important ones from EDA rather than monitoring
# all 280+ features — monitoring too many creates false positives.
MONITORED_FEATURES = [
    "TransactionAmt",
    "hour_of_day",
    "day_of_week",
]

default_args = {
    "owner": "fraud-mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Calculates the Population Stability Index (PSI) between two distributions.

    PSI measures how much a feature's distribution has shifted between
    the training data (expected) and the live data (actual).

    The formula compares the proportion of values in each bucket between
    the two distributions. A large difference in proportions = high PSI = drift.

    Args:
        expected: feature values from training data
        actual: feature values from recent live data
        bins: number of equal-width buckets to compare

    Returns:
        PSI score — higher means more drift
    """
    # create buckets based on the training data distribution
    # we use the training data to define buckets so they are consistent
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1
    )

    # count what proportion of values fall in each bucket
    expected_counts = np.histogram(expected, breakpoints)[0]
    actual_counts   = np.histogram(actual, breakpoints)[0]

    # convert counts to proportions — add small epsilon to avoid log(0)
    epsilon = 1e-6
    expected_pct = (expected_counts + epsilon) / (len(expected) + epsilon * bins)
    actual_pct   = (actual_counts + epsilon)   / (len(actual) + epsilon * bins)

    # PSI formula: sum of (actual% - expected%) * ln(actual% / expected%)
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return float(psi)


@dag(
    dag_id="fraud_drift_detection",

    # Run every day at 6am UTC
    schedule_interval="0 6 * * *",

    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["fraud-detection", "monitoring", "drift"],
    description="Daily drift detection — triggers retraining if needed",
)
def fraud_drift_detection():

    @task
    def load_recent_predictions() -> dict:
        """
        Task 1 — Load the last 24 hours of prediction logs from S3.

        The Kafka consumer saves prediction logs to S3 in batches.
        We read all log files from the last 24 hours and combine them
        into a single dataframe.

        Returns a dict with the recent data statistics.
        """
        log.info("Task 1: Loading recent prediction logs from S3...")

        s3 = boto3.client("s3")

        # list all log files in the predictions prefix
        response = s3.list_objects_v2(
            Bucket=BUCKET,
            Prefix=LOGS_PREFIX,
        )

        if "Contents" not in response:
            raise AirflowSkipException(
                "No prediction logs found in S3. "
                "The consumer may not have run yet."
            )

        # filter to files from the last 24 hours
        cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_files = [
            obj["Key"] for obj in response["Contents"]
            if obj["LastModified"].replace(tzinfo=None) > cutoff
        ]

        if not recent_files:
            raise AirflowSkipException(
                "No prediction logs from the last 24 hours. "
                "Skipping drift check."
            )

        log.info(f"  Found {len(recent_files)} log files from last 24 hours")

        # load and combine all recent log files
        dfs = []
        for key in recent_files:
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            df = pd.read_csv(obj["Body"])
            dfs.append(df)

        recent_df = pd.concat(dfs, ignore_index=True)

        log.info(f"  Total recent predictions: {len(recent_df):,}")
        log.info(f"  Recent fraud rate: {recent_df['is_fraud'].mean():.2%}")

        # return statistics rather than the full dataframe
        # XCom has a size limit — we pass stats not raw data
        return {
            "n_predictions": len(recent_df),
            "fraud_rate": float(recent_df["is_fraud"].mean()),
            "avg_score": float(recent_df["fraud_score"].mean()),
            "transaction_amt_mean": float(
                recent_df["price"].astype(float).multiply(
                    recent_df["quantity"].astype(float)
                ).mean()
            ),
            "hour_of_day_mean": float(
                pd.to_datetime(recent_df["timestamp"]).dt.hour.mean()
            ),
        }


    @task
    def load_training_baseline() -> dict:
        """
        Task 2 — Load baseline statistics from the training data.

        We compare live data against these baselines to detect drift.
        Rather than loading the full 590k row CSV every day, we load
        the feature metadata which contains the statistics we need.

        If the metadata does not have baseline stats yet, we calculate
        them from the features.csv and save them for future runs.
        """
        log.info("Task 2: Loading training data baseline statistics...")

        s3 = boto3.client("s3")

        # try to load cached baseline stats first
        baseline_key = "data/features/baseline_stats.json"
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=baseline_key)
            baseline = json.loads(obj["Body"].read())
            log.info("  Loaded cached baseline stats")
            return baseline
        except Exception:
            log.info("  No cached baseline found — calculating from features.csv")

        # load the features CSV to calculate baseline
        obj = s3.get_object(Bucket=BUCKET, Key=FEATURES_KEY)

        # only load columns we monitor to keep memory usage low
        cols_to_load = MONITORED_FEATURES + ["isFraud"]
        available_cols = [
            c for c in cols_to_load
            if c in pd.read_csv(obj["Body"], nrows=1).columns
        ]

        obj = s3.get_object(Bucket=BUCKET, Key=FEATURES_KEY)
        df = pd.read_csv(obj["Body"], usecols=available_cols)

        baseline = {
            "n_rows": len(df),
            "fraud_rate": float(df["isFraud"].mean()),
        }

        # calculate mean and std for each monitored feature
        for feature in MONITORED_FEATURES:
            if feature in df.columns:
                baseline[f"{feature}_mean"] = float(df[feature].mean())
                baseline[f"{feature}_std"]  = float(df[feature].std())
                baseline[f"{feature}_values"] = df[feature].dropna().tolist()[:10000]

        # cache the baseline so we do not recalculate every day
        s3.put_object(
            Bucket=BUCKET,
            Key=baseline_key,
            Body=json.dumps(baseline).encode("utf-8"),
        )

        log.info(f"  Baseline calculated from {len(df):,} training rows")
        return baseline


    @task
    def check_for_drift(recent_stats: dict, baseline: dict) -> dict:
        """
        Task 3 — Compare recent data against training baseline.

        We calculate the Population Stability Index for each monitored
        feature. If any feature's PSI exceeds PSI_THRESHOLD, we flag drift.

        We also check for a sudden spike in the fraud flag rate which
        could indicate either a real fraud wave or model degradation.

        Returns a drift report that the next task uses to decide
        whether to trigger retraining.
        """
        log.info("Task 3: Checking for drift...")

        drift_detected = False
        drift_report = {
            "checked_at": datetime.utcnow().isoformat(),
            "n_recent_predictions": recent_stats["n_predictions"],
            "feature_psi": {},
            "fraud_rate_change": 0.0,
            "drift_detected": False,
            "reason": [],
        }

        # check fraud rate change
        recent_fraud_rate = recent_stats.get("fraud_rate", 0)
        baseline_fraud_rate = baseline.get("fraud_rate", 0.035)
        fraud_rate_change = abs(recent_fraud_rate - baseline_fraud_rate)

        drift_report["fraud_rate_change"] = float(fraud_rate_change)
        log.info(f"  Baseline fraud rate : {baseline_fraud_rate:.2%}")
        log.info(f"  Recent fraud rate   : {recent_fraud_rate:.2%}")
        log.info(f"  Change              : {fraud_rate_change:.2%}")

        # if fraud rate has changed by more than 5 percentage points, flag it
        if fraud_rate_change > 0.05:
            drift_detected = True
            drift_report["reason"].append(
                f"Fraud rate changed by {fraud_rate_change:.2%} "
                f"(from {baseline_fraud_rate:.2%} to {recent_fraud_rate:.2%})"
            )

        # check PSI for each monitored feature
        # note: for full PSI we need the actual value arrays from both datasets
        # here we use a simplified z-score approach since we only have summary stats
        for feature in MONITORED_FEATURES:
            baseline_mean = baseline.get(f"{feature}_mean")
            baseline_std  = baseline.get(f"{feature}_std")
            recent_mean   = recent_stats.get(f"{feature}_mean")

            if None in (baseline_mean, baseline_std, recent_mean):
                continue

            if baseline_std == 0:
                continue

            # z-score measures how many standard deviations the recent mean
            # has shifted from the baseline mean
            z_score = abs(recent_mean - baseline_mean) / baseline_std

            drift_report["feature_psi"][feature] = round(z_score, 4)
            log.info(f"  {feature}: z-score={z_score:.2f} "
                     f"(baseline mean={baseline_mean:.2f}, "
                     f"recent mean={recent_mean:.2f})")

            # a z-score above 3 means the mean has shifted by 3 standard
            # deviations — very unlikely by chance, likely real drift
            if z_score > 3.0:
                drift_detected = True
                drift_report["reason"].append(
                    f"{feature} mean shifted by {z_score:.1f} standard deviations"
                )

        drift_report["drift_detected"] = drift_detected

        if drift_detected:
            log.warning(f"  DRIFT DETECTED: {drift_report['reason']}")
        else:
            log.info("  No significant drift detected. Model is stable.")

        return drift_report


    @task.branch
    def decide_retraining(drift_report: dict) -> str:
        """
        Task 4 — Branch: trigger retraining or do nothing.

        If drift was detected, return the task_id that triggers retraining.
        If no drift, return the task_id that logs stability and stops.
        """
        if drift_report["drift_detected"]:
            log.warning("Drift confirmed — triggering retraining pipeline.")
            return "trigger_retraining"
        else:
            log.info("No drift — model is stable. No retraining needed.")
            return "log_stability"


    @task
    def trigger_retraining(drift_report: dict):
        """
        Task 5a — Trigger the training DAG to retrain the model.

        Airflow allows one DAG to trigger another using the local client.
        This is what makes the system self-healing — drift is detected
        and retraining starts automatically without any human action.

        We also save the drift report to S3 for auditing purposes.
        """
        log.warning("Triggering fraud_detection_training DAG...")

        # save drift report to S3 for auditing
        s3 = boto3.client("s3")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        s3.put_object(
            Bucket=BUCKET,
            Key=f"logs/drift_reports/{timestamp}.json",
            Body=json.dumps(drift_report, indent=2).encode("utf-8"),
        )

        # trigger the training DAG
        # this is equivalent to clicking "Trigger DAG" in the UI
        client = Client(None, None)
        client.trigger_dag(
            dag_id="fraud_detection_training",
            run_id=f"drift_triggered_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        )

        log.info("Training DAG triggered successfully.")
        log.info(f"Drift report saved to: logs/drift_reports/{timestamp}.json")


    @task
    def log_stability(drift_report: dict):
        """
        Task 5b — Log that the model is stable and no retraining is needed.

        We save the stability report to S3 so you have a daily audit trail
        showing the model was checked and found to be healthy.
        """
        log.info("Model is stable — no retraining triggered.")

        s3 = boto3.client("s3")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        s3.put_object(
            Bucket=BUCKET,
            Key=f"logs/stability_reports/{timestamp}.json",
            Body=json.dumps(drift_report, indent=2).encode("utf-8"),
        )

        log.info(f"Stability report saved to: logs/stability_reports/{timestamp}.json")


    # ── wire up the tasks ──────────────────────────────────────────────────────
    recent      = load_recent_predictions()
    baseline    = load_training_baseline()
    drift       = check_for_drift(recent, baseline)
    branch      = decide_retraining(drift)

    retrain     = trigger_retraining(drift)
    stable      = log_stability(drift)

    branch >> [retrain, stable]


fraud_drift_detection()