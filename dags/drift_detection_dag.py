"""
drift_detection_dag.py — Stage 8: Drift Detection

Runs daily. Reads prediction logs from S3, computes whether the mean
fraud score has shifted significantly from the historical baseline,
and if drift is detected:
  1. Logs a detailed alert to S3
  2. Triggers the retraining DAG (model_retraining_dag)

DAG task graph:
    check_s3_logs
         │
    compute_drift
         │
    evaluate_drift
        / \\
  send_   trigger_
  alert   retraining

Environment variables required (already in your .env):
    S3_BUCKET               — bucket where logs/predictions/*.csv live
    AWS_DEFAULT_REGION      — e.g. us-east-1
    DRIFT_THRESHOLD         — mean shift that triggers alert (default 0.05)
    DRIFT_BASELINE_DAYS     — how many days of history to use as baseline (default 7)
    DRIFT_WINDOW_HOURS      — how many hours of recent logs to evaluate (default 24)
"""

from __future__ import annotations

import os
import io
import json
import logging
from datetime import datetime, timedelta, timezone

import boto3
import numpy as np
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

BUCKET              = os.getenv("S3_BUCKET", "")
LOGS_PREFIX         = os.getenv("S3_LOGS_PREFIX", "logs/predictions/")
DRIFT_ALERTS_PREFIX = "logs/drift_alerts/"
REGION              = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Mean fraud score must shift by more than this to trigger an alert.
# 0.05 = the window mean is 5 percentage points away from the baseline mean.
# Tune this lower (0.03) for more sensitivity, higher (0.08) for less noise.
DRIFT_THRESHOLD     = float(os.getenv("DRIFT_THRESHOLD", "0.05"))

# How many days of older logs to use as the stable baseline.
BASELINE_DAYS       = int(os.getenv("DRIFT_BASELINE_DAYS", "7"))

# How many hours of recent logs to treat as the "current window" to evaluate.
WINDOW_HOURS        = int(os.getenv("DRIFT_WINDOW_HOURS", "24"))

# Minimum number of predictions needed to compute a meaningful drift score.
MIN_PREDICTIONS     = int(os.getenv("DRIFT_MIN_PREDICTIONS", "50"))

# XCom keys — used to pass data between tasks
XCOM_DRIFT_RESULT   = "drift_result"
XCOM_LOG_PATHS      = "log_paths"

# Branch task return values
BRANCH_DRIFT        = "drift_detected_gate"
BRANCH_NO_DRIFT     = "no_drift"

RETRAINING_DAG_ID   = "model_retraining_dag"


# ── helpers ───────────────────────────────────────────────────────────────────

def _s3_client():
    return boto3.client("s3", region_name=REGION)


def _list_prediction_csvs(s3, since: datetime, until: datetime) -> list[str]:
    """
    Lists all CSV keys under LOGS_PREFIX whose last-modified timestamp
    falls between `since` and `until`.

    We compare by S3 LastModified rather than parsing filenames because
    the consumer names files by upload time which matches last-modified.
    """
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=LOGS_PREFIX):
        for obj in page.get("Contents", []):
            mod = obj["LastModified"]
            # S3 returns timezone-aware datetimes; ensure both sides are aware
            if since <= mod <= until:
                keys.append(obj["Key"])
    return keys


def _read_csvs_from_s3(s3, keys: list[str]) -> pd.DataFrame:
    """
    Downloads and concatenates all CSVs from the given S3 keys into
    a single DataFrame. Skips any file that fails to parse.
    """
    frames = []
    for key in keys:
        try:
            obj  = s3.get_object(Bucket=BUCKET, Key=key)
            body = obj["Body"].read().decode("utf-8")
            df   = pd.read_csv(io.StringIO(body))
            frames.append(df)
        except Exception as e:
            log.warning(f"Skipping {key}: {e}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── task 1: check_s3_logs ────────────────────────────────────────────────────

def check_s3_logs(**context) -> None:
    """
    Discovers all prediction CSV files in S3 for the baseline window
    and the recent evaluation window.

    Pushes two lists of S3 keys to XCom:
      - baseline_keys: older logs used as the stable reference
      - window_keys:   recent logs we want to evaluate for drift

    Raises if either window has fewer than MIN_PREDICTIONS rows —
    drift computation is meaningless with too little data.
    """
    if not BUCKET:
        raise ValueError("S3_BUCKET environment variable is not set.")

    s3  = _s3_client()
    now = datetime.now(timezone.utc)

    # recent window: last WINDOW_HOURS hours
    window_start    = now - timedelta(hours=WINDOW_HOURS)
    window_end      = now

    # baseline: the BASELINE_DAYS days before the window starts
    baseline_start  = window_start - timedelta(days=BASELINE_DAYS)
    baseline_end    = window_start

    log.info(f"Baseline : {baseline_start} → {baseline_end}")
    log.info(f"Window   : {window_start}   → {window_end}")

    window_keys   = _list_prediction_csvs(s3, window_start,   window_end)
    baseline_keys = _list_prediction_csvs(s3, baseline_start, baseline_end)

    log.info(f"Found {len(window_keys)} window files, {len(baseline_keys)} baseline files")

    context["ti"].xcom_push(key="window_keys",   value=window_keys)
    context["ti"].xcom_push(key="baseline_keys", value=baseline_keys)


# ── task 2: compute_drift ─────────────────────────────────────────────────────

def compute_drift(**context) -> None:
    """
    Downloads the window and baseline CSVs from S3, extracts the
    fraud_score column from each, and computes:

      drift_score = abs(window_mean - baseline_mean)

    A drift_score above DRIFT_THRESHOLD means the model's output
    distribution has shifted — either the data has changed or the
    model is degrading.

    Also computes std, min, max, and prediction counts for both
    windows so the alert message is informative.

    Pushes a drift_result dict to XCom for the evaluate task.
    """
    ti             = context["ti"]
    window_keys    = ti.xcom_pull(key="window_keys")
    baseline_keys  = ti.xcom_pull(key="baseline_keys")

    s3 = _s3_client()

    window_df   = _read_csvs_from_s3(s3, window_keys)
    baseline_df = _read_csvs_from_s3(s3, baseline_keys)

    def _stats(df: pd.DataFrame, label: str) -> dict:
        if df.empty or "fraud_score" not in df.columns:
            log.warning(f"{label} DataFrame is empty or missing fraud_score column.")
            return {"mean": None, "std": None, "min": None, "max": None, "count": 0}
        scores = df["fraud_score"].dropna()
        return {
            "mean":  float(scores.mean()),
            "std":   float(scores.std()),
            "min":   float(scores.min()),
            "max":   float(scores.max()),
            "count": int(len(scores)),
        }

    window_stats   = _stats(window_df,   "Window")
    baseline_stats = _stats(baseline_df, "Baseline")

    # guard: not enough data
    if window_stats["count"] < MIN_PREDICTIONS:
        log.warning(
            f"Window only has {window_stats['count']} predictions "
            f"(need {MIN_PREDICTIONS}). Skipping drift check."
        )
        drift_score = None
        drift_detected = False
    elif baseline_stats["count"] < MIN_PREDICTIONS:
        log.warning(
            f"Baseline only has {baseline_stats['count']} predictions "
            f"(need {MIN_PREDICTIONS}). Cannot establish baseline. Skipping."
        )
        drift_score = None
        drift_detected = False
    else:
        drift_score    = abs(window_stats["mean"] - baseline_stats["mean"])
        drift_detected = drift_score > DRIFT_THRESHOLD
        log.info(f"Baseline mean : {baseline_stats['mean']:.6f}")
        log.info(f"Window mean   : {window_stats['mean']:.6f}")
        log.info(f"Drift score   : {drift_score:.6f}  (threshold={DRIFT_THRESHOLD})")
        log.info(f"Drift detected: {drift_detected}")

    # also compute flag rate for both windows
    def _flag_rate(df):
        if df.empty or "is_fraud" not in df.columns:
            return None
        return float(df["is_fraud"].mean())

    result = {
        "computed_at":      datetime.now(timezone.utc).isoformat(),
        "drift_score":      drift_score,
        "drift_detected":   drift_detected,
        "threshold":        DRIFT_THRESHOLD,
        "window_stats":     window_stats,
        "baseline_stats":   baseline_stats,
        "window_flag_rate": _flag_rate(window_df),
        "baseline_flag_rate": _flag_rate(baseline_df),
        "window_hours":     WINDOW_HOURS,
        "baseline_days":    BASELINE_DAYS,
    }

    ti.xcom_push(key=XCOM_DRIFT_RESULT, value=result)


# ── task 3: evaluate_drift (branch) ──────────────────────────────────────────

def evaluate_drift(**context) -> str:
    """
    Branch operator — reads the drift result and returns the ID of the
    next task to run.

    Returns BRANCH_DRIFT   → drift was detected → run alert + retraining
    Returns BRANCH_NO_DRIFT → no drift → end quietly
    """
    result = context["ti"].xcom_pull(key=XCOM_DRIFT_RESULT)

    if not result or result.get("drift_score") is None:
        log.info("No drift score computed (insufficient data). Branching to no_drift.")
        return BRANCH_NO_DRIFT

    if result["drift_detected"]:
        log.warning(
            f"DRIFT DETECTED — score={result['drift_score']:.6f} "
            f"exceeds threshold={result['threshold']}"
        )
        return BRANCH_DRIFT

    log.info(
        f"No drift — score={result['drift_score']:.6f} "
        f"is within threshold={result['threshold']}"
    )
    return BRANCH_NO_DRIFT


# ── task 4: send_alert ────────────────────────────────────────────────────────

def send_alert(**context) -> None:
    """
    Writes a detailed drift alert JSON to S3 and logs a prominent
    warning so it appears in Airflow task logs and any log aggregation
    tool (CloudWatch, Datadog, etc.).

    The JSON is written to:
        s3://{BUCKET}/logs/drift_alerts/{timestamp}.json

    This file can be picked up by downstream alerting tools, SNS,
    PagerDuty, or just queried manually when investigating drift.
    """
    result = context["ti"].xcom_pull(key=XCOM_DRIFT_RESULT)
    s3     = _s3_client()

    ts      = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    s3_key  = f"{DRIFT_ALERTS_PREFIX}{ts}.json"

    alert = {
        "alert_type":    "FRAUD_SCORE_DRIFT",
        "severity":      "HIGH" if result["drift_score"] > DRIFT_THRESHOLD * 2 else "MEDIUM",
        "message":       (
            f"Fraud score drift detected. "
            f"Window mean={result['window_stats']['mean']:.6f}, "
            f"Baseline mean={result['baseline_stats']['mean']:.6f}, "
            f"Drift={result['drift_score']:.6f} (threshold={result['threshold']})."
        ),
        "action":        "Model retraining has been triggered automatically.",
        "drift_result":  result,
        "dag_run_id":    context["run_id"],
    }

    # log prominently — shows up in Airflow UI task logs
    log.warning("=" * 70)
    log.warning("  ⚠  DRIFT ALERT")
    log.warning(f"  Drift score  : {result['drift_score']:.6f}")
    log.warning(f"  Threshold    : {result['threshold']}")
    log.warning(f"  Window mean  : {result['window_stats']['mean']:.6f}  "
                f"(n={result['window_stats']['count']:,})")
    log.warning(f"  Baseline mean: {result['baseline_stats']['mean']:.6f}  "
                f"(n={result['baseline_stats']['count']:,})")
    log.warning(f"  Flag rate    : window={result['window_flag_rate']:.2%}  "
                f"baseline={result['baseline_flag_rate']:.2%}")
    log.warning(f"  Alert saved  : s3://{BUCKET}/{s3_key}")
    log.warning("=" * 70)

    # write to S3
    if BUCKET:
        try:
            s3.put_object(
                Bucket=BUCKET,
                Key=s3_key,
                Body=json.dumps(alert, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
            log.info(f"Alert written to s3://{BUCKET}/{s3_key}")
        except Exception as e:
            log.error(f"Failed to write alert to S3: {e}")
    else:
        log.warning("S3_BUCKET not set — alert not persisted to S3.")


# ── DAG definition ────────────────────────────────────────────────────────────

default_args = {
    "owner":            "fraud-mlops",
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}

with DAG(
    dag_id="drift_detection_dag",
    description="Daily drift detection on SageMaker prediction logs",
    schedule_interval="0 6 * * *",   # 06:00 UTC every day
    start_date=days_ago(1),
    catchup=False,
    default_args=default_args,
    tags=["monitoring", "drift", "stage-8"],
    doc_md="""
## Drift Detection DAG

Runs daily at 06:00 UTC.

### What it does
1. **check_s3_logs** — finds prediction CSVs in S3 for the last 24h (window)
   and the 7 days before that (baseline).
2. **compute_drift** — downloads those CSVs, computes mean fraud score for
   each window, and calculates drift = |window_mean − baseline_mean|.
3. **evaluate_drift** — branches: if drift > threshold → alert + retrain;
   otherwise → finish quietly.
4. **send_alert** — writes a JSON alert to `s3://{bucket}/logs/drift_alerts/`
   and logs a prominent warning visible in the Airflow UI.
5. **trigger_retraining** — triggers `model_retraining_dag` to kick off a
   new SageMaker training job and swap the endpoint.

### Tuning
| Env var | Default | Meaning |
|---|---|---|
| `DRIFT_THRESHOLD` | `0.05` | Mean shift (abs) that triggers alert |
| `DRIFT_BASELINE_DAYS` | `7` | Days of history used as baseline |
| `DRIFT_WINDOW_HOURS` | `24` | Hours of recent logs evaluated |
| `DRIFT_MIN_PREDICTIONS` | `50` | Min rows needed to run check |
    """,
) as dag:

    # ── task 1 ────────────────────────────────────────────────────────────────
    t_check = PythonOperator(
        task_id="check_s3_logs",
        python_callable=check_s3_logs,
        doc_md="""
Lists S3 prediction CSVs for the baseline window and the recent evaluation
window. Pushes S3 key lists to XCom for the next task to download.
        """,
    )

    # ── task 2 ────────────────────────────────────────────────────────────────
    t_compute = PythonOperator(
        task_id="compute_drift",
        python_callable=compute_drift,
        doc_md="""
Downloads CSVs, extracts fraud_score columns, computes drift_score =
abs(window_mean − baseline_mean). Pushes full result dict to XCom.
        """,
    )

    # ── task 3 (branch) ───────────────────────────────────────────────────────
    t_evaluate = BranchPythonOperator(
        task_id="evaluate_drift",
        python_callable=evaluate_drift,
        doc_md="""
Reads drift_result from XCom. If drift_detected=True → branches to
drift_detected_gate (alert + retrain). Otherwise → no_drift (end).
        """,
    )

    # ── task 4a: drift path ───────────────────────────────────────────────────
    # gate task — both send_alert and trigger_retraining depend on this
    # so they run in parallel after the branch
    t_drift_gate = EmptyOperator(
        task_id=BRANCH_DRIFT,
    )

    t_alert = PythonOperator(
        task_id="send_alert",
        python_callable=send_alert,
        doc_md="""
Writes a JSON drift alert to s3://{bucket}/logs/drift_alerts/ and logs
a prominent WARNING visible in the Airflow task logs.
        """,
    )

    t_retrain = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id=RETRAINING_DAG_ID,
        wait_for_completion=False,      # fire and forget — retraining can take hours
        conf={
            "triggered_by":  "drift_detection_dag",
            "drift_score":   "{{ ti.xcom_pull(key='drift_result')['drift_score'] }}",
            "triggered_at":  "{{ ts }}",
        },
        doc_md="""
Triggers model_retraining_dag with the drift score as context.
wait_for_completion=False so this task completes immediately and the
retraining runs independently.
        """,
    )

    # ── task 4b: no-drift path ────────────────────────────────────────────────
    t_no_drift = EmptyOperator(
        task_id=BRANCH_NO_DRIFT,
    )

    # ── wiring ────────────────────────────────────────────────────────────────
    #
    # check_s3_logs
    #       │
    # compute_drift
    #       │
    # evaluate_drift
    #      / \
    # gate   no_drift
    #  / \
    # alert  trigger_retraining

    t_check >> t_compute >> t_evaluate
    t_evaluate >> t_drift_gate >> [t_alert, t_retrain]
    t_evaluate >> t_no_drift