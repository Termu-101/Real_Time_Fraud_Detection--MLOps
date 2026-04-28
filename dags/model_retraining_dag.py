"""
model_retraining_dag.py — Stage 8 (companion): Model Retraining Pipeline

Triggered automatically by drift_detection_dag when drift is detected.
Can also be triggered manually from the Airflow UI for scheduled retraining.

Task graph:
    validate_trigger
          │
    fetch_training_data        ← pulls latest features.csv from S3
          │
    launch_sagemaker_job       ← starts a new XGBoost training job
          │
    wait_for_training          ← polls until job completes
          │
    update_endpoint            ← deploys new model, swaps endpoint
          │
    verify_endpoint            ← smoke test: score one dummy record
          │
    log_completion             ← writes retraining report to S3
"""

from __future__ import annotations

import os
import json
import time
import logging
from datetime import datetime, timedelta, timezone

import boto3
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────

REGION          = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
BUCKET          = os.getenv("S3_BUCKET", "")
ROLE_ARN        = os.getenv("SAGEMAKER_ROLE_ARN", "")
ENDPOINT_NAME   = os.getenv("SAGEMAKER_ENDPOINT_NAME", "fraud-detection-endpoint")
FEATURES_KEY    = "data/features/features.csv"
REPORTS_PREFIX  = "logs/retraining_reports/"

# SageMaker training config
TRAINING_IMAGE  = f"683313688378.dkr.ecr.{REGION}.amazonaws.com/sagemaker-xgboost:1.7-1"
INSTANCE_TYPE   = os.getenv("SAGEMAKER_TRAIN_INSTANCE", "ml.m5.xlarge")
INSTANCE_COUNT  = 1

# XGBoost hyperparameters — same as Stage 5
HYPERPARAMETERS = {
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "num_round":        "200",
    "max_depth":        "6",
    "eta":              "0.1",
    "subsample":        "0.8",
    "colsample_bytree": "0.8",
    "min_child_weight": "5",
}

# how often (seconds) to poll the training job status
POLL_INTERVAL   = 60
# max time to wait for training (seconds)  — 3 hours
MAX_WAIT        = 3 * 60 * 60


# ── helpers ───────────────────────────────────────────────────────────────────

def _sm():
    return boto3.client("sagemaker", region_name=REGION)

def _s3():
    return boto3.client("s3", region_name=REGION)

def _sm_runtime():
    return boto3.client("sagemaker-runtime", region_name=REGION)


# ── task 1: validate_trigger ──────────────────────────────────────────────────

def validate_trigger(**context) -> None:
    """
    Logs who triggered this DAG run and what drift score caused it.
    If triggered manually (no conf), uses sensible defaults.
    Also checks that required env vars are set before doing any work.
    """
    conf = context.get("dag_run").conf or {}
    triggered_by  = conf.get("triggered_by",  "manual")
    drift_score   = conf.get("drift_score",   "N/A")
    triggered_at  = conf.get("triggered_at",  datetime.now(timezone.utc).isoformat())

    log.info("=" * 60)
    log.info("  Model Retraining Pipeline — Starting")
    log.info(f"  Triggered by : {triggered_by}")
    log.info(f"  Drift score  : {drift_score}")
    log.info(f"  Triggered at : {triggered_at}")
    log.info("=" * 60)

    missing = []
    if not BUCKET:       missing.append("S3_BUCKET")
    if not ROLE_ARN:     missing.append("SAGEMAKER_ROLE_ARN")
    if not ENDPOINT_NAME: missing.append("SAGEMAKER_ENDPOINT_NAME")

    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")

    context["ti"].xcom_push(key="trigger_conf", value=conf)
    log.info("Validation passed. Proceeding with retraining.")


# ── task 2: fetch_training_data ───────────────────────────────────────────────

def fetch_training_data(**context) -> None:
    """
    Verifies that features.csv exists in S3 and is recent enough to
    be worth training on. Pushes the S3 URI to XCom for the training job.

    We do NOT download the file — SageMaker reads directly from S3.
    We just confirm it exists and log its size and last-modified time.
    """
    s3  = _s3()
    try:
        head = s3.head_object(Bucket=BUCKET, Key=FEATURES_KEY)
        size_mb       = head["ContentLength"] / 1e6
        last_modified = head["LastModified"].isoformat()
        log.info(f"Training data: s3://{BUCKET}/{FEATURES_KEY}")
        log.info(f"  Size         : {size_mb:.1f} MB")
        log.info(f"  Last modified: {last_modified}")
    except Exception as e:
        raise FileNotFoundError(
            f"Training data not found at s3://{BUCKET}/{FEATURES_KEY}: {e}"
        )

    training_uri = f"s3://{BUCKET}/{FEATURES_KEY}"
    context["ti"].xcom_push(key="training_data_uri", value=training_uri)


# ── task 3: launch_sagemaker_job ──────────────────────────────────────────────

def launch_sagemaker_job(**context) -> None:
    """
    Starts a new SageMaker XGBoost training job.

    Job name includes a timestamp so every run is uniquely identifiable
    in the SageMaker console and can be compared to previous runs.

    The model artifacts (.tar.gz) are saved to:
        s3://{BUCKET}/models/{job_name}/output/model.tar.gz

    We push the job_name to XCom so downstream tasks can poll and
    deploy it without hardcoding names.
    """
    ti           = context["ti"]
    training_uri = ti.xcom_pull(key="training_data_uri")
    sm           = _sm()

    ts       = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"fraud-detection-retrain-{ts}"
    output_path = f"s3://{BUCKET}/models/{job_name}/output/"

    log.info(f"Launching training job: {job_name}")
    log.info(f"  Instance     : {INSTANCE_TYPE}")
    log.info(f"  Input data   : {training_uri}")
    log.info(f"  Output path  : {output_path}")

    sm.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage":     TRAINING_IMAGE,
            "TrainingInputMode": "File",
        },
        RoleArn=ROLE_ARN,
        InputDataConfig=[{
            "ChannelName":     "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType":             "S3Prefix",
                    "S3Uri":                  training_uri,
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
            "ContentType": "text/csv",
        }],
        OutputDataConfig={"S3OutputPath": output_path},
        ResourceConfig={
            "InstanceType":   INSTANCE_TYPE,
            "InstanceCount":  INSTANCE_COUNT,
            "VolumeSizeInGB": 30,
        },
        StoppingCondition={"MaxRuntimeInSeconds": MAX_WAIT},
        HyperParameters=HYPERPARAMETERS,
    )

    log.info(f"Training job launched: {job_name}")
    ti.xcom_push(key="job_name",    value=job_name)
    ti.xcom_push(key="output_path", value=output_path)


# ── task 4: wait_for_training ────────────────────────────────────────────────

def wait_for_training(**context) -> None:
    """
    Polls the SageMaker training job until it reaches a terminal state.

    Terminal states:
      Completed  → proceed to endpoint update
      Failed     → raise an exception (Airflow marks task as failed,
                   retries will re-poll, not re-launch)
      Stopped    → raise an exception

    We poll every POLL_INTERVAL seconds and log progress so the
    Airflow task log shows activity and doesn't look stuck.
    """
    ti       = context["ti"]
    job_name = ti.xcom_pull(key="job_name")
    sm       = _sm()

    log.info(f"Waiting for training job: {job_name}")
    elapsed = 0

    while elapsed < MAX_WAIT:
        resp   = sm.describe_training_job(TrainingJobName=job_name)
        status = resp["TrainingJobStatus"]
        log.info(f"  [{elapsed//60:3d} min] Status: {status}")

        if status == "Completed":
            log.info(f"Training job completed: {job_name}")
            model_artifact = resp["ModelArtifacts"]["S3ModelArtifacts"]
            ti.xcom_push(key="model_artifact", value=model_artifact)
            return

        if status in ("Failed", "Stopped"):
            reason = resp.get("FailureReason", "No reason provided")
            raise RuntimeError(
                f"Training job {job_name} ended with status={status}. "
                f"Reason: {reason}"
            )

        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    raise TimeoutError(
        f"Training job {job_name} did not complete within {MAX_WAIT//3600} hours."
    )


# ── task 5: update_endpoint ──────────────────────────────────────────────────

def update_endpoint(**context) -> None:
    """
    Creates a new SageMaker Model and EndpointConfig from the freshly
    trained artifact, then updates the existing endpoint to use it.

    SageMaker endpoint updates are blue/green — traffic shifts to the
    new model with zero downtime. The old model stays warm until the
    update completes, so the fraud-consumer keeps scoring trades
    throughout.

    Steps:
      1. Create Model (points at the new model.tar.gz)
      2. Create EndpointConfig (specifies instance type)
      3. Update Endpoint (swaps the config — triggers blue/green)
    """
    ti            = context["ti"]
    job_name      = ti.xcom_pull(key="job_name")
    model_artifact = ti.xcom_pull(key="model_artifact")
    sm            = _sm()

    ts           = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_name   = f"fraud-model-{ts}"
    config_name  = f"fraud-config-{ts}"

    log.info(f"Creating SageMaker model: {model_name}")
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image":           TRAINING_IMAGE,
            "ModelDataUrl":    model_artifact,
            "Environment":     {},
        },
        ExecutionRoleArn=ROLE_ARN,
    )

    log.info(f"Creating endpoint config: {config_name}")
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName":          "AllTraffic",
            "ModelName":            model_name,
            "InitialInstanceCount": 1,
            "InstanceType":         os.getenv("SAGEMAKER_DEPLOY_INSTANCE", "ml.m5.large"),
            "InitialVariantWeight": 1.0,
        }],
    )

    log.info(f"Updating endpoint: {ENDPOINT_NAME}")
    sm.update_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=config_name,
    )

    ti.xcom_push(key="model_name",  value=model_name)
    ti.xcom_push(key="config_name", value=config_name)
    log.info("Endpoint update initiated. SageMaker will perform blue/green swap.")


# ── task 6: verify_endpoint ──────────────────────────────────────────────────

def verify_endpoint(**context) -> None:
    """
    Waits for the endpoint to reach InService status after the update,
    then sends one dummy record to confirm it responds correctly.

    We poll up to 30 minutes for the endpoint to become InService.
    A dummy all-zero feature vector should return a score near 0
    (not fraud) — we just confirm the response is a valid float.
    """
    sm         = _sm()
    sm_runtime = _sm_runtime()
    ti         = context["ti"]

    log.info(f"Waiting for endpoint {ENDPOINT_NAME} to reach InService...")
    for attempt in range(60):          # up to 30 minutes
        resp   = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = resp["EndpointStatus"]
        log.info(f"  [attempt {attempt+1:2d}] Endpoint status: {status}")

        if status == "InService":
            break
        if status == "Failed":
            raise RuntimeError(f"Endpoint update failed: {resp.get('FailureReason')}")
        time.sleep(30)
    else:
        raise TimeoutError("Endpoint did not reach InService within 30 minutes.")

    # smoke test — send a dummy record
    ti_data  = ti.xcom_pull(key="job_name")
    n_features = 408   # matches your feature engineering output
    dummy_csv  = ",".join(["0.0"] * n_features)

    try:
        response = sm_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=dummy_csv,
        )
        score = float(response["Body"].read().decode().strip())
        log.info(f"Smoke test passed — endpoint returned score: {score:.6f}")
        ti.xcom_push(key="smoke_test_score", value=score)
    except Exception as e:
        raise RuntimeError(f"Smoke test failed: {e}")


# ── task 7: log_completion ────────────────────────────────────────────────────

def log_completion(**context) -> None:
    """
    Writes a retraining report JSON to S3 summarising the full run:
    job name, model artifact, drift score that triggered it, and timing.

    This creates a retraining history in S3 that can be queried later
    to understand how often the model is being retrained and why.
    """
    ti           = context["ti"]
    s3           = _s3()
    trigger_conf = ti.xcom_pull(key="trigger_conf") or {}

    report = {
        "completed_at":    datetime.now(timezone.utc).isoformat(),
        "dag_run_id":      context["run_id"],
        "job_name":        ti.xcom_pull(key="job_name"),
        "model_artifact":  ti.xcom_pull(key="model_artifact"),
        "model_name":      ti.xcom_pull(key="model_name"),
        "endpoint_name":   ENDPOINT_NAME,
        "smoke_test_score": ti.xcom_pull(key="smoke_test_score"),
        "triggered_by":    trigger_conf.get("triggered_by",  "manual"),
        "drift_score":     trigger_conf.get("drift_score",   None),
    }

    ts      = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    s3_key  = f"{REPORTS_PREFIX}{ts}.json"

    log.info("=" * 60)
    log.info("  Retraining pipeline complete")
    log.info(f"  Job         : {report['job_name']}")
    log.info(f"  Model       : {report['model_name']}")
    log.info(f"  Endpoint    : {report['endpoint_name']}")
    log.info(f"  Smoke test  : {report['smoke_test_score']}")
    log.info(f"  Report      : s3://{BUCKET}/{s3_key}")
    log.info("=" * 60)

    if BUCKET:
        try:
            s3.put_object(
                Bucket=BUCKET,
                Key=s3_key,
                Body=json.dumps(report, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
        except Exception as e:
            log.error(f"Failed to write report to S3: {e}")


# ── DAG definition ────────────────────────────────────────────────────────────

default_args = {
    "owner":             "fraud-mlops",
    "retries":           1,
    "retry_delay":       timedelta(minutes=10),
    "execution_timeout": timedelta(hours=4),
}

with DAG(
    dag_id="model_retraining_dag",
    description="Retrains XGBoost fraud model on SageMaker when drift is detected",
    schedule_interval=None,       # only triggered by drift_detection_dag or manually
    start_date=days_ago(1),
    catchup=False,
    default_args=default_args,
    tags=["retraining", "sagemaker", "stage-8"],
    doc_md="""
## Model Retraining DAG

Triggered automatically by `drift_detection_dag` when fraud score drift
exceeds the configured threshold. Can also be run manually.

### What it does
1. **validate_trigger** — checks env vars and logs what triggered this run
2. **fetch_training_data** — confirms features.csv exists in S3
3. **launch_sagemaker_job** — starts a new XGBoost training job
4. **wait_for_training** — polls until job completes (up to 3h)
5. **update_endpoint** — blue/green swap of the SageMaker endpoint
6. **verify_endpoint** — waits for InService + smoke test
7. **log_completion** — writes retraining report to S3

### Zero downtime
SageMaker endpoint updates are blue/green. The fraud-consumer keeps
scoring live Binance trades throughout the model swap.
    """,
) as dag:

    t1 = PythonOperator(task_id="validate_trigger",      python_callable=validate_trigger)
    t2 = PythonOperator(task_id="fetch_training_data",   python_callable=fetch_training_data)
    t3 = PythonOperator(task_id="launch_sagemaker_job",  python_callable=launch_sagemaker_job)
    t4 = PythonOperator(task_id="wait_for_training",     python_callable=wait_for_training)
    t5 = PythonOperator(task_id="update_endpoint",       python_callable=update_endpoint)
    t6 = PythonOperator(task_id="verify_endpoint",       python_callable=verify_endpoint)
    t7 = PythonOperator(task_id="log_completion",        python_callable=log_completion)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7



    