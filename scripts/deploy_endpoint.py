"""
deploy_endpoint.py — One-time script to train and deploy the model.

Run this once manually to create your first model and endpoint.
After this Airflow handles all future retraining automatically.

What this script does:
  1. Triggers a SageMaker training job using features.csv from S3
  2. Waits for training to complete (10-20 minutes)
  3. Creates a SageMaker model from the trained artifact
  4. Creates an endpoint configuration
  5. Deploys the endpoint (the Kafka consumer calls this)
  6. Saves the model metrics to S3 for the Airflow evaluation gate

Run it with:
  python scripts/deploy_endpoint.py
"""

import os
import json
import time
import boto3
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# ── configuration ─────────────────────────────────────────────────────────────
BUCKET        = os.getenv("S3_BUCKET")
FEATURES_KEY  = "data/features/features.csv"
MODELS_PREFIX = "models/"
ROLE_ARN      = os.getenv("SAGEMAKER_ROLE_ARN")
ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "fraud-detection-endpoint")
REGION        = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# SageMaker's built-in XGBoost container image for us-east-1
# This is a managed container — no custom Dockerfile needed for training
XGBOOST_IMAGE = (
    f"683313688378.dkr.ecr.{REGION}.amazonaws.com"
    f"/sagemaker-xgboost:1.7-1"
)

def prepare_training_data(s3_client) -> str:
    """
    Reformats features.csv for SageMaker's built-in XGBoost container.

    SageMaker's XGBoost expects:
    - No header row
    - Label column (isFraud) as the FIRST column
    - All other feature columns after it

    We download features.csv, reformat it, and upload it to a
    separate S3 key so the original file stays intact.

    Returns the S3 prefix where the reformatted file is saved.
    """
    import io
    print("Preparing training data for SageMaker...")

    # download features.csv
    obj = s3_client.get_object(Bucket=BUCKET, Key=FEATURES_KEY)
    df = pd.read_csv(obj["Body"])

    print(f"  Original shape: {df.shape}")
    print(f"  Moving isFraud to first column and removing header...")

    # move isFraud to the first column
    cols = ["isFraud"] + [c for c in df.columns if c != "isFraud"]
    df = df[cols]

    # convert to CSV string with no header
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, header=False)

    # upload to a separate key for SageMaker
    sagemaker_key = "data/sagemaker/train/features_sagemaker.csv"
    s3_client.put_object(
        Bucket=BUCKET,
        Key=sagemaker_key,
        Body=csv_buffer.getvalue().encode("utf-8"),
    )

    print(f"  Uploaded to: s3://{BUCKET}/{sagemaker_key}")
    return "data/sagemaker/train/"


def validate_config():
    """
    Checks that all required environment variables are set before starting.
    Fails immediately with a clear message rather than a confusing AWS error.
    """
    print("Validating configuration...")

    missing = []
    if not BUCKET:
        missing.append("S3_BUCKET")
    if not ROLE_ARN:
        missing.append("SAGEMAKER_ROLE_ARN")

    if missing:
        raise ValueError(
            f"Missing required environment variables: {missing}\n"
            f"Make sure your .env file is filled in correctly.\n"
            f"SAGEMAKER_ROLE_ARN should look like:\n"
            f"arn:aws:iam::245987718424:role/SageMakerRole"
        )

    print(f"  Bucket        : {BUCKET}")
    print(f"  Role ARN      : {ROLE_ARN}")
    print(f"  Endpoint name : {ENDPOINT_NAME}")
    print(f"  Region        : {REGION}")
    print("  Configuration valid.\n")


def trigger_training_job(sm_client) -> tuple:
    """
    Starts a SageMaker training job and waits for it to complete.

    SageMaker manages all the infrastructure — it spins up an ml.m4.xlarge
    instance, runs the training, saves the model to S3, and shuts down.
    You only pay for the minutes the instance runs (~15 minutes = ~$0.03).

    Returns the job name and model artifact S3 path.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    job_name = f"fraud-detection-{timestamp}"

    print(f"Starting training job: {job_name}")
    print(f"Training data: s3://{BUCKET}/{FEATURES_KEY}")
    print("This takes 10-20 minutes...\n")

    sm_client.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": XGBOOST_IMAGE,
            "TrainingInputMode": "File",
        },
        RoleArn=ROLE_ARN,
        InputDataConfig=[{
            "ChannelName": "train",
            # SageMaker downloads everything at this S3 prefix
            # to /opt/ml/input/data/train/ on the training instance
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{BUCKET}/data/sagemaker/train/",
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
            "ContentType": "text/csv",
        }],
        OutputDataConfig={
            # SageMaker saves model.tar.gz here after training
            "S3OutputPath": f"s3://{BUCKET}/{MODELS_PREFIX}",
        },
        ResourceConfig={
            # ml.m4.xlarge: 4 vCPUs, 16GB RAM — free tier eligible
            # sufficient for training on 590k rows
            "InstanceType": "ml.m4.xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 20,
        },
        StoppingCondition={
            # safety limit — stop after 2 hours maximum
            # prevents runaway costs if something goes wrong
            "MaxRuntimeInSeconds": 7200,
        },
        HyperParameters={
            "objective":        "binary:logistic",
            "eval_metric":      "auc",
            "num_round":        "200",
            "max_depth":        "6",
            "eta":              "0.1",
            "subsample":        "0.8",
            "colsample_bytree": "0.8",
            # 28 = ratio of non-fraud to fraud cases (~96.5% / 3.5%)
            # this makes XGBoost treat each fraud case as 28x more important
            "scale_pos_weight": "28",
        },
    )

    # poll training job status every 30 seconds
    # print progress so you know it is still running
    while True:
        response = sm_client.describe_training_job(TrainingJobName=job_name)
        status = response["TrainingJobStatus"]
        seconds_running = response.get("TrainingTimeInSeconds", 0)

        print(f"  Status: {status} | Running: {seconds_running}s")

        if status == "Completed":
            model_artifact = response["ModelArtifacts"]["S3ModelArtifacts"]
            metrics = {
                m["MetricName"]: m["Value"]
                for m in response.get("FinalMetricDataList", [])
            }
            auc = metrics.get("validation:auc", 0.0)
            print(f"\nTraining complete!")
            print(f"  AUC           : {auc:.4f}")
            print(f"  Model artifact: {model_artifact}")
            return job_name, model_artifact, auc

        elif status in ("Failed", "Stopped"):
            reason = response.get("FailureReason", "Unknown")
            raise RuntimeError(
                f"Training job failed with status: {status}\n"
                f"Reason: {reason}\n"
                f"Check the SageMaker console for full logs."
            )

        time.sleep(30)


def create_model(sm_client, model_artifact: str, timestamp: str) -> str:
    """
    Creates a SageMaker model object from the trained artifact.

    A SageMaker model object is a pointer that says:
    'use this container image + this model artifact to serve predictions'.
    It is not the endpoint itself — that comes next.
    """
    model_name = f"fraud-model-{timestamp}"
    print(f"\nCreating SageMaker model: {model_name}")

    sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": XGBOOST_IMAGE,
            # the trained model artifact from the training job
            "ModelDataUrl": model_artifact,
        },
        ExecutionRoleArn=ROLE_ARN,
    )

    print(f"  Model created: {model_name}")
    return model_name


def create_endpoint_config(sm_client, model_name: str, timestamp: str) -> str:
    """
    Creates an endpoint configuration specifying what instance serves the model.

    ml.t2.medium is the smallest SageMaker inference instance and is
    free-tier eligible. It is sufficient for our use case since the
    Kafka consumer sends one trade at a time, not batch requests.
    """
    config_name = f"fraud-config-{timestamp}"
    print(f"Creating endpoint config: {config_name}")

    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "primary",
            "ModelName": model_name,
            "InstanceType": "ml.t2.medium",
            "InitialInstanceCount": 1,
            # InitialVariantWeight controls traffic percentage
            # 1 means 100% of traffic goes to this variant
            "InitialVariantWeight": 1,
        }],
    )

    print(f"  Config created: {config_name}")
    return config_name


def deploy_endpoint(sm_client, config_name: str):
    """
    Creates or updates the SageMaker endpoint.

    If the endpoint already exists we update it with the new config.
    This is a blue-green deployment — SageMaker keeps the old endpoint
    running until the new one is fully ready, then switches traffic over.
    No downtime.

    If the endpoint does not exist we create it fresh.
    Either way we wait until it is InService before returning.
    """
    print(f"\nDeploying endpoint: {ENDPOINT_NAME}")

    try:
        # check if endpoint exists
        sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        # if we get here, it exists — update it
        print("  Endpoint exists — updating (blue-green deployment)...")
        sm_client.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name,
        )
    except sm_client.exceptions.ClientError:
        # endpoint does not exist — create it
        print("  Creating new endpoint...")
        sm_client.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name,
        )

    # wait for endpoint to be InService
    print("  Waiting for endpoint to be InService...")
    print("  This takes 5-10 minutes...\n")

    while True:
        response = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = response["EndpointStatus"]
        print(f"  Endpoint status: {status}")

        if status == "InService":
            print(f"\nEndpoint is live: {ENDPOINT_NAME}")
            return
        elif status in ("Failed", "RollingBack"):
            reason = response.get("FailureReason", "Unknown")
            raise RuntimeError(
                f"Endpoint deployment failed: {status}\n"
                f"Reason: {reason}"
            )

        time.sleep(30)


def save_model_metrics(s3_client, job_name: str, auc: float):
    """
    Saves the deployed model's metrics to S3.

    The Airflow evaluation gate reads this file to compare the current
    model's AUC against a newly trained model's AUC. If the new model
    is not better, Airflow skips deployment and keeps this one running.
    """
    metrics = {
        "auc": auc,
        "job_name": job_name,
        "deployed_at": datetime.utcnow().isoformat(),
        "endpoint_name": ENDPOINT_NAME,
    }

    s3_client.put_object(
        Bucket=BUCKET,
        Key=f"{MODELS_PREFIX}current_model_metrics.json",
        Body=json.dumps(metrics, indent=2).encode("utf-8"),
    )

    print(f"\nModel metrics saved to S3:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Job: {job_name}")


def run():
    print("=" * 60)
    print("  Fraud Detection — Train and Deploy")
    print("=" * 60 + "\n")

    validate_config()

    s3_client = boto3.client("s3")
    sm_client = boto3.client("sagemaker", region_name=REGION)

    sagemaker_data_prefix = prepare_training_data(s3_client)


    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    # step 1 — train
    job_name, model_artifact, auc = trigger_training_job(sm_client)

    # step 2 — create model
    model_name = create_model(sm_client, model_artifact, timestamp)

    # step 3 — create endpoint config
    config_name = create_endpoint_config(sm_client, model_name, timestamp)

    # step 4 — deploy endpoint
    deploy_endpoint(sm_client, config_name)

    # step 5 — save metrics for Airflow evaluation gate
    save_model_metrics(s3_client, job_name, auc)

    print("\n" + "=" * 60)
    print("  Stage 7 complete.")
    print("=" * 60)
    print(f"\n  Endpoint name : {ENDPOINT_NAME}")
    print(f"  Model AUC     : {auc:.4f}")
    print(f"\n  The Kafka consumer will now score live Binance trades.")
    print(f"  The SageMaker error in the consumer logs will stop.")
    print(f"\n  Next: Stage 8 — Monitoring + drift detection")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run()