"""
training_dag.py — Full retraining pipeline DAG

This DAG runs the complete model retraining cycle:
  1. Validate raw data in S3
  2. Run feature engineering
  3. Trigger SageMaker training job
  4. Evaluate the new model against the current one
  5. Deploy if the new model is better, skip if not

Schedule: every Sunday at midnight (weekly retraining).
Can also be triggered manually from the Airflow UI or by the drift DAG.

Every task in this DAG is a Python function decorated with @task.
This is Airflow's TaskFlow API — the modern way to write DAGs.
It is cleaner than the older Operator-based approach because you
write plain Python functions and Airflow handles the rest.
"""

import os
import json
import boto3
import logging
import pandas as pd
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException

log = logging.getLogger(__name__)

# ── DAG default arguments ──────────────────────────────────────────────────────
# These settings apply to every task in the DAG unless overridden.
# owner: who is responsible for this DAG
# retries: how many times to retry a failed task before giving up
# retry_delay: how long to wait between retries
# email_on_failure: we set False because we have no email configured yet
default_args = {
    "owner": "fraud-mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

# ── environment variables ──────────────────────────────────────────────────────
BUCKET         = os.getenv("S3_BUCKET")
RAW_KEY        = "data/raw/merged_raw.csv"
FEATURES_KEY   = "data/features/features.csv"
METADATA_KEY   = "data/features/feature_metadata.json"
MODELS_PREFIX  = "models/"
ROLE_ARN       = os.getenv("SAGEMAKER_ROLE_ARN")
ENDPOINT_NAME  = os.getenv("SAGEMAKER_ENDPOINT_NAME", "fraud-detection-endpoint")
REGION         = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Minimum AUC improvement required to replace the current model.
# If the new model's AUC is not at least this much better,
# we keep the existing model and skip deployment.
# This prevents deploying a slightly worse model due to random variation.
MIN_AUC_IMPROVEMENT = 0.005


@dag(
    dag_id="fraud_detection_training",

    # Run every Sunday at midnight.
    # Cron format: minute hour day-of-month month day-of-week
    # "0 0 * * 0" = at 00:00 on Sunday
    schedule_interval="0 0 * * 0",

    # The first date this DAG is eligible to run.
    # Setting it to today prevents Airflow from trying to backfill
    # all the missed runs since the beginning of time.
    start_date=datetime(2024, 1, 1),

    # Do not run missed schedules when the DAG is first activated.
    catchup=False,

    default_args=default_args,
    tags=["fraud-detection", "training", "sagemaker"],
    description="Weekly retraining pipeline for fraud detection model",
)
def fraud_detection_training():
    """
    The @dag decorator turns this function into an Airflow DAG.
    All @task functions defined inside it become tasks in the DAG.
    The dependencies between tasks are defined by how we call them
    at the bottom of this function.
    """

    @task
    def validate_raw_data() -> dict:
        """
        Task 1 — Check that the raw data in S3 looks correct.

        Before running expensive feature engineering and training,
        we verify the data is there and has the right shape.
        If something is wrong we fail loudly here instead of
        discovering it an hour into a training job.

        Returns a dict of validation stats that the next task can use.
        In Airflow's TaskFlow API, returning a value from a @task
        function automatically passes it to the next task as XCom
        (cross-communication between tasks).
        """
        log.info("Task 1: Validating raw data in S3...")

        s3 = boto3.client("s3")

        # check the file exists in S3
        try:
            response = s3.head_object(Bucket=BUCKET, Key=RAW_KEY)
            file_size_mb = response["ContentLength"] / 1e6
            log.info(f"  Raw file found: {file_size_mb:.1f} MB")
        except Exception as e:
            raise ValueError(
                f"Raw data not found at s3://{BUCKET}/{RAW_KEY}. "
                f"Run ingest.py and upload the file first. Error: {e}"
            )

        # download a sample to validate schema
        # we only download 1000 rows to keep this task fast
        log.info("  Downloading sample to validate schema...")
        obj = s3.get_object(Bucket=BUCKET, Key=RAW_KEY)
        df_sample = pd.read_csv(obj["Body"], nrows=1000)

        # check required columns exist
        required_cols = ["isFraud", "TransactionAmt"]
        for col in required_cols:
            assert col in df_sample.columns, \
                f"Required column '{col}' missing from raw data"

        fraud_rate = df_sample["isFraud"].mean()
        log.info(f"  Sample fraud rate: {fraud_rate:.2%} (expected ~3.5%)")
        log.info("  Validation passed.")

        return {
            "file_size_mb": file_size_mb,
            "sample_fraud_rate": fraud_rate,
            "validation_passed": True,
        }


    @task
    def run_feature_engineering(validation_result: dict) -> dict:
        """
        Task 2 — Run the full feature engineering pipeline.

        We import and call the same feature_engineering.py functions
        from Stage 3. This is why we mount src/ into the Airflow container —
        so DAGs can import from your source code directly.

        This task downloads raw data from S3, transforms it, and uploads
        features.csv and feature_metadata.json back to S3.

        Returns the path to the features file in S3.
        """
        log.info("Task 2: Running feature engineering...")
        log.info(f"  Previous task result: {validation_result}")

        import sys
        sys.path.insert(0, "/opt/airflow")

        from features.feature_engineering import (
            download_from_s3,
            extract_time_features,
            drop_high_null_columns,
            log_transform,
            encode_categoricals,
            fill_remaining_nulls,
            scale_numerical_features,
            calculate_class_weight,
            save_metadata,
            upload_to_s3,
            RAW_LOCAL,
            FEATURES_LOCAL,
            METADATA_LOCAL,
            TARGET_COL,
        )

        # run the full pipeline
        download_from_s3(RAW_KEY, RAW_LOCAL)

        df = pd.read_csv(RAW_LOCAL)
        target = df[TARGET_COL].copy()
        df = df.drop(columns=[TARGET_COL])

        df = extract_time_features(df)
        df = drop_high_null_columns(df)
        df = log_transform(df)
        df, encoders = encode_categoricals(df)
        df = fill_remaining_nulls(df)
        df, scaler, feature_cols = scale_numerical_features(df, target)

        class_weight = calculate_class_weight(target)
        df[TARGET_COL] = target.values

        df.to_csv(FEATURES_LOCAL, index=False)
        save_metadata(encoders, scaler, feature_cols, class_weight)

        upload_to_s3(FEATURES_LOCAL, FEATURES_KEY)
        upload_to_s3(METADATA_LOCAL, METADATA_KEY)

        log.info("  Feature engineering complete.")
        return {
            "features_s3_key": FEATURES_KEY,
            "n_features": len(feature_cols),
            "class_weight": class_weight,
        }


    @task
    def trigger_sagemaker_training(feature_result: dict) -> dict:
        """
        Task 3 — Start a SageMaker training job.

        SageMaker manages the training infrastructure — it spins up
        a machine, runs your training script, saves the model to S3,
        and shuts the machine down. You only pay for the minutes it runs.

        We use SageMaker's built-in XGBoost algorithm so we do not need
        to write a custom training script — we just pass hyperparameters.

        This task starts the job and waits for it to finish before
        returning. Training typically takes 10-20 minutes on ml.m4.xlarge.
        """
        log.info("Task 3: Starting SageMaker training job...")

        sm = boto3.client("sagemaker", region_name=REGION)

        # unique job name using timestamp — SageMaker requires unique names
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        job_name = f"fraud-detection-{timestamp}"

        # the features.csv in S3 is the training data
        training_data_uri = f"s3://{BUCKET}/{FEATURES_KEY}"

        # where SageMaker saves the trained model artifact
        output_uri = f"s3://{BUCKET}/{MODELS_PREFIX}"

        # SageMaker's built-in XGBoost image URI for us-east-1
        # This is a managed container — no Dockerfile needed
        xgboost_image = (
            f"683313688378.dkr.ecr.{REGION}.amazonaws.com"
            f"/sagemaker-xgboost:1.7-1"
        )

        log.info(f"  Job name: {job_name}")
        log.info(f"  Training data: {training_data_uri}")

        sm.create_training_job(
            TrainingJobName=job_name,
            AlgorithmSpecification={
                "TrainingImage": xgboost_image,
                "TrainingInputMode": "File",
            },
            RoleArn=ROLE_ARN,
            InputDataConfig=[{
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": training_data_uri,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "text/csv",
            }],
            OutputDataConfig={
                "S3OutputPath": output_uri,
            },
            ResourceConfig={
                # ml.m4.xlarge is free-tier eligible for training
                "InstanceType": "ml.m4.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 10,
            },
            StoppingCondition={
                # stop after 2 hours maximum to prevent runaway costs
                "MaxRuntimeInSeconds": 7200,
            },
            HyperParameters={
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "num_round": "200",
                "max_depth": "6",
                "eta": "0.1",
                "subsample": "0.8",
                "colsample_bytree": "0.8",
                # class weight to handle fraud imbalance
                # we use the value calculated in feature engineering
                "scale_pos_weight": str(
                    int(feature_result.get("class_weight", 28))
                ),
            },
        )

        log.info("  Training job started. Waiting for completion...")

        # wait for the training job to finish
        # this blocks the task until training is done
        waiter = sm.get_waiter("training_job_completed_or_stopped")
        waiter.wait(
            TrainingJobName=job_name,
            WaiterConfig={"Delay": 30, "MaxAttempts": 120},
        )

        # get the final status and model artifact location
        response = sm.describe_training_job(TrainingJobName=job_name)
        status = response["TrainingJobStatus"]
        model_artifact = response["ModelArtifacts"]["S3ModelArtifacts"]

        if status != "Completed":
            raise ValueError(f"Training job failed with status: {status}")

        # extract the AUC from the final metric
        metrics = {
            m["MetricName"]: m["Value"]
            for m in response.get("FinalMetricDataList", [])
        }
        auc = metrics.get("validation:auc", 0.0)

        log.info(f"  Training complete. AUC: {auc:.4f}")
        log.info(f"  Model artifact: {model_artifact}")

        return {
            "job_name": job_name,
            "model_artifact": model_artifact,
            "auc": auc,
            "status": status,
        }


    @task.branch
    def evaluate_model(training_result: dict) -> str:
        """
        Task 4 — Decide whether to deploy the new model or skip.

        @task.branch is a special task type that returns the task_id
        of the next task to run. This is how Airflow implements branching.

        We compare the new model's AUC against the currently deployed
        model's AUC (stored in S3 as a JSON file).

        If the new model is better by at least MIN_AUC_IMPROVEMENT,
        we return "deploy_model" to proceed to deployment.
        If not, we return "skip_deployment" to stop here.

        This gate prevents deploying a worse model — which could happen
        if the training data had issues or the model overfit.
        """
        log.info("Task 4: Evaluating new model...")

        new_auc = training_result.get("auc", 0.0)
        log.info(f"  New model AUC: {new_auc:.4f}")

        # try to load the current model's AUC from S3
        # if no model has been deployed yet, default to 0
        current_auc = 0.0
        s3 = boto3.client("s3")

        try:
            obj = s3.get_object(
                Bucket=BUCKET,
                Key=f"{MODELS_PREFIX}current_model_metrics.json"
            )
            metrics = json.loads(obj["Body"].read())
            current_auc = metrics.get("auc", 0.0)
            log.info(f"  Current model AUC: {current_auc:.4f}")
        except s3.exceptions.NoSuchKey:
            log.info("  No current model found — this is the first deployment.")

        improvement = new_auc - current_auc
        log.info(f"  AUC improvement: {improvement:.4f} "
                 f"(threshold: {MIN_AUC_IMPROVEMENT})")

        if improvement >= MIN_AUC_IMPROVEMENT or current_auc == 0.0:
            log.info("  Decision: DEPLOY new model")
            return "deploy_model"
        else:
            log.info("  Decision: SKIP deployment — new model is not better enough")
            return "skip_deployment"


    @task
    def deploy_model(training_result: dict):
        """
        Task 5a — Deploy the new model to the SageMaker endpoint.

        This creates or updates the SageMaker endpoint with the new
        model artifact. The endpoint is a REST API that the Kafka
        consumer calls to score each incoming Binance trade.

        If the endpoint already exists, we update it (blue-green style).
        If it does not exist, we create it fresh.

        After deploying, we save the new model's metrics to S3 so the
        next evaluation task can compare against it.
        """
        log.info("Task 5a: Deploying model to SageMaker endpoint...")

        sm = boto3.client("sagemaker", region_name=REGION)
        s3 = boto3.client("s3")

        model_artifact = training_result["model_artifact"]
        job_name = training_result["job_name"]
        new_auc = training_result["auc"]

        # SageMaker requires unique names for models and endpoint configs
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        model_name = f"fraud-model-{timestamp}"
        config_name = f"fraud-config-{timestamp}"

        xgboost_image = (
            f"683313688378.dkr.ecr.{REGION}.amazonaws.com"
            f"/sagemaker-xgboost:1.7-1"
        )

        # step 1: create a SageMaker model object pointing to the artifact
        log.info(f"  Creating SageMaker model: {model_name}")
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": xgboost_image,
                "ModelDataUrl": model_artifact,
            },
            ExecutionRoleArn=ROLE_ARN,
        )

        # step 2: create an endpoint configuration
        # this defines what instance type serves the model
        log.info(f"  Creating endpoint config: {config_name}")
        sm.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[{
                "VariantName": "primary",
                "ModelName": model_name,
                # ml.t2.medium is free-tier eligible for inference
                "InstanceType": "ml.t2.medium",
                "InitialInstanceCount": 1,
            }],
        )

        # step 3: create or update the endpoint
        try:
            # try to update existing endpoint first
            log.info(f"  Updating endpoint: {ENDPOINT_NAME}")
            sm.update_endpoint(
                EndpointName=ENDPOINT_NAME,
                EndpointConfigName=config_name,
            )
        except sm.exceptions.ClientError:
            # endpoint does not exist yet — create it
            log.info(f"  Creating new endpoint: {ENDPOINT_NAME}")
            sm.create_endpoint(
                EndpointName=ENDPOINT_NAME,
                EndpointConfigName=config_name,
            )

        # wait for the endpoint to be ready
        log.info("  Waiting for endpoint to be InService...")
        waiter = sm.get_waiter("endpoint_in_service")
        waiter.wait(
            EndpointName=ENDPOINT_NAME,
            WaiterConfig={"Delay": 30, "MaxAttempts": 60},
        )

        # save the new model's metrics to S3 for the next evaluation
        metrics = {
            "auc": new_auc,
            "job_name": job_name,
            "model_name": model_name,
            "deployed_at": datetime.utcnow().isoformat(),
        }
        s3.put_object(
            Bucket=BUCKET,
            Key=f"{MODELS_PREFIX}current_model_metrics.json",
            Body=json.dumps(metrics).encode("utf-8"),
        )

        log.info(f"  Endpoint {ENDPOINT_NAME} is live.")
        log.info(f"  New model AUC: {new_auc:.4f}")


    @task
    def skip_deployment():
        """
        Task 5b — Log that we skipped deployment and why.

        This task runs when the new model is not good enough to replace
        the current one. We raise AirflowSkipException which marks the
        task as "skipped" (shown in yellow in the Airflow UI) rather
        than failed or succeeded.
        """
        log.info("Deployment skipped — new model did not improve enough.")
        raise AirflowSkipException(
            "New model AUC did not improve by the required threshold. "
            "Keeping the current model."
        )


    # ── wire up the tasks ──────────────────────────────────────────────────────
    # This is where we define the order and dependencies.
    # Each line passes the output of one task as input to the next.
    # Airflow reads these assignments to build the dependency graph.

    validation   = validate_raw_data()
    features     = run_feature_engineering(validation)
    training     = trigger_sagemaker_training(features)
    branch       = evaluate_model(training)

    # after the branch decision, run either deploy or skip
    # we pass training to deploy so it knows which model artifact to use
    deploy       = deploy_model(training)
    skip         = skip_deployment()

    # tell Airflow that branch leads to either deploy or skip
    branch >> [deploy, skip]


# instantiate the DAG
fraud_detection_training()