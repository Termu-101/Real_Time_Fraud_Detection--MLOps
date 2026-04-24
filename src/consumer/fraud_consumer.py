import os
import csv
import json
import time
import boto3
import logging
import numpy as np
from io import StringIO
from datetime import datetime
from kafka import KafkaConsumer
from dotenv import load_dotenv

load_dotenv()

# import the mapper we built in Stage 3
from features.binance_mapper import load_metadata, feature_vector

# ── configuration ─────────────────────────────────────────────────────────────

BOOTSTRAP_SERVERS   = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
INPUT_TOPIC         = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "transactions")
ALERTS_TOPIC        = os.getenv("KAFKA_TOPIC_ALERTS", "fraud-alerts")
BUCKET              = os.getenv("S3_BUCKET")
LOGS_PREFIX         = os.getenv("S3_LOGS_PREFIX", "logs/predictions/")
ENDPOINT_NAME       = os.getenv("SAGEMAKER_ENDPOINT_NAME", "fraud-detection-endpoint")

# Fraud probability threshold — if the model returns a score above this,
# we flag the trade as suspicious. 0.5 is the default but in practice
# you tune this based on how many false positives you can tolerate.
# A lower threshold catches more fraud but also flags more legitimate trades.
FRAUD_THRESHOLD     = float(os.getenv("FRAUD_THRESHOLD", "0.5"))

# How many predictions to accumulate before flushing logs to S3.
# We batch to avoid making thousands of tiny S3 uploads — one upload
# per 100 predictions is much more efficient than one per prediction.
LOG_BATCH_SIZE      = int(os.getenv("LOG_BATCH_SIZE", "100"))

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CONSUMER] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── kafka consumer setup ──────────────────────────────────────────────────────

def create_kafka_consumer() -> KafkaConsumer:
    """
    Creates and returns a KafkaConsumer that reads from the transactions topic.

    group_id groups multiple consumer instances together — if you ran
    two consumer containers, Kafka would split the partitions between them
    so each trade is processed exactly once. With one consumer it has no
    practical effect but it is good practice to always set it.

    auto_offset_reset="earliest" means if this consumer has never run before
    (no committed offset), start from the beginning of the topic.
    This is important for catching up after the consumer was down.

    value_deserializer reverses what the producer's value_serializer did —
    it converts raw bytes back to a Python dict automatically.
    """
    while True:
        try:
            consumer = KafkaConsumer(
                INPUT_TOPIC,
                bootstrap_servers=BOOTSTRAP_SERVERS,
                group_id="fraud-detection-group",
                auto_offset_reset="earliest",
                value_deserializer=lambda b: json.loads(b.decode("utf-8")),
                # how long to wait for new messages before returning empty
                consumer_timeout_ms=1000,
            )
            log.info(f"Connected to Kafka at {BOOTSTRAP_SERVERS}")
            log.info(f"Listening on topic: {INPUT_TOPIC}")
            return consumer
        except Exception as e:
            log.warning(f"Kafka not ready: {e}. Retrying in 5s...")
            time.sleep(5)


# ── sagemaker scoring ─────────────────────────────────────────────────────────

def score_trade(trade: dict, metadata: dict, sagemaker_client) -> float:
    """
    Sends a trade's features to the SageMaker endpoint and returns
    the fraud probability score (a float between 0 and 1).

    The endpoint expects a CSV string where each value corresponds to
    one feature column in the exact order the model was trained on.
    This is why feature_vector() returns a list in the correct order.

    A score close to 1.0 means the model thinks it is very likely fraud.
    A score close to 0.0 means the model thinks it is likely legitimate.
    """
    try:
        # convert trade to the feature vector the model expects
        vector = feature_vector(trade, metadata)

        # convert to CSV string — SageMaker's built-in XGBoost endpoint
        # accepts CSV as the input content type
        csv_body = ",".join(str(v) for v in vector)

        response = sagemaker_client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=csv_body,
        )

        # the response body is a stream — we read it and parse the score
        score = float(response["Body"].read().decode("utf-8").strip())
        return score

    except Exception as e:
        log.error(f"SageMaker scoring failed: {e}")
        # return -1 to indicate scoring failed
        # this lets us log the failure without crashing the consumer
        return -1.0


# ── s3 logging ────────────────────────────────────────────────────────────────

def flush_logs_to_s3(logs: list, s3_client):
    """
    Writes a batch of prediction logs to S3 as a CSV file.

    We use the current timestamp in the filename so each batch gets
    a unique file. This avoids overwriting previous logs and makes
    it easy to query logs by time period in Athena or Spark later.

    Each log row contains the trade details, the fraud score, and
    whether it was flagged. This is what Airflow's drift detection
    DAG will read in Stage 10.

    We write to S3 using StringIO (in-memory string buffer) rather
    than saving to disk first. This is cleaner inside a container
    where we do not want to manage temporary files.
    """
    if not logs:
        return

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    s3_key = f"{LOGS_PREFIX}{timestamp}.csv"

    # write logs to an in-memory CSV buffer
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=logs[0].keys())
    writer.writeheader()
    writer.writerows(logs)

    try:
        s3_client.put_object(
            Bucket=BUCKET,
            Key=s3_key,
            Body=buffer.getvalue().encode("utf-8"),
            ContentType="text/csv",
        )
        log.info(f"Flushed {len(logs)} predictions to s3://{BUCKET}/{s3_key}")
    except Exception as e:
        log.error(f"Failed to write logs to S3: {e}")


# ── main loop ─────────────────────────────────────────────────────────────────

def run():
    """
    Main loop — reads trades from Kafka and scores them forever.

    For each trade:
    1. Transform it using the Binance mapper from Stage 3
    2. Send it to the SageMaker endpoint for scoring
    3. Log the result (trade + score + flag) to an in-memory buffer
    4. If the score exceeds FRAUD_THRESHOLD, publish to the alerts topic
    5. Every LOG_BATCH_SIZE predictions, flush the buffer to S3

    The consumer runs forever — it is designed to be always-on.
    If it crashes, Docker Compose restarts it automatically (restart: always).
    """
    log.info("Starting fraud consumer...")
    log.info(f"Fraud threshold: {FRAUD_THRESHOLD}")
    log.info(f"Log batch size : {LOG_BATCH_SIZE}")

    # load feature metadata saved by feature_engineering.py in Stage 3
    metadata = load_metadata()
    log.info(f"Loaded metadata with {len(metadata['feature_cols'])} features")

    consumer       = create_kafka_consumer()
    s3_client      = boto3.client("s3")
    sagemaker_client = boto3.client("sagemaker-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))


    pending_logs   = []
    total_scored   = 0
    total_flagged  = 0

    log.info("Waiting for trades...")

    for message in consumer:
        trade = message.value

        # score the trade using SageMaker
        score = score_trade(trade, metadata, sagemaker_client)

        # determine if it is flagged as fraud
        is_fraud = score >= FRAUD_THRESHOLD and score != -1.0

        if is_fraud:
            total_flagged += 1
            log.warning(
                f"FRAUD ALERT | score={score:.4f} | "
                f"symbol={trade.get('s')} | "
                f"price={trade.get('p')} | "
                f"qty={trade.get('q')}"
            )

        # build a log record for this prediction
        log_record = {
            "timestamp":    datetime.utcnow().isoformat(),
            "trade_id":     trade.get("t"),
            "symbol":       trade.get("s"),
            "price":        trade.get("p"),
            "quantity":     trade.get("q"),
            "trade_time":   trade.get("T"),
            "is_buyer_maker": trade.get("m"),
            "fraud_score":  round(score, 6),
            "is_fraud":     int(is_fraud),
            "threshold":    FRAUD_THRESHOLD,
        }

        pending_logs.append(log_record)
        total_scored += 1

        # flush logs to S3 every LOG_BATCH_SIZE predictions
        if len(pending_logs) >= LOG_BATCH_SIZE:
            flush_logs_to_s3(pending_logs, s3_client)
            pending_logs = []
            log.info(
                f"Total scored: {total_scored:,} | "
                f"Total flagged: {total_flagged:,} | "
                f"Flag rate: {total_flagged/total_scored:.2%}"
            )


if __name__ == "__main__":
    run()