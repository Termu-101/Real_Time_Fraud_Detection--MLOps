"""
kafka_reader.py — background thread that consumes the Kafka transactions
topic, calls SageMaker, and writes results to the SQLite store.

This runs in a daemon thread so Streamlit's main thread stays free for
rendering. We use threading.Event to signal a clean shutdown.

Drift is computed every DRIFT_WINDOW predictions using a simple
statistical approach: compare the mean fraud score of the latest window
to the overall historical mean. A large deviation signals drift.
"""

import os
import sys
import json
import time
import logging
import threading
import numpy as np
import boto3
from datetime import datetime
from kafka import KafkaConsumer

# ensure features/ is importable inside the container
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app")

import store

# ── config ────────────────────────────────────────────────────────────────────

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
INPUT_TOPIC       = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "transactions")
ENDPOINT_NAME     = os.getenv("SAGEMAKER_ENDPOINT_NAME", "fraud-detection-endpoint")
FRAUD_THRESHOLD   = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
AWS_REGION        = os.getenv("AWS_REGION", "us-east-1")

# compute a drift score every N predictions
DRIFT_WINDOW = int(os.getenv("DRIFT_WINDOW", "50"))

log = logging.getLogger("kafka_reader")

# ── module-level state ────────────────────────────────────────────────────────

_thread: threading.Thread | None = None
_stop_event = threading.Event()
_status = {"running": False, "total": 0, "errors": 0, "last_ts": None}
_status_lock = threading.Lock()

# sliding window of recent scores for drift computation
_score_window: list[float] = []
_all_scores:   list[float] = []


# ── sagemaker ─────────────────────────────────────────────────────────────────

def _score_trade(trade: dict, metadata: dict, sm_client) -> float:
    try:
        from features.binance_mapper import feature_vector
        vector   = feature_vector(trade, metadata)
        csv_body = ",".join(str(v) for v in vector)
        resp     = sm_client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=csv_body,
        )
        return float(resp["Body"].read().decode().strip())
    except Exception as e:
        log.error(f"SageMaker error: {e}")
        return -1.0


# ── drift ─────────────────────────────────────────────────────────────────────

def _compute_drift(window: list[float], all_scores: list[float]) -> dict:
    """
    Drift score = absolute difference between window mean and global mean,
    normalised by global std. A value > 1.0 is worth watching; > 2.0 is
    a clear signal that the score distribution has shifted.
    """
    if len(all_scores) < 2:
        return None

    global_mean = float(np.mean(all_scores))
    global_std  = float(np.std(all_scores)) or 1e-6
    window_mean = float(np.mean(window))
    window_std  = float(np.std(window))
    drift_score = abs(window_mean - global_mean) / global_std

    return {
        "ts":          datetime.utcnow().isoformat(),
        "drift_score": round(drift_score, 6),
        "mean_score":  round(window_mean, 6),
        "std_score":   round(window_std, 6),
        "window_size": len(window),
    }


# ── main reader loop ──────────────────────────────────────────────────────────

def _reader_loop():
    global _score_window, _all_scores

    store.init_db()

    # load metadata
    try:
        from features.binance_mapper import load_metadata
        metadata = load_metadata()
        log.info(f"Metadata loaded: {len(metadata['feature_cols'])} features")
    except Exception as e:
        log.error(f"Could not load metadata: {e}. Running in score=0 demo mode.")
        metadata = None

    # build clients
    try:
        sm_client = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    except Exception as e:
        log.warning(f"SageMaker client failed: {e}. Scores will be simulated.")
        sm_client = None

    # connect to Kafka (retry until ready)
    consumer = None
    while not _stop_event.is_set():
        try:
            consumer = KafkaConsumer(
                INPUT_TOPIC,
                bootstrap_servers=BOOTSTRAP_SERVERS,
                group_id="dashboard-consumer-group",
                auto_offset_reset="latest",
                value_deserializer=lambda b: json.loads(b.decode("utf-8")),
                # NO consumer_timeout_ms — loop blocks forever waiting for
                # messages rather than exiting after 1s of silence
            )
            log.info(f"Dashboard consumer connected to {BOOTSTRAP_SERVERS}")
            break
        except Exception as e:
            log.warning(f"Kafka not ready: {e}. Retrying in 5s...")
            time.sleep(5)

    if consumer is None:
        return

    with _status_lock:
        _status["running"] = True

    try:
        for message in consumer:
            if _stop_event.is_set():
                break

            trade = message.value

            # score — use SageMaker if available, else simulate
            if sm_client and metadata:
                score = _score_trade(trade, metadata, sm_client)
            else:
                # simulate a realistic-looking score for demo / no-endpoint mode
                price = float(trade.get("p", 0) or 0)
                score = float(np.clip(np.random.beta(1, 20) + (price % 1) * 0.05, 0, 1))

            if score < 0:
                with _status_lock:
                    _status["errors"] += 1
                continue

            is_fraud = score >= FRAUD_THRESHOLD

            record = {
                "ts":          datetime.utcnow().isoformat(),
                "symbol":      trade.get("s", "BTCUSDT"),
                "price":       float(trade.get("p", 0) or 0),
                "quantity":    float(trade.get("q", 0) or 0),
                "fraud_score": round(score, 6),
                "is_fraud":    int(is_fraud),
                "threshold":   FRAUD_THRESHOLD,
            }
            store.insert_prediction(record)

            # update drift window
            _score_window.append(score)
            _all_scores.append(score)

            if len(_score_window) >= DRIFT_WINDOW:
                drift = _compute_drift(_score_window, _all_scores)
                if drift:
                    store.insert_drift(drift)
                _score_window = []          # reset window

            with _status_lock:
                _status["total"]  += 1
                _status["last_ts"] = record["ts"]

    finally:
        consumer.close()
        with _status_lock:
            _status["running"] = False
        log.info("Dashboard Kafka reader stopped.")


# ── public API ────────────────────────────────────────────────────────────────

def start():
    global _thread
    if _thread and _thread.is_alive():
        return
    _stop_event.clear()
    _thread = threading.Thread(target=_reader_loop, daemon=True, name="kafka-reader")
    _thread.start()
    log.info("Kafka reader thread started.")


def stop():
    _stop_event.set()


def get_status() -> dict:
    with _status_lock:
        return dict(_status)