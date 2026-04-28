"""
kafka_reader.py — background thread that feeds the dashboard with trade data.

Two modes — selected automatically:

  DEMO MODE  (DEMO_MODE=true, or Kafka unreachable after MAX_KAFKA_RETRIES)
    Generates synthetic BTC/USDT trades at ~2/second with realistic prices
    and a ~3% fraud rate. No Kafka, no AWS required. Perfect for local dev.

  LIVE MODE  (Kafka reachable)
    Consumes the real "transactions" Kafka topic. Scores via SageMaker if
    the endpoint is configured; otherwise simulates scores with a beta
    distribution (same as demo mode).

Drift is computed every DRIFT_WINDOW predictions and written to SQLite.
"""

import os
import sys
import json
import time
import random
import logging
import threading
import numpy as np
import boto3
from datetime import datetime, timezone
from pathlib import Path

# Support both Docker (/app layout) and local dev (repo root layout)
_here = Path(__file__).resolve().parent          # src/dashboard/
_root = _here.parent.parent                      # project root
for _p in ["/app", "/app/src", str(_root), str(_root / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import store

# Optional Kafka import — only available inside Docker or when kafka-python is installed
try:
    from kafka import KafkaConsumer
    _KAFKA_AVAILABLE = True
except ImportError:
    KafkaConsumer = None        # type: ignore[assignment,misc]
    _KAFKA_AVAILABLE = False

# ── config ────────────────────────────────────────────────────────────────────

BOOTSTRAP_SERVERS  = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
INPUT_TOPIC        = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "transactions")
ENDPOINT_NAME      = os.getenv("SAGEMAKER_ENDPOINT_NAME", "fraud-detection-endpoint")
FRAUD_THRESHOLD    = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
AWS_REGION         = os.getenv("AWS_REGION", "us-east-1")
DRIFT_WINDOW       = int(os.getenv("DRIFT_WINDOW", "50"))

# Set DEMO_MODE=true to skip Kafka and generate synthetic trades locally.
# Auto-enabled when kafka-python is not installed.
DEMO_MODE = (
    os.getenv("DEMO_MODE", "false").lower() == "true"
    or not _KAFKA_AVAILABLE
)

# How many times to try connecting to Kafka before falling back to demo mode
MAX_KAFKA_RETRIES = int(os.getenv("KAFKA_CONNECT_RETRIES", "3"))

# Seconds between synthetic trades in demo mode (~2 trades/sec)
DEMO_TRADE_INTERVAL = float(os.getenv("DEMO_TRADE_INTERVAL_S", "0.5"))

log = logging.getLogger("kafka_reader")

# ── module-level state ────────────────────────────────────────────────────────

_thread: threading.Thread | None = None
_stop_event = threading.Event()
_status = {"running": False, "total": 0, "errors": 0, "last_ts": None, "mode": "starting", "error": None}
_status_lock = threading.Lock()

_score_window: list[float] = []
_all_scores:   list[float] = []

# Realistic BTC price range for demo trades (USD)
_BTC_BASE   = 65_000.0
_BTC_SPREAD = 3_000.0


# ── demo trade generator ──────────────────────────────────────────────────────

def _fake_trade(trade_id: int) -> dict:
    """
    Generates a synthetic Binance-style trade event.
    Price walks randomly around _BTC_BASE to simulate a live market.
    """
    global _BTC_BASE
    _BTC_BASE += random.gauss(0, 20)
    _BTC_BASE  = max(50_000, min(80_000, _BTC_BASE))

    return {
        "e": "trade",
        "s": "BTCUSDT",
        "t": trade_id,
        "p": str(round(_BTC_BASE + random.uniform(-_BTC_SPREAD, _BTC_SPREAD), 2)),
        "q": str(round(random.uniform(0.001, 0.15), 5)),
        "T": int(time.time() * 1000),
        "m": random.choice([True, False]),
    }


def _simulate_score(trade: dict) -> float:
    """
    Returns a simulated fraud score that mirrors the real training distribution.

    ~3.5% of trades are fraudulent (matching the IEEE-CIS dataset base rate).
    - Legit trades:  beta(1.5, 15)  → mean ~0.09, almost never exceeds 0.5
    - Fraud trades:  beta(8,  2)    → mean ~0.80, almost always exceeds 0.5

    Large-quantity trades get a small extra boost (a real fraud signal).
    """
    qty_boost = min(float(trade.get("q", 0) or 0) * 0.3, 0.08)

    if random.random() < 0.035:
        # fraud trade — score peaks between 0.6 and 0.99
        score = float(np.random.beta(8, 2)) + qty_boost
    else:
        # legit trade — score concentrated near 0
        score = float(np.random.beta(1.5, 15)) + qty_boost

    return float(np.clip(score, 0.0, 1.0))


# ── sagemaker scoring ─────────────────────────────────────────────────────────

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

def _compute_drift(window: list[float], all_scores: list[float]) -> dict | None:
    if len(all_scores) < 2:
        return None
    global_mean = float(np.mean(all_scores))
    global_std  = float(np.std(all_scores)) or 1e-6
    window_mean = float(np.mean(window))
    window_std  = float(np.std(window))
    drift_score = abs(window_mean - global_mean) / global_std
    return {
        "ts":          datetime.now(timezone.utc).isoformat(),
        "drift_score": round(drift_score, 6),
        "mean_score":  round(window_mean, 6),
        "std_score":   round(window_std, 6),
        "window_size": len(window),
    }


# ── shared record processing ──────────────────────────────────────────────────

def _process_trade(trade: dict, sm_client, metadata) -> None:
    """Scores one trade, writes it to SQLite, updates drift window."""
    global _score_window, _all_scores

    # Demo mode always simulates — never calls the real SageMaker endpoint.
    # Without this guard, locally-loaded metadata + a valid boto3 client
    # would send fake BTC trades (mostly -999 features) to the real model,
    # which outputs near-1.0 fraud scores for out-of-distribution data.
    if not DEMO_MODE and sm_client and metadata:
        score = _score_trade(trade, metadata, sm_client)
    else:
        score = _simulate_score(trade)

    if score < 0:
        with _status_lock:
            _status["errors"] += 1
        return

    is_fraud = score >= FRAUD_THRESHOLD
    record = {
        "ts":          datetime.now(timezone.utc).isoformat(),
        "symbol":      trade.get("s", "BTCUSDT"),
        "price":       float(trade.get("p", 0) or 0),
        "quantity":    float(trade.get("q", 0) or 0),
        "fraud_score": round(score, 6),
        "is_fraud":    int(is_fraud),
        "threshold":   FRAUD_THRESHOLD,
    }
    store.insert_prediction(record)

    _score_window.append(score)
    _all_scores.append(score)

    if len(_score_window) >= DRIFT_WINDOW:
        drift = _compute_drift(_score_window, _all_scores)
        if drift:
            store.insert_drift(drift)
        _score_window = []

    with _status_lock:
        _status["total"]   += 1
        _status["last_ts"]  = record["ts"]


# ── reader loops ──────────────────────────────────────────────────────────────

def _demo_loop(sm_client, metadata) -> None:
    """Generates synthetic trades and processes them at DEMO_TRADE_INTERVAL."""
    log.info("Demo mode active — generating synthetic BTC/USDT trades.")
    with _status_lock:
        _status["running"] = True
        _status["mode"]    = "demo"
    trade_id = 1
    try:
        while not _stop_event.is_set():
            trade = _fake_trade(trade_id)
            _process_trade(trade, sm_client, metadata)
            trade_id += 1
            time.sleep(DEMO_TRADE_INTERVAL)
    finally:
        with _status_lock:
            _status["running"] = False
        log.info("Demo loop stopped.")


def _kafka_loop(sm_client, metadata) -> None:
    """Connects to Kafka and processes real trades. Falls back to demo on failure."""
    retries = 0
    consumer = None

    while not _stop_event.is_set() and retries < MAX_KAFKA_RETRIES:
        try:
            consumer = KafkaConsumer(
                INPUT_TOPIC,
                bootstrap_servers=BOOTSTRAP_SERVERS,
                group_id="dashboard-consumer-group",
                auto_offset_reset="latest",
                value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            )
            log.info(f"Dashboard connected to Kafka at {BOOTSTRAP_SERVERS}")
            break
        except Exception as e:
            retries += 1
            log.warning(f"Kafka not ready ({retries}/{MAX_KAFKA_RETRIES}): {e}. Retrying in 5s…")
            time.sleep(5)

    if consumer is None:
        log.warning(f"Kafka unreachable after {MAX_KAFKA_RETRIES} retries — switching to demo mode.")
        _demo_loop(sm_client, metadata)
        return

    with _status_lock:
        _status["running"] = True
        _status["mode"]    = "live"

    try:
        for message in consumer:
            if _stop_event.is_set():
                break
            _process_trade(message.value, sm_client, metadata)
    finally:
        consumer.close()
        with _status_lock:
            _status["running"] = False
        log.info("Kafka reader stopped.")


# ── main thread entry ─────────────────────────────────────────────────────────

def _reader_loop():
    # try to load real feature metadata (only needed for SageMaker scoring)
    metadata = None
    try:
        from features.binance_mapper import load_metadata
        metadata = load_metadata()
        log.info(f"Metadata loaded: {len(metadata['feature_cols'])} features")
    except Exception as e:
        log.warning(f"Feature metadata not found ({e}) — simulated scores will be used.")

    # try to build a SageMaker client (silently skip if no credentials)
    sm_client = None
    try:
        sm_client = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    except Exception as e:
        log.warning(f"SageMaker client unavailable ({e}) — simulated scores will be used.")

    while not _stop_event.is_set():
        try:
            store.init_db()
            if DEMO_MODE:
                _demo_loop(sm_client, metadata)
            else:
                _kafka_loop(sm_client, metadata)
        except Exception as e:
            log.error(f"Reader loop crashed: {e}", exc_info=True)
            with _status_lock:
                _status["running"] = False
                _status["error"]   = str(e)
            if not _stop_event.is_set():
                log.info("Restarting reader loop in 5 s…")
                time.sleep(5)


# ── public API ────────────────────────────────────────────────────────────────

def start():
    global _thread
    if _thread and _thread.is_alive():
        return
    _stop_event.clear()
    _thread = threading.Thread(target=_reader_loop, daemon=True, name="kafka-reader")
    _thread.start()
    log.info(f"Kafka reader thread started (demo_mode={DEMO_MODE}).")


def stop():
    _stop_event.set()


def get_status() -> dict:
    with _status_lock:
        return dict(_status)
