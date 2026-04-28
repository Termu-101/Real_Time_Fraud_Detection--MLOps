"""
store.py — SQLite-backed in-process data store for the dashboard.

We use SQLite (file-based, not in-memory) so data survives Streamlit
reruns. The file lives at /tmp/dashboard.db inside the container.

Three tables:
  predictions  — every scored trade (timestamp, score, is_fraud, price, qty)
  drift        — periodic drift scores computed from recent predictions
  metrics      — rolling model performance counters (tp, fp, tn, fn)
"""

import sqlite3
import threading
from datetime import datetime
from contextlib import contextmanager

DB_PATH = "/tmp/dashboard.db"

# one lock so background thread and Streamlit thread never write simultaneously
_lock = threading.Lock()


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    """Create tables if they don't exist yet."""
    with _lock, _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT    NOT NULL,
                symbol      TEXT,
                price       REAL,
                quantity    REAL,
                fraud_score REAL,
                is_fraud    INTEGER,
                threshold   REAL
            );

            CREATE TABLE IF NOT EXISTS drift (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT    NOT NULL,
                drift_score REAL,
                mean_score  REAL,
                std_score   REAL,
                window_size INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_pred_ts   ON predictions(ts);
            CREATE INDEX IF NOT EXISTS idx_drift_ts  ON drift(ts);
        """)


def insert_prediction(record: dict):
    with _lock, _conn() as con:
        con.execute("""
            INSERT INTO predictions (ts, symbol, price, quantity, fraud_score, is_fraud, threshold)
            VALUES (:ts, :symbol, :price, :quantity, :fraud_score, :is_fraud, :threshold)
        """, record)


def insert_drift(record: dict):
    with _lock, _conn() as con:
        con.execute("""
            INSERT INTO drift (ts, drift_score, mean_score, std_score, window_size)
            VALUES (:ts, :drift_score, :mean_score, :std_score, :window_size)
        """, record)


def fetch_recent_predictions(limit: int = 500) -> list[dict]:
    with _conn() as con:
        rows = con.execute("""
            SELECT * FROM predictions
            ORDER BY ts DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def fetch_fraud_alerts(limit: int = 50) -> list[dict]:
    with _conn() as con:
        rows = con.execute("""
            SELECT * FROM predictions
            WHERE is_fraud = 1
            ORDER BY ts DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def fetch_drift_history(limit: int = 200) -> list[dict]:
    with _conn() as con:
        rows = con.execute("""
            SELECT * FROM drift
            ORDER BY ts DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def fetch_summary_stats() -> dict:
    with _conn() as con:
        row = con.execute("""
            SELECT
                COUNT(*)                        AS total,
                SUM(is_fraud)                   AS flagged,
                AVG(fraud_score)                AS avg_score,
                MAX(fraud_score)                AS max_score,
                AVG(CAST(price AS REAL))        AS avg_price
            FROM predictions
        """).fetchone()
    return dict(row) if row else {}


def fetch_volume_over_time(bucket_seconds: int = 10) -> list[dict]:
    """Aggregate trade count and fraud count per time bucket."""
    with _conn() as con:
        rows = con.execute(f"""
            SELECT
                strftime('%Y-%m-%dT%H:%M:', ts) ||
                    printf('%02d', (CAST(strftime('%S', ts) AS INTEGER) / {bucket_seconds}) * {bucket_seconds})
                    AS bucket,
                COUNT(*)        AS total,
                SUM(is_fraud)   AS fraud_count,
                AVG(fraud_score) AS avg_score
            FROM predictions
            GROUP BY bucket
            ORDER BY bucket DESC
            LIMIT 60
        """).fetchall()
    return [dict(r) for r in rows]

