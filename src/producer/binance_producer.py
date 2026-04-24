import os
import json
import time
import logging
import websocket
from kafka import KafkaProducer
from dotenv import load_dotenv

load_dotenv()

# ── configuration ─────────────────────────────────────────────────────────────

# The Kafka broker address. Inside Docker this is kafka:29092.
# When running locally for testing this is localhost:9092.
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

# The Kafka topic we publish trades to.
# The consumer reads from this same topic.
TOPIC = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "transactions")

# The Binance WebSocket URL for trade streams.
# wss:// means it is a secure WebSocket connection (like https for WebSocket).
# btcusdt@trade subscribes to every trade on the BTC/USDT trading pair.
# This is completely free and requires no API key — Binance makes it public.
BINANCE_WS_URL = "wss://stream.binance.us:9443/ws/btcusdt@trade"

# How many seconds to wait before reconnecting if the WebSocket drops.
# Binance disconnects idle connections after 24 hours so reconnection
# handling is essential for a production-grade producer.
RECONNECT_DELAY = 5

# ── logging ───────────────────────────────────────────────────────────────────

# We use Python's logging module instead of print() because:
# - It includes timestamps automatically
# - Log levels (INFO, WARNING, ERROR) let you filter noise
# - Docker captures it properly with python -u
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PRODUCER] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── kafka producer setup ──────────────────────────────────────────────────────

def create_kafka_producer() -> KafkaProducer:
    """
    Creates and returns a KafkaProducer instance.

    We retry in a loop because Kafka might not be fully ready when
    the producer container starts. depends_on with healthcheck helps
    but there is still a small window where Kafka is up but not ready.

    value_serializer converts our Python dict to JSON bytes automatically.
    Kafka messages are raw bytes — we need to serialize before sending.
    """
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers=BOOTSTRAP_SERVERS,
                # convert dict -> JSON string -> UTF-8 bytes automatically
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                # wait up to 5 seconds for Kafka to acknowledge each message
                request_timeout_ms=5000,
                # retry sending up to 3 times if it fails
                retries=3,
            )
            log.info(f"Connected to Kafka at {BOOTSTRAP_SERVERS}")
            return producer
        except Exception as e:
            log.warning(f"Kafka not ready yet: {e}. Retrying in {RECONNECT_DELAY}s...")
            time.sleep(RECONNECT_DELAY)


# ── websocket handlers ────────────────────────────────────────────────────────

def on_message(ws, message, producer: KafkaProducer):
    """
    Called every time a new trade arrives from Binance.

    Each message is a JSON string representing one trade.
    We parse it, add a source field so downstream consumers know
    where this data came from, and publish it to the Kafka topic.

    producer.send() is non-blocking — it puts the message in a local
    buffer and a background thread handles the actual sending.
    This is why the producer can handle high-frequency trade streams
    without falling behind.
    """
    try:
        trade = json.loads(message)

        # add a source label so consumers know this is a Binance trade
        # and not data from some other source
        trade["source"] = "binance"

        # send to Kafka — key=symbol so all BTC trades go to the same partition
        # this preserves ordering within a trading pair
        producer.send(
            TOPIC,
            value=trade,
            key=trade.get("s", "BTCUSDT").encode("utf-8"),
        )

        # log every 100th trade to avoid flooding the logs
        # at ~10 trades/second this logs roughly every 10 seconds
        trade_id = trade.get("t", 0)
        if trade_id % 100 == 0:
            price = trade.get("p", "?")
            qty   = trade.get("q", "?")
            log.info(f"Trade {trade_id} | BTC/USDT {price} | qty {qty} -> Kafka")

    except json.JSONDecodeError as e:
        log.error(f"Failed to parse message: {e}")
    except Exception as e:
        log.error(f"Failed to publish to Kafka: {e}")


def on_error(ws, error):
    """
    Called when the WebSocket connection encounters an error.
    We log it and let the reconnection loop handle restarting.
    """
    log.error(f"WebSocket error: {error}")


def on_close(ws, close_status_code, close_msg):
    """
    Called when the WebSocket connection closes.
    Binance closes connections after 24 hours so this is expected.
    The reconnection loop in run() will restart the connection.
    """
    log.warning(f"WebSocket closed. Status: {close_status_code} | {close_msg}")


def on_open(ws):
    """
    Called when the WebSocket connection opens successfully.
    """
    log.info(f"Connected to Binance WebSocket: {BINANCE_WS_URL}")
    log.info(f"Streaming BTC/USDT trades to Kafka topic: {TOPIC}")


# ── main loop ─────────────────────────────────────────────────────────────────

def run():
    """
    Main loop — connects to Binance WebSocket and keeps it alive forever.

    WebSocket connections drop occasionally — network hiccups, Binance
    server maintenance, the 24-hour limit. The while True loop catches
    these drops and reconnects automatically so the stream never stops.

    This is what makes the producer production-grade — it self-heals.
    """
    log.info("Starting Binance producer...")
    producer = create_kafka_producer()

    while True:
        try:
            log.info(f"Connecting to {BINANCE_WS_URL}...")

            # websocket.WebSocketApp manages the connection lifecycle.
            # We pass our handler functions as callbacks.
            # on_message needs the producer so we use a lambda to pass it in.
            ws = websocket.WebSocketApp(
                BINANCE_WS_URL,
                on_open=on_open,
                on_message=lambda ws, msg: on_message(ws, msg, producer),
                on_error=on_error,
                on_close=on_close,
            )

            # run_forever() blocks here and processes messages until
            # the connection closes. ping_interval keeps the connection
            # alive by sending a ping every 30 seconds.
            ws.run_forever(ping_interval=30, ping_timeout=10)

        except Exception as e:
            log.error(f"Unexpected error: {e}")

        # if we reach here the connection has closed
        # wait a few seconds then reconnect
        log.info(f"Reconnecting in {RECONNECT_DELAY} seconds...")
        time.sleep(RECONNECT_DELAY)


if __name__ == "__main__":
    run()