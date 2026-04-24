import json
import numpy as np
from pathlib import Path
from datetime import datetime

METADATA_PATH = Path("data/feature_metadata.json")


def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found at {METADATA_PATH}.\n"
            "Run feature_engineering.py first to generate it."
        )
 
    with open(METADATA_PATH, "r") as f:
        return json.load(f)
    


def map_binance_to_features(trade: dict, metadata: dict) -> dict:
    """
    Converts a raw Binance trade event into a feature vector the model can score.
 
    This is the core of the mapper. For each feature the model expects,
    we either map it from a real Binance field or fill it with the null
    fill value (-999) to indicate the information is not available.
 
    The mapping logic:
    - TransactionAmt -> price * quantity (total trade value in USD)
      This is the closest equivalent to a transaction amount.
    - hour_of_day -> extracted from the trade timestamp
    - day_of_week -> extracted from the trade timestamp
    - isBuyerMaker -> 1 if True, 0 if False
      This tells us the direction of the trade (buy vs sell pressure).
    - Everything else -> -999 (null fill value from metadata)
      These are bank-specific features (card numbers, email domains etc)
      that simply do not exist for crypto trades. The model was trained
      to treat -999 as "missing" which is the honest representation.
 
    Args:
        trade: raw Binance WebSocket trade event as a dict
        metadata: feature metadata loaded from feature_metadata.json
 
    Returns:
        dict of feature_name -> value, matching exactly what the model expects
    """
    # extract fields from the raw Binance event
    # float() is needed because Binance sends price and quantity as strings
    price    = float(trade.get("p", 0))
    quantity = float(trade.get("q", 0))
    trade_time_ms = trade.get("T", trade.get("E", 0))
    is_buyer_maker = trade.get("m", False)
 
    # convert millisecond timestamp to a datetime object
    trade_dt = datetime.fromtimestamp(trade_time_ms / 1000)
 
    # total trade value — this maps to TransactionAmt
    # price * quantity gives the USD value of the trade
    total_value = price * quantity
 
    # apply the same log transform that was applied to TransactionAmt 
    # we must apply this here because the model was trained on log-transformed values
    if "TransactionAmt" in metadata.get("log_transform_cols", []):
        total_value = np.log1p(total_value)
 
    null_val = metadata.get("null_fill_value", -999)
    feature_cols = metadata.get("feature_cols", [])
 
    # start with null fill value for every feature the model expects
    # this is the safe default — we override the ones we can actually map
    features = {col: null_val for col in feature_cols}
 
    # override with real values where we have a Binance equivalent
    if "TransactionAmt" in features:
        features["TransactionAmt"] = total_value
 
    if "hour_of_day" in features:
        features["hour_of_day"] = trade_dt.hour
 
    if "day_of_week" in features:
        # Monday=0, Sunday=6 — consistent with how we extracted it in Stage 3
        features["day_of_week"] = trade_dt.weekday()
 
    # isBuyerMaker tells us trade direction
    # True means the buyer was the market maker (sell-side aggressor)
    # False means the seller was the market maker (buy-side aggressor)
    # We encode True as 1 and False as 0 to make it numerical
    if "isBuyerMaker" in features:
        features["isBuyerMaker"] = int(is_buyer_maker)
 
    # add metadata about the trade for logging purposes
    # these are not model features — they are for traceability in S3 logs
    features["_trade_id"]    = trade.get("t", "unknown")
    features["_symbol"]      = trade.get("s", "unknown")
    features["_price"]       = price
    features["_quantity"]    = quantity
    features["_timestamp"]   = trade_dt.isoformat()
 
    return features

def feature_vector(trade: dict, metadata: dict) -> list:
    """
    Returns a feature vector as a list in the exact column order the model expects.
 
    The model does not accept a dict — it needs a list where every position
    corresponds to a specific feature column. The order must match exactly
    what was used during training.
 
    This is why we save feature_cols in the metadata — it records the
    exact order of columns. We use it here to build the list in the right order.
 
    Args:
        trade: raw Binance WebSocket trade event as a dict
        metadata: feature metadata from feature_metadata.json
 
    Returns:
        list of float values in the exact order the model was trained on
    """
    features = map_binance_to_features(trade, metadata)
    feature_cols = metadata.get("feature_cols", [])
 
    # build the vector in the exact column order from training
    # private fields starting with _ are excluded from the model input
    return [
        float(features.get(col, metadata.get("null_fill_value", -999)))
        for col in feature_cols
    ]


if __name__ == "__main__":
    # quick test with a fake Binance trade to verify the mapper works
    fake_trade = {
        "e": "trade",
        "E": 1704067200000,
        "s": "BTCUSDT",
        "t": 99999,
        "p": "42500.50",
        "q": "0.02300",
        "T": 1704067200000,
        "m": False,
    }
 
    print("Testing Binance mapper with a fake trade...")
    print(f"\nRaw trade: {json.dumps(fake_trade, indent=2)}")
 
    try:
        metadata = load_metadata()
        features = map_binance_to_features(fake_trade, metadata)
        vector = feature_vector(fake_trade, metadata)
 
        print(f"\nMapped features (first 10):")
        for k, v in list(features.items())[:10]:
            print(f"  {k}: {v}")
 
        print(f"\nFeature vector length: {len(vector)}")
        print(f"First 5 values: {vector[:5]}")
        print("\nMapper is working correctly.")
 
    except FileNotFoundError as e:
        print(f"\nNote: {e}")
        print("Run feature_engineering.py first to generate metadata.")
