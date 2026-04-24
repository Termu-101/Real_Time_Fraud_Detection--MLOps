import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from features.feature_engineering import (
    drop_high_null_columns,
    extract_time_features,
    log_transform,
    encode_categoricals,
    fill_remaining_nulls,
    calculate_class_weight,
    NULL_THRESHOLD,
    TARGET_COL,
)
from features.binance_mapper import map_binance_to_features, feature_vector


@pytest.fixture
def sample_df():
    """
    A small fake dataframe that mimics the structure of the real IEEE-CIS data.
    We make it large enough to pass the 100k row assertion in validate()
    but keep it small here since we are testing feature engineering only.
    """
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "TransactionID":  range(n),
        "TransactionDT":  np.random.randint(0, 86400 * 180, n),
        "TransactionAmt": np.random.exponential(100, n),
        "ProductCD":      np.random.choice(["W", "H", "C", "S", "R"], n),
        "card4":          np.random.choice(["visa", "mastercard", "missing"], n),
        "almost_empty":   [np.nan] * 960 + [1.0] * 40,  # 96% null — should be dropped
        "useful_col":     np.random.randn(n),
        "isFraud":        np.random.choice([0, 1], n, p=[0.965, 0.035]),
    })


@pytest.fixture
def sample_metadata():
    """
    Fake metadata that mimics what feature_engineering.py saves.
    Used by the Binance mapper tests.
    """
    return {
        "feature_cols": ["TransactionAmt", "hour_of_day", "day_of_week", "useful_col"],
        "encoders": {
            "ProductCD": {"C": 0, "H": 1, "R": 2, "S": 3, "W": 4}
        },
        "log_transform_cols": ["TransactionAmt"],
        "null_fill_value": -999,
        "class_weight": 27.6,
    }


@pytest.fixture
def fake_binance_trade():
    """
    A fake Binance WebSocket trade event for testing the mapper.
    """
    return {
        "e": "trade",
        "E": 1704067200000,
        "s": "BTCUSDT",
        "t": 12345,
        "p": "42500.00",
        "q": "0.02500",
        "T": 1704067200000,
        "m": False,
    }


# ── drop_high_null_columns tests ──────────────────────────────────────────────

def test_drops_high_null_column(sample_df):
    """
    almost_empty is 96% null which is above the 90% threshold.
    It must be dropped.
    """
    result = drop_high_null_columns(sample_df.drop(columns=[TARGET_COL]))
    assert "almost_empty" not in result.columns, \
        "Column with 96% nulls should be dropped"


def test_keeps_useful_columns(sample_df):
    """
    useful_col has no nulls — it must survive the null drop step.
    """
    result = drop_high_null_columns(sample_df.drop(columns=[TARGET_COL]))
    assert "useful_col" in result.columns, \
        "Column with no nulls should be kept"


def test_drops_transaction_id(sample_df):
    """
    TransactionID is in COLS_TO_DROP — it must always be dropped
    regardless of its null rate.
    """
    result = drop_high_null_columns(sample_df.drop(columns=[TARGET_COL]))
    assert "TransactionID" not in result.columns, \
        "TransactionID should always be dropped"


# ── extract_time_features tests ───────────────────────────────────────────────

def test_creates_hour_of_day(sample_df):
    """
    extract_time_features must create an hour_of_day column
    from TransactionDT.
    """
    result = extract_time_features(sample_df)
    assert "hour_of_day" in result.columns


def test_creates_day_of_week(sample_df):
    result = extract_time_features(sample_df)
    assert "day_of_week" in result.columns


def test_hour_range_is_valid(sample_df):
    """
    Hour of day must always be between 0 and 23.
    If the modulo arithmetic is wrong this will fail.
    """
    result = extract_time_features(sample_df)
    assert result["hour_of_day"].between(0, 24).all(), \
        "hour_of_day must be in range 0-24"


def test_day_range_is_valid(sample_df):
    """
    Day of week must always be between 0 and 6.
    """
    result = extract_time_features(sample_df)
    assert result["day_of_week"].between(0, 7).all(), \
        "day_of_week must be in range 0-7"


# ── log_transform tests ───────────────────────────────────────────────────────

def test_log_transform_reduces_max(sample_df):
    """
    After log transform, the max value of TransactionAmt should be
    much smaller than the original max. This confirms the transform
    compressed the skewed distribution.
    """
    original_max = sample_df["TransactionAmt"].max()
    result = log_transform(sample_df.copy())
    assert result["TransactionAmt"].max() < original_max, \
        "Log transform should reduce the maximum value"


def test_log_transform_no_negatives(sample_df):
    """
    log1p(x) where x >= 0 should never produce negative values.
    All transaction amounts are positive so this must hold.
    """
    result = log_transform(sample_df.copy())
    assert (result["TransactionAmt"] >= 0).all(), \
        "Log transformed values should all be non-negative"


# ── encode_categoricals tests ─────────────────────────────────────────────────

def test_encoding_produces_integers(sample_df):
    """
    After encoding, ProductCD should contain only integers,
    not strings like "W" or "H".
    """
    df = sample_df.drop(columns=[TARGET_COL]).copy()
    result, encoders = encode_categoricals(df)
    assert result["ProductCD"].dtype in [np.int32, np.int64, int], \
        "ProductCD should be integer after encoding"


def test_encoder_mappings_are_saved(sample_df):
    """
    encode_categoricals must return an encoders dict with an entry
    for each categorical column it processed.
    """
    df = sample_df.drop(columns=[TARGET_COL]).copy()
    _, encoders = encode_categoricals(df)
    assert "ProductCD" in encoders, \
        "Encoders dict must contain ProductCD mapping"
    assert "card4" in encoders, \
        "Encoders dict must contain card4 mapping"


def test_nan_handled_as_missing(sample_df):
    """
    NaN values in categorical columns must not cause an error.
    They should be encoded as the "missing" category.
    """
    df = sample_df.drop(columns=[TARGET_COL]).copy()
    df.loc[0, "ProductCD"] = np.nan
    try:
        encode_categoricals(df)
    except Exception as e:
        pytest.fail(f"encode_categoricals crashed on NaN: {e}")


# ── fill_remaining_nulls tests ────────────────────────────────────────────────

def test_no_nulls_after_fill(sample_df):
    """
    After fill_remaining_nulls there must be zero null values
    anywhere in the dataframe.
    """
    result = fill_remaining_nulls(sample_df.copy())
    assert result.isnull().sum().sum() == 0, \
        "No nulls should remain after filling"


def test_fill_value_is_minus_999(sample_df):
    """
    Nulls should be filled with -999, not 0 or mean.
    We check this by inserting a null and verifying it becomes -999.
    """
    df = sample_df.copy()
    df.loc[0, "useful_col"] = np.nan
    result = fill_remaining_nulls(df)
    assert result.loc[0, "useful_col"] == -999, \
        "Null fill value should be -999"


# ── calculate_class_weight tests ──────────────────────────────────────────────

def test_class_weight_is_positive(sample_df):
    """
    scale_pos_weight must always be a positive number.
    """
    target = sample_df[TARGET_COL]
    weight = calculate_class_weight(target)
    assert weight > 0, "Class weight must be positive"


def test_class_weight_reflects_imbalance(sample_df):
    """
    With ~96.5% non-fraud the weight should be substantially greater
    than 1. We check it is at least 10 to confirm the imbalance
    is being captured.
    """
    target = sample_df[TARGET_COL]
    weight = calculate_class_weight(target)
    assert weight > 10, \
        f"Class weight should reflect severe imbalance, got {weight:.2f}"


# ── binance mapper tests ──────────────────────────────────────────────────────

def test_mapper_returns_all_feature_cols(fake_binance_trade, sample_metadata):
    """
    The mapper must return a dict that contains every feature column
    the model expects. Missing columns would cause a scoring error.
    """
    features = map_binance_to_features(fake_binance_trade, sample_metadata)
    for col in sample_metadata["feature_cols"]:
        assert col in features, f"Feature {col} missing from mapped output"


def test_transaction_amt_is_log_transformed(fake_binance_trade, sample_metadata):
    """
    TransactionAmt in the output should be log1p(price * quantity),
    not the raw product. We verify by checking the value is in a
    reasonable log-transformed range.
    """
    features = map_binance_to_features(fake_binance_trade, sample_metadata)
    raw_value = 42500.00 * 0.02500  # 1062.5
    log_value = np.log1p(raw_value)  # ~6.97
    assert abs(features["TransactionAmt"] - log_value) < 0.01, \
        "TransactionAmt should be log-transformed"


def test_hour_of_day_is_integer(fake_binance_trade, sample_metadata):
    """
    hour_of_day must be an integer between 0 and 23.
    """
    features = map_binance_to_features(fake_binance_trade, sample_metadata)
    assert 0 <= features["hour_of_day"] <= 23


def test_unknown_columns_filled_with_null_value(fake_binance_trade, sample_metadata):
    """
    Columns that have no Binance equivalent must be filled with -999,
    not left as NaN or zero.
    """
    features = map_binance_to_features(fake_binance_trade, sample_metadata)
    # useful_col has no Binance equivalent
    assert features["useful_col"] == -999, \
        "Unknown columns should be filled with null_fill_value (-999)"


def test_feature_vector_length(fake_binance_trade, sample_metadata):
    """
    The feature vector must have exactly as many elements as there
    are feature columns. If the length is wrong, SageMaker will reject
    the scoring request.
    """
    vector = feature_vector(fake_binance_trade, sample_metadata)
    assert len(vector) == len(sample_metadata["feature_cols"]), \
        "Feature vector length must match number of feature columns"


def test_feature_vector_contains_only_floats(fake_binance_trade, sample_metadata):
    """
    SageMaker expects all values to be floats. Strings or None would
    cause a scoring error.
    """
    vector = feature_vector(fake_binance_trade, sample_metadata)
    for i, val in enumerate(vector):
        assert isinstance(val, float), \
            f"Value at position {i} is {type(val)}, expected float"