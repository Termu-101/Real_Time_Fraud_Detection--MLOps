"""
test_pipeline.py — Integration tests for the full pipeline.

These tests verify that all the pieces of the pipeline connect correctly.
They run in CI on every push and pull request.

We use unittest.mock to replace real AWS and Kafka calls with fake ones.
This means tests run in seconds without needing real infrastructure,
and they work in GitHub Actions without real AWS credentials.

The goal is not to test AWS itself — Amazon tests that.
The goal is to test YOUR code — does it call AWS correctly,
does it handle errors gracefully, does it produce the right output.
"""

import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path


# ── test that feature engineering produces valid output ────────────────────────

class TestFeatureEngineeringPipeline:
    """
    Tests that the feature engineering pipeline produces output
    in the exact format the model and consumer expect.
    """

    @pytest.fixture
    def raw_df(self):
        """
        Fake raw dataframe mimicking the merged IEEE-CIS data.
        Large enough to pass validation checks.
        """
        np.random.seed(42)
        n = 500
        return pd.DataFrame({
            "TransactionID":  range(n),
            "TransactionDT":  np.random.randint(0, 86400 * 180, n),
            "TransactionAmt": np.random.exponential(100, n),
            "ProductCD":      np.random.choice(["W", "H", "C"], n),
            "card4":          np.random.choice(["visa", "mastercard"], n),
            "card6":          np.random.choice(["credit", "debit"], n),
            "P_emaildomain":  np.random.choice(["gmail.com", "yahoo.com"], n),
            "R_emaildomain":  np.random.choice(["gmail.com", "missing"], n),
            "useful_num":     np.random.randn(n),
            "isFraud":        np.random.choice([0, 1], n, p=[0.965, 0.035]),
        })

    def test_output_has_no_nulls(self, raw_df):
        """
        After feature engineering, there must be zero nulls.
        The model cannot handle NaN — it would produce garbage predictions.
        """
        from features.feature_engineering import (
            extract_time_features,
            drop_high_null_columns,
            log_transform,
            encode_categoricals,
            fill_remaining_nulls,
            TARGET_COL,
        )

        target = raw_df[TARGET_COL].copy()
        df = raw_df.drop(columns=[TARGET_COL])

        df = extract_time_features(df)
        df = drop_high_null_columns(df)
        df = log_transform(df)
        df, _ = encode_categoricals(df)
        df = fill_remaining_nulls(df)

        assert df.isnull().sum().sum() == 0, \
            "Feature engineering output must have zero nulls"

    def test_transaction_amt_is_transformed(self, raw_df):
        """
        TransactionAmt must be log-transformed in the output.
        We verify this by checking the max is much smaller than the raw max.
        """
        from features.feature_engineering import log_transform, TARGET_COL

        raw_max = raw_df["TransactionAmt"].max()
        df = log_transform(raw_df.drop(columns=[TARGET_COL]).copy())
        transformed_max = df["TransactionAmt"].max()

        assert transformed_max < raw_max, \
            "TransactionAmt must be log-transformed (max should decrease)"

    def test_time_features_created(self, raw_df):
        """
        extract_time_features must create hour_of_day and day_of_week.
        These are important fraud signals — missing them hurts model quality.
        """
        from features.feature_engineering import extract_time_features, TARGET_COL

        df = extract_time_features(raw_df.copy())
        assert "hour_of_day" in df.columns
        assert "day_of_week" in df.columns

    def test_encoders_are_consistent(self, raw_df):
        """
        Encoding the same data twice must produce the same result.
        If it does not, live data encoded differently than training data
        would produce nonsense predictions.
        """
        from features.feature_engineering import encode_categoricals, TARGET_COL

        df1 = raw_df.drop(columns=[TARGET_COL]).copy()
        df2 = raw_df.drop(columns=[TARGET_COL]).copy()

        result1, encoders1 = encode_categoricals(df1)
        result2, encoders2 = encode_categoricals(df2)

        assert encoders1["ProductCD"] == encoders2["ProductCD"], \
            "Encoders must produce consistent mappings for the same data"


# ── test that the Binance mapper produces valid feature vectors ────────────────

class TestBinanceMapperPipeline:
    """
    Tests that the Binance mapper correctly converts raw trade events
    into feature vectors the model can score.
    """

    @pytest.fixture
    def metadata(self):
        return {
            "feature_cols": [
                "TransactionAmt", "hour_of_day", "day_of_week", "card1"
            ],
            "log_transform_cols": ["TransactionAmt"],
            "null_fill_value": -999,
            "encoders": {},
        }

    @pytest.fixture
    def binance_trade(self):
        return {
            "e": "trade",
            "s": "BTCUSDT",
            "t": 99999,
            "p": "65000.00",
            "q": "0.01500",
            "T": 1704067200000,
            "m": False,
        }

    def test_vector_has_correct_length(self, binance_trade, metadata):
        """
        The feature vector must have exactly as many elements as feature_cols.
        A wrong-length vector causes SageMaker to reject the request.
        """
        from features.binance_mapper import feature_vector

        vector = feature_vector(binance_trade, metadata)
        assert len(vector) == len(metadata["feature_cols"]), \
            f"Vector length {len(vector)} must match feature_cols length {len(metadata['feature_cols'])}"

    def test_all_values_are_float(self, binance_trade, metadata):
        """
        Every value in the feature vector must be a float.
        SageMaker's CSV endpoint rejects non-numeric values.
        """
        from features.binance_mapper import feature_vector

        vector = feature_vector(binance_trade, metadata)
        for i, val in enumerate(vector):
            assert isinstance(val, float), \
                f"Position {i} has type {type(val).__name__}, expected float"

    def test_missing_features_filled_with_null_value(self, binance_trade, metadata):
        """
        Features that have no Binance equivalent (like card1) must be
        filled with the null_fill_value (-999), not left as NaN.
        """
        from features.binance_mapper import map_binance_to_features

        features = map_binance_to_features(binance_trade, metadata)
        assert features["card1"] == -999, \
            "card1 has no Binance equivalent and must be -999"

    def test_transaction_amt_is_log_transformed(self, binance_trade, metadata):
        """
        TransactionAmt in the feature vector must be log1p(price * qty).
        If we forget the log transform, the model sees raw dollar values
        it was never trained on and produces wrong predictions.
        """
        from features.binance_mapper import map_binance_to_features
        import numpy as np

        features = map_binance_to_features(binance_trade, metadata)
        expected = np.log1p(65000.00 * 0.01500)
        assert abs(features["TransactionAmt"] - expected) < 0.001


# ── test that the consumer handles SageMaker errors gracefully ────────────────

class TestConsumerErrorHandling:
    """
    Tests that the fraud consumer handles errors gracefully
    instead of crashing. A consumer that crashes on one bad trade
    stops processing all subsequent trades.
    """

    def test_scoring_failure_returns_minus_one(self):
        """
        When SageMaker is unavailable (endpoint not found, network error),
        score_trade must return -1.0 instead of raising an exception.
        Returning -1.0 lets the consumer log the failure and continue
        processing the next trade.
        """
        from src.consumer.fraud_consumer import score_trade

        mock_metadata = {
            "feature_cols": ["TransactionAmt", "hour_of_day"],
            "log_transform_cols": ["TransactionAmt"],
            "null_fill_value": -999,
        }

        mock_trade = {
            "p": "50000.00", "q": "0.01",
            "T": 1704067200000, "m": False, "s": "BTCUSDT"
        }

        # mock the SageMaker client to simulate an endpoint not found error
        mock_sm = MagicMock()
        mock_sm.invoke_endpoint.side_effect = Exception(
            "Endpoint fraud-detection-endpoint not found"
        )

        result = score_trade(mock_trade, mock_metadata, mock_sm)

        assert result == -1.0, \
            "score_trade must return -1.0 when SageMaker call fails"

    def test_scoring_returns_float_between_zero_and_one(self):
        """
        When SageMaker is available, score_trade must return a float
        between 0.0 and 1.0. A score outside this range would indicate
        the response was parsed incorrectly.
        """
        from src.consumer.fraud_consumer import score_trade

        mock_metadata = {
            "feature_cols": ["TransactionAmt", "hour_of_day"],
            "log_transform_cols": ["TransactionAmt"],
            "null_fill_value": -999,
        }

        mock_trade = {
            "p": "50000.00", "q": "0.01",
            "T": 1704067200000, "m": False, "s": "BTCUSDT"
        }

        # mock a successful SageMaker response returning score 0.87
        mock_response = {
            "Body": MagicMock(
                read=MagicMock(return_value=b"0.87")
            )
        }
        mock_sm = MagicMock()
        mock_sm.invoke_endpoint.return_value = mock_response

        result = score_trade(mock_trade, mock_metadata, mock_sm)

        assert 0.0 <= result <= 1.0, \
            f"Score {result} must be between 0.0 and 1.0"
        assert abs(result - 0.87) < 0.001


# ── test that S3 log flushing works correctly ─────────────────────────────────

class TestS3LogFlushing:
    """
    Tests that prediction logs are correctly formatted and uploaded to S3.
    """

    def test_flush_uploads_csv_to_s3(self):
        """
        flush_logs_to_s3 must call s3.put_object exactly once
        with a CSV-formatted body containing all log records.
        """
        from src.consumer.fraud_consumer import flush_logs_to_s3

        mock_s3 = MagicMock()

        logs = [
            {
                "timestamp": "2024-01-01T00:00:00",
                "trade_id": 1,
                "symbol": "BTCUSDT",
                "price": "50000",
                "quantity": "0.01",
                "trade_time": 1704067200000,
                "is_buyer_maker": False,
                "fraud_score": 0.12,
                "is_fraud": 0,
                "threshold": 0.5,
            }
        ]

        flush_logs_to_s3(logs, mock_s3)

        # verify put_object was called exactly once
        assert mock_s3.put_object.call_count == 1, \
            "flush_logs_to_s3 must call s3.put_object exactly once"

        # verify the body is CSV-formatted (contains the header)
        call_kwargs = mock_s3.put_object.call_args.kwargs
        body = call_kwargs["Body"].decode("utf-8")
        assert "fraud_score" in body, \
            "CSV body must contain the fraud_score column header"
        assert "0.12" in body, \
            "CSV body must contain the actual fraud score value"

    def test_flush_does_nothing_when_logs_empty(self):
        """
        flush_logs_to_s3 must not call s3.put_object when the log list
        is empty. Making an empty S3 upload wastes API calls.
        """
        from src.consumer.fraud_consumer import flush_logs_to_s3

        mock_s3 = MagicMock()
        flush_logs_to_s3([], mock_s3)

        assert mock_s3.put_object.call_count == 0, \
            "flush_logs_to_s3 must not upload when logs list is empty"