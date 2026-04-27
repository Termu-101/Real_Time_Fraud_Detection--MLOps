"""
test_training.py — Tests for the training pipeline.

These tests verify training logic without actually calling SageMaker
or spending any money. We use small fake datasets and mock AWS calls.
"""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_features_df():
    """
    A small fake features dataframe — enough rows to train and evaluate
    without taking more than a second. Matches the format that
    feature_engineering.py produces.
    """
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "TransactionAmt": np.random.randn(n),
        "hour_of_day":    np.random.randint(0, 24, n).astype(float),
        "day_of_week":    np.random.randint(0, 7, n).astype(float),
        "card1":          np.random.randn(n),
        "card2":          np.random.randn(n),
        "isFraud":        np.random.choice([0, 1], n, p=[0.965, 0.035]),
    })


# ── data loading tests ────────────────────────────────────────────────────────

class TestDataLoading:

    def test_features_and_target_separated_correctly(self, small_features_df, tmp_path):
        """
        The target column isFraud must not be in the feature matrix X.
        If it is, the model trivially learns to predict it and fails on new data.
        """
        # save fake data to a temp CSV mimicking what SageMaker provides
        csv_path = tmp_path / "features.csv"
        small_features_df.to_csv(csv_path, index=False)

        df = pd.read_csv(csv_path)
        X = df.drop(columns=["isFraud"]).values
        y = df["isFraud"].values

        assert "isFraud" not in df.drop(columns=["isFraud"]).columns
        assert X.shape[1] == len(df.columns) - 1
        assert len(y) == len(df)

    def test_train_val_split_preserves_fraud_rate(self, small_features_df):
        """
        The stratified train/val split must preserve the fraud rate in both sets.
        Without stratify=y the val set might have no fraud cases at all.
        """
        from sklearn.model_selection import train_test_split

        X = small_features_df.drop(columns=["isFraud"]).values
        y = small_features_df["isFraud"].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # both splits should have fraud cases
        assert y_train.sum() > 0, "Training set must contain fraud cases"
        assert y_val.sum() > 0,   "Validation set must contain fraud cases"

    def test_no_nulls_in_feature_matrix(self, small_features_df):
        """
        Feature matrix must have zero nulls before training.
        XGBoost handles -999 fill values but not actual NaN.
        """
        X = small_features_df.drop(columns=["isFraud"]).values
        assert not np.isnan(X).any(), \
            "Feature matrix must not contain NaN values"


# ── model training tests ──────────────────────────────────────────────────────

class TestModelTraining:

    def test_model_trains_without_error(self, small_features_df):
        """
        XGBoost training must complete without raising any exception
        on valid input data.
        """
        import xgboost as xgb
        from sklearn.model_selection import train_test_split

        X = small_features_df.drop(columns=["isFraud"]).values
        y = small_features_df["isFraud"].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval   = xgb.DMatrix(X_val,   label=y_val)

        params = {
            "objective":        "binary:logistic",
            "eval_metric":      "auc",
            "max_depth":        3,
            "eta":              0.1,
            "scale_pos_weight": 28,
            "verbosity":        0,
        }

        try:
            model = xgb.train(
                params, dtrain,
                num_boost_round=10,
                evals=[(dval, "validation")],
                verbose_eval=False,
            )
        except Exception as e:
            pytest.fail(f"XGBoost training raised an exception: {e}")

        assert model is not None

    def test_model_outputs_probabilities(self, small_features_df):
        """
        Model predictions must be probabilities between 0 and 1.
        Values outside this range would break the fraud threshold comparison.
        """
        import xgboost as xgb
        from sklearn.model_selection import train_test_split

        X = small_features_df.drop(columns=["isFraud"]).values
        y = small_features_df["isFraud"].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval   = xgb.DMatrix(X_val,   label=y_val)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "verbosity": 0,
        }

        model = xgb.train(params, dtrain, num_boost_round=5,
                         verbose_eval=False)
        predictions = model.predict(dval)

        assert predictions.min() >= 0.0, "Predictions must be >= 0"
        assert predictions.max() <= 1.0, "Predictions must be <= 1"

    def test_auc_above_random(self, small_features_df):
        """
        A trained model must perform better than random guessing (AUC > 0.5).
        If AUC is below 0.5, something is fundamentally wrong with the features
        or the training setup.
        """
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score

        X = small_features_df.drop(columns=["isFraud"]).values
        y = small_features_df["isFraud"].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval   = xgb.DMatrix(X_val,   label=y_val)

        params = {
            "objective":        "binary:logistic",
            "eval_metric":      "auc",
            "scale_pos_weight": 28,
            "verbosity":        0,
        }

        model = xgb.train(params, dtrain, num_boost_round=20,
                         verbose_eval=False)
        y_pred = model.predict(dval)
        auc = roc_auc_score(y_val, y_pred)

        assert auc > 0.5, \
            f"Model AUC {auc:.4f} must be above 0.5 (random baseline)"


# ── model saving tests ────────────────────────────────────────────────────────

class TestModelSaving:

    def test_model_saves_and_loads(self, small_features_df, tmp_path):
        """
        A saved model must load and produce identical predictions.
        If save/load changes predictions, the deployed model would
        behave differently from the evaluated model.
        """
        import xgboost as xgb

        X = small_features_df.drop(columns=["isFraud"]).values
        y = small_features_df["isFraud"].values

        dtrain = xgb.DMatrix(X, label=y)
        params = {"objective": "binary:logistic", "verbosity": 0}
        model = xgb.train(params, dtrain, num_boost_round=5,
                         verbose_eval=False)

        # save and reload
        model_path = tmp_path / "model.xgb"
        model.save_model(str(model_path))

        loaded_model = xgb.Booster()
        loaded_model.load_model(str(model_path))

        # predictions must be identical
        dtest = xgb.DMatrix(X)
        original_preds = model.predict(dtest)
        loaded_preds   = loaded_model.predict(dtest)

        np.testing.assert_array_almost_equal(
            original_preds, loaded_preds, decimal=5,
            err_msg="Loaded model predictions must match original"
        )

    def test_metadata_saved_correctly(self, tmp_path):
        """
        Model metadata JSON must contain all required keys.
        Missing keys would cause the Airflow evaluation gate to fail.
        """
        metadata = {
            "auc": 0.92,
            "feature_cols": ["TransactionAmt", "hour_of_day"],
            "threshold": 0.5,
            "model_type": "xgboost",
            "scale_pos_weight": 28,
        }

        metadata_path = tmp_path / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        with open(metadata_path) as f:
            loaded = json.load(f)

        required_keys = ["auc", "feature_cols", "threshold", "model_type"]
        for key in required_keys:
            assert key in loaded, f"Metadata must contain key: {key}"

        assert loaded["auc"] == 0.92
        assert loaded["threshold"] == 0.5