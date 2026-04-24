import os
import json
import boto3
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder, StandardScaler
 
load_dotenv()

# S3 Locations

BUCKET          = os.getenv("S3_BUCKET")
RAW_KEY         = "data/raw/merged_raw.csv"
FEATURES_KEY    = "data/features/features.csv"
METADATA_KEY    = "data/features/feature_metadata.json"

# Local Paths - we download from S3, process locally and then reupload

DATA_DIR        = Path("data")
RAW_LOCAL       = DATA_DIR / "merged_raw.csv"
FEATURES_LOCAL  = DATA_DIR / "features.csv"
METADATA_LOCAL  = DATA_DIR / "feature_metadata.json"


# Constants ---------------------------------------------------------------------------------------

# Any column where more than these fractions of values are null gets dropped
# 0.9 means - drop if the nulls values are more than 90%
# We found this threshold from the EDA - more than 90% null values bring noise

NULL_THRESHOLD = 0.9

# Columns we drop regardless of the null rate
# TransactionId and TransactionDT it has no predictive value
# We just extract (hour of the day) and then drop the raw column

COLS_TO_DROP = ["TransactionID", "TransactionDT"]

# Columns where the values are highly skewed 
# We apply log transformation on these values - it makes them normal 
# This helps the tree based models like XGBoost split them better
# From EDA we got to know that TransactionAmt is highly skewed

LOG_TRANSFORM_COLS = ["TransactionAmt"]

# Categorical columsn that contains strings like 'visa', 'mastercard', 'gmail.com', 'w'
# Models can't read strings - LabelEncoder can be used them to convert to numbers

CATEGORICAL_COLS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType",
    "DeviceInfo",
]

# The target column
# We seperate the target column from our feauture columns

TARGET_COL = "isFraud"

# S3 helpers ---------------------------------------------------------------------------------------------------

def download_from_s3(s3_key: str, local_path: Path):
    """
    Downloads a file from S3 to a local path.
 
    We download to process locally because pandas operations on local files
    are much faster than streaming from S3. After processing we re-upload.
    """
    DATA_DIR.mkdir(exist_ok=True)
 
    if local_path.exists():
        print(f"  Already exists locally, skipping download: {local_path}")
        return
 
    print(f"  Downloading s3://{BUCKET}/{s3_key} -> {local_path}")
    s3 = boto3.client("s3")
    s3.download_file(BUCKET, s3_key, str(local_path))
    print(f"  Download complete.")
 
 
def upload_to_s3(local_path: Path, s3_key: str):
    """
    Uploads a local file to S3.
 
    This is called at the end after all processing is done.
    The processed features and metadata both go to S3 so Airflow
    and SageMaker can access them in later stages.
    """
    print(f"  Uploading {local_path} -> s3://{BUCKET}/{s3_key}")
    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), BUCKET, s3_key)
    print(f"  Upload complete.")


# Feature Engineering Steps ----------------------------------------------------------------------------------------------

def drop_high_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns where more than NULL_THRESHOLD (90%) of values are null.
 
    A column that is 90% empty tells the model almost nothing. Keeping it
    adds noise and slows down training. We calculate the null rate for every
    column and drop the ones above the threshold.
 
    We also drop the fixed columns in COLS_TO_DROP.
    """
    print("\nStep 1 — dropping high-null and irrelevant columns...")
 
    # Calculate null rate for every column
    null_rates = df.isnull().mean()
 
    # Find columns above the threshold
    high_null_cols = null_rates[null_rates > NULL_THRESHOLD].index.tolist()
 
    print(f"  Dropping {len(high_null_cols)} columns with >{NULL_THRESHOLD:.0%} nulls")
    print(f"  Dropping {len(COLS_TO_DROP)} fixed columns: {COLS_TO_DROP}")
 
    # Combine both lists, only dropping columns that actually exist
    all_to_drop = [
        col for col in high_null_cols + COLS_TO_DROP
        if col in df.columns
    ]
 
    df = df.drop(columns=all_to_drop)
 
    print(f"  Columns remaining: {df.shape[1]}")
    return df


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts useful features from TransactionDT before dropping it.
 
    TransactionDT is a time delta in seconds from some reference point.
    The raw number is not meaningful — but we can extract:
    - hour of day (fraud tends to happen at unusual hours)
    - day of week (fraud patterns differ on weekdays vs weekends)
 
    We do this BEFORE dropping TransactionDT in the previous step,
    so we call this function first in the pipeline.
    """
    print("\nStep 2 — extracting time features from TransactionDT...")
 
    if "TransactionDT" not in df.columns:
        print("  TransactionDT not found, skipping.")
        return df
 
    df["hour_of_day"] = (df["TransactionDT"] / 3600) % 24
    df["day_of_week"] = (df["TransactionDT"] / 86400) % 7
 
    print(f"  Created: hour_of_day, day_of_week")
    return df

def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies log(x + 1) transform to heavily skewed numerical columns.
 
    TransactionAmt has a very long right tail — most transactions are
    small but a few are enormous. This squashes the scale so the model
    treats a $10 vs $100 difference similarly to a $1000 vs $10000 one,
    which is more meaningful for fraud detection.
 
    """
    print("\nStep 3 — applying log transform to skewed columns...")
 
    for col in LOG_TRANSFORM_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col])
            print(f"  log1p applied to: {col}")
 
    return df

def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Converts string categorical columns to integers using LabelEncoder.
 
    XGBoost cannot process strings. LabelEncoder assigns a unique integer
    to each unique string value. For example in card4:
    "visa" -> 3, "mastercard" -> 1, "american express" -> 0, etc.
 
    NaN values are filled with the string "missing" before encoding
    so they become a valid category rather than causing errors.
 
    We save the encoder mappings to metadata so we can apply the exact
    same encoding to live Binance data in Stage 4. If we don't save this,
    "visa" might get a different number next time and the model breaks.
 
    Returns:
        df: dataframe with encoded columns
        encoders: dict of {column_name: {original_value: encoded_int}}
    """
    print("\nStep 4 — encoding categorical columns...")
 
    encoders = {}
 
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
 
        # fill NaN with "missing" so encoder doesn't crash
        df[col] = df[col].fillna("missing").astype(str)
 
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
 
        encoders[col] = {
            str(original): int(encoded)
            for encoded, original in enumerate(le.classes_)
        }
 
        print(f"  Encoded {col}: {len(le.classes_)} unique values")
 
    return df, encoders


def fill_remaining_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills any remaining null values after the previous steps.
 
    After dropping high-null columns and encoding categoricals, some
    numerical columns still have scattered nulls. We fill them with -999
    which is a value XGBoost handles well — it learns to treat -999 as
    "this value was missing" which is actually informative for fraud detection.
 
    We do NOT use mean/median imputation here because -999 is a more
    honest representation of missingness. Mean imputation would pretend
    we know the value when we don't.
    """
    print("\nStep 5 — filling remaining nulls with -999...")
 
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
 
    if len(cols_with_nulls) == 0:
        print("  No remaining nulls found.")
        return df
 
    print(f"  Filling nulls in {len(cols_with_nulls)} columns")
    df = df.fillna(-999)
 
    return df


def scale_numerical_features(
    df: pd.DataFrame,
    target: pd.Series
) -> tuple[pd.DataFrame, object, list]:
    """
    Scales numerical features to have mean=0 and standard deviation=1.
 
    We separate the target column before scaling so isFraud (0 or 1)
    never gets scaled — a scaled label would break the model completely.
 
    Returns:
        df: scaled feature dataframe (without target)
        scaler: fitted StandardScaler object
        feature_cols: list of column names in the scaled dataframe
    """
    print("\nStep 6 — scaling numerical features...")
 
    # get all numerical columns except the target
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COL in numerical_cols:
        numerical_cols.remove(TARGET_COL)
 
    # fit the scaler on training data and transform
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
 
    feature_cols = numerical_cols
    print(f"  Scaled {len(numerical_cols)} numerical columns")
 
    return df, scaler, feature_cols


def calculate_class_weight(target: pd.Series) -> float:
    """
    Calculates the scale_pos_weight parameter for XGBoost.
 
    XGBoost has a parameter called scale_pos_weight that tells it how
    much more to penalise missing a fraud case vs a non-fraud case.
 
    The formula is: count of negative class / count of positive class.
    With ~96.5% non-fraud and ~3.5% fraud this gives roughly 28.
    This means XGBoost treats each fraud case as 28x more important,
    which balances the class imbalance during training.

    """
    n_negative = (target == 0).sum()
    n_positive = (target == 1).sum()
    weight = n_negative / n_positive
 
    print(f"\nClass weight calculation:")
    print(f"  Non-fraud cases : {n_negative:,}")
    print(f"  Fraud cases     : {n_positive:,}")
    print(f"  scale_pos_weight: {weight:.2f}")
    print(f"  (XGBoost will treat each fraud case as {weight:.1f}x more important)")
 
    return weight


def save_metadata(
    encoders: dict,
    scaler: object,
    feature_cols: list,
    class_weight: float
    ):
    
    import joblib
 
    metadata = {
        "encoders": encoders,
        "feature_cols": feature_cols,
        "class_weight": class_weight,
        "log_transform_cols": LOG_TRANSFORM_COLS,
        "null_fill_value": -999,
        "null_threshold": NULL_THRESHOLD,
    }
 
    # save JSON metadata
    with open(METADATA_LOCAL, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata saved to: {METADATA_LOCAL}")
 
    # save scaler as a pickle file — JSON cannot store Python objects
    scaler_path = DATA_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved to: {scaler_path}")


def run():
    """
    Main function — runs all feature engineering steps in order.
 
    The order matters:
    1. Extract time features BEFORE dropping TransactionDT
    2. Drop high-null columns BEFORE encoding 
    3. Encode categoricals BEFORE filling nulls (fill uses -999 for numerics)
    4. Fill nulls BEFORE scaling (scaler cannot handle NaN)
    5. Scale AFTER everything else (scale the final clean features)
    6. Save metadata LAST (captures the final state of all transformations)
    """
    print("=" * 60)
    print("  Stage 3 — Feature Engineering")
    print("=" * 60)
 
    # download raw data from S3
    print("\nDownloading raw data from S3...")
    download_from_s3(RAW_KEY, RAW_LOCAL)
 
    # load into pandas
    print("Loading data...")
    df = pd.read_csv(RAW_LOCAL)
    print(f"  Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
 
    # separate target before any transformations
    # this ensures isFraud never gets modified by any step below
    target = df[TARGET_COL].copy()
    df = df.drop(columns=[TARGET_COL])
 
    # run all feature engineering steps in order
    df = extract_time_features(df)
    df = drop_high_null_columns(df)
    df = log_transform(df)
    df, encoders = encode_categoricals(df)
    df = fill_remaining_nulls(df)
    df, scaler, feature_cols = scale_numerical_features(df, target)
 
    # calculate class weight for XGBoost
    class_weight = calculate_class_weight(target)
 
    # add target back to the dataframe for saving
    # the target goes in last so it is always the final column
    df[TARGET_COL] = target.values
 
    # save processed features locally
    print(f"\nSaving processed features...")
    df.to_csv(FEATURES_LOCAL, index=False)
    size_mb = FEATURES_LOCAL.stat().st_size / 1e6
    print(f"  Saved to : {FEATURES_LOCAL}")
    print(f"  Shape    : {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"  Size     : {size_mb:.1f} MB")
 
    # save all metadata
    save_metadata(encoders, scaler, feature_cols, class_weight)
 
    # upload everything to S3
    print("\nUploading to S3...")
    upload_to_s3(FEATURES_LOCAL, FEATURES_KEY)
    upload_to_s3(METADATA_LOCAL, METADATA_KEY)
    upload_to_s3(DATA_DIR / "scaler.pkl", "data/features/scaler.pkl")
 
    print("\n" + "=" * 60)
    print("  Stage 3 complete.")
    print("=" * 60)
    print(f"\n  Files in S3:")
    print(f"  s3://{BUCKET}/{FEATURES_KEY}")
    print(f"  s3://{BUCKET}/{METADATA_KEY}")
    print(f"  s3://{BUCKET}/data/features/scaler.pkl")
    print("\n  Next: Stage 4 — Docker + Kafka setup")
    print("=" * 60 + "\n")
 
 
if __name__ == "__main__":
    run()
