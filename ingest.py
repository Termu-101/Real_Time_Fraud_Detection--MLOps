import os
import zipfile
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# load_dotenv() reads your .env file and makes AWS_ACCESS_KEY_ID,
# AWS_SECRET_ACCESS_KEY, and S3_BUCKET available via os.getenv().
# It must be called before any os.getenv() call otherwise you get None.
load_dotenv()

# Path("data") points to a data/ folder inside whatever directory
# you run this script from. Always cd into your project folder first
# before running — otherwise this folder gets created in the wrong place.
DATA_DIR = Path("data")

# The IEEE-CIS competition gives you two CSV files.
# train_transaction.csv — the main file with 590k rows and the isFraud label.
# train_identity.csv — extra device/identity info for ~24% of transactions.
# They share TransactionID which is how we join them together.
TRAIN_TRANSACTION = "train_transaction.csv"
TRAIN_IDENTITY = "train_identity.csv"


def download_from_kaggle():
    """
    Downloads the IEEE-CIS dataset from Kaggle and extracts it.

    Two things were fixed here for Windows:
    1. The original used os.system("unzip ...") which does not exist on Windows.
       We now use Python's built-in zipfile module which works everywhere.
    2. Kaggle authentication must be set up first — kaggle.json must be at
       C:\\Users\\yourname\\.kaggle\\kaggle.json before this will work.

    The function checks if the CSV already exists before downloading.
    This means you can safely re-run the script without re-downloading 500MB.
    """
    # Create the data/ folder if it doesn't already exist.
    # exist_ok=True means don't throw an error if it's already there.
    DATA_DIR.mkdir(exist_ok=True)

    # If the transaction CSV already exists, skip the whole download.
    # This saves time when you re-run the script after fixing an error.
    if (DATA_DIR / TRAIN_TRANSACTION).exists():
        print("Dataset already downloaded, skipping download.")
        return

    print("Downloading IEEE-CIS dataset from Kaggle...")
    print("This is ~500MB and may take a few minutes.\n")

    # os.system() runs a shell command exactly like typing it in the terminal.
    # The kaggle CLI reads your credentials from ~/.kaggle/kaggle.json automatically.
    # -c specifies the competition name, -p specifies where to save the zip file.
    exit_code = os.system(
        f"kaggle competitions download -c ieee-fraud-detection -p {DATA_DIR}"
    )

    # exit_code is 0 if the command succeeded, non-zero if it failed.
    # If Kaggle authentication is not set up, exit_code will be non-zero
    # and we stop here with a clear error message instead of a confusing crash.
    if exit_code != 0:
        raise RuntimeError(
            "\nKaggle download failed. Make sure you have done the following:\n"
            "1. Go to kaggle.com -> profile -> Settings -> API -> Create New Token\n"
            "2. Place kaggle.json at: C:\\Users\\yourname\\.kaggle\\kaggle.json\n"
            "3. Accept the competition rules at:\n"
            "   https://www.kaggle.com/competitions/ieee-fraud-detection/rules\n"
        )

    # The downloaded file is a zip. We use Python's zipfile module to extract it.
    # This replaces the original unzip shell command which does not exist on Windows.
    zip_path = DATA_DIR / "ieee-fraud-detection.zip"

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Expected zip file at {zip_path} but it was not found.\n"
            "The Kaggle download may have failed silently."
        )

    print("\nExtracting zip file...")

    # zipfile.ZipFile opens the zip, and extractall() dumps all contents
    # into DATA_DIR. The 'with' block closes the file automatically when done,
    # even if an error occurs partway through extraction.
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)

    print("Extraction complete.")


def validate(transactions: pd.DataFrame, identity: pd.DataFrame):
    """
    Runs sanity checks on the raw data before we do anything with it.

    In a real pipeline this is a gate — if the data looks wrong the script
    stops here with a clear error instead of silently training a broken model.

    assert stops execution and prints the message if the condition is False.
    A loud failure is always better than a quiet bad model.
    """
    print("\nValidating data...")

    # isFraud is the target column — the thing we are trying to predict.
    # Without it we have nothing to train on.
    assert "isFraud" in transactions.columns, \
        "isFraud column is missing from transactions — wrong file?"

    # TransactionID is the key we use to join the two tables.
    # Without it the merge in the next step will fail.
    assert "TransactionID" in transactions.columns, \
        "TransactionID column is missing — wrong file?"

    # isFraud should only contain 0 (not fraud) and 1 (fraud).
    assert transactions["isFraud"].nunique() == 2, \
        "isFraud should be binary (0 and 1 only)"

    # The real dataset has 590k rows. If we have far fewer,
    # the download or extraction probably failed partway through.
    assert len(transactions) > 100_000, \
        f"Transactions table looks too small: {len(transactions)} rows. " \
        f"Download may be incomplete."

    print(f"  Transactions : {transactions.shape[0]:,} rows, "
          f"{transactions.shape[1]} columns")
    print(f"  Identity     : {identity.shape[0]:,} rows, "
          f"{identity.shape[1]} columns")
    print(f"  Fraud rate   : {transactions['isFraud'].mean():.2%}")

    # ~43% null rate is expected — many identity columns are optional.
    # We handle nulls in Stage 3, not here.
    print(f"  Null rate    : {transactions.isnull().mean().mean():.2%} "
          f"(high is expected)")
    print("  Validation passed.\n")


def merge_datasets(
    transactions: pd.DataFrame,
    identity: pd.DataFrame
) -> pd.DataFrame:
    """
    Joins transactions with identity on TransactionID using a left join.

    A left join keeps ALL rows from transactions and brings in matching
    identity columns where they exist. If a transaction has no identity
    record, those columns are NaN — the row is still kept.

    We never use inner join here because only ~24% of transactions have
    a matching identity record. Inner join would silently drop 76% of
    the data including many fraud cases which would ruin the model.
    """
    print("Merging transactions + identity...")

    merged = transactions.merge(identity, on="TransactionID", how="left")

    print(f"  Merged shape : {merged.shape[0]:,} rows, "
          f"{merged.shape[1]} columns")

    # A left join must never drop rows from the left table.
    assert len(merged) == len(transactions), \
        "Merge dropped rows — this should never happen with a left join."

    return merged


def save_locally(df: pd.DataFrame) -> Path:
    """
    Saves the merged dataframe as a CSV in the data/ folder.

    index=False prevents pandas from writing row numbers (0, 1, 2...)
    as a column. Without this you get an unwanted 'Unnamed: 0' column
    every time you read the file back.

    This file is what you will manually upload to S3 via the AWS Console.
    """
    output_path = DATA_DIR / "merged_raw.csv"

    print("Saving merged file...")
    df.to_csv(output_path, index=False)

    size_mb = output_path.stat().st_size / 1e6
    print(f"  Saved to : {output_path.resolve()}")
    print(f"  Size     : {size_mb:.1f} MB")

    return output_path


def run():
    """
    Main function — runs all steps in order.

    Step 1: download IEEE-CIS from Kaggle and extract the zip
    Step 2: load both CSV files into pandas dataframes
    Step 3: validate the data looks correct
    Step 4: merge the two files into one combined dataframe
    Step 5: save the merged file to data/merged_raw.csv

    After this finishes, manually upload merged_raw.csv to S3 via
    the AWS Console under data/raw/ in your bucket.
    """
    download_from_kaggle()

    print("Loading CSV files into memory...")
    transactions = pd.read_csv(DATA_DIR / TRAIN_TRANSACTION)
    identity = pd.read_csv(DATA_DIR / TRAIN_IDENTITY)

    validate(transactions, identity)

    merged = merge_datasets(transactions, identity)

    output_path = save_locally(merged)

    bucket = os.getenv("S3_BUCKET", "your-bucket-name")
    print("\n" + "=" * 60)
    print("  Stage 1 complete.")
    print("=" * 60)
    print("\n  Next — upload to S3 manually:")
    print(f"  1. Go to https://s3.console.aws.amazon.com")
    print(f"  2. Open your bucket : {bucket}")
    print(f"  3. Create folder   : data/raw/")
    print(f"  4. Upload this file: {output_path.resolve()}")
    print("\n  Then run your tests:")
    print("  pytest tests/test_ingestion.py -v")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # This block only runs when you execute this file directly with
    # python ingest.py. If another file imports this module (like the
    # tests or Airflow), this block is skipped automatically.
    run()