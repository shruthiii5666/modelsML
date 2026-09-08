# ============================================================
# data_loader.py
# Large CSV Dataset Loader
# ============================================================

import os
import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    CHUNK_SIZE,
    REQUIRED_COLUMNS
)


# ============================================================
# 1. CHECK DATASET
# ============================================================

def check_dataset_exists(file_path=RAW_DATA_PATH):
    """
    Check whether the input dataset exists.
    """

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"\nDataset not found!\n"
            f"Expected location:\n"
            f"{os.path.abspath(file_path)}\n\n"
            f"Please place 2019-Oct.csv inside:\n"
            f"data/raw/"
        )

    # File size
    file_size_bytes = os.path.getsize(file_path)

    file_size_gb = (
        file_size_bytes /
        (1024 ** 3)
    )

    print("\nDataset found successfully.")

    print(
        f"File: {file_path}"
    )

    print(
        f"Size: {file_size_gb:.2f} GB"
    )

    return True


# ============================================================
# 2. READ DATASET IN CHUNKS
# ============================================================

def read_dataset_in_chunks(
    file_path=RAW_DATA_PATH,
    chunk_size=CHUNK_SIZE
):
    """
    Read the large CSV file in chunks.

    This prevents the complete dataset from being loaded
    into RAM.
    """

    check_dataset_exists(file_path)

    print(
        f"\nReading dataset in chunks of "
        f"{chunk_size:,} rows..."
    )

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            file_path,
            chunksize=chunk_size,
            low_memory=True
        ),
        start=1
    ):

        print(
            f"Processing chunk {chunk_number}: "
            f"{len(chunk):,} rows"
        )

        yield chunk


# ============================================================
# 3. VALIDATE COLUMNS
# ============================================================

def validate_columns(df):
    """
    Check whether all required columns exist.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nThe dataset is missing the following "
            f"required columns:\n{missing_columns}\n\n"

            f"Available columns are:\n"
            f"{list(df.columns)}"
        )

    return True


# ============================================================
# 4. READ FIRST CHUNK FOR INSPECTION
# ============================================================

def load_first_chunk(
    file_path=RAW_DATA_PATH,
    chunk_size=CHUNK_SIZE
):
    """
    Load only the first chunk.

    Useful for checking the dataset before processing
    the entire file.
    """

    check_dataset_exists(file_path)

    df = pd.read_csv(
        file_path,
        nrows=chunk_size,
        low_memory=True
    )

    validate_columns(df)

    return df


# ============================================================
# 5. GET DATASET INFORMATION
# ============================================================

def inspect_first_chunk(
    file_path=RAW_DATA_PATH,
    chunk_size=CHUNK_SIZE
):
    """
    Display basic information about the first chunk.
    """

    df = load_first_chunk(
        file_path,
        chunk_size
    )

    print("\n")
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    print(
        f"\nRows in inspected chunk: "
        f"{len(df):,}"
    )

    print(
        "\nColumns:"
    )

    for column in df.columns:
        print(
            f"  - {column}"
        )

    print(
        "\nData types:"
    )

    print(
        df.dtypes
    )

    print(
        "\nFirst 5 records:"
    )

    print(
        df.head()
    )

    print(
        "\nEvent types:"
    )

    if "event_type" in df.columns:

        print(
            df["event_type"]
            .value_counts()
        )

    print(
        "\nMissing values in inspected chunk:"
    )

    print(
        df.isnull().sum()
    )

    print(
        "\n" + "=" * 60
    )

    return df

# ============================================================
# 6. READ AND PREPROCESS CHUNKS
# ============================================================

def read_preprocessed_chunks(
    file_path=RAW_DATA_PATH,
    chunk_size=CHUNK_SIZE
):
    """
    Read the large CSV file chunk-by-chunk and preprocess
    every chunk before yielding it.
    """

    from src.preprocessing import preprocess_chunk

    for chunk_number, chunk in enumerate(
        read_dataset_in_chunks(
            file_path,
            chunk_size
        ),
        start=1
    ):

        cleaned_chunk = preprocess_chunk(
            chunk
        )

        print(
            f"Cleaned chunk {chunk_number}: "
            f"{len(cleaned_chunk):,} rows"
        )

        yield cleaned_chunk