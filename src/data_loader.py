"""
Data loading and raw schema validation module for WaterWise.
Loads real household water-consumption CSV files using Apache PySpark.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


# Expected required columns in HSB Living Lab dataset
EXPECTED_COLUMNS = [
    "timestamp",
    "value",
    "sensor_id",
    "room_type",
    "type",
    "attached_to"
]

ZENODO_URL = "https://zenodo.org/records/18971107"
DATA_INSTRUCTIONS = f"""
[ERROR] HSB Living Lab dataset not found!
Please follow these steps:
1. Download the dataset archive from Zenodo:
   {ZENODO_URL}
   Direct link: https://zenodo.org/records/18971107/files/HSB_Living_Lab_Data_Package.zip?download=1
2. Extract the archive into the project's 'data/' directory.
3. Ensure 'HSB_Living_Lab_Water_Consumption_Anonymized.csv' exists at:
   data/HSB_Living_Lab_Water_Consumption_Anonymized.csv
"""


def locate_dataset_file(data_dir: str = "data") -> Path:
    """
    Locates the primary HSB water consumption CSV file in data_dir.
    Raises FileNotFoundError with explicit setup instructions if missing.
    """
    target_dir = Path(data_dir).resolve()
    if not target_dir.exists():
        raise FileNotFoundError(
            f"Data directory '{target_dir}' does not exist.\n{DATA_INSTRUCTIONS}"
        )

    # Primary expected filename
    primary_file = target_dir / "HSB_Living_Lab_Water_Consumption_Anonymized.csv"
    if primary_file.exists() and primary_file.is_file():
        return primary_file

    # Check for any .csv file with water consumption keywords
    candidates = list(target_dir.glob("*.csv")) + list(target_dir.glob("**/*.csv"))
    for cand in candidates:
        name_lower = cand.name.lower()
        if "water" in name_lower or "consumption" in name_lower or "hsb" in name_lower:
            return cand

    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"No valid CSV dataset found in '{target_dir}'.\n{DATA_INSTRUCTIONS}"
    )


def load_raw_data(
    spark: SparkSession,
    data_dir: str = "data",
    max_rows: Optional[int] = None
) -> Tuple[DataFrame, Dict[str, Any]]:
    """
    Reads the raw CSV file into a PySpark DataFrame without schema casting
    and returns initial file metadata and raw validation metrics.
    """
    csv_path = locate_dataset_file(data_dir)
    file_size_mb = round(os.path.getsize(csv_path) / (1024 * 1024), 2)

    # Read with PySpark: delimiter is semicolon (;)
    raw_df = (
        spark.read
        .option("header", "true")
        .option("sep", ";")
        .option("inferSchema", "false")
        .csv(str(csv_path))
    )

    if max_rows and max_rows > 0:
        raw_df = raw_df.limit(max_rows)

    detected_cols = raw_df.columns

    # Verify critical columns exist
    missing_critical = [c for c in EXPECTED_COLUMNS if c not in detected_cols]
    if missing_critical:
        raise ValueError(
            f"CSV file '{csv_path.name}' is missing expected critical columns: {missing_critical}. "
            f"Detected columns: {detected_cols}"
        )

    raw_count = raw_df.count()

    metadata = {
        "file_path": str(csv_path),
        "file_name": csv_path.name,
        "file_size_mb": file_size_mb,
        "detected_columns": detected_cols,
        "column_count": len(detected_cols),
        "raw_record_count": raw_count
    }

    return raw_df, metadata


def compute_missing_value_stats(raw_df: DataFrame) -> Dict[str, int]:
    """
    Computes count of null or empty values per column using Spark SQL aggregations.
    """
    null_exprs = [
        F.count(
            F.when(
                F.col(c).isNull() | 
                (F.trim(F.col(c)) == "") | 
                (F.lower(F.col(c)) == "nan") | 
                (F.lower(F.col(c)) == "null"),
                1
            )
        ).alias(c)
        for c in raw_df.columns
    ]

    result_row = raw_df.select(null_exprs).collect()[0]
    return {c: int(result_row[c]) for c in raw_df.columns}
