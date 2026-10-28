"""
PySpark Preprocessing and Data Cleaning Pipeline for WaterWise.
Transforms raw readings into clean, structured DataFrames with rich temporal features
and exports optimized Parquet storage.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType
)


PHYSICAL_UPPER_LIMIT_M3 = 0.3  # 300 Liters per 10-minute interval (sensor saturation)


def preprocess_and_clean(
    raw_df: DataFrame,
    output_dir: str = "outputs",
    save_parquet: bool = True
) -> Tuple[DataFrame, Dict[str, Any]]:
    """
    Executes the PySpark cleaning and feature engineering pipeline.

    Rules applied:
    1. Parse and validate UTC timestamps (format: 'yyyy-MM-dd HH:mm:ss').
    2. Cast 'value' and 'aggregated_value' to DoubleType.
    3. Filter out zero or negative readings (value > 0).
    4. Filter out physical telemetry corruption (value <= 0.3 m3).
    5. Drop duplicate readings based on true composite key (timestamp, sensor_id).
    6. Impute missing unit/cluster identifiers with 'Shared Facility'.
    7. Generate derived temporal features (date, hour, day_name, month, year, is_weekend).
    8. Convert volume to both cubic meters (m3) and Liters (L).
    """
    # Ensure optional contextual columns exist in raw DataFrame
    working_df = raw_df
    if "aggregated_value" not in working_df.columns:
        working_df = working_df.withColumn("aggregated_value", F.lit(None).cast(StringType()))
    if "apartment" not in working_df.columns:
        working_df = working_df.withColumn("apartment", F.lit("Shared Facility"))
    if "cluster_name" not in working_df.columns:
        working_df = working_df.withColumn("cluster_name", F.lit("Building-Wide"))
    if "room_number" not in working_df.columns:
        working_df = working_df.withColumn("room_number", F.lit("Unknown"))
    if "room_type" not in working_df.columns:
        working_df = working_df.withColumn("room_type", F.lit("Other"))
    if "type" not in working_df.columns:
        working_df = working_df.withColumn("type", F.lit("Standard"))
    if "attached_to" not in working_df.columns:
        working_df = working_df.withColumn("attached_to", F.lit("General Tap"))

    # Step 1: Type casting & timestamp parsing
    staged_df = (
        working_df
        .withColumn("parsed_timestamp", F.try_to_timestamp(F.col("timestamp"), F.lit("yyyy-MM-dd HH:mm:ss")))
        .withColumn("raw_value", F.col("value").try_cast(DoubleType()))
        .withColumn("raw_agg_value", F.col("aggregated_value").try_cast(DoubleType()))
    )

    # Step 2: Quality validation with mutually exclusive classification
    stats_row = staged_df.select(
        F.count(F.lit(1)).alias("total_raw"),
        F.sum(F.when(F.col("parsed_timestamp").isNull(), 1).otherwise(0)).alias("bad_time"),
        F.sum(
            F.when(
                F.col("parsed_timestamp").isNotNull() &
                (F.col("raw_value").isNull() | (F.col("raw_value") <= 0.0)),
                1
            ).otherwise(0)
        ).alias("zero_neg"),
        F.sum(
            F.when(
                F.col("parsed_timestamp").isNotNull() &
                F.col("raw_value").isNotNull() &
                (F.col("raw_value") > PHYSICAL_UPPER_LIMIT_M3),
                1
            ).otherwise(0)
        ).alias("outlier")
    ).collect()[0]

    raw_count = int(stats_row["total_raw"] or 0)
    invalid_time_count = int(stats_row["bad_time"] or 0)
    zero_negative_count = int(stats_row["zero_neg"] or 0)
    physical_outliers_count = int(stats_row["outlier"] or 0)

    # Step 3: Filter valid telemetry and deduplicate on composite primary key (timestamp, sensor_id)
    filtered_df = staged_df.filter(
        F.col("parsed_timestamp").isNotNull() &
        F.col("raw_value").isNotNull() &
        (F.col("raw_value") > 0.0) &
        (F.col("raw_value") <= PHYSICAL_UPPER_LIMIT_M3)
    )
    filtered_df.cache()
    pre_dedup_count = filtered_df.count()

    deduped_df = filtered_df.dropDuplicates(["parsed_timestamp", "sensor_id"])
    deduped_df.cache()
    cleaned_count = deduped_df.count()
    duplicates_count = max(0, pre_dedup_count - cleaned_count)

    # Step 4: Feature engineering & standardization
    cleaned_df = (
        deduped_df
        .withColumn("timestamp", F.col("parsed_timestamp"))
        .withColumn("date", F.to_date(F.col("parsed_timestamp")))
        .withColumn("hour", F.hour(F.col("parsed_timestamp")).cast(IntegerType()))
        .withColumn("day_of_week", ((F.dayofweek(F.col("parsed_timestamp")) + 5) % 7).cast(IntegerType())) # 0=Mon, 6=Sun
        .withColumn("day_name", F.date_format(F.col("parsed_timestamp"), "EEEE"))
        .withColumn("month", F.month(F.col("parsed_timestamp")).cast(IntegerType()))
        .withColumn("year", F.year(F.col("parsed_timestamp")).cast(IntegerType()))
        .withColumn(
            "is_weekend",
            F.when(F.col("day_of_week").isin([5, 6]), F.lit(1)).otherwise(F.lit(0))
        )
        .withColumn("volume_m3", F.col("raw_value"))
        .withColumn("volume_liters", F.round(F.col("raw_value") * 1000.0, 3))
        .withColumn("cumulative_m3", F.col("raw_agg_value"))
        .withColumn("sensor_id", F.trim(F.col("sensor_id")))
        .withColumn(
            "apartment",
            F.when(
                F.col("apartment").isNull() | 
                (F.trim(F.col("apartment")) == "") | 
                (F.lower(F.col("apartment")) == "nan"),
                F.lit("Shared Facility")
            ).otherwise(F.trim(F.col("apartment")))
        )
        .withColumn(
            "cluster_name",
            F.when(
                F.col("cluster_name").isNull() | 
                (F.trim(F.col("cluster_name")) == "") | 
                (F.lower(F.col("cluster_name")) == "nan"),
                F.lit("Building-Wide")
            ).otherwise(F.trim(F.col("cluster_name")))
        )
        .withColumn(
            "room_type",
            F.when(
                F.col("room_type").isNull() | (F.trim(F.col("room_type")) == ""),
                F.lit("Other")
            ).otherwise(F.trim(F.col("room_type")))
        )
        .withColumn(
            "water_type",
            F.when(
                F.col("type").isNull() | (F.trim(F.col("type")) == ""),
                F.lit("Standard")
            ).otherwise(F.trim(F.col("type")))
        )
        .withColumn(
            "fixture",
            F.when(
                F.col("attached_to").isNull() | (F.trim(F.col("attached_to")) == ""),
                F.lit("General Tap")
            ).otherwise(F.trim(F.col("attached_to")))
        )
        .select(
            "timestamp",
            "date",
            "hour",
            "day_of_week",
            "day_name",
            "month",
            "year",
            "is_weekend",
            "sensor_id",
            "apartment",
            "cluster_name",
            "room_number",
            "room_type",
            "water_type",
            "fixture",
            "volume_m3",
            "volume_liters",
            "cumulative_m3"
        )
    )

    filtered_df.unpersist()
    retention_pct = round((cleaned_count / raw_count) * 100, 2) if raw_count > 0 else 0.0

    # Save to Parquet if requested
    parquet_path = Path(output_dir) / "parquet" / "cleaned_water_data"
    if save_parquet:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cleaned_df.write.mode("overwrite").parquet(str(parquet_path))
        except Exception as e:
            print(f"[WARN] Parquet write encountered an issue: {e}")

    summary = {
        "raw_record_count": raw_count,
        "cleaned_record_count": cleaned_count,
        "invalid_timestamps_dropped": invalid_time_count,
        "zero_negative_dropped": zero_negative_count,
        "physical_outliers_dropped": physical_outliers_count,
        "duplicates_dropped": duplicates_count,
        "total_records_dropped": raw_count - cleaned_count,
        "retention_percentage": retention_pct,
        "parquet_path": str(parquet_path)
    }

    return cleaned_df, summary
