"""
Statistical Anomaly Detection Pipeline for WaterWise.
Implements group-level Interquartile Range (IQR) anomaly detection
using Apache PySpark distributed aggregations.
"""

from typing import Dict, Any, Tuple
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
    TimestampType,
    DateType
)


EXPLANATION_DISCLAIMER = """
[STATISTICAL INTERPRETATION NOTE]
Readings flagged as anomalies represent statistical outliers (exceeding Q3 + 1.5 × IQR
within their specific fixture category). They do NOT definitively prove a water pipe burst
or mechanical leak. Legitimate reasons for high usage include extended multi-person showers,
filling buckets/aquariums, intensive cleaning, or concurrent fixture usage.
"""

MIN_GROUP_OBSERVATIONS = 5

ANOMALY_SCHEMA = StructType([
    StructField("timestamp", TimestampType(), True),
    StructField("date", DateType(), True),
    StructField("hour", IntegerType(), True),
    StructField("apartment", StringType(), True),
    StructField("cluster_name", StringType(), True),
    StructField("room_number", StringType(), True),
    StructField("room_type", StringType(), True),
    StructField("fixture", StringType(), True),
    StructField("water_type", StringType(), True),
    StructField("sensor_id", StringType(), True),
    StructField("volume_liters", DoubleType(), True),
    StructField("volume_m3", DoubleType(), True),
    StructField("threshold_liters", DoubleType(), True),
    StructField("excess_liters", DoubleType(), True),
    StructField("anomaly_ratio", DoubleType(), True),
])


def detect_fixture_anomalies(
    spark: SparkSession,
    df: DataFrame,
    iqr_multiplier: float = 1.5
) -> Tuple[DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Computes fixture-stratified IQR thresholds and flags anomalous consumption readings.

    Statistical Formula:
    - Q1: 25th percentile volume per fixture
    - Q3: 75th percentile volume per fixture
    - IQR = Q3 - Q1
    - Upper Threshold = Q3 + (iqr_multiplier * IQR)

    Handles:
    - Groups with insufficient observations (< 5) via conservative guard.
    - Zero IQR (common for discretized meters with identical frequent pulses) via stddev fallback.
    """
    df.createOrReplaceTempView("v_records_for_anomaly")

    # Step 1: Compute group percentiles using Spark SQL percentile_approx
    threshold_sql = """
    SELECT
        fixture,
        COUNT(*) AS group_count,
        ROUND(COALESCE(percentile_approx(volume_liters, 0.25, 1000), 0.0), 3) AS q1_liters,
        ROUND(COALESCE(percentile_approx(volume_liters, 0.50, 1000), 0.0), 3) AS median_liters,
        ROUND(COALESCE(percentile_approx(volume_liters, 0.75, 1000), 0.0), 3) AS q3_liters,
        ROUND(COALESCE(STDDEV(volume_liters), 0.0), 3) AS stddev_liters,
        ROUND(COALESCE(AVG(volume_liters), 0.0), 3) AS avg_liters,
        ROUND(COALESCE(MAX(volume_liters), 0.0), 3) AS max_liters
    FROM v_records_for_anomaly
    GROUP BY fixture
    """
    thresholds_pdf = spark.sql(threshold_sql).toPandas()

    if thresholds_pdf.empty:
        empty_spark_df = spark.createDataFrame([], ANOMALY_SCHEMA)
        empty_summary = {
            "total_anomalies": 0,
            "overall_anomaly_pct": 0.0,
            "iqr_multiplier": iqr_multiplier,
            "disclaimer": EXPLANATION_DISCLAIMER.strip(),
            "fixtures_evaluated": 0
        }
        for col, col_type in [
            ("iqr", float),
            ("threshold_liters", float),
            ("detection_method", str),
            ("anomaly_count", int),
            ("anomaly_pct", float)
        ]:
            if col not in thresholds_pdf.columns:
                thresholds_pdf[col] = pd.Series(dtype=col_type)
        return empty_spark_df, thresholds_pdf, empty_summary

    # Calculate IQR and dynamic upper threshold
    thresholds_pdf["iqr"] = (thresholds_pdf["q3_liters"] - thresholds_pdf["q1_liters"]).round(3)
    
    def calculate_threshold_and_method(row):
        cnt = row["group_count"]
        iqr = row["iqr"]
        q3 = row["q3_liters"]
        std = row["stddev_liters"]
        max_val = row["max_liters"]

        if cnt < MIN_GROUP_OBSERVATIONS:
            # Insufficient sample size to compute reliable IQR percentiles
            return pd.Series([round(max_val, 3), "Insufficient Data (Max Guard)"])
        elif iqr > 0:
            return pd.Series([round(q3 + (iqr_multiplier * iqr), 3), f"IQR ({iqr_multiplier}x)"])
        elif std > 0:
            return pd.Series([round(q3 + (2.5 * std), 3), "Zero-IQR (StdDev Fallback)"])
        else:
            fallback = round(q3 * 1.5 if q3 > 0 else 5.0, 3)
            return pd.Series([fallback, "Zero-Variance Fallback"])

    applied = thresholds_pdf.apply(calculate_threshold_and_method, axis=1)
    thresholds_pdf["threshold_liters"] = pd.to_numeric(applied[0], errors="coerce").fillna(0.0).round(3)
    thresholds_pdf["detection_method"] = applied[1].astype(str)

    # Convert threshold lookup back to Spark DataFrame for distributed join
    spark_thresholds = spark.createDataFrame(
        thresholds_pdf[["fixture", "q1_liters", "q3_liters", "iqr", "threshold_liters"]]
    )
    spark_thresholds.createOrReplaceTempView("v_fixture_thresholds")

    # Step 2: Join and flag anomalies
    anomaly_sql = """
    SELECT
        r.timestamp,
        r.date,
        r.hour,
        r.apartment,
        r.cluster_name,
        r.room_number,
        r.room_type,
        r.fixture,
        r.water_type,
        r.sensor_id,
        r.volume_liters,
        r.volume_m3,
        t.threshold_liters,
        ROUND(r.volume_liters - t.threshold_liters, 2) AS excess_liters,
        ROUND(r.volume_liters / NULLIF(t.threshold_liters, 0), 2) AS anomaly_ratio
    FROM v_records_for_anomaly r
    INNER JOIN v_fixture_thresholds t
        ON r.fixture = t.fixture
    WHERE r.volume_liters > t.threshold_liters
    ORDER BY excess_liters DESC
    """
    anomalies_spark_df = spark.sql(anomaly_sql)

    # Step 3: Compute anomaly counts per fixture for report
    anomalies_spark_df.createOrReplaceTempView("v_flagged_anomalies")
    counts_sql = """
    SELECT
        fixture,
        COUNT(*) AS anomaly_count
    FROM v_flagged_anomalies
    GROUP BY fixture
    """
    counts_pdf = spark.sql(counts_sql).toPandas()

    if not counts_pdf.empty:
        thresholds_pdf = thresholds_pdf.merge(counts_pdf, on="fixture", how="left").fillna({"anomaly_count": 0})
        thresholds_pdf["anomaly_count"] = thresholds_pdf["anomaly_count"].astype(int)
        thresholds_pdf["anomaly_pct"] = (
            (thresholds_pdf["anomaly_count"] / thresholds_pdf["group_count"]) * 100
        ).round(2)
    else:
        thresholds_pdf["anomaly_count"] = 0
        thresholds_pdf["anomaly_pct"] = 0.0

    total_valid = int(thresholds_pdf["group_count"].sum())
    total_anomalies = int(thresholds_pdf["anomaly_count"].sum())
    overall_anomaly_pct = round((total_anomalies / total_valid) * 100, 2) if total_valid > 0 else 0.0

    summary = {
        "total_anomalies": total_anomalies,
        "overall_anomaly_pct": overall_anomaly_pct,
        "iqr_multiplier": iqr_multiplier,
        "disclaimer": EXPLANATION_DISCLAIMER.strip(),
        "fixtures_evaluated": len(thresholds_pdf)
    }

    return anomalies_spark_df, thresholds_pdf, summary
