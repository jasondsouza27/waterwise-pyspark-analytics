"""
Unit and Edge-Case Tests for WaterWise.
Verifies error paths, schema mismatch handling, zero-IQR edge cases,
and de-duplication logic.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.spark_session import get_spark_session, stop_spark_session
from src.data_loader import locate_dataset_file, load_raw_data
from src.preprocessing import preprocess_and_clean
from src.analytics import (
    compute_kpis,
    get_hourly_trends,
    get_daily_trends,
    get_day_of_week_trends,
    get_fixture_distribution,
    get_water_type_distribution,
    get_apartment_comparison,
    identify_peak_periods,
    get_monthly_trends,
    export_processed_outputs
)
from src.anomaly_detection import detect_fixture_anomalies
from src.forecasting import train_daily_forecast_model
from streamlit.testing.v1 import AppTest
import pandas as pd


@pytest.fixture(scope="module")
def spark():
    s = get_spark_session(app_name="WaterWise-EdgeCaseTests")
    yield s
    stop_spark_session()


def test_missing_data_directory_raises_error():
    """Verify locate_dataset_file raises FileNotFoundError with explicit guidance when directory is missing."""
    with pytest.raises(FileNotFoundError) as exc_info:
        locate_dataset_file("non_existent_directory_xyz")
    assert "Zenodo" in str(exc_info.value)
    assert "HSB Living Lab" in str(exc_info.value)


def test_missing_critical_columns_raises_value_error(spark, tmp_path):
    """Verify that CSV with invalid columns triggers clear ValueError."""
    bad_csv = tmp_path / "bad_dataset.csv"
    with open(bad_csv, "w") as f:
        f.write("col_a;col_b;col_c\n1;2;3\n")
    
    with pytest.raises(ValueError) as exc_info:
        load_raw_data(spark, data_dir=str(tmp_path))
    assert "missing expected critical columns" in str(exc_info.value)


def test_deduplication_logic_preserves_concurrent_different_sensors(spark):
    """
    Verify that readings with identical timestamps but DIFFERENT sensors are retained,
    while identical (timestamp, sensor_id) pairs are deduplicated.
    """
    test_data = [
        ("2020-01-01 10:00:00", "0.010", "0.010", "S001", "APT_1", "CL_A", "R1", "Bathroom", "cold", "Toilet"),
        ("2020-01-01 10:00:00", "0.015", "0.025", "S002", "APT_2", "CL_B", "R2", "Kitchen", "hot", "Kitchen tap"),  # Concurrent, different sensor -> RETAIN
        ("2020-01-01 10:00:00", "0.010", "0.010", "S001", "APT_1", "CL_A", "R1", "Bathroom", "cold", "Toilet"),   # True duplicate -> DROP
    ]
    cols = [
        "timestamp", "value", "aggregated_value", "sensor_id", "apartment",
        "cluster_name", "room_number", "room_type", "type", "attached_to"
    ]
    raw_df = spark.createDataFrame(test_data, cols)
    cleaned_df, summary = preprocess_and_clean(raw_df, output_dir="outputs", save_parquet=False)

    assert summary["raw_record_count"] == 3
    assert summary["duplicates_dropped"] == 1
    assert summary["cleaned_record_count"] == 2


def test_zero_and_negative_value_filtering(spark):
    """Verify that zero and negative values are dropped while positive within threshold are kept."""
    test_data = [
        ("2020-01-01 10:00:00", "0.000", "0.0", "S001", "APT_1", "CL_A", "R1", "Bathroom", "cold", "Toilet"), # Zero -> DROP
        ("2020-01-01 10:10:00", "-0.005", "0.0", "S001", "APT_1", "CL_A", "R1", "Bathroom", "cold", "Toilet"), # Negative -> DROP
        ("2020-01-01 10:20:00", "0.500", "0.5", "S001", "APT_1", "CL_A", "R1", "Bathroom", "cold", "Toilet"), # >0.3 m3 -> DROP (Physical outlier)
        ("2020-01-01 10:30:00", "0.025", "0.525", "S001", "APT_1", "CL_A", "R1", "Bathroom", "cold", "Toilet"), # Valid -> KEEP
    ]
    cols = [
        "timestamp", "value", "aggregated_value", "sensor_id", "apartment",
        "cluster_name", "room_number", "room_type", "type", "attached_to"
    ]
    raw_df = spark.createDataFrame(test_data, cols)
    cleaned_df, summary = preprocess_and_clean(raw_df, output_dir="outputs", save_parquet=False)

    assert summary["zero_negative_dropped"] == 2
    assert summary["physical_outliers_dropped"] == 1
    assert summary["cleaned_record_count"] == 1


def test_anomaly_detection_handles_zero_iqr(spark):
    """
    Verify that when all readings in a fixture category are identical (IQR == 0),
    the system applies a fallback threshold without throwing division by zero or errors.
    """
    # 20 readings with identical value of 6.0 Liters
    identical_data = [
        ("2020-01-01 12:00:00", "2020-01-01", 12, 4, "Wednesday", 1, 2020, 0, f"S{i:03d}", "APT_01", "CL_A", "R01", "Bathroom", "cold", "FixedFlushToilet", 0.006, 6.0, 100.0)
        for i in range(20)
    ]
    # Add 1 spike reading
    identical_data.append(
        ("2020-01-01 12:10:00", "2020-01-01", 12, 4, "Wednesday", 1, 2020, 0, "S999", "APT_01", "CL_A", "R01", "Bathroom", "cold", "FixedFlushToilet", 0.050, 50.0, 150.0)
    )
    cols = [
        "timestamp", "date", "hour", "day_of_week", "day_name", "month", "year", "is_weekend",
        "sensor_id", "apartment", "cluster_name", "room_number", "room_type", "water_type",
        "fixture", "volume_m3", "volume_liters", "cumulative_m3"
    ]
    df = spark.createDataFrame(identical_data, cols)
    anomalies_df, th_df, summary = detect_fixture_anomalies(spark, df, iqr_multiplier=1.5)

    assert len(th_df) == 1
    assert th_df.iloc[0]["threshold_liters"] > 6.0
    assert summary["total_anomalies"] == 1


def test_forecasting_graceful_on_insufficient_data():
    """Verify that forecasting returns error dictionary when observations < 30."""
    short_series = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=10, freq="D"),
        "total_liters": [100.0] * 10
    })
    test_eval, forecast, metrics = train_daily_forecast_model(short_series)
    assert test_eval is None
    assert forecast is None
    assert "error" in metrics


def test_preprocessing_handles_missing_optional_columns(spark):
    """
    Verify preprocessing succeeds even if optional contextual columns
    (such as aggregated_value, apartment, cluster_name, room_number) are omitted from input.
    """
    minimal_data = [
        ("2020-01-01 10:00:00", "0.015", "S001", "Kitchen", "cold water consumption", "Kitchen tap")
    ]
    cols = ["timestamp", "value", "sensor_id", "room_type", "type", "attached_to"]
    raw_df = spark.createDataFrame(minimal_data, cols)
    cleaned_df, summary = preprocess_and_clean(raw_df, save_parquet=False)

    assert summary["cleaned_record_count"] == 1
    row = cleaned_df.collect()[0]
    assert row["apartment"] == "Shared Facility"
    assert row["cluster_name"] == "Building-Wide"
    assert row["room_number"] == "Unknown"
    assert row["volume_liters"] == 15.0


def test_analytics_and_anomaly_detection_on_empty_dataset(spark):
    """
    Verify all analytics functions and anomaly detection handle an empty DataFrame
    gracefully without throwing uncaught exceptions.
    """
    sample_data = [
        ("2020-01-01 10:00:00", "0.010", "0.010", "S001", "APT_1", "CL_A", "R1", "Bathroom", "cold", "Toilet")
    ]
    cols = [
        "timestamp", "value", "aggregated_value", "sensor_id", "apartment",
        "cluster_name", "room_number", "room_type", "type", "attached_to"
    ]
    raw_df = spark.createDataFrame(sample_data, cols)
    cleaned_df, _ = preprocess_and_clean(raw_df, save_parquet=False)
    empty_df = cleaned_df.filter("1 = 0")

    kpis = compute_kpis(spark, empty_df)
    assert kpis["total_events"] == 0
    assert kpis["total_volume_liters"] == 0.0
    assert kpis["total_days"] == 0
    assert kpis["daily_avg_liters"] == 0.0
    assert kpis["min_date"] == "N/A"
    assert kpis["max_date"] == "N/A"

    hourly = get_hourly_trends(spark, empty_df)
    assert hourly.empty

    daily = get_daily_trends(spark, empty_df)
    assert daily.empty
    assert "rolling_7d_avg" in daily.columns

    dow = get_day_of_week_trends(spark, empty_df)
    assert dow.empty

    fixtures = get_fixture_distribution(spark, empty_df)
    assert fixtures.empty
    assert "percentage" in fixtures.columns

    water_types = get_water_type_distribution(spark, empty_df)
    assert water_types.empty
    assert "percentage" in water_types.columns

    apts = get_apartment_comparison(spark, empty_df)
    assert apts.empty

    monthly = get_monthly_trends(spark, empty_df)
    assert monthly.empty

    peaks = identify_peak_periods(spark, empty_df)
    assert peaks["peak_hour"] is None

    anom_df, th_df, anom_sum = detect_fixture_anomalies(spark, empty_df)
    assert anom_sum["total_anomalies"] == 0
    assert "threshold_liters" in th_df.columns
    assert "iqr" in th_df.columns


def test_export_processed_outputs_on_empty_dataset(spark, tmp_path):
    """
    Verify export_processed_outputs creates valid Parquet files and summary
    reports even when the input DataFrame has 0 rows (no CANNOT_INFER_EMPTY_SCHEMA crash).
    """
    sample_data = [
        ("2020-01-01 10:00:00", "0.010", "0.010", "S001", "APT_1", "CL_A", "R1", "Bathroom", "cold", "Toilet")
    ]
    cols = [
        "timestamp", "value", "aggregated_value", "sensor_id", "apartment",
        "cluster_name", "room_number", "room_type", "type", "attached_to"
    ]
    raw_df = spark.createDataFrame(sample_data, cols)
    cleaned_df, summary = preprocess_and_clean(raw_df, save_parquet=False)
    empty_df = cleaned_df.filter("1 = 0")
    anom_df, _, _ = detect_fixture_anomalies(spark, empty_df)

    export_dir = tmp_path / "test_outputs"
    artifacts = export_processed_outputs(
        spark,
        empty_df,
        output_dir=str(export_dir),
        summary_meta=summary,
        anomalies_df=anom_df
    )

    assert "hourly_aggregates" in artifacts
    assert "daily_aggregates" in artifacts
    assert "fixture_aggregates" in artifacts
    assert "pipeline_summary" in artifacts
    assert "statistical_anomalies" in artifacts

    # Verify Parquet datasets are readable by Spark
    spark_hourly = spark.read.parquet(artifacts["hourly_aggregates"])
    assert spark_hourly.count() == 0
    assert "hour" in spark_hourly.columns

    spark_daily = spark.read.parquet(artifacts["daily_aggregates"])
    assert spark_daily.count() == 0
    assert "total_liters" in spark_daily.columns


def test_streamlit_dashboard_runs_end_to_end_without_exceptions():
    """
    Verify that the Streamlit dashboard initializes, runs, and renders end-to-end
    without raising NameError or any other unhandled exception across filter interactions.
    """
    app_file = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_file, default_timeout=120)
    at.run(timeout=120)
    assert len(at.exception) == 0, f"Streamlit raised exceptions on load: {[e.value for e in at.exception]}"

    # Interactive test 1: Modify fixture multiselect
    if len(at.multiselect) > 0:
        at.multiselect[0].set_value(["Shower"]).run(timeout=120)
        assert len(at.exception) == 0, f"Exceptions after filter selection: {[e.value for e in at.exception]}"

        # Empty multiselect test
        at.multiselect[0].set_value([]).run(timeout=120)
        assert len(at.exception) == 0, f"Exceptions after empty multiselect: {[e.value for e in at.exception]}"

    # Interactive test 2: Adjust sensitivity slider
    if len(at.slider) > 0:
        at.slider[0].set_value(2.0).run(timeout=120)
        assert len(at.exception) == 0, f"Exceptions after slider adjustment: {[e.value for e in at.exception]}"

    # Interactive test 3: Click export artifacts button
    export_btn = None
    for b in at.button:
        if "Export Parquet" in b.label:
            export_btn = b
            break
    if export_btn:
        export_btn.click().run(timeout=120)
        assert len(at.exception) == 0, f"Exceptions after export click: {[e.value for e in at.exception]}"
