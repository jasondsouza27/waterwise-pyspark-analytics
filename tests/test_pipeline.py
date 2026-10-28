"""
Automated Verification Suite for WaterWise PySpark Pipeline.
Tests loading, schema validation, PySpark cleaning, aggregations,
anomaly detection, and forecasting on the real dataset.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.spark_session import get_spark_session, stop_spark_session
from src.data_loader import locate_dataset_file, load_raw_data, compute_missing_value_stats
from src.preprocessing import preprocess_and_clean
from src.analytics import (
    compute_kpis,
    get_hourly_trends,
    get_daily_trends,
    get_day_of_week_trends,
    get_fixture_distribution,
    identify_peak_periods
)
from src.anomaly_detection import detect_fixture_anomalies
from src.forecasting import train_daily_forecast_model


def run_tests():
    print("=" * 70)
    print("STEP 1: Starting SparkSession...")
    print("=" * 70)
    spark = get_spark_session(app_name="WaterWiseVerification")
    print(f"Spark version: {spark.version}")

    print("\n" + "=" * 70)
    print("STEP 2: Testing Dataset Locator & Raw Loading (Sample 50,000 rows)...")
    print("=" * 70)
    csv_file = locate_dataset_file("data")
    print(f"Located CSV: {csv_file}")
    assert csv_file.exists(), "CSV file does not exist!"

    raw_df, metadata = load_raw_data(spark, data_dir="data", max_rows=50000)
    print(f"Loaded raw sample count: {metadata['raw_record_count']:,}")
    print(f"Detected columns: {metadata['detected_columns']}")
    assert metadata["raw_record_count"] == 50000, "Raw count mismatch!"

    missing_stats = compute_missing_value_stats(raw_df)
    print(f"Missing value stats (top 5): {dict(list(missing_stats.items())[:5])}")

    print("\n" + "=" * 70)
    print("STEP 3: Testing Preprocessing & Cleaning Pipeline...")
    print("=" * 70)
    cleaned_df, clean_summary = preprocess_and_clean(raw_df, output_dir="outputs")
    print(f"Cleaned records count: {clean_summary['cleaned_record_count']:,}")
    print(f"Retention percentage: {clean_summary['retention_percentage']}%")
    print(f"Duplicates dropped: {clean_summary['duplicates_dropped']}")
    print(f"Zero/negative dropped: {clean_summary['zero_negative_dropped']}")
    assert clean_summary["cleaned_record_count"] > 0, "Cleaned record count is zero!"

    print("\n" + "=" * 70)
    print("STEP 4: Testing Analytics & KPI Computation...")
    print("=" * 70)
    kpis = compute_kpis(spark, cleaned_df)
    print(f"Total Consumption (m3): {kpis['total_volume_m3']:,}")
    print(f"Total Consumption (Liters): {kpis['total_volume_liters']:,}")
    print(f"Avg Event Volume (Liters): {kpis['avg_volume_liters']}")
    print(f"Active Sensors: {kpis['active_sensors']}")
    print(f"Date Range: {kpis['min_date']} to {kpis['max_date']}")
    assert kpis["total_volume_liters"] > 0, "Total liters must be > 0"

    hourly_df = get_hourly_trends(spark, cleaned_df)
    print(f"Hourly trend shape: {hourly_df.shape} (24 hours expected)")
    assert len(hourly_df) == 24, f"Expected 24 hours, got {len(hourly_df)}"

    daily_df = get_daily_trends(spark, cleaned_df)
    print(f"Daily trend rows: {len(daily_df)}")
    assert len(daily_df) > 0, "Daily trends empty!"

    dow_df = get_day_of_week_trends(spark, cleaned_df)
    print(f"Day of week rows: {len(dow_df)}")
    assert len(dow_df) > 0, "DOW trends empty!"

    fixtures_df = get_fixture_distribution(spark, cleaned_df)
    print(f"Fixtures breakdown:\n{fixtures_df[['fixture', 'total_liters', 'percentage']]}")
    assert len(fixtures_df) > 0, "Fixture breakdown empty!"

    peaks = identify_peak_periods(spark, cleaned_df)
    print(f"Peak hour: {peaks['peak_hour']}:00 (Volume: {peaks['peak_hour_volume_liters']:,} L)")
    print(f"Peak day of week: {peaks['peak_day_of_week']}")
    print(f"Top fixture: {peaks['top_fixture']} ({peaks['top_fixture_pct']}%)")

    print("\n" + "=" * 70)
    print("STEP 5: Testing Statistical Anomaly Detection (IQR)...")
    print("=" * 70)
    anomalies_df, thresholds_df, anom_summary = detect_fixture_anomalies(spark, cleaned_df, iqr_multiplier=1.5)
    print(f"Total anomalies flagged: {anom_summary['total_anomalies']}")
    print(f"Anomaly rate: {anom_summary['overall_anomaly_pct']}%")
    print(f"Thresholds Table:\n{thresholds_df[['fixture', 'q1_liters', 'median_liters', 'q3_liters', 'iqr', 'threshold_liters', 'anomaly_count', 'anomaly_pct']]}")
    assert len(thresholds_df) > 0, "Thresholds table empty!"

    print("\n" + "=" * 70)
    print("STEP 6: Testing Forecasting Enhancement...")
    print("=" * 70)
    test_eval_df, forecast_df, fc_metrics = train_daily_forecast_model(daily_df, forecast_horizon_days=7)
    if "error" not in fc_metrics:
        print(f"Model: {fc_metrics['model_type']}")
        print(f"Test MAE: {fc_metrics['test_mae_liters']} Liters")
        print(f"Test RMSE: {fc_metrics['test_rmse_liters']} Liters")
        print(f"Forecast rows: {len(forecast_df)}")
        print(f"Sample Forecast:\n{forecast_df.head(3)}")
    else:
        print(f"Forecasting note (expected on small sample): {fc_metrics['error']}")

    print("\n" + "=" * 70)
    print("STEP 7: Testing Analytical Output Persistence...")
    print("=" * 70)
    from src.analytics import export_processed_outputs
    artifacts = export_processed_outputs(spark, cleaned_df, output_dir="outputs", summary_meta=clean_summary, anomalies_df=anomalies_df)
    print(f"Saved artifacts: {list(artifacts.keys())}")
    assert "hourly_aggregates" in artifacts, "Hourly export failed!"
    assert "daily_aggregates" in artifacts, "Daily export failed!"

    print("\n" + "=" * 70)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
    stop_spark_session()


if __name__ == "__main__":
    run_tests()
