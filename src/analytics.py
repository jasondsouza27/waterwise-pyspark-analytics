"""
PySpark Analytics Module for WaterWise.
Performs large-scale aggregation, trend discovery, peak period analysis,
and SQL-based queries on cleaned water consumption DataFrames.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def apply_filters(
    df: DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fixtures: Optional[List[str]] = None,
    water_types: Optional[List[str]] = None,
    apartments: Optional[List[str]] = None
) -> DataFrame:
    """
    Applies user-selected filters on the PySpark DataFrame.
    Filters are pushed down into Spark execution plan before aggregations.
    """
    filtered_df = df

    if start_date:
        filtered_df = filtered_df.filter(F.col("date") >= F.to_date(F.lit(start_date)))
    if end_date:
        filtered_df = filtered_df.filter(F.col("date") <= F.to_date(F.lit(end_date)))
    if fixtures and len(fixtures) > 0 and "All" not in fixtures:
        filtered_df = filtered_df.filter(F.col("fixture").isin(fixtures))
    if water_types and len(water_types) > 0 and "All" not in water_types:
        filtered_df = filtered_df.filter(F.col("water_type").isin(water_types))
    if apartments and len(apartments) > 0 and "All" not in apartments:
        filtered_df = filtered_df.filter(F.col("apartment").isin(apartments))

    return filtered_df


def compute_kpis(spark: SparkSession, df: DataFrame) -> Dict[str, Any]:
    """
    Computes top-level Key Performance Indicators (KPIs) using Spark SQL.
    """
    df.createOrReplaceTempView("v_water_kpi")

    query = """
    SELECT
        COUNT(*) AS total_events,
        COALESCE(SUM(volume_m3), 0.0) AS total_volume_m3,
        COALESCE(SUM(volume_liters), 0.0) AS total_volume_liters,
        COALESCE(AVG(volume_liters), 0.0) AS avg_volume_liters,
        COALESCE(STDDEV(volume_liters), 0.0) AS std_volume_liters,
        COALESCE(MIN(volume_liters), 0.0) AS min_volume_liters,
        COALESCE(MAX(volume_liters), 0.0) AS max_volume_liters,
        COUNT(DISTINCT sensor_id) AS active_sensors,
        COUNT(DISTINCT apartment) AS active_apartments,
        COUNT(DISTINCT cluster_name) AS active_clusters,
        MIN(date) AS min_date,
        MAX(date) AS max_date,
        DATEDIFF(MAX(date), MIN(date)) + 1 AS total_days
    FROM v_water_kpi
    """
    row = spark.sql(query).collect()[0]

    total_events = int(row["total_events"])
    if total_events == 0:
        return {
            "total_events": 0,
            "total_volume_m3": 0.0,
            "total_volume_liters": 0.0,
            "avg_volume_liters": 0.0,
            "std_volume_liters": 0.0,
            "min_volume_liters": 0.0,
            "max_volume_liters": 0.0,
            "active_sensors": 0,
            "active_apartments": 0,
            "active_clusters": 0,
            "min_date": "N/A",
            "max_date": "N/A",
            "total_days": 0,
            "daily_avg_liters": 0.0
        }

    total_days = max(1, row["total_days"] or 1)
    daily_avg_liters = row["total_volume_liters"] / total_days

    return {
        "total_events": total_events,
        "total_volume_m3": round(float(row["total_volume_m3"]), 3),
        "total_volume_liters": round(float(row["total_volume_liters"]), 2),
        "avg_volume_liters": round(float(row["avg_volume_liters"]), 2),
        "std_volume_liters": round(float(row["std_volume_liters"]), 2),
        "min_volume_liters": round(float(row["min_volume_liters"]), 2),
        "max_volume_liters": round(float(row["max_volume_liters"]), 2),
        "active_sensors": int(row["active_sensors"]),
        "active_apartments": int(row["active_apartments"]),
        "active_clusters": int(row["active_clusters"]),
        "min_date": str(row["min_date"]),
        "max_date": str(row["max_date"]),
        "total_days": int(total_days),
        "daily_avg_liters": round(daily_avg_liters, 2)
    }


def get_hourly_trends(spark: SparkSession, df: DataFrame) -> pd.DataFrame:
    """
    Computes 24-hour diurnal consumption patterns across the entire building.
    Returns a small Pandas DataFrame for Plotly charting.
    """
    df.createOrReplaceTempView("v_water_hourly")

    query = """
    SELECT
        hour,
        ROUND(SUM(volume_liters), 2) AS total_liters,
        ROUND(AVG(volume_liters), 2) AS avg_liters_per_event,
        COUNT(*) AS event_count
    FROM v_water_hourly
    GROUP BY hour
    ORDER BY hour ASC
    """
    return spark.sql(query).toPandas()


def get_daily_trends(spark: SparkSession, df: DataFrame) -> pd.DataFrame:
    """
    Computes time-series daily water consumption (Liters and m3).
    """
    df.createOrReplaceTempView("v_water_daily")

    query = """
    SELECT
        date,
        ROUND(SUM(volume_liters), 2) AS total_liters,
        ROUND(SUM(volume_m3), 3) AS total_m3,
        COUNT(*) AS event_count
    FROM v_water_daily
    GROUP BY date
    ORDER BY date ASC
    """
    pdf = spark.sql(query).toPandas()
    if not pdf.empty and "date" in pdf.columns:
        pdf["date"] = pd.to_datetime(pdf["date"])
        # Compute 7-day rolling average for smoother trend visualization
        pdf["rolling_7d_avg"] = pdf["total_liters"].rolling(window=7, min_periods=1).mean().round(2)
    elif "rolling_7d_avg" not in pdf.columns:
        pdf["rolling_7d_avg"] = pd.Series(dtype=float)
    return pdf


def get_day_of_week_trends(spark: SparkSession, df: DataFrame) -> pd.DataFrame:
    """
    Computes consumption aggregated by Day of Week.
    """
    df.createOrReplaceTempView("v_water_dow")

    query = """
    SELECT
        day_of_week,
        day_name,
        ROUND(SUM(volume_liters), 2) AS total_liters,
        ROUND(AVG(volume_liters), 2) AS avg_event_liters,
        COUNT(*) AS event_count
    FROM v_water_dow
    GROUP BY day_of_week, day_name
    ORDER BY day_of_week ASC
    """
    return spark.sql(query).toPandas()


def get_fixture_distribution(spark: SparkSession, df: DataFrame) -> pd.DataFrame:
    """
    Computes water consumption distribution categorized by fixture type.
    """
    df.createOrReplaceTempView("v_water_fixture")

    query = """
    SELECT
        fixture,
        ROUND(SUM(volume_liters), 2) AS total_liters,
        ROUND(SUM(volume_m3), 3) AS total_m3,
        ROUND(AVG(volume_liters), 2) AS avg_event_liters,
        COUNT(*) AS event_count
    FROM v_water_fixture
    GROUP BY fixture
    ORDER BY total_liters DESC
    """
    pdf = spark.sql(query).toPandas()
    if not pdf.empty:
        total = pdf["total_liters"].sum()
        pdf["percentage"] = ((pdf["total_liters"] / total) * 100).round(2) if total > 0 else 0.0
    elif "percentage" not in pdf.columns:
        pdf["percentage"] = pd.Series(dtype=float)
    return pdf


def get_water_type_distribution(spark: SparkSession, df: DataFrame) -> pd.DataFrame:
    """
    Computes breakdown between cold water and hot water consumption.
    """
    df.createOrReplaceTempView("v_water_type")

    query = """
    SELECT
        water_type,
        ROUND(SUM(volume_liters), 2) AS total_liters,
        COUNT(*) AS event_count
    FROM v_water_type
    GROUP BY water_type
    ORDER BY total_liters DESC
    """
    pdf = spark.sql(query).toPandas()
    if not pdf.empty:
        total = pdf["total_liters"].sum()
        pdf["percentage"] = ((pdf["total_liters"] / total) * 100).round(2) if total > 0 else 0.0
    elif "percentage" not in pdf.columns:
        pdf["percentage"] = pd.Series(dtype=float)
    return pdf


def get_apartment_comparison(spark: SparkSession, df: DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Ranks top consuming residential units / apartments.
    """
    df.createOrReplaceTempView("v_water_apt")

    query = f"""
    SELECT
        apartment,
        ROUND(SUM(volume_liters), 2) AS total_liters,
        ROUND(AVG(volume_liters), 2) AS avg_event_liters,
        COUNT(*) AS event_count
    FROM v_water_apt
    GROUP BY apartment
    ORDER BY total_liters DESC
    LIMIT {top_n}
    """
    return spark.sql(query).toPandas()


def identify_peak_periods(spark: SparkSession, df: DataFrame) -> Dict[str, Any]:
    """
    Identifies peak consumption hours, peak days of week, and peak fixtures.
    """
    hourly_df = get_hourly_trends(spark, df)
    dow_df = get_day_of_week_trends(spark, df)
    fixture_df = get_fixture_distribution(spark, df)

    peak_hour = None
    peak_hour_volume = 0.0
    if not hourly_df.empty:
        max_h_row = hourly_df.loc[hourly_df["total_liters"].idxmax()]
        peak_hour = int(max_h_row["hour"])
        peak_hour_volume = float(max_h_row["total_liters"])

    peak_dow = None
    peak_dow_volume = 0.0
    if not dow_df.empty:
        max_d_row = dow_df.loc[dow_df["total_liters"].idxmax()]
        peak_dow = str(max_d_row["day_name"])
        peak_dow_volume = float(max_d_row["total_liters"])

    top_fixture = None
    top_fixture_pct = 0.0
    if not fixture_df.empty:
        top_f_row = fixture_df.iloc[0]
        top_fixture = str(top_f_row["fixture"])
        top_fixture_pct = float(top_f_row["percentage"])

    return {
        "peak_hour": peak_hour,
        "peak_hour_volume_liters": peak_hour_volume,
        "peak_day_of_week": peak_dow,
        "peak_day_volume_liters": peak_dow_volume,
        "top_fixture": top_fixture,
        "top_fixture_pct": top_fixture_pct
    }


def get_monthly_trends(spark: SparkSession, df: DataFrame) -> pd.DataFrame:
    """
    Computes monthly aggregated water consumption (Liters and m3).
    """
    df.createOrReplaceTempView("v_water_monthly")

    query = """
    SELECT
        year,
        month,
        ROUND(SUM(volume_liters), 2) AS monthly_liters,
        ROUND(SUM(volume_m3), 3) AS monthly_m3,
        COUNT(*) AS event_count
    FROM v_water_monthly
    GROUP BY year, month
    ORDER BY year ASC, month ASC
    """
    return spark.sql(query).toPandas()


def export_processed_outputs(
    spark: SparkSession,
    df: DataFrame,
    output_dir: str = "outputs",
    summary_meta: Optional[Dict[str, Any]] = None,
    anomalies_df: Optional[DataFrame] = None
) -> Dict[str, str]:
    """
    Persists analytical aggregates to local Parquet and report directories.
    Matches the schema documented in outputs/README.md.
    """
    base_path = Path(output_dir).resolve()
    parquet_path = base_path / "parquet"
    reports_path = base_path / "reports"
    parquet_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    saved_artifacts = {}
    df.createOrReplaceTempView("v_export_active")

    # 1. Hourly Aggregates Parquet
    try:
        h_dest = parquet_path / "hourly_aggregates"
        spark.sql("""
            SELECT
                hour,
                ROUND(SUM(volume_liters), 2) AS total_liters,
                ROUND(AVG(volume_liters), 2) AS avg_liters_per_event,
                COUNT(*) AS event_count
            FROM v_export_active
            GROUP BY hour
            ORDER BY hour ASC
        """).write.mode("overwrite").parquet(str(h_dest))
        saved_artifacts["hourly_aggregates"] = str(h_dest)
    except Exception as e:
        print(f"[WARN] Failed to export hourly aggregates: {e}")

    # 2. Daily Aggregates Parquet
    try:
        d_dest = parquet_path / "daily_aggregates"
        spark.sql("""
            SELECT
                date,
                ROUND(SUM(volume_liters), 2) AS total_liters,
                ROUND(SUM(volume_m3), 3) AS total_m3,
                COUNT(*) AS event_count
            FROM v_export_active
            GROUP BY date
            ORDER BY date ASC
        """).write.mode("overwrite").parquet(str(d_dest))
        saved_artifacts["daily_aggregates"] = str(d_dest)
    except Exception as e:
        print(f"[WARN] Failed to export daily aggregates: {e}")

    # 3. Fixture Aggregates Parquet
    try:
        f_dest = parquet_path / "fixture_aggregates"
        total_liters_row = spark.sql("SELECT COALESCE(SUM(volume_liters), 0.0) AS grand_total FROM v_export_active").collect()[0]
        grand_total = float(total_liters_row["grand_total"])

        if grand_total > 0:
            spark.sql(f"""
                SELECT
                    fixture,
                    ROUND(SUM(volume_liters), 2) AS total_liters,
                    ROUND(SUM(volume_m3), 3) AS total_m3,
                    ROUND(AVG(volume_liters), 2) AS avg_event_liters,
                    COUNT(*) AS event_count,
                    ROUND((SUM(volume_liters) / {grand_total}) * 100, 2) AS percentage
                FROM v_export_active
                GROUP BY fixture
                ORDER BY total_liters DESC
            """).write.mode("overwrite").parquet(str(f_dest))
        else:
            spark.sql("""
                SELECT
                    fixture,
                    ROUND(SUM(volume_liters), 2) AS total_liters,
                    ROUND(SUM(volume_m3), 3) AS total_m3,
                    ROUND(AVG(volume_liters), 2) AS avg_event_liters,
                    COUNT(*) AS event_count,
                    CAST(0.0 AS DOUBLE) AS percentage
                FROM v_export_active
                GROUP BY fixture
                ORDER BY total_liters DESC
            """).write.mode("overwrite").parquet(str(f_dest))
        saved_artifacts["fixture_aggregates"] = str(f_dest)
    except Exception as e:
        print(f"[WARN] Failed to export fixture aggregates: {e}")

    # 4. Pipeline Summary JSON Report
    if summary_meta:
        try:
            summary_dest = reports_path / "pipeline_summary.json"
            with open(summary_dest, "w", encoding="utf-8") as f:
                json.dump(summary_meta, f, indent=2, default=str)
            saved_artifacts["pipeline_summary"] = str(summary_dest)
        except Exception as e:
            print(f"[WARN] Failed to export summary JSON: {e}")

    # 5. Statistical Anomalies CSV Report
    if anomalies_df is not None:
        try:
            top_anomalies_pdf = anomalies_df.limit(1000).toPandas()
            anom_dest = reports_path / "statistical_anomalies.csv"
            top_anomalies_pdf.to_csv(anom_dest, index=False)
            saved_artifacts["statistical_anomalies"] = str(anom_dest)
        except Exception as e:
            print(f"[WARN] Failed to export anomalies CSV: {e}")

    return saved_artifacts
