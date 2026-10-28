"""
Optional Forecasting Module for WaterWise.
Implements chronological daily water consumption forecasting using lag features
and scikit-learn regressor models with out-of-sample evaluation.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def train_daily_forecast_model(
    daily_df: pd.DataFrame,
    forecast_horizon_days: int = 14,
    test_split_ratio: float = 0.20
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Trains a chronological time-series forecasting model on daily consumption aggregates.

    Design Principles:
    - Pure chronological train/test split to eliminate future data leakage.
    - Lag features (lag 1, lag 7, 7-day rolling mean, day of week, month).
    - Reports MAE, RMSE, and R2 on held-out test records.
    - Clearly demarcates predictions as statistical estimates.
    """
    if daily_df is None or len(daily_df) < 30:
        return None, None, {"error": "Insufficient daily observations for time series modeling (minimum 30 days required)."}

    # Prepare time-series dataframe
    df = daily_df[["date", "total_liters"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    # Reindex to continuous daily calendar frequency to guarantee lag_1 and lag_7 strictly represent 1-day and 7-day lags
    min_date = df["date"].min()
    max_date = df["date"].max()
    full_date_range = pd.date_range(min_date, max_date, freq="D")
    df = (
        df.set_index("date")
        .reindex(full_date_range)
        .rename_axis("date")
        .reset_index()
    )
    df["total_liters"] = df["total_liters"].interpolate(method="linear").bfill().ffill().round(2)

    # Feature engineering
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["lag_1"] = df["total_liters"].shift(1)
    df["lag_7"] = df["total_liters"].shift(7)
    df["rolling_mean_7"] = df["total_liters"].shift(1).rolling(7).mean()

    # Drop initial NaN rows caused by lag shifting
    clean_ts = df.dropna().reset_index(drop=True)
    if len(clean_ts) < 20:
        return None, None, {"error": "Insufficient data points after creating lag features."}

    feature_cols = ["day_of_week", "month", "is_weekend", "lag_1", "lag_7", "rolling_mean_7"]
    X = clean_ts[feature_cols]
    y = clean_ts["total_liters"]

    # Chronological Train / Test Split
    split_idx = int(len(clean_ts) * (1.0 - test_split_ratio))
    if split_idx >= len(clean_ts) - 1:
        split_idx = len(clean_ts) - 2
    if split_idx < 5:
        return None, None, {"error": "Insufficient training samples after split."}

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_dates = clean_ts["date"].iloc[split_idx:]

    # Train model
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Evaluate on held-out test data
    y_test_pred = model.predict(X_test)
    y_test_pred = np.clip(y_test_pred, a_min=0, a_max=None) # water volume cannot be negative

    mae = float(mean_absolute_error(y_test, y_test_pred))
    rmse = float(root_mean_squared_error(y_test, y_test_pred))
    try:
        r2 = float(r2_score(y_test, y_test_pred))
        if np.isnan(r2):
            r2 = 0.0
    except Exception:
        r2 = 0.0

    test_eval_df = pd.DataFrame({
        "date": test_dates.values,
        "actual_liters": y_test.values.round(2),
        "predicted_liters": y_test_pred.round(2),
        "residual_liters": (y_test.values - y_test_pred).round(2)
    })

    # Generate forward out-of-sample forecast
    last_date = clean_ts["date"].iloc[-1]
    last_known_vals = list(clean_ts["total_liters"].iloc[-7:].values)
    future_rows = []

    curr_date = last_date
    for _ in range(forecast_horizon_days):
        curr_date += pd.Timedelta(days=1)
        dow = curr_date.dayofweek
        month = curr_date.month
        is_wknd = 1 if dow in [5, 6] else 0
        lag_1_val = last_known_vals[-1]
        lag_7_val = last_known_vals[-7]
        roll_val = float(np.mean(last_known_vals[-7:]))

        feat_vector = pd.DataFrame([{
            "day_of_week": dow,
            "month": month,
            "is_weekend": is_wknd,
            "lag_1": lag_1_val,
            "lag_7": lag_7_val,
            "rolling_mean_7": roll_val
        }])

        pred_val = float(np.clip(model.predict(feat_vector)[0], 0, None))
        future_rows.append({
            "date": curr_date,
            "forecast_estimated_liters": round(pred_val, 2),
            "forecast_estimated_m3": round(pred_val / 1000.0, 3)
        })
        last_known_vals.append(pred_val)

    forecast_df = pd.DataFrame(future_rows)

    metrics = {
        "model_type": "Ridge Time-Series Regressor with Autoregressive Lag Features",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_mae_liters": round(mae, 2),
        "test_rmse_liters": round(rmse, 2),
        "test_r2_score": round(r2, 3),
        "forecast_days": forecast_horizon_days,
        "disclaimer": "Predictions are statistical projections based on historical patterns and should be treated as estimates, not actual measurements."
    }

    return test_eval_df, forecast_df, metrics
