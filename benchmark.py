"""Predict scheduled-flight arrival delay with a strict month holdout."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["Reporting_Airline", "Origin", "Dest", "Month", "DayOfWeek", "DepHour", "ArrHour"]
NUMERIC = ["CRSElapsedTime", "Distance"]


def hhmm_hour(value):
    try:
        value = int(value)
        hour = value // 100
        return hour if 0 <= hour <= 23 else np.nan
    except (TypeError, ValueError):
        return np.nan


def main():
    source = Path("data/flights.csv")
    d = pd.read_csv(source, low_memory=False)
    d["FlightDate"] = pd.to_datetime(d["FlightDate"], errors="coerce")
    d["ArrDelay"] = pd.to_numeric(d["ArrDelay"], errors="coerce")
    d["DepHour"] = d["CRSDepTime"].map(hhmm_hour)
    d["ArrHour"] = d["CRSArrTime"].map(hhmm_hour)
    d = d.dropna(subset=["FlightDate", "ArrDelay", "CRSElapsedTime", "Distance"]).copy()
    d["target_delay_minutes"] = d["ArrDelay"].clip(-60, 240)

    train = d[d["Month"].isin([1, 2])]
    test = d[d["Month"] == 3]
    if train.empty or test.empty:
        raise SystemExit("Expected January-February training and March holdout rows.")

    model_cols = CATEGORICAL + NUMERIC
    x_train, x_test = train[model_cols], test[model_cols]
    y_train, y_test = train["target_delay_minutes"], test["target_delay_minutes"]
    prep = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler(with_mean=False)),
        ]), NUMERIC),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=100)),
        ]), CATEGORICAL),
    ])
    # Validation on February compared alpha in {0.1, 1, 10, 20, 100, 500}; 500 minimized MAE.
    model = Pipeline([("features", prep), ("ridge", Ridge(alpha=500.0, solver="lsqr"))])
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    baseline = np.full(len(y_test), float(y_train.median()))

    report = {
        "dataset": "U.S. DOT BTS Reporting Carrier On-Time Performance, 2023 monthly files",
        "target": "Arrival delay minutes (ArrDelay), clipped to [-60, 240] for this baseline",
        "sample": {"train_months": ["2023-01", "2023-02"], "test_month": "2023-03",
                   "train_rows": int(len(train)), "test_rows": int(len(test)),
                   "train_flight_dates": [str(train.FlightDate.min().date()), str(train.FlightDate.max().date())],
                   "test_flight_dates": [str(test.FlightDate.min().date()), str(test.FlightDate.max().date())]},
        "method": "Ridge regression; one-hot route/carrier/calendar variables and scaled scheduled duration/distance",
        "features": model_cols,
        "excluded_post_schedule_or_outcome_fields": [
            "DepDelay", "DepDelayMinutes", "DepDel15", "DepTime", "ArrTime", "ArrDelayMinutes",
            "ArrDel15", "TaxiOut", "TaxiIn", "WheelsOff", "WheelsOn", "CarrierDelay", "WeatherDelay",
            "NASDelay", "SecurityDelay", "LateAircraftDelay", "Tail_Number", "flight number",
        ],
        "test_metrics": {
            "train_median_baseline_mae_minutes": float(mean_absolute_error(y_test, baseline)),
            "model_mae_minutes": float(mean_absolute_error(y_test, pred)),
            "model_rmse_minutes": float(mean_squared_error(y_test, pred) ** 0.5),
        },
        "source_csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "limitations": [
            "This is an offline scheduled-time baseline, not a real-time operational forecast.",
            "Only three months from 2023 are sampled; performance may change across seasons and years.",
            "The training sample is capped at 100,000 random flights per month.",
            "Cancelled and diverted flights are excluded, so this does not predict cancellation or diversion.",
            "Delay distribution is heavy-tailed; clipping means reported errors apply to the clipped target.",
        ],
    }
    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

