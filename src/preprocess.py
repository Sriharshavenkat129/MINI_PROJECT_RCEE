"""Cleaning + feature engineering (shared by training and the app)."""
import pandas as pd
from .config import FEATURES, TARGET, PEAK_MONTHS

RAW_COLS = ["airline", "source", "destination", "total_stops", "dep_time",
            "duration_min", "booking_date", "journey_date", "price"]


def clean(df: pd.DataFrame, report: bool = False):
    """Remove duplicates / nulls / invalid rows / extreme outliers."""
    log = {"rows_raw": len(df)}
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    df = df[RAW_COLS].copy()

    df = df.drop_duplicates()
    log["after_duplicates"] = len(df)

    df = df.dropna()
    log["after_nulls"] = len(df)

    df["booking_date"] = pd.to_datetime(df["booking_date"], errors="coerce")
    df["journey_date"] = pd.to_datetime(df["journey_date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["booking_date", "journey_date", "price"])
    df = df[(df["journey_date"] > df["booking_date"]) & (df["price"] > 0)
            & (df["source"] != df["destination"])]
    log["after_validity"] = len(df)

    # outliers: 3 * IQR fence within each route
    route = df["source"] + "-" + df["destination"]
    q1 = df.groupby(route)["price"].transform(lambda s: s.quantile(.25))
    q3 = df.groupby(route)["price"].transform(lambda s: s.quantile(.75))
    iqr = q3 - q1
    df = df[(df["price"] >= q1 - 3 * iqr) & (df["price"] <= q3 + 3 * iqr)].reset_index(drop=True)
    log["after_outliers"] = len(df)
    return (df, log) if report else df


def _period(h):
    if h < 6:
        return "Early Morning"
    if h < 12:
        return "Morning"
    if h < 17:
        return "Afternoon"
    if h < 21:
        return "Evening"
    return "Night"


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features from cleaned booking data."""
    out = df.copy()
    out["booking_date"] = pd.to_datetime(out["booking_date"])
    out["journey_date"] = pd.to_datetime(out["journey_date"])
    out["days_left"] = (out["journey_date"] - out["booking_date"]).dt.days
    out["dep_hour"] = out["dep_time"].astype(str).str.slice(0, 2).astype(int)
    out["dep_period"] = out["dep_hour"].map(_period)
    out["journey_dow"] = out["journey_date"].dt.dayofweek
    out["journey_month"] = out["journey_date"].dt.month
    out["is_weekend"] = out["journey_dow"].isin([4, 5, 6]).astype(int)
    out["is_peak_month"] = out["journey_month"].isin(PEAK_MONTHS.keys()).astype(int)
    out["total_stops"] = out["total_stops"].astype(int)
    return out


def split_xy(df: pd.DataFrame):
    return df[FEATURES], df[TARGET]
