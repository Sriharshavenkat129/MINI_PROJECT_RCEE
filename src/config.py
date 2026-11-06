"""Central configuration: paths, route knowledge, feature lists."""
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "flight_bookings.csv"
MODEL_PATH = ROOT / "models" / "gb_fare_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
IMPORTANCE_PATH = ROOT / "models" / "feature_importance.csv"
TESTPRED_PATH = ROOT / "models" / "test_predictions.csv"
ALERTS_PATH = ROOT / "data" / "alerts.json"

# City -> (lat, lon)
CITIES = {
    "Delhi": (28.61, 77.21),
    "Mumbai": (19.08, 72.88),
    "Bangalore": (12.97, 77.59),
    "Hyderabad": (17.39, 78.49),
    "Chennai": (13.08, 80.27),
    "Kolkata": (22.57, 88.36),
    "Vijayawada": (16.51, 80.65),
}

AIRLINES = ["IndiGo", "Air India", "SpiceJet", "Akasa Air", "Air India Express"]

# Relative price level of each airline (used by the synthetic data generator)
AIRLINE_FACTOR = {
    "IndiGo": 1.00,
    "Air India": 1.22,
    "SpiceJet": 0.92,
    "Akasa Air": 0.95,
    "Air India Express": 0.90,
}

# Festival / peak-season months (Oct-Nov Diwali, Dec holidays, Apr-May summer)
PEAK_MONTHS = {4: 1.10, 5: 1.15, 10: 1.12, 11: 1.10, 12: 1.25, 1: 1.08}

CATEGORICAL = ["airline", "source", "destination", "dep_period"]
NUMERIC = [
    "total_stops", "duration_min", "days_left", "dep_hour",
    "journey_dow", "journey_month", "is_weekend", "is_peak_month",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "price"


def haversine_km(a: str, b: str) -> float:
    (la1, lo1), (la2, lo2) = CITIES[a], CITIES[b]
    p1, p2 = math.radians(la1), math.radians(la2)
    dphi, dl = p2 - p1, math.radians(lo2 - lo1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def estimate_duration_min(source: str, destination: str, stops: int) -> int:
    """Rough flight duration: cruise at ~750 km/h + 45 min ground/climb + 110 min per stop."""
    km = haversine_km(source, destination)
    return int(km / 750 * 60 + 45 + 110 * stops)
