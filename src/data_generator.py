"""
Generates a realistic *historical airline booking* dataset.

Use this when you do not have a real dataset (e.g. the Kaggle "Flight Fare
Prediction" / "Indian Airlines Ticket Prices" CSV). Schema produced:

airline, source, destination, total_stops, dep_time, duration_min,
booking_date, journey_date, price

If you have a real dataset, map its columns to this schema and replace
data/flight_bookings.csv (or upload it from the app's "Data & Retrain" page).
"""
import numpy as np
import pandas as pd
from itertools import permutations
from .config import (CITIES, AIRLINES, AIRLINE_FACTOR, PEAK_MONTHS, DATA_PATH,
                     haversine_km, estimate_duration_min)


def _days_left_multiplier(d):
    """Fare rises steeply in the last 2 weeks, is cheapest ~3-8 weeks out."""
    d = np.asarray(d, dtype=float)
    mult = np.where(d <= 3, 1.85,
           np.where(d <= 7, 1.55,
           np.where(d <= 14, 1.30,
           np.where(d <= 21, 1.12,
           np.where(d <= 45, 1.00, 1.07)))))
    return mult + 0.02 * np.sin(d / 3.0)


def _hour_probs():
    p = np.array([1, 1, 1, 1, 2, 3, 6, 7, 7, 6, 5, 4, 4, 4, 4, 4, 5, 6, 7, 7, 6, 4, 3, 2], float)
    return p / p.sum()


def generate(n_rows: int = 40000, seed: int = 42, start="2026-11-01", end="2027-03-31"):
    rng = np.random.default_rng(seed)
    routes = list(permutations(CITIES.keys(), 2))
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    span = (end - start).days
    hp = _hour_probs()

    rows = []
    for _ in range(n_rows):
        src, dst = routes[rng.integers(len(routes))]
        airline = AIRLINES[rng.choice(len(AIRLINES), p=[.38, .22, .12, .12, .16])]
        stops = int(rng.choice([0, 1, 2], p=[.72, .24, .04]))
        dep_hour = int(rng.choice(24, p=hp))
        dep_min = int(rng.choice([0, 15, 30, 45]))
        journey = start + pd.Timedelta(days=int(rng.integers(0, span)))
        days_left = int(rng.integers(1, 90))
        booking = journey - pd.Timedelta(days=days_left)
        duration = estimate_duration_min(src, dst, stops) + int(rng.integers(-10, 25))

        km = haversine_km(src, dst)
        price = (2300 + km * 3.1) * AIRLINE_FACTOR[airline]
        price *= {0: 1.0, 1: 0.88, 2: 0.82}[stops]
        price *= float(_days_left_multiplier(days_left))
        price *= PEAK_MONTHS.get(journey.month, 1.0)
        if journey.dayofweek in (4, 6):          # Fri / Sun
            price *= 1.10
        elif journey.dayofweek == 1:             # Tue
            price *= 0.94
        if 6 <= dep_hour <= 9 or 17 <= dep_hour <= 21:
            price *= 1.08
        elif dep_hour <= 4:
            price *= 0.92
        price *= rng.lognormal(0, 0.07)
        rows.append((airline, src, dst, stops, f"{dep_hour:02d}:{dep_min:02d}",
                     duration, booking.date(), journey.date(), round(price, 0)))

    df = pd.DataFrame(rows, columns=["airline", "source", "destination", "total_stops",
                                     "dep_time", "duration_min", "booking_date",
                                     "journey_date", "price"])
    # add real-world mess so the cleaning step has something to do
    idx = rng.choice(len(df), size=int(0.01 * len(df)), replace=False)
    third = len(idx) // 3
    df.loc[idx[:third], "price"] = np.nan
    df.loc[idx[third:2 * third], "airline"] = None
    df = pd.concat([df, df.sample(120, random_state=1)], ignore_index=True)  # duplicates
    return df


if __name__ == "__main__":
    df = generate()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df):,} rows -> {DATA_PATH}")
