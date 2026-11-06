"""Inference helpers + the fare-alert logic."""
import json
from datetime import date, timedelta
import joblib
import pandas as pd
from .config import MODEL_PATH, ALERTS_PATH, FEATURES, estimate_duration_min
from .preprocess import engineer

_model = None


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def reload_model():
    global _model
    _model = None
    return load_model()


def _frame(airline, source, destination, stops, dep_time, journey_dates, booking_dates):
    n = len(journey_dates)
    dur = estimate_duration_min(source, destination, stops)
    df = pd.DataFrame({
        "airline": [airline] * n, "source": [source] * n, "destination": [destination] * n,
        "total_stops": [stops] * n, "dep_time": [dep_time] * n, "duration_min": [dur] * n,
        "booking_date": pd.to_datetime(list(booking_dates)),
        "journey_date": pd.to_datetime(list(journey_dates)),
    })
    return engineer(df)


def predict_price(airline, source, destination, stops, dep_time, journey_date, booking_date=None):
    booking_date = booking_date or date.today()
    X = _frame(airline, source, destination, stops, dep_time, [journey_date], [booking_date])
    return float(load_model().predict(X[FEATURES])[0])


def booking_curve(airline, source, destination, stops, dep_time, journey_date, today=None):
    """Predicted fare if you booked on each day from today until the day before travel."""
    today = today or date.today()
    n = (journey_date - today).days
    if n < 1:
        raise ValueError("Journey date must be after today.")
    bookings = [today + timedelta(days=i) for i in range(n)]
    X = _frame(airline, source, destination, stops, dep_time, [journey_date] * n, bookings)
    prices = load_model().predict(X[FEATURES])
    return pd.DataFrame({"booking_date": pd.to_datetime(bookings),
                         "days_left": X["days_left"].values,
                         "predicted_fare": prices.round(0)})


def date_scan(airline, source, destination, stops, dep_time, start, end, today=None):
    """Predicted fare for every travel date in [start, end] if booked today."""
    today = today or date.today()
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    days = [d for d in days if d > today]
    if not days:
        raise ValueError("No valid travel dates after today.")
    X = _frame(airline, source, destination, stops, dep_time, days, [today] * len(days))
    prices = load_model().predict(X[FEATURES])
    out = pd.DataFrame({"journey_date": pd.to_datetime(days), "predicted_fare": prices.round(0)})
    out["day"] = out["journey_date"].dt.day_name()
    return out


def fare_alert(curve: pd.DataFrame, target_price: float) -> dict:
    """Turn a booking curve into a BUY_NOW / WAIT / TARGET_LOW recommendation."""
    now = curve.iloc[0]
    best = curve.loc[curve["predicted_fare"].idxmin()]
    below = curve[curve["predicted_fare"] <= target_price]
    week = curve.iloc[: min(8, len(curve))]
    rising = len(week) > 1 and week["predicted_fare"].iloc[-1] > week["predicted_fare"].iloc[0] * 1.03

    res = {"current": float(now["predicted_fare"]), "best_price": float(best["predicted_fare"]),
           "best_date": best["booking_date"].date(), "target": float(target_price)}
    if now["predicted_fare"] <= target_price:
        res.update(status="BUY_NOW", message=f"Today's predicted fare (₹{now['predicted_fare']:,.0f}) is at or below your target of ₹{target_price:,.0f}. Book now!")
    elif not below.empty:
        d = below.iloc[0]
        res.update(status="WAIT", alert_date=d["booking_date"].date(),
                   message=f"Fare is predicted to drop to ₹{d['predicted_fare']:,.0f} on {d['booking_date']:%d %b %Y}. Wait until then to book.")
    elif rising:
        res.update(status="BUY_NOW", message=f"Your target is below every predicted fare (lowest ₹{best['predicted_fare']:,.0f}) and prices are trending up. Booking soon is safer.")
    else:
        res.update(status="TARGET_LOW", message=f"Your target is below the lowest predicted fare of ₹{best['predicted_fare']:,.0f} (around {best['booking_date']:%d %b %Y}). Consider raising it.")
    return res


# ---- simple persistent alert storage ------------------------------------
def load_alerts() -> list:
    if ALERTS_PATH.exists():
        try:
            return json.loads(ALERTS_PATH.read_text())
        except Exception:
            return []
    return []


def save_alert(alert: dict):
    alerts = load_alerts()
    alerts.append(alert)
    ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALERTS_PATH.write_text(json.dumps(alerts, indent=2, default=str))


def delete_alert(i: int):
    alerts = load_alerts()
    if 0 <= i < len(alerts):
        alerts.pop(i)
        ALERTS_PATH.write_text(json.dumps(alerts, indent=2, default=str))
