"""
✈️  Air Ticket Fare Prediction & Fare-Alert Travel Planner
Gradient Boosting Regression  |  Streamlit prototype
Run:  streamlit run app.py
"""
import json
from datetime import date, timedelta, time as dtime

import pandas as pd
import streamlit as st

from src.config import (CITIES, AIRLINES, DATA_PATH, MODEL_PATH, METRICS_PATH,
                        IMPORTANCE_PATH, TESTPRED_PATH, estimate_duration_min)
from src import predict as P

st.set_page_config(page_title="Fare Alert Travel Planner", page_icon="✈️", layout="wide")

# ----------------------------------------------------------------- bootstrap
if not DATA_PATH.exists():
    from src.data_generator import generate
    DATA_PATH.parent.mkdir(exist_ok=True)
    generate().to_csv(DATA_PATH, index=False)
if not MODEL_PATH.exists():
    from src.train import train
    with st.spinner("First run: training the Gradient Boosting model…"):
        train(verbose=False)


@st.cache_data
def load_raw():
    return pd.read_csv(DATA_PATH)


def fmt_inr(x):
    return f"₹{x:,.0f}"


# ------------------------------------------------------------------- sidebar
st.sidebar.title("✈️ Fare Alert Planner")
page = st.sidebar.radio("Navigate", ["🔔 Fare Alert Planner", "📅 Cheapest Travel Date",
                                     "📊 Data Explorer", "🧠 Model Performance",
                                     "⚙️ Data & Retrain", "ℹ️ About"])
st.sidebar.caption("Gradient Boosting Regressor · scikit-learn · Streamlit")


def route_inputs(key):
    c1, c2, c3 = st.columns(3)
    src = c1.selectbox("From", list(CITIES), index=3, key=f"src{key}")
    dst_opts = [c for c in CITIES if c != src]
    dst = c2.selectbox("To", dst_opts, index=0, key=f"dst{key}")
    airline = c3.selectbox("Airline", AIRLINES, key=f"air{key}")
    c4, c5 = st.columns(2)
    stops = c4.selectbox("Stops", [0, 1, 2], format_func=lambda s: "Non-stop" if s == 0 else f"{s} stop(s)", key=f"stp{key}")
    dep = c5.time_input("Departure time", dtime(7, 30), key=f"dep{key}")
    return src, dst, airline, stops, dep.strftime("%H:%M")


# ============================================================ FARE ALERT PAGE
if page.startswith("🔔"):
    st.title("🔔 Fare Alert Travel Planner")
    st.write("Tell us your trip and your target price. The model forecasts the fare for **every day you "
             "could book** and tells you whether to **buy now or wait**.")
    src, dst, airline, stops, dep = route_inputs("a")
    c1, c2 = st.columns(2)
    journey = c1.date_input("Travel date", date.today() + timedelta(days=45),
                            min_value=date.today() + timedelta(days=2), key="jd")
    dur = estimate_duration_min(src, dst, stops)
    c2.metric("Estimated flight duration", f"{dur // 60}h {dur % 60:02d}m")

    curve = P.booking_curve(airline, src, dst, stops, dep, journey)
    default_target = int(round(curve["predicted_fare"].min() * 1.05, -2))
    target = st.number_input("🎯 Your target fare (₹)", min_value=1000, max_value=100000,
                             value=default_target, step=100)
    alert = P.fare_alert(curve, target)

    box = {"BUY_NOW": st.success, "WAIT": st.warning, "TARGET_LOW": st.info}[alert["status"]]
    label = {"BUY_NOW": "✅ BUY NOW", "WAIT": "⏳ WAIT", "TARGET_LOW": "ℹ️ TARGET TOO LOW"}[alert["status"]]
    box(f"**{label}** — {alert['message']}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted fare today", fmt_inr(alert["current"]))
    m2.metric("Lowest predicted fare", fmt_inr(alert["best_price"]),
              delta=fmt_inr(alert["best_price"] - alert["current"]), delta_color="inverse")
    m3.metric("Best day to book", alert["best_date"].strftime("%d %b %Y"))
    m4.metric("Your target", fmt_inr(target))

    st.subheader("Predicted fare vs. booking day")
    chart = curve.set_index("booking_date")[["predicted_fare"]].copy()
    chart["your_target"] = target
    st.line_chart(chart)
    st.caption("Each point = predicted price if you book on that day for the selected travel date.")

    with st.expander("See the numbers"):
        show = curve.copy()
        show["booking_date"] = show["booking_date"].dt.strftime("%d %b %Y")
        st.dataframe(show.rename(columns={"booking_date": "Booking date", "days_left": "Days before flight",
                                          "predicted_fare": "Predicted fare (₹)"}),
                     use_container_width=True, hide_index=True)

    if st.button("💾 Save this fare alert"):
        P.save_alert({"airline": airline, "source": src, "destination": dst, "stops": stops,
                      "dep_time": dep, "journey_date": str(journey), "target": target,
                      "created": str(date.today())})
        st.toast("Alert saved!", icon="🔔")

    st.divider()
    st.subheader("📋 My saved alerts (re-evaluated with today's date)")
    alerts = P.load_alerts()
    if not alerts:
        st.caption("No alerts saved yet.")
    for i, a in enumerate(alerts):
        jd = date.fromisoformat(a["journey_date"])
        cols = st.columns([5, 4, 1])
        title = f"**{a['source']} → {a['destination']}** · {a['airline']} · {jd:%d %b %Y} · target {fmt_inr(a['target'])}"
        cols[0].markdown(title)
        if jd <= date.today():
            cols[1].caption("Expired")
        else:
            res = P.fare_alert(P.booking_curve(a["airline"], a["source"], a["destination"],
                                               a["stops"], a["dep_time"], jd), a["target"])
            icon = {"BUY_NOW": "✅", "WAIT": "⏳", "TARGET_LOW": "ℹ️"}[res["status"]]
            cols[1].markdown(f"{icon} {res['status'].replace('_', ' ')} · now {fmt_inr(res['current'])}")
        if cols[2].button("🗑️", key=f"del{i}"):
            P.delete_alert(i)
            st.rerun()

# ======================================================== CHEAPEST DATE PAGE
elif page.startswith("📅"):
    st.title("📅 Cheapest Travel Date Finder")
    st.write("Flexible on dates? See the predicted fare for each travel day if you book **today**.")
    src, dst, airline, stops, dep = route_inputs("b")
    c1, c2 = st.columns(2)
    start = c1.date_input("From date", date.today() + timedelta(days=7), min_value=date.today() + timedelta(days=1))
    end = c2.date_input("To date", date.today() + timedelta(days=37), min_value=start)
    if (end - start).days > 120:
        st.warning("Please choose a window of 120 days or less.")
    else:
        scan = P.date_scan(airline, src, dst, stops, dep, start, end)
        best = scan.loc[scan["predicted_fare"].idxmin()]
        worst = scan.loc[scan["predicted_fare"].idxmax()]
        a, b, c = st.columns(3)
        a.metric("Cheapest day", f"{best['journey_date']:%a, %d %b}", fmt_inr(best["predicted_fare"]))
        b.metric("Costliest day", f"{worst['journey_date']:%a, %d %b}", fmt_inr(worst["predicted_fare"]))
        c.metric("Potential saving", fmt_inr(worst["predicted_fare"] - best["predicted_fare"]))
        st.bar_chart(scan.set_index("journey_date")["predicted_fare"])
        st.dataframe(scan.sort_values("predicted_fare").head(10).assign(
            journey_date=lambda d: d["journey_date"].dt.strftime("%d %b %Y")
        ).rename(columns={"journey_date": "Travel date", "predicted_fare": "Predicted fare (₹)", "day": "Day"}),
            use_container_width=True, hide_index=True)
        st.caption("Top 10 cheapest travel dates in your window.")

# ======================================================= DATA EXPLORER PAGE
elif page.startswith("📊"):
    st.title("📊 Data Explorer")
    from src.preprocess import clean, engineer
    raw = load_raw()
    df, log = clean(raw, report=True)
    df = engineer(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw rows", f"{log['rows_raw']:,}")
    c2.metric("After cleaning", f"{log['after_outliers']:,}")
    c3.metric("Avg fare", fmt_inr(df["price"].mean()))
    c4.metric("Routes", df.groupby(["source", "destination"]).ngroups)

    t1, t2, t3, t4 = st.tabs(["Days before flight", "Airline", "Month / Weekday", "Cleaning log"])
    with t1:
        d = df.groupby("days_left")["price"].mean()
        st.line_chart(d)
        st.caption("Average fare by how many days ahead the ticket was booked — fares climb sharply in the last two weeks.")
    with t2:
        st.bar_chart(df.groupby("airline")["price"].mean())
        st.bar_chart(df.groupby("total_stops")["price"].mean())
    with t3:
        st.bar_chart(df.groupby("journey_month")["price"].mean())
        dow = df.groupby("journey_dow")["price"].mean()
        dow.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        st.bar_chart(dow)
    with t4:
        st.json(log)
        st.dataframe(df.head(50), use_container_width=True)

# ================================================== MODEL PERFORMANCE PAGE
elif page.startswith("🧠"):
    st.title("🧠 Model Performance")
    m = json.loads(METRICS_PATH.read_text())
    gb, lr = m["gradient_boosting"], m["linear_baseline"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² (test)", f"{gb['R2']:.3f}", f"{gb['R2'] - lr['R2']:+.3f} vs linear")
    c2.metric("MAE", fmt_inr(gb["MAE"]), fmt_inr(gb["MAE"] - lr["MAE"]), delta_color="inverse")
    c3.metric("RMSE", fmt_inr(gb["RMSE"]), fmt_inr(gb["RMSE"] - lr["RMSE"]), delta_color="inverse")
    c4.metric("MAPE", f"{gb['MAPE_pct']:.1f}%")
    st.caption(f"3-fold CV R² = {m['cv_r2_mean']:.3f} ± {m['cv_r2_std']:.3f} · "
               f"train rows {m['n_train']:,} · test rows {m['n_test']:,}")
    st.subheader("Gradient Boosting vs Linear Regression baseline")
    st.dataframe(pd.DataFrame({"Gradient Boosting": gb, "Linear Regression": lr}).round(3), use_container_width=True)
    st.subheader("Actual vs predicted (test sample)")
    st.scatter_chart(pd.read_csv(TESTPRED_PATH), x="actual", y="predicted")
    st.subheader("Feature importance")
    imp = pd.read_csv(IMPORTANCE_PATH).head(15).set_index("feature")
    st.bar_chart(imp)

# ====================================================== DATA & RETRAIN PAGE
elif page.startswith("⚙️"):
    st.title("⚙️ Data & Retrain")
    st.write("Replace the demo data with **real historical booking data** and retrain the model.")
    st.code("airline, source, destination, total_stops, dep_time (HH:MM), duration_min,\n"
            "booking_date, journey_date, price", language="text")
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        new = pd.read_csv(up)
        st.dataframe(new.head(), use_container_width=True)
        if st.button("Use this data & retrain", type="primary"):
            try:
                from src.train import train
                with st.spinner("Training…"):
                    res = train(new, verbose=False)
                new.to_csv(DATA_PATH, index=False)
                P.reload_model()
                load_raw.clear()
                st.success(f"Retrained. R² = {res['gradient_boosting']['R2']:.3f}, "
                           f"MAE = {fmt_inr(res['gradient_boosting']['MAE'])}")
            except Exception as e:
                st.error(f"Could not train: {e}")
    st.divider()
    if st.button("Regenerate demo dataset & retrain"):
        from src.data_generator import generate
        from src.train import train
        with st.spinner("Working…"):
            d = generate()
            d.to_csv(DATA_PATH, index=False)
            train(d, verbose=False)
            P.reload_model()
            load_raw.clear()
        st.success("Done.")

# ============================================================== ABOUT PAGE
else:
    st.title("ℹ️ About this project")
    st.markdown("""
**Design and Development of a Working Prototype for Air Ticket Fare Prediction Using a Gradient Boosting Regression Model**

*Course:* Machine Learning & Predictive Analytics

**Pipeline**
1. **Data collection** – historical airline booking records (airline, route, stops, departure time, booking date, journey date, fare).
2. **Cleaning** – drop duplicates / nulls / invalid dates / per-route price outliers.
3. **Transformation** – days-before-departure, day-of-week, month, peak-season flag, departure period, one-hot encoding.
4. **Model** – `GradientBoostingRegressor` inside a scikit-learn `Pipeline`, compared against a Linear Regression baseline.
5. **Deployment** – this Streamlit **fare-alert travel planner**: forecast fare for every booking day → *Buy now / Wait* advice against your target price.

**Disclaimer:** the bundled dataset is *synthetic* (generated to mimic real fare behaviour) so the prototype runs out-of-the-box.
Upload real data on the *Data & Retrain* page for real predictions.
""")
