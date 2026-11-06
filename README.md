# ✈️ Air Ticket Fare Prediction – Fare-Alert Travel Planner

**Roll No:** 23ME1A5413 · **Name:** Duggirala Sri Harsha Venkat · **Guide:** Mr. K. Rajendra
**Course:** Machine Learning & Predictive Analytics

**Title:** Design and Development of a Working Prototype for Air Ticket Fare Prediction Using a Gradient Boosting Regression Model

## Quick start
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
The data + trained model are already included. To rebuild them:
```bash
python -m src.data_generator   # creates data/flight_bookings.csv
python -m src.train            # trains & saves models/gb_fare_model.joblib
```

## Project structure
```
airfare_predictor/
├── app.py                 # Streamlit app (6 pages)
├── requirements.txt
├── .streamlit/config.toml
├── data/flight_bookings.csv   # demo (synthetic) historical bookings
├── models/                    # trained model, metrics, feature importance
└── src/
    ├── config.py          # routes, airlines, feature lists
    ├── data_generator.py  # synthetic booking data
    ├── preprocess.py      # cleaning + feature engineering
    ├── train.py           # Gradient Boosting training + evaluation
    └── predict.py         # inference, booking curve, fare-alert logic
```

## App pages
| Page | What it does |
|---|---|
| Fare Alert Planner | Forecasts fare for each booking day, gives BUY NOW / WAIT advice vs your target price, saves alerts |
| Cheapest Travel Date | Predicted fare for each travel date in a window |
| Data Explorer | EDA charts + cleaning log |
| Model Performance | R², MAE, RMSE, MAPE, vs Linear baseline, feature importance |
| Data & Retrain | Upload a real CSV and retrain |
| About | Project summary |

## Using real data
Map your CSV (e.g. Kaggle *Flight Fare Prediction* / *Indian Airlines Ticket Prices*) to:
`airline, source, destination, total_stops, dep_time, duration_min, booking_date, journey_date, price`
then upload it on the **Data & Retrain** page. Add your cities to `src/config.py → CITIES`.

## Results on the bundled demo data
Gradient Boosting: R² ≈ 0.947, MAE ≈ ₹442 (Linear Regression baseline: R² ≈ 0.738, MAE ≈ ₹945).
