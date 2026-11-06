"""Train + evaluate the Gradient Boosting regression model.

Run:  python -m src.train
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import (DATA_PATH, MODEL_PATH, METRICS_PATH, IMPORTANCE_PATH,
                     TESTPRED_PATH, CATEGORICAL, NUMERIC)
from .preprocess import clean, engineer, split_xy


def build_pipeline(**gb_params) -> Pipeline:
    params = dict(n_estimators=400, learning_rate=0.08, max_depth=4,
                  subsample=0.85, min_samples_leaf=10, random_state=42)
    params.update(gb_params)
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    return Pipeline([("prep", pre), ("gb", GradientBoostingRegressor(**params))])


def _metrics(y, p):
    y = np.asarray(y)
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
        "MAPE_pct": float(np.mean(np.abs((y - p) / y)) * 100),
    }


def train(df_raw=None, verbose: bool = True) -> dict:
    if df_raw is None:
        df_raw = pd.read_csv(DATA_PATH)
    df, log = clean(df_raw, report=True)
    df = engineer(df)
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baseline (Linear Regression) for comparison
    base = build_pipeline()
    base.steps[-1] = ("gb", LinearRegression())
    base.fit(X_tr, y_tr)
    base_m = _metrics(y_te, base.predict(X_te))

    model = build_pipeline()
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    gb_m = _metrics(y_te, pred)
    cv = cross_val_score(build_pipeline(n_estimators=150), X_tr, y_tr, cv=3, scoring="r2")

    names = model.named_steps["prep"].get_feature_names_out()
    imp = pd.DataFrame({"feature": names, "importance": model.named_steps["gb"].feature_importances_})
    imp["feature"] = imp["feature"].str.replace(r"^(cat|num)__", "", regex=True)
    imp = imp.sort_values("importance", ascending=False)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    imp.to_csv(IMPORTANCE_PATH, index=False)
    pd.DataFrame({"actual": y_te.values, "predicted": pred}).sample(
        min(1500, len(y_te)), random_state=1).to_csv(TESTPRED_PATH, index=False)

    out = {"cleaning": log, "gradient_boosting": gb_m, "linear_baseline": base_m,
           "cv_r2_mean": float(cv.mean()), "cv_r2_std": float(cv.std()),
           "n_train": int(len(X_tr)), "n_test": int(len(X_te))}
    METRICS_PATH.write_text(json.dumps(out, indent=2))
    if verbose:
        print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    train()
