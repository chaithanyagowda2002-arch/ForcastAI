"""Retail demand forecasting: seasonal-naive baseline vs gradient boosting.

Data: Walmart Recruiting - Store Sales Forecasting (public Kaggle dataset,
45 stores, weekly sales per department, Feb 2010 - Oct 2012). This script
uses a cleaned/merged copy hosted on GitHub and downloads it on first run.

Validation is TIME-BASED: we train on the past and test on the final 26
weeks. Random splits would leak the future into training.
"""
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

HERE = Path(__file__).parent
DATA = HERE / "data" / "walmart_clean.csv"
URL = ("https://raw.githubusercontent.com/ezgigm/Project4_Store_Sales_Forecasting/"
       "master/clean_data.csv")
VAL_WEEKS = 26

FEATURES = ["Store", "Dept", "week", "month", "IsHoliday", "Size", "Type_code",
            "Temperature", "Fuel_Price", "CPI", "Unemployment", "lag_52", "lag_avg"]


def load() -> pd.DataFrame:
    if not DATA.exists():
        DATA.parent.mkdir(exist_ok=True)
        print("Downloading dataset (~60 MB, first run only)...")
        urllib.request.urlretrieve(URL, DATA)
    cols = ["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday", "Temperature",
            "Fuel_Price", "CPI", "Unemployment", "Type", "Size", "week", "month"]
    df = pd.read_csv(DATA, usecols=cols, parse_dates=["Date"])
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    df["Type_code"] = df["Type"].map({"A": 0, "B": 1, "C": 2})
    return df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Sales from the same week last year (52 weeks back).

    Safe for a 26-week horizon: when we forecast a week, the value from 52
    weeks earlier is already known. We also average weeks 51-53 because
    holidays shift by a few days between years.
    """
    key = ["Store", "Dept", "Date"]
    base = df[key + ["Weekly_Sales"]]
    shifted = []
    for k in (51, 52, 53):
        s = base.copy()
        s["Date"] = s["Date"] + pd.Timedelta(weeks=k)
        shifted.append(s.rename(columns={"Weekly_Sales": f"lag_{k}"}))
    for s in shifted:
        df = df.merge(s, on=key, how="left")
    df["lag_avg"] = df[["lag_51", "lag_52", "lag_53"]].mean(axis=1)
    return df.drop(columns=["lag_51", "lag_53"])


def wmae(y, pred, holiday) -> float:
    """Kaggle's metric: holiday weeks count 5x because they matter most."""
    w = np.where(holiday == 1, 5, 1)
    return float(np.sum(w * np.abs(y - pred)) / np.sum(w))


def chain_error_pct(val: pd.DataFrame, pred: np.ndarray) -> float:
    """Business view: total chain sales per week, forecast vs actual."""
    t = val[["Date", "Weekly_Sales"]].assign(pred=pred).groupby("Date").sum()
    return float((np.abs(t["pred"] - t["Weekly_Sales"]) / t["Weekly_Sales"]).mean() * 100)


def main():
    df = add_lags(load())
    cutoff = df["Date"].max() - pd.Timedelta(weeks=VAL_WEEKS - 1)
    train, val = df[df["Date"] < cutoff], df[df["Date"] >= cutoff]
    val = val[val["lag_52"].notna()].reset_index(drop=True)  # rows every method can score
    print(f"Train: {len(train):,} rows up to {train['Date'].max().date()}")
    print(f"Validation: {len(val):,} rows, {val['Date'].min().date()} to {val['Date'].max().date()}\n")

    # Baseline 1: same week last year
    base_naive = val["lag_52"].to_numpy()
    # Baseline 2: average of each store-dept's history in the training period
    hist_mean = train.groupby(["Store", "Dept"])["Weekly_Sales"].mean().rename("hist_mean")
    base_mean = val.join(hist_mean, on=["Store", "Dept"])["hist_mean"].fillna(train["Weekly_Sales"].mean()).to_numpy()

    # Model: gradient boosting (handles missing lags natively)
    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
        categorical_features=["Store", "Dept"], random_state=42)
    model.fit(train[FEATURES], train["Weekly_Sales"])
    pred = model.predict(val[FEATURES])

    y, h = val["Weekly_Sales"].to_numpy(), val["IsHoliday"].to_numpy()
    print(f"{'Method':<28}{'WMAE ($)':>10}{'MAE ($)':>10}{'Chain error %':>15}")
    for name, p in [("Store-dept average", base_mean),
                    ("Same week last year", base_naive),
                    ("Gradient boosting", pred)]:
        print(f"{name:<28}{wmae(y, p, h):>10,.0f}{np.mean(np.abs(y - p)):>10,.0f}{chain_error_pct(val, p):>15.1f}")

    # Which features matter? (shuffle one feature, see how much error grows)
    sample = val.sample(min(15000, len(val)), random_state=0)
    imp = permutation_importance(model, sample[FEATURES], sample["Weekly_Sales"], n_repeats=3,
                                 random_state=0, scoring="neg_mean_absolute_error")
    print("\nFeature importance (increase in MAE when shuffled):")
    for f, v in sorted(zip(FEATURES, imp.importances_mean), key=lambda x: -x[1])[:6]:
        print(f"  {f:<14}{v:>10,.0f}")


if __name__ == "__main__":
    main()
