
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# =====================================================
# 1. CONFIGURATION
# =====================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_PATH = (
    PROJECT_ROOT / "2. Data"
    / "4. Feature Engineering" / "sales.csv"
)

VALIDATION_START = pd.Timestamp("2024-10-03")
VALIDATION_END = pd.Timestamp("2024-12-31")
HORIZON = 30


# =====================================================
# 2. PREPARE DAILY SALES
# =====================================================

def load_daily():

    df = pd.read_csv(
        DATA_PATH,
        usecols=["order_date", "order_status", "net_sales"]
    )

    df["order_date"] = pd.to_datetime(df["order_date"])
    df["net_sales"] = pd.to_numeric(df["net_sales"])

    if df[["order_date", "order_status", "net_sales"]].isna().any().any():
        raise ValueError("Missing critical sales values.")

    calendar = pd.date_range(
        df["order_date"].min(),
        df["order_date"].max(),
        freq="D"
    )

    # Missing entire transaction days are not automatically zero.
    observed_days = pd.DatetimeIndex(
        df["order_date"].dt.normalize().unique()
    )

    if len(calendar.difference(observed_days)) > 0:
        raise ValueError("Missing transaction dates detected.")

    completed = df[df["order_status"] == "Completed"].copy()

    daily = (
        completed.groupby("order_date")["net_sales"]
        .sum()
        .reindex(calendar, fill_value=0)
        .astype(float)
    )

    daily.index.name = "date"
    daily.name = "sales"

    return daily


# =====================================================
# 3. TIME-SERIES FEATURES
# =====================================================

def make_features(series):

    x = pd.DataFrame(index=series.index)
    past = series.shift(1)

    for lag in [1, 7, 14, 28, 56]:
        x[f"lag_{lag}"] = series.shift(lag)

    for window in [7, 14, 28, 56]:
        x[f"mean_{window}"] = past.rolling(window).mean()

    for window in [7, 28]:
        x[f"std_{window}"] = (
            past.rolling(window).std(ddof=0)
        )

    dates = x.index

    x["day_of_week"] = dates.dayofweek
    x["day_of_month"] = dates.day
    x["month"] = dates.month
    x["week_of_year"] = (
        dates.isocalendar().week.astype(int).to_numpy()
    )

    x["sin_year"] = np.sin(
        2 * np.pi * dates.dayofyear / 365.25
    )
    x["cos_year"] = np.cos(
        2 * np.pi * dates.dayofyear / 365.25
    )

    return x


# =====================================================
# 4. CANDIDATE MODELS
# =====================================================

def candidates():

    return {
        "Seasonal Naive": None,

        "XGBoost A": XGBRegressor(
            n_estimators=350,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost B": XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        ),

        "LightGBM A": LGBMRegressor(
            n_estimators=350,
            max_depth=5,
            num_leaves=15,
            min_child_samples=25,
            learning_rate=0.04,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=8,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        ),

        "LightGBM B": LGBMRegressor(
            n_estimators=500,
            max_depth=7,
            num_leaves=25,
            min_child_samples=25,
            learning_rate=0.03,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=8,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )
    }


# =====================================================
# 5. RECURSIVE 30-DAY FORECAST
# =====================================================

def forecast_window(model, history, start, days=30):

    history = history.copy()
    predictions = []

    forecast_dates = pd.date_range(
        start, periods=days, freq="D"
    )

    for date in forecast_dates:

        if model is None:
            # Weekly seasonal naive baseline.
            predicted = history.loc[
                date - pd.Timedelta(days=7)
            ]

        else:
            # Append unknown future target.
            temp = pd.concat([
                history,
                pd.Series([np.nan], index=[date])
            ])

            # Features use past sales only.
            next_features = make_features(temp).iloc[[-1]]

            predicted = model.predict(next_features)[0]

        predicted = max(0.0, float(predicted))

        predictions.append(predicted)
        history.loc[date] = predicted

    return pd.Series(
        predictions,
        index=forecast_dates,
        name="predicted_sales"
    )


# =====================================================
# 6. ROLLING-ORIGIN VALIDATION
# =====================================================

def backtest(model, daily, start, end, horizon=30):

    all_predictions = []
    block_start = start

    while block_start <= end:

        days = min(
            horizon,
            (end - block_start).days + 1
        )

        # Only observations available before this block.
        history = daily.loc[
            :block_start - pd.Timedelta(days=1)
        ]

        forecast = forecast_window(
            model, history, block_start, days
        )

        all_predictions.append(forecast)

        block_start += pd.Timedelta(days=days)

    predictions = pd.concat(all_predictions)
    actual = daily.loc[predictions.index]

    mae = mean_absolute_error(actual, predictions)

    rmse = np.sqrt(
        mean_squared_error(actual, predictions)
    )

    wape = (
        np.abs(actual - predictions).sum()
        / np.abs(actual).sum()
        * 100
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_pct": wape
    }


# =====================================================
# 7. TRAINING AND MODEL COMPARISON
# =====================================================

def main():

    daily = load_daily()

    # Do not use 2025 during model selection.
    development = daily.loc[:"2024-12-31"]

    train = development.loc[
        :VALIDATION_START - pd.Timedelta(days=1)
    ]

    x_train = make_features(train).dropna()
    y_train = train.loc[x_train.index]

    print("\n" + "=" * 65)
    print("SALES FORECASTING - DATA SUMMARY")
    print("=" * 65)

    print("Target       : Completed-order daily net sales")
    print("Training     :", train.index.min().date(),
          "to", train.index.max().date())
    print("Validation   :", VALIDATION_START.date(),
          "to", VALIDATION_END.date())
    print("Train days   :", len(train))
    print("Train samples:", len(x_train))
    print("Forecast     : 30 days per window")
    print("2025 Test    : Reserved")

    results = []

    for name, model in candidates().items():

        print(f"\nEvaluating {name}...")

        if model is not None:
            model.fit(x_train, y_train)

        metrics = backtest(
            model,
            development,
            VALIDATION_START,
            VALIDATION_END,
            HORIZON
        )

        results.append({
            "Model": name,
            **metrics
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        ["WAPE_pct", "MAE", "RMSE"]
    ).reset_index(drop=True)

    print("\n" + "=" * 65)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 65)

    print(
        results_df.round(2).to_string(index=False)
    )

    best_name = results_df.iloc[0]["Model"]

    print("\n" + "=" * 65)
    print("SELECTED MODEL")
    print("=" * 65)

    print("Model:", best_name)
    print(
        "Validation WAPE:",
        f"{results_df.iloc[0]['WAPE_pct']:.2f}%"
    )

    selected = candidates()[best_name]

    if selected is not None:
        print("\nSelected Parameters:")
        print(selected.get_params())

    print("\nTraining and validation completed.")


if __name__ == "__main__":
    main()
