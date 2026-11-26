
from pathlib import Path
from sklearn.metrics import r2_score
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from s_train import load_daily, make_features, forecast_window


# =====================================================
# 1. PATHS AND CONFIGURATION
# =====================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

MODEL_PATH = SCRIPT_DIR / "s_model.pkl"

PREDICTION_PATH = (
    PROJECT_ROOT / "3. Predictive Modeling (ML)"
    / "4. Prediction" / "sales_pred.csv"
)

TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2025-12-31")
HORIZON = 30


def build_model():
    return XGBRegressor(
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
    )


# =====================================================
# 2. LOAD AND SPLIT DATA
# =====================================================

daily = load_daily()

development = daily.loc[:"2024-12-31"]
test = daily.loc[TEST_START:TEST_END]

if len(test) != 365:
    raise ValueError("Expected 365 test days in 2025.")

X_train = make_features(development).dropna()
y_train = development.loc[X_train.index]

print("\n" + "=" * 60)
print("FINAL SALES FORECASTING EVALUATION")
print("=" * 60)

print("Model      : XGBoost A")
print("Training   : 2021-2024")
print("Testing    : 2025")
print("Horizon    : 30 days")


# =====================================================
# 3. TRAIN FROZEN MODEL
# =====================================================

test_model = build_model()
test_model.fit(X_train, y_train)


# =====================================================
# 4. 2025 ROLLING 30-DAY BACKTEST
# =====================================================

records = []
block_start = TEST_START

while block_start <= TEST_END:

    days = min(
        HORIZON,
        (TEST_END - block_start).days + 1
    )

    # Earlier actual observations are available.
    # Actual sales inside this forecast block are not used.
    history = daily.loc[
        :block_start - pd.Timedelta(days=1)
    ]

    predictions = forecast_window(
        test_model,
        history,
        block_start,
        days
    )

    origin = block_start - pd.Timedelta(days=1)

    for horizon_day, (date, predicted) in enumerate(
        predictions.items(), start=1
    ):
        actual = float(test.loc[date])

        records.append({
            "date": date,
            "record_type": "2025_TEST",
            "forecast_origin": origin,
            "horizon_day": horizon_day,
            "actual_sales": actual,
            "predicted_sales": predicted,
            "absolute_error": abs(actual - predicted)
        })

    block_start += pd.Timedelta(days=days)

test_predictions = pd.DataFrame(records)

actual = test_predictions["actual_sales"]
predicted = test_predictions["predicted_sales"]

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
wape = 100 * np.abs(actual - predicted).sum() / actual.sum()
r2 = r2_score(actual, predicted)
bias = (predicted - actual).mean()


# =====================================================
# 5. DISPLAY FINAL TEST RESULTS
# =====================================================

print("\n" + "=" * 60)
print("FINAL 2025 TEST RESULTS")
print("=" * 60)

print(f"MAE       : {mae:.2f}")
print(f"RMSE      : {rmse:.2f}")
print(f"WAPE      : {wape:.2f}%")
print(f"R2 Score  : {r2:.4f}")
print(f"Bias (ME) : {bias:.2f}")
print(f"Test Days : {len(test_predictions)}")

# =====================================================
# 6. REFIT ON ALL AVAILABLE DATA
# =====================================================

print("\nTraining deployment model on 2021-2025...")

X_all = make_features(daily).dropna()
y_all = daily.loc[X_all.index]

final_model = build_model()
final_model.fit(X_all, y_all)


# =====================================================
# 7. PROJECT 30 DAYS AFTER DATA CUTOFF
# =====================================================

future_start = daily.index.max() + pd.Timedelta(days=1)

future = forecast_window(
    final_model,
    daily,
    future_start,
    HORIZON
)

future_records = []

for horizon_day, (date, value) in enumerate(
    future.items(), start=1
):
    future_records.append({
        "date": date,
        "record_type": "POST_CUTOFF_PROJECTION",
        "forecast_origin": daily.index.max(),
        "horizon_day": horizon_day,
        "actual_sales": np.nan,
        "predicted_sales": value,
        "absolute_error": np.nan
    })

future_predictions = pd.DataFrame(future_records)


# =====================================================
# 8. SAVE FINAL MODEL
# =====================================================

model_package = {
    "model": final_model,
    "model_name": "XGBoost A",
    "feature_columns": X_all.columns.tolist(),
    "recent_history": daily.tail(56),
    "trained_through": str(daily.index.max().date()),
    "forecast_horizon": HORIZON,
    "target": "completed_order_daily_net_sales"
}

joblib.dump(model_package, MODEL_PATH)


# =====================================================
# 9. SAVE PREDICTIONS
# =====================================================

PREDICTION_PATH.parent.mkdir(
    parents=True, exist_ok=True
)

output = pd.concat(
    [test_predictions, future_predictions],
    ignore_index=True
)

output.to_csv(PREDICTION_PATH, index=False)

print("\nFiles Saved")
print("-" * 60)
print("Model       :", MODEL_PATH)
print("Predictions :", PREDICTION_PATH)

print("\nProjection Period:")
print(future.index.min().date(), "to", future.index.max().date())

print("\nEvaluation completed successfully.")
