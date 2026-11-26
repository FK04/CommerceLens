from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from lightgbm import LGBMClassifier


# =========================================================
# 1. PATHS
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "2. Data"
    / "4. Feature Engineering"
    / "sales.csv"
)

MODEL_PATH = SCRIPT_DIR / "r_model.pkl"

PREDICTION_PATH = (
    PROJECT_ROOT
    / "3. Predictive Modeling (ML)"
    / "4. Prediction"
    / "return_pred.csv"
)


# =========================================================
# 2. LOAD DATA
# =========================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"sales.csv not found at:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

df["order_date"] = pd.to_datetime(
    df["order_date"]
)

print("\nDataset shape:", df.shape)


# =========================================================
# 3. TARGET + FEATURES
# =========================================================

target = "is_returned"

numeric_features = [
    "customer_age",
    "quantity",
    "gross_sales",
    "discount_amount",
    "tax_amount",
    "shipping_cost",
    "order_year",
    "order_quarter",
    "order_month",
    "order_day",
    "is_weekend",
    "effective_discount_rate"
]

categorical_features = [
    "gender",
    "customer_segment",
    "customer_state",
    "customer_country",
    "region",
    "sales_channel",
    "payment_method",
    "shipping_method",
    "warehouse",
    "marketing_channel",
    "campaign_name",
    "coupon_code",
    "order_day_name"
]

features = numeric_features + categorical_features


# =========================================================
# 4. FINAL TRAIN / TEST SPLIT
# =========================================================

# All 2021-2024 data is now used for final training.
train_final = df[
    df["order_year"] < 2025
].copy()

# 2025 remains the untouched final test set.
test_final = df[
    df["order_year"] == 2025
].copy()

X_train = train_final[features]
y_train = train_final[target]

X_test = test_final[features]
y_test = test_final[target]

print("\nFinal Evaluation Split")
print("-" * 40)
print("Training (2021-2024):", len(train_final))
print("Testing  (2025)     :", len(test_final))


# =========================================================
# 5. CLASS IMBALANCE
# =========================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive


# =========================================================
# 6. PREPROCESSING
# =========================================================

numeric_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])

preprocessor = ColumnTransformer([
    (
        "num",
        numeric_transformer,
        numeric_features
    ),
    (
        "cat",
        categorical_transformer,
        categorical_features
    )
])


# =========================================================
# 7. FINAL LIGHTGBM MODEL
# =========================================================

# Frozen configuration selected in r_train.py.

final_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.10,
    max_depth=7,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbosity=-1,
    n_jobs=-1
)

final_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", final_model)
])


# =========================================================
# 8. TRAIN FINAL MODEL
# =========================================================

print("\nTraining final LightGBM model...")

final_pipeline.fit(
    X_train,
    y_train
)


# =========================================================
# 9. FINAL 2025 EVALUATION
# =========================================================

FINAL_THRESHOLD = 0.31

probability = final_pipeline.predict_proba(
    X_test
)[:, 1]

prediction = (
    probability >= FINAL_THRESHOLD
).astype(int)

accuracy = accuracy_score(y_test, prediction)

precision = precision_score(
    y_test,
    prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    prediction,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probability
)

pr_auc = average_precision_score(
    y_test,
    probability
)


# =========================================================
# 10. CONFUSION MATRIX
# =========================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    prediction
).ravel()


# =========================================================
# 11. DISPLAY FINAL RESULTS
# =========================================================

print("\n" + "=" * 60)
print("FINAL 2025 TEST RESULTS")
print("=" * 60)

print(f"Model      : LightGBM")
print(f"Threshold  : {FINAL_THRESHOLD:.2f}")

print("\nPerformance")
print("-" * 40)

print(f"Accuracy   : {accuracy:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"ROC-AUC    : {roc_auc:.4f}")
print(f"PR-AUC     : {pr_auc:.4f}")

print("\nConfusion Matrix")
print("-" * 40)

print(f"True Negative  : {tn}")
print(f"False Positive : {fp}")
print(f"False Negative : {fn}")
print(f"True Positive  : {tp}")


# =========================================================
# 12. SAVE FINAL MODEL
# =========================================================

joblib.dump(
    final_pipeline,
    MODEL_PATH
)

print(
    "\nFinal model saved:",
    MODEL_PATH
)


# =========================================================
# 13. SAVE 2025 PREDICTIONS
# =========================================================

PREDICTION_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

prediction_output = pd.DataFrame({
    "order_id": test_final["order_id"].values,
    "actual_return": y_test.values,
    "return_probability": probability,
    "predicted_return": prediction
})

prediction_output.to_csv(
    PREDICTION_PATH,
    index=False
)

print(
    "Predictions saved:",
    PREDICTION_PATH
)

print("\nFinal evaluation completed successfully.")