from pathlib import Path
import pandas as pd
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# =========================================================
# 1. LOAD DATA
# =========================================================

# r_train.py is inside:
# Project/3. Machine Learning/1. Return Prediction/r_train.py

SCRIPT_DIR = Path(__file__).resolve().parent

# Go up:
# 1. Return Prediction -> 3. Machine Learning -> Project
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "2. Data"
    / "4. Feature Engineering"
    / "sales.csv"
)

# print("Project root :", PROJECT_ROOT)
# print("Reading      :", DATA_PATH)
# print("File exists  :", DATA_PATH.exists())

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"sales.csv not found at:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)


# =========================================================
# 2. TARGET
# =========================================================

target = "is_returned"

print("\nTarget distribution:\n")
print(df[target].value_counts())

print(
    f"Return rate: {df[target].mean() * 100:.2f}%"
)


# =========================================================
# 3. FEATURES
# =========================================================

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
# 4. TIME-BASED SPLIT
# =========================================================

# Keep 2025 completely separate for later final evaluation.
development = df[df["order_year"] < 2025].copy()
test_2025 = df[df["order_year"] == 2025].copy()

# Sort development data chronologically.
development["order_date"] = pd.to_datetime(
    development["order_date"]
)

development = development.sort_values("order_date")

# Last 20% of 2021-2024 = validation data.
split = int(len(development) * 0.80)

train = development.iloc[:split]
validation = development.iloc[split:]

X_train = train[features]
y_train = train[target]

X_val = validation[features]
y_val = validation[target]


print("\nData Split")
print("-" * 40)
print("Train      :", len(train))
print("Validation :", len(validation))
print("2025 Test  :", len(test_2025))

print("\nReturn Rates")
print("-" * 40)
print(f"Train      : {y_train.mean()*100:.2f}%")
print(f"Validation : {y_val.mean()*100:.2f}%")
print(
    f"2025 Test  : "
    f"{test_2025[target].mean()*100:.2f}%"
)


# =========================================================
# 5. CLASS IMBALANCE
# =========================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print(
    "\nscale_pos_weight:",
    round(scale_pos_weight, 2)
)


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
# 7. MODELS
# =========================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ),

    "LightGBM":
        LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=25,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )
}


# =========================================================
# 8. TRAIN + PREDICT + COMPARE
# =========================================================

results = []
trained_models = {}
validation_probabilities = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    probability = pipeline.predict_proba(X_val)[:, 1]
    prediction = (probability >= 0.50).astype(int)

    # Keep fitted model and probabilities for later steps
    trained_models[name] = pipeline
    validation_probabilities[name] = probability

    results.append({
        "Model": name,
        "Precision": precision_score(
            y_val, prediction, zero_division=0
        ),
        "Recall": recall_score(
            y_val, prediction, zero_division=0
        ),
        "F1": f1_score(
            y_val, prediction, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_val, probability
        ),
        "PR_AUC": average_precision_score(
            y_val, probability
        )
    })


# =========================================================
# 9. INITIAL MODEL COMPARISON
# =========================================================

results = pd.DataFrame(results)

print("\n" + "=" * 75)
print("INITIAL MODEL COMPARISON")
print("=" * 75)

print(
    results.round(4).to_string(index=False)
)

candidates = results[
    results["Model"].isin(
        ["XGBoost", "LightGBM"]
    )
].copy()

# print("\nBaseline:")
# print(
#     results[
#         results["Model"] == "Logistic Regression"
#     ].round(4).to_string(index=False)
# )

# print("\nAdvanced Models:")
# print(
#     candidates.round(4).to_string(index=False)
# )

# =========================================================
# 10. THRESHOLD ANALYSIS
# =========================================================

thresholds = [
    0.50, 0.55, 0.60, 0.65, 0.70,
    0.75, 0.80, 0.85, 0.90
]

threshold_results = []

for name in ["XGBoost", "LightGBM"]:

    probability = validation_probabilities[name]

    for threshold in thresholds:

        prediction = (
            probability >= threshold
        ).astype(int)

        threshold_results.append({
            "Model": name,
            "Threshold": threshold,
            "Precision": precision_score(
                y_val,
                prediction,
                zero_division=0
            ),
            "Recall": recall_score(
                y_val,
                prediction,
                zero_division=0
            ),
            "F1": f1_score(
                y_val,
                prediction,
                zero_division=0
            )
        })

threshold_results = pd.DataFrame(
    threshold_results
)

# print("\n" + "=" * 75)
# print("THRESHOLD ANALYSIS")
# print("=" * 75)

# print(
#     threshold_results.round(4)
#     .to_string(index=False)
# )

# =========================================================
# 11. FEATURE SIGNAL CHECK
# =========================================================

signal = (
    development[
        numeric_features + [target]
    ]
    .corr(numeric_only=True)[target]
    .drop(target)
    .abs()
    .sort_values(ascending=False)
)

# print("\n" + "=" * 75)
# print("NUMERIC FEATURE SIGNAL")
# print("=" * 75)

#print(signal.round(4))

print("\n" + "=" * 75)
print("CATEGORICAL RETURN-RATE RANGE")
print("=" * 75)

for column in categorical_features:

    rates = development.groupby(
        column,
        observed=True
    )[target].mean()

    print(
        f"{column:25s} "
        f"min={rates.min():.3f}  "
        f"max={rates.max():.3f}"
    )


# =========================================================
# 12. XGBOOST + LIGHTGBM TUNING
# =========================================================

tscv = TimeSeriesSplit(n_splits=3)


# ---------------------------------------------------------
# XGBOOST
# ---------------------------------------------------------

xgb_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ))
])

xgb_params = {
    "model__n_estimators": [200, 300, 500],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [3, 5, 7],
    "model__min_child_weight": [3, 5, 7],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0]
}

xgb_search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=xgb_params,
    n_iter=15,
    scoring="average_precision",
    cv=tscv,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("\nTuning XGBoost...")

xgb_search.fit(
    development[features],
    development[target]
)


# ---------------------------------------------------------
# LIGHTGBM
# ---------------------------------------------------------

lgb_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LGBMClassifier(
        scale_pos_weight=scale_pos_weight,
        subsample_freq=1,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    ))
])

lgb_params = {
    "model__n_estimators": [200, 300, 500],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [3, 5, 7],
    "model__num_leaves": [15, 25, 31],
    "model__min_child_samples": [20, 30, 50],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0]
}

lgb_search = RandomizedSearchCV(
    estimator=lgb_pipeline,
    param_distributions=lgb_params,
    n_iter=15,
    scoring="average_precision",
    cv=tscv,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("\nTuning LightGBM...")

lgb_search.fit(
    development[features],
    development[target]
)


print("\n" + "=" * 75)
print("BEST PARAMETERS")
print("=" * 75)

print("\nXGBoost:")
print(xgb_search.best_params_)
print(
    "CV PR-AUC:",
    round(xgb_search.best_score_, 4)
)

print("\nLightGBM:")
print(lgb_search.best_params_)
print(
    "CV PR-AUC:",
    round(lgb_search.best_score_, 4)
)

# =========================================================
# 13. VALIDATE TUNED MODELS
# =========================================================

tuned_models = {
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5,
        subsample=1.0,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ),

    "LightGBM": LGBMClassifier(
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
}

tuned_results = []
tuned_pipelines = {}
tuned_probabilities = {}

for name, model in tuned_models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    probability = pipeline.predict_proba(X_val)[:, 1]
    prediction = (probability >= 0.50).astype(int)

    tuned_pipelines[name] = pipeline
    tuned_probabilities[name] = probability

    tuned_results.append({
        "Model": name,
        "Precision": precision_score(
            y_val, prediction, zero_division=0
        ),
        "Recall": recall_score(
            y_val, prediction, zero_division=0
        ),
        "F1": f1_score(
            y_val, prediction, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_val, probability
        ),
        "PR_AUC": average_precision_score(
            y_val, probability
        )
    })

tuned_results = pd.DataFrame(tuned_results)

print("\n" + "=" * 75)
print("TUNED MODEL VALIDATION")
print("=" * 75)

print(
    tuned_results.round(4).to_string(index=False)
)

# =========================================================
# 14. THRESHOLD OPTIMIZATION
# =========================================================

threshold_results = []

for name, probability in tuned_probabilities.items():

    for threshold in [
        x / 100 for x in range(30, 96)
    ]:

        prediction = (
            probability >= threshold
        ).astype(int)

        threshold_results.append({
            "Model": name,
            "Threshold": threshold,
            "Precision": precision_score(
                y_val, prediction, zero_division=0
            ),
            "Recall": recall_score(
                y_val, prediction, zero_division=0
            ),
            "F1": f1_score(
                y_val, prediction, zero_division=0
            )
        })

threshold_results = pd.DataFrame(
    threshold_results
)

best_thresholds = (
    threshold_results
    .sort_values("F1", ascending=False)
    .groupby("Model")
    .first()
    .reset_index()
)

print("\n" + "=" * 75)
print("BEST VALIDATION THRESHOLD")
print("=" * 75)

print(
    best_thresholds.round(4)
    .to_string(index=False)
)