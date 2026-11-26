
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =====================================================
# 1. PATHS
# =====================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_PATH = (
    PROJECT_ROOT / "2. Data"
    / "4. Feature Engineering" / "sales.csv"
)

MODEL_PATH = SCRIPT_DIR / "c_model.pkl"

OUTPUT_PATH = (
    PROJECT_ROOT / "3. Predictive Modeling (ML)"
    / "4. Prediction" / "Customer_seg.csv"
)

# =====================================================
# 2. PREPARE RFM DATA
# =====================================================

df = pd.read_csv(
    DATA_PATH,
    usecols=[
        "customer_id", "order_id", "order_date",
        "order_status", "net_sales"
    ]
)

df["order_date"] = pd.to_datetime(df["order_date"])
snapshot = df["order_date"].max()

sales = df[df["order_status"] == "Completed"].copy()

rfm = sales.groupby("customer_id").agg(
    last_purchase=("order_date", "max"),
    frequency=("order_id", "nunique"),
    monetary=("net_sales", "sum")
)

rfm["recency"] = (snapshot - rfm["last_purchase"]).dt.days

features = ["recency", "frequency", "monetary"]

# Same transformation as c_train.py
scaler = StandardScaler()
X = scaler.fit_transform(np.log1p(rfm[features]))

# =====================================================
# 3. FINAL K-MEANS MODEL
# =====================================================

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

rfm["cluster"] = model.fit_predict(X)

# =====================================================
# 4. ASSIGN SEGMENT NAMES
# =====================================================

profile = rfm.groupby("cluster").agg(
    customers=("frequency", "size"),
    median_recency=("recency", "median"),
    median_frequency=("frequency", "median"),
    median_monetary=("monetary", "median")
)

active = profile["median_recency"].idxmin()
low = profile["median_monetary"].idxmin()

if active == low:
    raise ValueError("Review cluster profiles before naming.")

established = next(
    c for c in profile.index if c not in [active, low]
)

segment_map = {
    int(active): "Active High-Value",
    int(low): "Low Engagement",
    int(established): "Established Customers"
}

rfm["segment"] = rfm["cluster"].map(segment_map)

# =====================================================
# 5. INCLUDE ALL CUSTOMERS
# =====================================================

customers = df[["customer_id"]].drop_duplicates()

output = customers.merge(
    rfm.reset_index(),
    on="customer_id",
    how="left"
)

output["frequency"] = output["frequency"].fillna(0).astype(int)
output["monetary"] = output["monetary"].fillna(0)
output["segment"] = output["segment"].fillna(
    "No Completed Purchase"
)

# =====================================================
# 6. INACTIVITY-BASED CHURN RISK
# =====================================================

output["inactivity_risk"] = np.select(
    [
        output["recency"].isna(),
        output["recency"] > 676,
        output["recency"] > 420
    ],
    [
        "No Completed Purchase",
        "Very High Inactivity",
        "Elevated Inactivity"
    ],
    default="Within Typical Gap"
)

output["recency"] = output["recency"].astype("Int64")
output["cluster"] = output["cluster"].astype("Int64")

# =====================================================
# 7. SAVE MODEL AND RESULTS
# =====================================================

model_package = {
    "model": model,
    "scaler": scaler,
    "features": features,
    "segment_map": segment_map,
    "snapshot_date": str(snapshot.date()),
    "inactivity_thresholds": [420, 676]
}

joblib.dump(model_package, MODEL_PATH)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

output = output.rename(columns={
    "recency": "recency_days",
    "frequency": "order_frequency",
    "monetary": "total_spending"
})

output.to_csv(OUTPUT_PATH, index=False)

# =====================================================
# 8. FINAL RESULTS
# =====================================================

print("\nFINAL CUSTOMER SEGMENTATION")
print("Model: K-Means")
print("Clusters: 3")
print("Snapshot:", snapshot.date())
print("Total customers:", len(output))

print("\nSEGMENT DISTRIBUTION")
print(output["segment"].value_counts().to_string())

print("\nINACTIVITY RISK DISTRIBUTION")
print(output["inactivity_risk"].value_counts().to_string())

print("\nCLUSTER PROFILES")
print(profile.round(2).to_string())

print("\nFiles saved:")
print(MODEL_PATH)
print(OUTPUT_PATH)

print("\nEvaluation completed successfully.")
