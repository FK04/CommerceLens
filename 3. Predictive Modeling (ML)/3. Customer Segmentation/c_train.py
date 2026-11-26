from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

# =====================================================
# 1. LOAD DATA
# =====================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_PATH = (
    PROJECT_ROOT / "2. Data"
    / "4. Feature Engineering" / "sales.csv"
)

df = pd.read_csv(
    DATA_PATH,
    usecols=[
        "customer_id", "order_id", "order_date",
        "order_status", "net_sales"
    ]
)

df["order_date"] = pd.to_datetime(df["order_date"])
snapshot = df["order_date"].max()

# =====================================================
# 2. RFM FEATURE ENGINEERING
# =====================================================

sales = df[df["order_status"] == "Completed"].copy()

rfm = sales.groupby("customer_id").agg(
    last_purchase=("order_date", "max"),
    frequency=("order_id", "nunique"),
    monetary=("net_sales", "sum")
)

rfm["recency"] = (
    snapshot - rfm["last_purchase"]
).dt.days

features = ["recency", "frequency", "monetary"]

# Log transformation reduces skewness
X_log = np.log1p(rfm[features])

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X_log)

# =====================================================
# 3. MODEL COMPARISON
# =====================================================

results = []

for k in [3, 4, 5]:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    silhouette = silhouette_score(
        X, labels,
        sample_size=5000,
        random_state=42
    )

    db_score = davies_bouldin_score(X, labels)

    results.append({
        "Clusters": k,
        "Silhouette": silhouette,
        "Davies_Bouldin": db_score,
        "Min_Cluster_Size": pd.Series(labels).value_counts().min()
    })

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

# =====================================================
# 4. SELECT BEST MODEL
# =====================================================

best_k = int(
    results_df.sort_values(
        "Silhouette", ascending=False
    ).iloc[0]["Clusters"]
)

final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

rfm["cluster"] = final_model.fit_predict(X)

print("\nSELECTED MODEL")
print("Algorithm: K-Means")
print("Clusters :", best_k)

# =====================================================
# 5. CUSTOMER SEGMENT PROFILES
# =====================================================

profile = rfm.groupby("cluster").agg(
    customers=("frequency", "size"),
    median_recency=("recency", "median"),
    median_frequency=("frequency", "median"),
    median_monetary=("monetary", "median")
)

profile["customer_pct"] = (
    profile["customers"] / len(rfm) * 100
)

print("\nCLUSTER PROFILES")
print(profile.round(2).to_string())

print("\nTraining comparison completed.")
