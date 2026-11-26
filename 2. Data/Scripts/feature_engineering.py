"""
Feature Engineering Pipeline
Creates ONLY the requested engineered features.

Expected structure:
Data/
├── 2. Cleaned/
│   ├── ecommerce_sales_customer_analytics_150k_cln.csv
│   ├── customer_master_cln.csv
│   ├── product_catalog_cln.csv
│   └── order_items_cln.csv
└── 4. Scripts/
    └── feature_engineering.py

Outputs are written to:
Data/4. Feature Engineering/
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent

CLEAN_DIR = DATA_ROOT / "2. Cleaned"
OUTPUT_DIR = DATA_ROOT / "4. Feature Engineering"

SALES_FILE = CLEAN_DIR / "ecommerce_sales_customer_analytics_150k_cln.csv"
CUSTOMER_FILE = CLEAN_DIR / "customer_master_cln.csv"
PRODUCT_FILE = CLEAN_DIR / "product_catalog_cln.csv"
ITEM_FILE = CLEAN_DIR / "order_items_cln.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================
def safe_divide(numerator, denominator):
    """Return NaN when division is undefined instead of inf/error."""
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return numerator.div(denominator.replace(0, np.nan))


def check_files():
    required = [SALES_FILE, CUSTOMER_FILE, PRODUCT_FILE, ITEM_FILE]
    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing cleaned input file(s):\n- " + "\n- ".join(missing)
        )


# ============================================================
# A. SALES / ORDERS FEATURES
# ============================================================
def engineer_sales(df):
    df = df.copy()

    # Convert required numeric fields safely.
    numeric_cols = [
        "gross_sales",
        "discount_amount",
        "delivery_days",
        "estimated_delivery_days",
        "loyalty_points_earned",
        "loyalty_points_redeemed",
        "profit",
        "quantity",
        "net_sales",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---------- Date / time ----------
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    df["order_year"] = df["order_date"].dt.year.astype("Int64")
    df["order_quarter"] = df["order_date"].dt.quarter.astype("Int64")
    df["order_month"] = df["order_date"].dt.month.astype("Int64")
    df["order_month_name"] = df["order_date"].dt.month_name()
    df["order_day"] = df["order_date"].dt.day.astype("Int64")
    df["order_day_name"] = df["order_date"].dt.day_name()

    df["is_weekend"] = (
        df["order_date"].dt.dayofweek.isin([5, 6]).astype(int)
    )

    # ---------- Discount ----------
    df["effective_discount_rate"] = safe_divide(
        df["discount_amount"],
        df["gross_sales"]
    ).round(4)

    # ---------- Delivery / SLA ----------
    df["delivery_delay_days"] = (
        df["delivery_days"] - df["estimated_delivery_days"]
    ).round(1)

    # Keep missing delivery information as missing rather than falsely saying SLA=0.
    df["sla_breached"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    known_delivery = df["delivery_delay_days"].notna()

    df.loc[known_delivery, "sla_breached"] = (
        df.loc[known_delivery, "delivery_delay_days"] > 0
    ).astype(int)

    # ---------- Loyalty ----------
    earned = df["loyalty_points_earned"]
    redeemed = df["loyalty_points_redeemed"]

    df["loyalty_burn_rate"] = np.where(
        earned > 0,
        redeemed / earned,
        0.0
    )

    df["loyalty_burn_rate"] = (
        pd.to_numeric(df["loyalty_burn_rate"], errors="coerce").round(4)
    )

    # ---------- Return target ----------
    df["is_returned"] = (
        df["return_status"]
        .astype("string")
        .str.strip()
        .str.lower()
        .eq("returned")
        .astype(int)
    )

    # ---------- Profitability ----------
    df["profit_per_unit"] = safe_divide(
        df["profit"],
        df["quantity"]
    ).round(2)

    df["profit_margin_rate"] = safe_divide(
        df["profit"],
        df["net_sales"]
    ).round(4)

    return df


# ============================================================
# B. CUSTOMER MASTER / RFM FEATURES
# ============================================================
def engineer_customers(customer_df, sales_df):
    customers = customer_df.copy()

    customers["customer_age"] = pd.to_numeric(
        customers["customer_age"],
        errors="coerce"
    )

    # Age segmentation requested by user.
    customers["age_group"] = pd.cut(
        customers["customer_age"],
        bins=[18, 26, 36, 51, np.inf],
        labels=["18-25", "26-35", "36-50", "50+"],
        right=False,
        include_lowest=True
    )

    # Snapshot = one day after latest observed order.
    snapshot_date = sales_df["order_date"].max() + pd.Timedelta(days=1)

    # "frequency_orders" is explicitly completed-order count.
    completed = (
        sales_df["order_status"]
        .astype("string")
        .str.strip()
        .str.lower()
        .eq("completed")
    )

    # Recency and monetary metrics use observed transaction history.
    base = sales_df.groupby("customer_id").agg(
        last_order_date=("order_date", "max"),
        total_net_spend=("net_sales", "sum"),
        returned_orders=("is_returned", "sum"),
        total_orders=("order_id", "nunique"),
    ).reset_index()

    completed_frequency = (
        sales_df.loc[completed]
        .groupby("customer_id")["order_id"]
        .nunique()
        .rename("frequency_orders")
        .reset_index()
    )

    customer_features = base.merge(
        completed_frequency,
        on="customer_id",
        how="left"
    )

    customer_features["frequency_orders"] = (
        customer_features["frequency_orders"]
        .fillna(0)
        .astype("Int64")
    )

    customer_features["recency_days"] = (
        snapshot_date - customer_features["last_order_date"]
    ).dt.days.astype("Int64")

    # AOV uses completed-order count, as requested.
    customer_features["average_order_value"] = safe_divide(
        customer_features["total_net_spend"],
        customer_features["frequency_orders"]
    ).round(2)

    customer_features["customer_return_rate"] = safe_divide(
        customer_features["returned_orders"],
        customer_features["total_orders"]
    ).round(4)

    customer_features["total_net_spend"] = (
        customer_features["total_net_spend"].round(2)
    )

    # Only keep the requested derived columns for the merge.
    customer_features = customer_features[
        [
            "customer_id",
            "recency_days",
            "frequency_orders",
            "total_net_spend",
            "average_order_value",
            "customer_return_rate",
        ]
    ]

    # Left join preserves valid customers with no order history.
    customers = customers.merge(
        customer_features,
        on="customer_id",
        how="left"
    )

    customers["frequency_orders"] = (
        customers["frequency_orders"].fillna(0).astype("Int64")
    )
    customers["total_net_spend"] = (
        customers["total_net_spend"].fillna(0).round(2)
    )

    return customers


# ============================================================
# C. PRODUCT CATALOG FEATURES
# ============================================================
def engineer_products(df):
    df = df.copy()

    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["product_cost"] = pd.to_numeric(df["product_cost"], errors="coerce")

    # Unit margin
    df["unit_margin"] = (
        df["unit_price"] - df["product_cost"]
    ).round(2)

    # Relative price segmentation based on the current product catalogue.
    df["price_band"] = pd.qcut(
        df["unit_price"],
        q=3,
        labels=["Budget", "Mid-Range", "Premium"],
        duplicates="drop"
    )

    return df


# ============================================================
# D. ORDER ITEMS FEATURE
# ============================================================
def engineer_order_items(df):
    df = df.copy()

    df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    df["line_profit_per_unit"] = safe_divide(
        df["profit"],
        df["quantity"]
    ).round(2)

    return df


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    check_files()

    print("=" * 72)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 72)

    sales_raw = pd.read_csv(SALES_FILE, low_memory=False)
    customers_raw = pd.read_csv(CUSTOMER_FILE, low_memory=False)
    products_raw = pd.read_csv(PRODUCT_FILE, low_memory=False)
    items_raw = pd.read_csv(ITEM_FILE, low_memory=False)

    sales = engineer_sales(sales_raw)
    customers = engineer_customers(customers_raw, sales)
    products = engineer_products(products_raw)
    items = engineer_order_items(items_raw)

    sales_output = OUTPUT_DIR / "sales.csv"
    customer_output = OUTPUT_DIR / "customers.csv"
    product_output = OUTPUT_DIR / "products.csv"
    item_output = OUTPUT_DIR / "order_items.csv"

    sales.to_csv(sales_output, index=False)
    customers.to_csv(customer_output, index=False)
    products.to_csv(product_output, index=False)
    items.to_csv(item_output, index=False)

    print("\nFeature engineering completed successfully.")
    print(f"Sales     : {sales.shape} -> {sales_output.name}")
    print(f"Customers : {customers.shape} -> {customer_output.name}")
    print(f"Products  : {products.shape} -> {product_output.name}")
    print(f"OrderItems: {items.shape} -> {item_output.name}")

    print("\nEngineered features:")
    print(
        "Sales: order_year, order_quarter, order_month, order_month_name, "
        "order_day, order_day_name, is_weekend, effective_discount_rate, "
        "delivery_delay_days, sla_breached, loyalty_burn_rate, is_returned, "
        "profit_per_unit, profit_margin_rate"
    )
    print(
        "Customers: recency_days, frequency_orders, total_net_spend, "
        "average_order_value, customer_return_rate, age_group"
    )
    print("Products: unit_margin, price_band")
    print("Order Items: line_profit_per_unit")
    print("=" * 72)


if __name__ == "__main__":
    main()
