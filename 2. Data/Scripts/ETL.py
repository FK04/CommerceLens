"""
Corrected E-Commerce ETL Pipeline
---------------------------------
Real-world rule used here:
- Preserve valid business events (Cancelled, Returned, Pending, negative-profit orders).
- Impute only when the business meaning is reasonably clear.
- Keep legitimate missing values as nullable where filling would invent information.
- Reject rows only for structural/critical problems (missing keys, invalid essential values,
  impossible quantities/prices, or exact duplicate business keys).
- Produce Cleaned, Rejected and Quality Report outputs.

Folder structure expected:
    Data/
    ├── 1. Raw/
    ├── 2. Cleaned/
    ├── 3. Deleted/
    └── 4. Scripts/
        └── ETL_pipeline.py

Run from any directory:
    python ETL_pipeline.py

The script locates the Data folder automatically from its own location.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

FILES = {
    "sales": "ecommerce_sales_customer_analytics_150k.csv",
    "customers": "customer_master.csv",
    "items": "order_items.csv",
    "products": "product_catalog.csv",
    "statistics": "dataset_statistics.csv",
}

def clean_text(df):
    """Trim text without converting real missing values into the string 'nan'."""
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("string").str.strip()
        df[c] = df[c].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return df

def to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def add_reason(reason, mask, text):
    """Append rejection reasons without losing an earlier reason."""
    current = reason.fillna("")
    reason.loc[mask] = np.where(
        current.loc[mask].eq(""), text, current.loc[mask] + "; " + text
    )

def split_valid(df, reason):
    bad = reason.fillna("").ne("")
    rejected = df.loc[bad].copy()
    rejected["rejection_reason"] = reason.loc[bad].values
    clean = df.loc[~bad].copy()
    return clean.reset_index(drop=True), rejected.reset_index(drop=True)

def process_sales(df):
    df = clean_text(df.copy())
    numeric = [
        "customer_age", "delivery_days", "estimated_delivery_days", "customer_rating",
        "loyalty_points_earned", "loyalty_points_redeemed", "quantity", "gross_sales",
        "discount_amount", "tax_amount", "shipping_cost", "net_sales", "product_cost",
        "profit", "profit_margin_percentage", "customer_lifetime_value",
        "customer_order_count"
    ]
    df = to_numeric(df, numeric)

    if "order_date" in df:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    if "order_time" in df:
        df["order_time"] = pd.to_datetime(df["order_time"], format="%H:%M:%S", errors="coerce").dt.time

    # Context-aware missing values: these are business labels, not invented measurements.
    if "return_status" in df:
        df["return_status"] = df["return_status"].fillna("Not Returned")
    if "return_reason" in df:
        df["return_reason"] = df["return_reason"].fillna("Not Applicable")
    if "coupon_code" in df:
        df["coupon_code"] = df["coupon_code"].fillna("NO_COUPON_RECORDED")
    if "campaign_name" in df:
        df["campaign_name"] = df["campaign_name"].fillna("NO_CAMPAIGN_RECORDED")
    if "customer_review" in df:
        df["customer_review"] = df["customer_review"].fillna("No Review")
    if "review_sentiment" in df:
        df["review_sentiment"] = df["review_sentiment"].fillna("No Review")

    # IMPORTANT: ratings and delivery durations remain NA if genuinely unavailable.
    # Filling them with zero would create false observations.

    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    if "order_id" in df:
        add_reason(reason, df["order_id"].isna(), "Missing order_id")
        add_reason(reason, df["order_id"].duplicated(keep="first"), "Duplicate order_id")
    if "customer_id" in df:
        add_reason(reason, df["customer_id"].isna(), "Missing customer_id")
    if "order_date" in df:
        add_reason(reason, df["order_date"].isna(), "Invalid/missing order_date")
    if "quantity" in df:
        add_reason(reason, df["quantity"].isna() | (df["quantity"] <= 0), "Invalid quantity")
    for c in ["gross_sales", "net_sales", "product_cost"]:
        if c in df:
            add_reason(reason, df[c].isna() | (df[c] < 0), f"Invalid {c}")

    clean, rejected = split_valid(df, reason)

    money = ["gross_sales","discount_amount","tax_amount","shipping_cost",
             "net_sales","product_cost","profit","customer_lifetime_value"]
    for c in money:
        if c in clean:
            clean[c] = clean[c].round(2)
    if "profit_margin_percentage" in clean:
        clean["profit_margin_percentage"] = clean["profit_margin_percentage"].round(2)
    if "quantity" in clean:
        clean["quantity"] = clean["quantity"].astype("Int64")
    if "customer_age" in clean:
        clean["customer_age"] = clean["customer_age"].astype("Int64")

    return clean, rejected

def process_customers(df):
    df = clean_text(df.copy())
    df = to_numeric(df, ["customer_age", "customer_acquisition_cost"])

    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    add_reason(reason, df["customer_id"].isna(), "Missing customer_id")
    add_reason(reason, df["customer_id"].duplicated(keep="first"), "Duplicate customer_id")
    if "customer_age" in df:
        add_reason(reason, df["customer_age"].notna() & ~df["customer_age"].between(18, 120),
                   "Invalid customer_age")

    clean, rejected = split_valid(df, reason)

    # Keep customers even if they have no orders: they are valid master-data records.
    for c in ["customer_city","customer_state","customer_country","region","customer_segment"]:
        if c in clean:
            clean[c] = clean[c].fillna("Unknown")
    if "customer_postal_code" in clean:
        clean["customer_postal_code"] = clean["customer_postal_code"].astype("string")
    if "customer_age" in clean:
        clean["customer_age"] = clean["customer_age"].astype("Int64")
    if "customer_acquisition_cost" in clean:
        clean["customer_acquisition_cost"] = clean["customer_acquisition_cost"].round(2)

    return clean, rejected

def process_products(df):
    df = clean_text(df.copy())
    df = to_numeric(df, ["unit_price", "product_cost", "product_rating"])

    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    add_reason(reason, df["product_id"].isna(), "Missing product_id")
    add_reason(reason, df["product_id"].duplicated(keep="first"), "Duplicate product_id")
    if "unit_price" in df:
        add_reason(reason, df["unit_price"].isna() | (df["unit_price"] <= 0), "Invalid unit_price")
    if "product_cost" in df:
        add_reason(reason, df["product_cost"].isna() | (df["product_cost"] < 0), "Invalid product_cost")
    if "product_rating" in df:
        add_reason(reason, df["product_rating"].notna() & ~df["product_rating"].between(0, 5),
                   "Invalid product_rating")

    clean, rejected = split_valid(df, reason)

    for c in ["product_category","product_subcategory","brand","supplier"]:
        if c in clean:
            clean[c] = clean[c].fillna("Unknown")
    for c in ["unit_price","product_cost"]:
        if c in clean:
            clean[c] = clean[c].round(2)
    if "product_rating" in clean:
        clean["product_rating"] = clean["product_rating"].round(1)

    return clean, rejected

def process_items(df, valid_orders, valid_products):
    df = clean_text(df.copy())
    numeric = ["quantity","unit_price","discount_percentage","discount_amount",
               "gross_sales","tax_amount","shipping_cost","net_sales","product_cost","profit"]
    df = to_numeric(df, numeric)

    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    add_reason(reason, df["order_id"].isna(), "Missing order_id")
    add_reason(reason, df["product_id"].isna(), "Missing product_id")
    add_reason(reason, ~df["order_id"].isin(valid_orders), "Unknown/rejected order_id")
    add_reason(reason, ~df["product_id"].isin(valid_products), "Unknown/rejected product_id")
    add_reason(reason, df["quantity"].isna() | (df["quantity"] <= 0), "Invalid quantity")
    add_reason(reason, df["unit_price"].isna() | (df["unit_price"] <= 0), "Invalid unit_price")

    clean, rejected = split_valid(df, reason)

    for c in ["discount_amount","tax_amount","shipping_cost"]:
        if c in clean:
            clean[c] = clean[c].fillna(0.0)

    money = ["unit_price","discount_amount","gross_sales","tax_amount",
             "shipping_cost","net_sales","product_cost","profit"]
    for c in money:
        if c in clean:
            clean[c] = clean[c].round(2)
    if "discount_percentage" in clean:
        clean["discount_percentage"] = clean["discount_percentage"].round(4)
    if "quantity" in clean:
        clean["quantity"] = clean["quantity"].astype("Int64")

    return clean, rejected

def build_statistics(sales, customers, items, products):
    order_count = sales["order_id"].nunique()
    returned = sales["return_status"].eq("Returned").sum() if "return_status" in sales else 0
    cancelled = sales["order_status"].eq("Cancelled").sum() if "order_status" in sales else 0

    return pd.DataFrame([{
        "Total Transactions": order_count,
        "Total Columns": sales.shape[1],
        "Total Customers": sales["customer_id"].nunique(),
        "Customers in Master": customers["customer_id"].nunique(),
        "Total Products Used": items["product_id"].nunique(),
        "Products in Catalog": products["product_id"].nunique(),
        "Date Range": (
            f"{sales['order_date'].min().date()} to {sales['order_date'].max().date()}"
            if "order_date" in sales and sales["order_date"].notna().any() else "N/A"
        ),
        "Total Revenue": round(sales["net_sales"].sum(), 2),
        "Total Profit": round(sales["profit"].sum(), 2),
        "Average Order Value": round(sales["net_sales"].mean(), 2),
        "Average Rating": round(sales["customer_rating"].mean(), 2),
        "Return Rate %": round(returned / order_count * 100, 2) if order_count else 0,
        "Cancellation Rate %": round(cancelled / order_count * 100, 2) if order_count else 0,
    }])

def quality_row(name, raw, clean, rejected):
    return {
        "dataset": name,
        "raw_rows": len(raw),
        "clean_rows": len(clean),
        "rejected_rows": len(rejected),
        "retention_pct": round(len(clean) / len(raw) * 100, 2) if len(raw) else 100,
        "raw_missing_cells": int(raw.isna().sum().sum()),
        "clean_missing_cells": int(clean.isna().sum().sum()),
        "duplicate_rows_raw": int(raw.duplicated().sum()),
        "duplicate_rows_clean": int(clean.duplicated().sum()),
    }

def save(name, clean, rejected, clean_dir, reject_dir):
    clean.to_csv(clean_dir / f"{name}_cln.csv", index=False)
    rejected.to_csv(reject_dir / f"{name}_rejected.csv", index=False)
    print(f"{name:<45} clean={len(clean):>7,} | rejected={len(rejected):>6,}")

def run():
    # ETL_pipeline.py is inside:
    # Data/4. Scripts/
    #
    # Therefore the parent of the script folder is the Data folder.
    script_dir = Path(__file__).resolve().parent
    data_root = script_dir.parent

    raw_dir = data_root / "1. Raw"
    clean_dir = data_root / "2. Cleaned"
    reject_dir = data_root / "3. Deleted"

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw folder not found: {raw_dir}\n"
            "Expected ETL_pipeline.py inside '4. Scripts' and "
            "'1. Raw' beside that folder."
        )

    missing = [
        filename
        for filename in FILES.values()
        if not (raw_dir / filename).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "The following expected input files were not found in "
            f"'{raw_dir}':\n- " + "\n- ".join(missing)
        )

    clean_dir.mkdir(parents=True, exist_ok=True)
    reject_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("REAL-WORLD E-COMMERCE ETL PIPELINE")
    print("=" * 72)
    print("Project root :", data_root)
    print("Raw folder  :", raw_dir)
    print("Clean folder:", clean_dir)
    print("Deleted     :", reject_dir)

    # --------------------------------------------------------
    # EXTRACT
    # --------------------------------------------------------
    raw_sales = pd.read_csv(
        raw_dir / FILES["sales"],
        low_memory=False
    )
    raw_customers = pd.read_csv(
        raw_dir / FILES["customers"],
        low_memory=False
    )
    raw_items = pd.read_csv(
        raw_dir / FILES["items"],
        low_memory=False
    )
    raw_products = pd.read_csv(
        raw_dir / FILES["products"],
        low_memory=False
    )

    # --------------------------------------------------------
    # TRANSFORM
    # --------------------------------------------------------
    sales, sales_rej = process_sales(raw_sales)
    customers, customers_rej = process_customers(raw_customers)
    products, products_rej = process_products(raw_products)

    # Child-table referential integrity checks.
    valid_orders = set(
        sales["order_id"].dropna()
    )
    valid_products = set(
        products["product_id"].dropna()
    )

    items, items_rej = process_items(
        raw_items,
        valid_orders,
        valid_products
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------
    save(
        "ecommerce_sales_customer_analytics_150k",
        sales,
        sales_rej,
        clean_dir,
        reject_dir
    )

    save(
        "customer_master",
        customers,
        customers_rej,
        clean_dir,
        reject_dir
    )

    save(
        "product_catalog",
        products,
        products_rej,
        clean_dir,
        reject_dir
    )

    save(
        "order_items",
        items,
        items_rej,
        clean_dir,
        reject_dir
    )

    # Rebuild statistics from the cleaned datasets.
    stats = build_statistics(
        sales,
        customers,
        items,
        products
    )

    stats.to_csv(
        clean_dir / "dataset_statistics_cln.csv",
        index=False
    )

    # Keep the quality report with the cleaned outputs so no
    # additional folder is required.
    quality = pd.DataFrame([
        quality_row(
            "ecommerce_sales",
            raw_sales,
            sales,
            sales_rej
        ),
        quality_row(
            "customer_master",
            raw_customers,
            customers,
            customers_rej
        ),
        quality_row(
            "product_catalog",
            raw_products,
            products,
            products_rej
        ),
        quality_row(
            "order_items",
            raw_items,
            items,
            items_rej
        ),
    ])

    quality.to_csv(
        clean_dir / "etl_quality_report.csv",
        index=False
    )

    print("-" * 72)
    print("ETL COMPLETED SUCCESSFULLY")
    print("-" * 72)
    print("Cleaned files :", clean_dir)
    print("Rejected rows :", reject_dir)
    print(
        "Quality report:",
        clean_dir / "etl_quality_report.csv"
    )
    print(
        "\nNote: Cancelled, Returned, Pending and "
        "negative-profit transactions are retained because "
        "they are valid real-world business events."
    )


def main():
    run()

if __name__ == "__main__":
    main()
