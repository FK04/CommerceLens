import os
import pandas as pd

# 1. Ingest Raw Datasets
raw_dir = "1. Raw" if os.path.exists("1. Raw") else "."
sales = pd.read_csv(os.path.join(raw_dir, "ecommerce_sales_customer_analytics_150k.csv"))
items = pd.read_csv(os.path.join(raw_dir, "order_items.csv"))
cust = pd.read_csv(os.path.join(raw_dir, "customer_master.csv"))
prod = pd.read_csv(os.path.join(raw_dir, "product_catalog.csv"))
stats = pd.read_csv(os.path.join(raw_dir, "dataset_statistics.csv"))

# 2. Executive Metrics & Macro Financials
net_rev = sales["net_sales"].sum()
net_profit = sales["profit"].sum()
margin_pct = (net_profit / net_rev) * 100
total_orders = len(sales)
total_cust = len(cust)
active_cust = sales["customer_id"].nunique()
inactive_cust = len(set(cust["customer_id"]) - set(sales["customer_id"]))

# 3. Operational Friction & Margin Leaks
cancel_count = (sales["order_status"] == "Cancelled").sum()
cancel_pct = (cancel_count / total_orders) * 100
return_count = (sales["return_status"] == "Returned").sum()
return_pct = (return_count / total_orders) * 100
neg_orders = sales[sales["profit"] < 0]
neg_loss = abs(neg_orders["profit"].sum())

# 4. Cross-Table Relational Discrepancies
agg_items = items.groupby("order_id")["gross_sales"].sum().reset_index()
order_sync = pd.merge(sales[["order_id", "gross_sales"]], agg_items, on="order_id")
mismatched_orders = (order_sync["gross_sales_x"] != order_sync["gross_sales_y"]).sum()

# 5. Output Summary Dashboard
print("=" * 82)
print("             RAW DATASET AUDIT & ENTERPRISE PERFORMANCE PROFILE            ")
print("=" * 82)
print(f"{'Strategic Dimension':<35} | {'Quantified Metric':<42}")
print("-" * 82)
print(f"{'Total Net Realized Revenue':<35} | ${net_rev:>18,.2f}")
print(f"{'Total Net Profit (Margin %)':<35} | ${net_profit:>18,.2f} ({margin_pct:.1f}%)")
print(f"{'Transaction Volume (2021-2025)':<35} | {total_orders:>18,} Orders")
print(f"{'Customer Base (Active / Master)':<35} | {active_cust:>9,} Active / {total_cust:,} Master")
print(f"{'Product Catalog Depth':<35} | {prod['product_id'].nunique():>18,} Unique SKUs")
print(f"{'Order Cancellation Rate':<35} | {cancel_count:>18,} Orders ({cancel_pct:.2f}%)")
print(f"{'Product Return Rate':<35} | {return_count:>18,} Orders ({return_pct:.2f}%)")
print(f"{'Negative Margin Orders (Cash Loss)':<35} | {len(neg_orders):>9,} Orders (-${neg_loss:,.2f})")
print(f"{'Line-Item Mismatch Orders':<35} | {mismatched_orders:>18,} Desynced Orders")
print(f"{'Inactive Registered Accounts':<35} | {inactive_cust:>18,} Zero-Order Users")
print("=" * 82)