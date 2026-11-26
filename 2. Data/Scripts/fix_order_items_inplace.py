"""Reconcile CommerceLens item batches without changing the source path or schema.
Place beside sales.csv and order_items.csv and run: python fix_order_items_inplace.py
Optional: python fix_order_items_inplace.py "path/to/sales.csv" "path/to/order_items.csv"
No external Python packages required.
"""
import argparse
import csv
import os
import shutil
import tempfile
from decimal import Decimal
from pathlib import Path

FIELDS = ('quantity', 'gross_sales', 'discount_amount', 'tax_amount',
          'shipping_cost', 'net_sales', 'product_cost', 'profit')

def values(row):
    out = []
    for i, field in enumerate(FIELDS):
        v = Decimal(row[field]) * (1 if i == 0 else 100)
        if not v.is_finite() or abs(v - v.to_integral_value()) > Decimal("0.00001"):
            raise ValueError(f'Invalid amount for {row["order_id"]}: {field}')
        out.append(int(v.to_integral_value()))
    return out

def reconcile(sales, items):
    sales, items = Path(sales).resolve(), Path(items).resolve()
    if sales == items:
        raise ValueError('The two source paths must be different.')
    expected = {}
    with sales.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            oid = row['order_id']
            if not oid.strip() or oid in expected:
                raise ValueError(f'Blank or duplicate order header: {oid}')
            expected[oid] = values(row)
    if not expected:
        raise ValueError('No order headers found.')
    with items.open('rb') as f:
        encoding = 'utf-8-sig' if f.read(3) == b'\xef\xbb\xbf' else 'utf-8'
    tmp = None
    try:
        actual = {}
        seen = set()
        excluded_orders = set()
        previous = None
        keep = False
        total = retained = 0
        with items.open(encoding=encoding, newline='') as src, tempfile.NamedTemporaryFile(mode='w', encoding=encoding, newline='', dir=items.parent, prefix='.order_items_', suffix='.tmp', delete=False) as dst:
            tmp = Path(dst.name)
            reader = csv.DictReader(src)
            if not reader.fieldnames or not {'order_id', *FIELDS}.issubset(reader.fieldnames):
                raise ValueError('Required item columns missing.')
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                total += 1
                oid = row['order_id']
                if oid not in expected:
                    raise ValueError(f'Item order has no header: {oid}')
                if oid != previous:
                    keep = oid not in seen
                    seen.add(oid)
                    previous = oid
                if keep:
                    nums = values(row)
                    sums = actual.setdefault(oid, [0] * len(FIELDS))
                    for i, v in enumerate(nums):
                        sums[i] += v
                    writer.writerow(row)
                    retained += 1
                else:
                    excluded_orders.add(oid)
        mismatches = sum(actual.get(oid) != nums for oid, nums in expected.items())
        if mismatches:
            raise ValueError(f'{mismatches:,} orders failed reconciliation. Original file unchanged. Check source row order and input files.')
        print(f'Orders reconciled: {len(expected):,}')
        print(f'Item rows: {total:,} -> {retained:,}')
        print(f'Excluded rows: {total-retained:,}; affected orders: {len(excluded_orders):,}')
        print('Mismatched orders across eight fields: 0')
        print(f'Net Sales: {Decimal(sum(v[5] for v in expected.values())) / 100:,.2f}')
        if total == retained:
            print('Already reconciled. No source changes needed.')
            return
        backup = items.with_name(items.stem + '_before_reconciliation' + items.suffix + '.bak')
        if backup.exists():
            raise FileExistsError(f'Backup already exists: {backup}. Original file unchanged.')
        shutil.copy2(items, backup)
        os.replace(tmp, items)
        tmp = None
        print(f'Backup: {backup}')
        print(f'Updated existing source: {items}')
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)

if __name__ == '__main__':
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sales', nargs='?', default=folder / 'sales.csv')
    parser.add_argument('items', nargs='?', default=folder / 'orders.csv')
    args = parser.parse_args()
    reconcile(args.sales, args.items)
