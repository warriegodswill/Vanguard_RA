"""
Vanguard RA - Sample Data Generator
------------------------------------
Simulates Bank, POS, and Inventory records for a retail merchant,
with deliberately injected mismatches so the reconciliation engine
has real leakages/anomalies to detect (mirrors the ₦2,013,800 demo
figure from the pitch deck).
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

N_DAYS = 30
TXNS_PER_DAY = 25
START_DATE = datetime(2025, 6, 1)

PRODUCTS = [
    ("Rice 50kg", 45000),
    ("Cooking Oil 5L", 12000),
    ("Sugar 1kg", 1200),
    ("Milk Powder 400g", 3500),
    ("Detergent 1kg", 2500),
    ("Bottled Water (crate)", 2000),
    ("Soft Drinks (crate)", 3200),
    ("Spaghetti (carton)", 9500),
]

PAYMENT_METHODS = ["POS Card", "Bank Transfer", "Cash"]


def generate_pos_records():
    """POS terminal transaction log — the merchant's point-of-sale truth."""
    records = []
    txn_id = 1000
    for day in range(N_DAYS):
        date = START_DATE + timedelta(days=day)
        for _ in range(TXNS_PER_DAY):
            product, unit_price = random.choice(PRODUCTS)
            qty = random.randint(1, 5)
            amount = unit_price * qty
            method = random.choices(PAYMENT_METHODS, weights=[0.55, 0.25, 0.20])[0]
            timestamp = date + timedelta(
                hours=random.randint(8, 20), minutes=random.randint(0, 59)
            )
            records.append({
                "txn_id": f"POS{txn_id}",
                "timestamp": timestamp,
                "product": product,
                "qty": qty,
                "unit_price": unit_price,
                "amount": amount,
                "payment_method": method,
            })
            txn_id += 1
    return pd.DataFrame(records)


def generate_bank_records(pos_df):
    """
    Bank/settlement feed. Most POS-card and transfer transactions should
    settle to the bank, but we inject realistic leakage scenarios:
      - missing settlements (POS batch never settled)
      - short settlements (bank fees / skimming shave the amount)
      - delayed settlement timestamps
      - a few phantom bank credits with no matching POS record
    """
    bank_records = []
    settleable = pos_df[pos_df["payment_method"] != "Cash"].copy()

    for _, row in settleable.iterrows():
        roll = random.random()

        if roll < 0.06:
            # Missing settlement entirely -> revenue leak
            continue
        elif roll < 0.12:
            # Short settlement (partial amount received) -> leak
            shortfall = round(row["amount"] * random.uniform(0.05, 0.25), -1)
            bank_records.append({
                "bank_ref": f"BNK{row['txn_id'][3:]}",
                "matched_txn_id": row["txn_id"],
                "settlement_time": row["timestamp"] + timedelta(hours=random.randint(1, 48)),
                "amount": row["amount"] - shortfall,
            })
        else:
            # Normal settlement, possibly delayed
            delay = timedelta(hours=random.randint(0, 6))
            bank_records.append({
                "bank_ref": f"BNK{row['txn_id'][3:]}",
                "matched_txn_id": row["txn_id"],
                "settlement_time": row["timestamp"] + delay,
                "amount": row["amount"],
            })

    # A few phantom/unmatched bank credits (e.g. misapplied transfers)
    for i in range(4):
        date = START_DATE + timedelta(days=random.randint(0, N_DAYS - 1))
        bank_records.append({
            "bank_ref": f"BNK-PHANTOM-{i}",
            "matched_txn_id": None,
            "settlement_time": date + timedelta(hours=random.randint(8, 20)),
            "amount": round(random.uniform(2000, 15000), -2),
        })

    return pd.DataFrame(bank_records)


def generate_inventory_records(pos_df):
    """
    Inventory depletion log. Should mirror units sold per product,
    but we inject shrinkage (theft/damage/miscount) and a couple of
    recording errors so inventory doesn't tie out to POS quantities.
    """
    daily_sold = (
        pos_df.assign(date=pos_df["timestamp"].dt.date)
        .groupby(["date", "product"])["qty"]
        .sum()
        .reset_index()
    )

    inventory_records = []
    for _, row in daily_sold.iterrows():
        roll = random.random()
        if roll < 0.10:
            # Shrinkage: fewer units left than POS says were sold implies
            # extra units vanished (theft/damage) beyond recorded sales
            shrinkage = random.randint(1, 3)
            depleted = row["qty"] + shrinkage
        elif roll < 0.15:
            # Recording error: inventory under-recorded vs actual sale
            depleted = max(row["qty"] - random.randint(1, 2), 0)
        else:
            depleted = row["qty"]

        inventory_records.append({
            "date": row["date"],
            "product": row["product"],
            "units_sold_pos": row["qty"],
            "units_depleted_inventory": depleted,
        })

    return pd.DataFrame(inventory_records)


def main():
    pos_df = generate_pos_records()
    bank_df = generate_bank_records(pos_df)
    inventory_df = generate_inventory_records(pos_df)

    pos_df.to_csv("pos_records.csv", index=False)
    bank_df.to_csv("bank_records.csv", index=False)
    inventory_df.to_csv("inventory_records.csv", index=False)

    print(f"POS records:       {len(pos_df)} rows -> pos_records.csv")
    print(f"Bank records:      {len(bank_df)} rows -> bank_records.csv")
    print(f"Inventory records: {len(inventory_df)} rows -> inventory_records.csv")


if __name__ == "__main__":
    main()
