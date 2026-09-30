# Vanguard RA — Prototype

AI-powered revenue assurance platform prototype: reconciles Bank, POS, and
Inventory records to detect revenue leakages, settlement mismatches, and
transaction anomalies, with a risk score and Streamlit dashboard.

## Files

- `generate_sample_data.py` — creates simulated `pos_records.csv`,
  `bank_records.csv`, `inventory_records.csv` with realistic injected
  leaks (missing/short settlements, phantom credits, inventory shrinkage).
- `reconciliation_engine.py` — the core `ReconciliationEngine` class.
  Matches POS ↔ Bank settlements and POS ↔ Inventory quantities, classifies
  every record as **Matched / Mismatch / Leak / Anomaly**, and computes a
  0–100 risk score.
- `app.py` — Streamlit dashboard: KPIs, risk banner, breakdown chart,
  leakage-over-time chart, and drill-down tables per category, with CSV
  download of the leaks report.
- `requirements.txt` — dependencies.

## Setup

```bash
pip install -r requirements.txt
python generate_sample_data.py    # creates the 3 demo CSVs
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Using your own data

In the dashboard sidebar, choose "Upload my own CSVs" and provide three files
matching this schema:

**pos_records.csv**: `txn_id, timestamp, product, qty, unit_price, amount, payment_method`

**bank_records.csv**: `bank_ref, matched_txn_id, settlement_time, amount`
(`matched_txn_id` blank/NaN for unmatched bank credits)

**inventory_records.csv**: `date, product, units_sold_pos, units_depleted_inventory`

## How reconciliation works

1. **Settlement matching (POS ↔ Bank)** — every non-cash POS transaction is
   looked up in the bank feed by `txn_id`:
   - No match found → **Leak** (missing settlement)
   - Bank amount < POS amount → **Leak** (short settlement / possible skim)
   - Bank amount > POS amount → **Mismatch**
   - Settled but delayed beyond threshold → **Anomaly**
   - Otherwise → **Matched**
   - Bank credits with no POS record behind them → **Anomaly** (phantom credit)

2. **Inventory matching (POS ↔ Inventory)** — daily units sold (POS) are
   compared to units depleted from inventory per product:
   - Inventory depletion > POS sales → **Leak** (shrinkage/theft/damage)
   - Inventory depletion < POS sales → **Mismatch** (recording error)

3. **Risk scoring** — a 0–100 weighted blend of leak rate (financial impact,
   up to 60 pts), mismatch count (up to 25 pts), and anomaly count (up to
   15 pts), bucketed into Low / Medium / High.

All thresholds (settlement tolerance, delay window, shrinkage tolerance) are
configurable constants at the top of `reconciliation_engine.py`.

## Next steps toward the full product

- Replace CSV ingestion with a live Open Banking API connector.
- Persist results to a database instead of recomputing per session.
- Add merchant-level auth and multi-tenant support (for the ALAT integration
  roadmap step).
- Add automated alerting (email/SMS/webhook) when new leaks are detected.
- Expand risk scoring with historical trend data per merchant.
