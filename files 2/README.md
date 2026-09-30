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
  every record as **Matched / Mismatch / Leak / Rule Anomaly**, runs the ML
  anomaly detector, initializes recovery tracking, and computes a 0–100
  risk score.
- `anomaly_detector.py` — **the actual "AI" in AI-powered.** An Isolation
  Forest fit fresh on each merchant's own settlement history, flagging
  transactions that are statistically unusual *for that merchant* — not
  against a generic fixed rule. This is separate from, and complements,
  the deterministic bookkeeping rules in the reconciliation engine, which
  are not ML and shouldn't be marketed as such.
- `recovery_tracker.py` — **the missing piece from the original pricing
  model.** Manages a leak's lifecycle (`detected → disputed → recovered /
  partial / written_off`) with an enforced state machine, and computes the
  0.1% success fee **only on confirmed-recovered amounts** — never on
  merely-detected leaks. Detecting a leak and recovering it are different
  events; this module is what keeps that distinction honest and billable.
- `app.py` — Streamlit dashboard: KPIs (now split into detected vs.
  actually-recovered vs. billable fee), risk banner, breakdown chart,
  leakage-over-time chart, a dedicated ML Anomalies tab, an interactive
  Recovery Tracking tab for walking leaks through their lifecycle, and
  drill-down tables per category with CSV/JSON export.
- `requirements.txt` — dependencies (now includes `scikit-learn`).

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

## The recovery model (why billing ≠ detection)

`total_leaked_amount` is what the reconciliation engine *found*. It is not
what should ever be invoiced. Every leak enters `recovery_tracker.py` as
`detected` and is worth ₦0 in fees until someone actually moves it through
the lifecycle:

```
detected -> disputed -> recovered   (full amount confirmed back)
                      -> partial    (some of it confirmed back)
                      -> written_off (bank/merchant couldn't recover it)
```

The state machine enforces valid transitions only (e.g. you can't jump
straight from `detected` to `recovered`, and terminal states can't be
reopened). The 0.1% success fee is calculated **only** on `recovered` +
the confirmed portion of `partial` — never on the raw detected total. In
the bundled demo data, ₦692,960 is detected but the fee stays at ₦0 until
you actually walk leaks through the dashboard's Recovery Tracking tab.

## Two anomaly layers — know the difference

- **Rule Anomalies** (in `reconciliation_engine.py`): deterministic
  thresholds — e.g. "settlement delayed more than 24h." This is
  bookkeeping logic, not AI, and shouldn't be marketed as such.
- **ML Anomalies** (in `anomaly_detector.py`): an Isolation Forest fit on
  each merchant's own transaction history, flagging what's statistically
  unusual *for them* — the actual machine-learning layer. It degrades
  gracefully (returns nothing) if a merchant has under ~20 settled
  transactions to learn from, rather than overfitting noise.

## Next steps toward the full product

- Replace CSV ingestion with a live Open Banking API connector.
- Persist results to a database instead of recomputing per session.
- Add merchant-level auth and multi-tenant support (for the ALAT integration
  roadmap step).
- Add automated alerting (email/SMS/webhook) when new leaks are detected.
- Expand risk scoring with historical trend data per merchant.
