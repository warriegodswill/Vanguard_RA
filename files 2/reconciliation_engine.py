"""
Vanguard RA - Reconciliation Engine
------------------------------------
Ingests Bank, POS, and Inventory records and reconciles them into:
    Matched | Mismatch | Leak | Anomaly

Also produces a per-merchant risk score and a summary report,
matching the pipeline described in the pitch deck:

    Open Banking API / CSV
        -> Transaction Processing
        -> Reconciliation Engine
        -> Matched | Mismatch | Leak | Anomaly
        -> Dashboard, Alerts & Reports
"""

from dataclasses import dataclass, field
from datetime import timedelta
import pandas as pd
import numpy as np

from anomaly_detector import AnomalyDetector
from recovery_tracker import RecoveryTracker


# ---------------------------------------------------------------------------
# Config / thresholds — tune these per merchant risk appetite
# ---------------------------------------------------------------------------

SETTLEMENT_SHORTFALL_TOLERANCE = 0.5   # naira tolerance for "exact match"
SETTLEMENT_DELAY_ANOMALY_HOURS = 24    # settlement later than this = anomaly
INVENTORY_SHRINKAGE_TOLERANCE = 0      # units difference allowed before flag


@dataclass
class ReconciliationResult:
    matched: pd.DataFrame
    mismatches: pd.DataFrame
    leaks: pd.DataFrame
    anomalies: pd.DataFrame
    ml_anomalies: pd.DataFrame = field(default_factory=pd.DataFrame)
    recovery_tracker: object = None
    summary: dict = field(default_factory=dict)


class ReconciliationEngine:
    def __init__(self, pos_df: pd.DataFrame, bank_df: pd.DataFrame, inventory_df: pd.DataFrame):
        self.pos_df = pos_df.copy()
        self.bank_df = bank_df.copy()
        self.inventory_df = inventory_df.copy()

        self.pos_df["timestamp"] = pd.to_datetime(self.pos_df["timestamp"])
        self.bank_df["settlement_time"] = pd.to_datetime(self.bank_df["settlement_time"])

    # ------------------------------------------------------------------
    # Stage 1: POS <-> Bank settlement reconciliation
    # ------------------------------------------------------------------
    def _reconcile_settlements(self):
        rows = []
        settleable_pos = self.pos_df[self.pos_df["payment_method"] != "Cash"]
        bank_by_txn = self.bank_df.set_index("matched_txn_id")

        for _, pos_row in settleable_pos.iterrows():
            txn_id = pos_row["txn_id"]

            if txn_id not in bank_by_txn.index:
                # No settlement found at all -> revenue leak
                rows.append({
                    "txn_id": txn_id,
                    "type": "leak",
                    "reason": "Missing bank settlement",
                    "pos_amount": pos_row["amount"],
                    "bank_amount": 0.0,
                    "leaked_amount": pos_row["amount"],
                    "timestamp": pos_row["timestamp"],
                })
                continue

            bank_row = bank_by_txn.loc[txn_id]
            if isinstance(bank_row, pd.DataFrame):
                bank_row = bank_row.iloc[0]

            diff = round(pos_row["amount"] - bank_row["amount"], 2)
            delay_hrs = (bank_row["settlement_time"] - pos_row["timestamp"]).total_seconds() / 3600

            if abs(diff) > SETTLEMENT_SHORTFALL_TOLERANCE:
                rows.append({
                    "txn_id": txn_id,
                    "type": "leak" if diff > 0 else "mismatch",
                    "reason": "Short settlement (possible fee skim / leakage)" if diff > 0
                              else "Bank amount exceeds POS amount",
                    "pos_amount": pos_row["amount"],
                    "bank_amount": bank_row["amount"],
                    "leaked_amount": max(diff, 0),
                    "timestamp": pos_row["timestamp"],
                })
            elif delay_hrs > SETTLEMENT_DELAY_ANOMALY_HOURS:
                rows.append({
                    "txn_id": txn_id,
                    "type": "anomaly",
                    "reason": f"Settlement delayed {delay_hrs:.1f}h (> {SETTLEMENT_DELAY_ANOMALY_HOURS}h threshold)",
                    "pos_amount": pos_row["amount"],
                    "bank_amount": bank_row["amount"],
                    "leaked_amount": 0.0,
                    "timestamp": pos_row["timestamp"],
                })
            else:
                rows.append({
                    "txn_id": txn_id,
                    "type": "matched",
                    "reason": "Settled correctly",
                    "pos_amount": pos_row["amount"],
                    "bank_amount": bank_row["amount"],
                    "leaked_amount": 0.0,
                    "timestamp": pos_row["timestamp"],
                })

        # Phantom bank credits with no POS record behind them
        unmatched_bank = self.bank_df[self.bank_df["matched_txn_id"].isna()]
        for _, bank_row in unmatched_bank.iterrows():
            rows.append({
                "txn_id": bank_row["bank_ref"],
                "type": "anomaly",
                "reason": "Bank credit with no matching POS transaction",
                "pos_amount": 0.0,
                "bank_amount": bank_row["amount"],
                "leaked_amount": 0.0,
                "timestamp": bank_row["settlement_time"],
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Stage 2: POS <-> Inventory reconciliation
    # ------------------------------------------------------------------
    def _reconcile_inventory(self):
        rows = []
        for _, row in self.inventory_df.iterrows():
            diff = row["units_depleted_inventory"] - row["units_sold_pos"]
            if diff > INVENTORY_SHRINKAGE_TOLERANCE:
                rows.append({
                    "date": row["date"],
                    "product": row["product"],
                    "type": "leak",
                    "reason": f"Inventory shrinkage: {diff} unit(s) unaccounted for",
                    "units_sold_pos": row["units_sold_pos"],
                    "units_depleted_inventory": row["units_depleted_inventory"],
                })
            elif diff < -INVENTORY_SHRINKAGE_TOLERANCE:
                rows.append({
                    "date": row["date"],
                    "product": row["product"],
                    "type": "mismatch",
                    "reason": f"Inventory under-recorded vs POS by {abs(diff)} unit(s)",
                    "units_sold_pos": row["units_sold_pos"],
                    "units_depleted_inventory": row["units_depleted_inventory"],
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run(self, run_ml_detection: bool = True) -> ReconciliationResult:
        settlement_results = self._reconcile_settlements()
        inventory_results = self._reconcile_inventory()

        matched = settlement_results[settlement_results["type"] == "matched"]
        mismatches = pd.concat([
            settlement_results[settlement_results["type"] == "mismatch"],
            inventory_results[inventory_results["type"] == "mismatch"],
        ], ignore_index=True)
        leaks = pd.concat([
            settlement_results[settlement_results["type"] == "leak"],
            inventory_results[inventory_results["type"] == "leak"],
        ], ignore_index=True)
        anomalies = settlement_results[settlement_results["type"] == "anomaly"]

        # Real ML layer: statistical anomalies in settlement behavior that
        # the deterministic rules above are not designed to catch (rules
        # catch missing/short/delayed against FIXED thresholds; this catches
        # what's unusual for THIS merchant's own history).
        ml_anomalies = pd.DataFrame()
        if run_ml_detection:
            detector = AnomalyDetector(contamination=0.05)
            ml_anomalies = detector.fit_predict(self.pos_df, self.bank_df)

        total_leak_amount = leaks["leaked_amount"].sum() if "leaked_amount" in leaks.columns else 0.0
        total_pos_revenue = self.pos_df["amount"].sum()

        risk_score = self._compute_risk_score(
            total_leak_amount=total_leak_amount,
            total_pos_revenue=total_pos_revenue,
            n_mismatches=len(mismatches),
            n_anomalies=len(anomalies) + len(ml_anomalies),
        )

        # Recovery tracking: every leak is registered as "detected" only.
        # Nothing here is billable until a human moves it through
        # disputed -> recovered. This is what keeps the success-fee number
        # honest instead of just restating total_leaked_amount.
        tracker = RecoveryTracker()
        tracker.register_leaks_from_df(leaks)
        recovery_summary = tracker.summary()

        summary = {
            "total_pos_revenue": round(total_pos_revenue, 2),
            "total_leaked_amount": round(total_leak_amount, 2),
            "leak_percentage": round(100 * total_leak_amount / total_pos_revenue, 3) if total_pos_revenue else 0,
            "n_matched": len(matched),
            "n_mismatches": len(mismatches),
            "n_leaks": len(leaks),
            "n_anomalies": len(anomalies),
            "n_ml_anomalies": len(ml_anomalies),
            "risk_score": risk_score,
            "risk_band": self._risk_band(risk_score),
            # NOTE: this is a THEORETICAL fee if every leak were recovered.
            # It is NOT what should ever be invoiced. Use
            # recovery_tracker.summary()["total_success_fees"] for billing —
            # that number only reflects money actually confirmed recovered.
            "theoretical_success_fee_if_fully_recovered": round(total_leak_amount * 0.001, 2),
            "recovery_status": recovery_summary,
        }

        return ReconciliationResult(
            matched=matched,
            mismatches=mismatches,
            leaks=leaks,
            anomalies=anomalies,
            ml_anomalies=ml_anomalies,
            recovery_tracker=tracker,
            summary=summary,
        )

    @staticmethod
    def _compute_risk_score(total_leak_amount, total_pos_revenue, n_mismatches, n_anomalies):
        """
        Simple, explainable 0-100 risk score. Weighted blend of:
          - leak rate (financial impact)
          - mismatch frequency
          - anomaly frequency
        Higher = riskier merchant / more urgent attention needed.
        """
        leak_rate = (total_leak_amount / total_pos_revenue) if total_pos_revenue else 0
        leak_component = min(leak_rate * 100 * 4, 60)          # up to 60 pts
        mismatch_component = min(n_mismatches * 2, 25)          # up to 25 pts
        anomaly_component = min(n_anomalies * 1.5, 15)          # up to 15 pts
        return round(leak_component + mismatch_component + anomaly_component, 1)

    @staticmethod
    def _risk_band(score):
        if score < 20:
            return "Low"
        elif score < 50:
            return "Medium"
        else:
            return "High"


if __name__ == "__main__":
    pos_df = pd.read_csv("pos_records.csv")
    bank_df = pd.read_csv("bank_records.csv")
    inventory_df = pd.read_csv("inventory_records.csv")

    engine = ReconciliationEngine(pos_df, bank_df, inventory_df)
    result = engine.run()

    print("=== Vanguard RA Reconciliation Summary ===")
    for k, v in result.summary.items():
        print(f"{k}: {v}")
