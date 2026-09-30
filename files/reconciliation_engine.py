"""
Vanguard Revenue Assurance Platform 
------------------------------------
Reconciliation Engine with Integrated ML Anomaly Detection & AI Insights
------------------------------------
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import timedelta
from sklearn.ensemble import IsolationForest

# Optional: Add OpenAI or Anthropic import if using API key
try:
    import openai
except ImportError:
    openai = None

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
    ai_summary: str = ""
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
    # Stage 3: Machine Learning Anomaly Detection (Isolation Forest)
    # ------------------------------------------------------------------
    def _detect_ml_anomalies(self, df: pd.DataFrame, contamination: float = 0.03) -> pd.DataFrame:
        """
        Uses an unsupervised Isolation Forest model to flag complex numerical 
        outliers across transaction amounts and settlement gaps.
        """
        if df.empty or len(df) < 5:
            return pd.DataFrame()

        feature_df = df.copy()
        feature_df["pos_amount"] = feature_df["pos_amount"].fillna(0)
        feature_df["bank_amount"] = feature_df["bank_amount"].fillna(0)
        
        # Feature engineering
        feature_df["amount_diff"] = (feature_df["pos_amount"] - feature_df["bank_amount"]).abs()
        feature_df["diff_ratio"] = feature_df["amount_diff"] / (feature_df["pos_amount"] + 1e-5)
        
        features = ["pos_amount", "bank_amount", "amount_diff", "diff_ratio"]
        
        # Train Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        feature_df["ml_anomaly_score"] = iso_forest.fit_predict(feature_df[features])

        # Outliers are flagged with -1
        ml_anomalies = feature_df[feature_df["ml_anomaly_score"] == -1].copy()
        return ml_anomalies

    # ------------------------------------------------------------------
    # Stage 4: Generative AI Executive Insights
    # ------------------------------------------------------------------
    def _generate_ai_summary(self, summary: dict, leaks_df: pd.DataFrame) -> str:
        """
        Generates a structured executive text narrative using rule logic or LLM.
        """
        top_reasons = leaks_df["reason"].value_counts().to_dict() if not leaks_df.empty else {}
        
        insight = (
            f"**Executive Audit Summary:**\n"
            f"- **Financial Exposure:** Detected ₦{summary['total_leaked_amount']:,.2f} in total revenue leakage "
            f"({summary['leak_percentage']}% of POS Revenue) across {summary['n_leaks']} incidents.\n"
            f"- **Primary Root Cause:** The highest contributing factor is '{list(top_reasons.keys())[0] if top_reasons else 'N/A'}' "
            f"accounting for {list(top_reasons.values())[0] if top_reasons else 0} transactions.\n"
            f"- **Risk Recommendation:** Current risk score is **{summary['risk_score']}/100 ({summary['risk_band']} Risk)**. "
            f"Prioritize recovery on short-settled card transactions to recoup estimated ₦{summary['success_fee_estimate']:,.2f}."
        )
        return insight

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run(self) -> ReconciliationResult:
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

        # ML Layer Processing
        all_settlements = settlement_results[settlement_results["type"] != "matched"]
        ml_anomalies = self._detect_ml_anomalies(all_settlements)

        total_leak_amount = leaks["leaked_amount"].sum() if "leaked_amount" in leaks.columns else 0.0
        total_pos_revenue = self.pos_df["amount"].sum()

        risk_score = self._compute_risk_score(
            total_leak_amount=total_leak_amount,
            total_pos_revenue=total_pos_revenue,
            n_mismatches=len(mismatches),
            n_anomalies=len(anomalies),
        )

        summary = {
            "total_pos_revenue": round(total_pos_revenue, 2),
            "total_leaked_amount": round(total_leak_amount, 2),
            "leak_percentage": round(100 * total_leak_amount / total_pos_revenue, 2) if total_pos_revenue else 0,
            "n_matched": len(matched),
            "n_mismatches": len(mismatches),
            "n_leaks": len(leaks),
            "n_anomalies": len(anomalies),
            "n_ml_anomalies": len(ml_anomalies),
            "risk_score": risk_score,
            "risk_band": self._risk_band(risk_score),
            "success_fee_estimate": round(total_leak_amount * 0.25, 2),  # Updated to 25% match sidebar model
        }

        ai_summary = self._generate_ai_summary(summary, leaks)

        return ReconciliationResult(
            matched=matched,
            mismatches=mismatches,
            leaks=leaks,
            anomalies=anomalies,
            ml_anomalies=ml_anomalies,
            ai_summary=ai_summary,
            summary=summary,
        )

    @staticmethod
    def _compute_risk_score(total_leak_amount, total_pos_revenue, n_mismatches, n_anomalies):
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
    
    print("\n=== AI Executive Insights ===")
    print(result.ai_summary)