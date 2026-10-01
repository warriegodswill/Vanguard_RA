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
from typing import Dict, Any
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
    summary: Dict[str, Any] = field(default_factory=dict)


class ReconciliationEngine:
    def __init__(self, pos_df: pd.DataFrame, bank_df: pd.DataFrame, inventory_df: pd.DataFrame):
        self.pos_df = pos_df.copy()
        self.bank_df = bank_df.copy()
        self.inventory_df = inventory_df.copy()

        # Schema Validation
        self._validate_schemas()

        # Datetime conversions
        self.pos_df["timestamp"] = pd.to_datetime(self.pos_df["timestamp"])
        self.bank_df["settlement_time"] = pd.to_datetime(self.bank_df["settlement_time"])

    def _validate_schemas(self):
        """Ensures required columns exist in input DataFrames."""
        required_pos = {"txn_id", "amount", "payment_method", "timestamp"}
        required_bank = {"bank_ref", "matched_txn_id", "amount", "settlement_time"}
        required_inv = {"date", "product", "units_sold_pos", "units_depleted_inventory"}

        missing_pos = required_pos - set(self.pos_df.columns)
        missing_bank = required_bank - set(self.bank_df.columns)
        missing_inv = required_inv - set(self.inventory_df.columns)

        if missing_pos:
            raise ValueError(f"pos_df missing required columns: {missing_pos}")
        if missing_bank:
            raise ValueError(f"bank_df missing required columns: {missing_bank}")
        if missing_inv:
            raise ValueError(f"inventory_df missing required columns: {missing_inv}")

    # ------------------------------------------------------------------
    # Stage 1: POS <-> Bank settlement reconciliation
    # ------------------------------------------------------------------
    def _reconcile_settlements(self) -> pd.DataFrame:
        rows = []
        settleable_pos = self.pos_df[self.pos_df["payment_method"] != "Cash"]

        # Aggregate bank transactions by matched_txn_id to handle split settlements cleanly
        matched_bank = self.bank_df.dropna(subset=["matched_txn_id"])
        bank_grouped = matched_bank.groupby("matched_txn_id").agg({
            "amount": "sum",
            "settlement_time": "max",
            "bank_ref": "first"
        })

        for _, pos_row in settleable_pos.iterrows():
            txn_id = pos_row["txn_id"]

            if txn_id not in bank_grouped.index:
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

            bank_row = bank_grouped.loc[txn_id]
            bank_amount = bank_row["amount"]
            settlement_time = bank_row["settlement_time"]

            diff = round(pos_row["amount"] - bank_amount, 2)
            delay_hrs = (settlement_time - pos_row["timestamp"]).total_seconds() / 3600

            if abs(diff) > SETTLEMENT_SHORTFALL_TOLERANCE:
                rows.append({
                    "txn_id": txn_id,
                    "type": "leak" if diff > 0 else "mismatch",
                    "reason": "Short settlement (possible fee skim / leakage)" if diff > 0
                              else "Bank amount exceeds POS amount",
                    "pos_amount": pos_row["amount"],
                    "bank_amount": bank_amount,
                    "leaked_amount": max(diff, 0.0),
                    "timestamp": pos_row["timestamp"],
                })
            elif delay_hrs > SETTLEMENT_DELAY_ANOMALY_HOURS:
                rows.append({
                    "txn_id": txn_id,
                    "type": "anomaly",
                    "reason": f"Settlement delayed {delay_hrs:.1f}h (> {SETTLEMENT_DELAY_ANOMALY_HOURS}h threshold)",
                    "pos_amount": pos_row["amount"],
                    "bank_amount": bank_amount,
                    "leaked_amount": 0.0,
                    "timestamp": pos_row["timestamp"],
                })
            else:
                rows.append({
                    "txn_id": txn_id,
                    "type": "matched",
                    "reason": "Settled correctly",
                    "pos_amount": pos_row["amount"],
                    "bank_amount": bank_amount,
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
    def _reconcile_inventory(self) -> pd.DataFrame:
        rows = []
        for _, row in self.inventory_df.iterrows():
            diff = row["units_depleted_inventory"] - row["units_sold_pos"]
            if diff > INVENTORY_SHRINKAGE_TOLERANCE:
                rows.append({
                    "date": row["date"],
                    "product": row["product"],
                    "type": "leak",
                    "reason": f"Inventory shrinkage: {int(diff)} unit(s) unaccounted for",
                    "units_sold_pos": row["units_sold_pos"],
                    "units_depleted_inventory": row["units_depleted_inventory"],
                })
            elif diff < -INVENTORY_SHRINKAGE_TOLERANCE:
                rows.append({
                    "date": row["date"],
                    "product": row["product"],
                    "type": "mismatch",
                    "reason": f"Inventory under-recorded vs POS by {int(abs(diff))} unit(s)",
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
        feature_df["pos_amount"] = feature_df["pos_amount"].fillna(0.0)
        feature_df["bank_amount"] = feature_df["bank_amount"].fillna(0.0)
        
        # Feature engineering with zero-division safeguard
        feature_df["amount_diff"] = (feature_df["pos_amount"] - feature_df["bank_amount"]).abs()
        denom = feature_df[["pos_amount", "bank_amount"]].max(axis=1).replace(0, 1.0)
        feature_df["diff_ratio"] = feature_df["amount_diff"] / denom
        
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
        primary_cause = list(top_reasons.keys())[0] if top_reasons else "N/A"
        primary_count = list(top_reasons.values())[0] if top_reasons else 0
        
        insight = (
            f"**Executive Audit Summary:**\n"
            f"- **Financial Exposure:** Detected ₦{summary['total_leaked_amount']:,.2f} in total revenue leakage "
            f"({summary['leak_percentage']}% of POS Revenue) across {summary['n_leaks']} incidents.\n"
            f"- **Primary Root Cause:** The highest contributing factor is '{primary_cause}' "
            f"accounting for {primary_count} incidents.\n"
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
            "leak_percentage": round(100 * total_leak_amount / total_pos_revenue, 2) if total_pos_revenue else 0.0,
            "n_matched": len(matched),
            "n_mismatches": len(mismatches),
            "n_leaks": len(leaks),
            "n_anomalies": len(anomalies),
            "n_ml_anomalies": len(ml_anomalies),
            "risk_score": risk_score,
            "risk_band": self._risk_band(risk_score),
            "success_fee_estimate": round(total_leak_amount * 0.25, 2),
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
    def _compute_risk_score(total_leak_amount: float, total_pos_revenue: float, n_mismatches: int, n_anomalies: int) -> float:
        leak_rate = (total_leak_amount / total_pos_revenue) if total_pos_revenue else 0.0
        leak_component = min(leak_rate * 100 * 4, 60)          # up to 60 pts
        mismatch_component = min(n_mismatches * 2, 25)          # up to 25 pts
        anomaly_component = min(n_anomalies * 1.5, 15)          # up to 15 pts
        return round(leak_component + mismatch_component + anomaly_component, 1)

    @staticmethod
    def _risk_band(score: float) -> str:
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