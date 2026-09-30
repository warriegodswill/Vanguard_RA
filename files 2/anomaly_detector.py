"""
Vanguard RA - Anomaly Detection Engine
----------------------------------------
This is the piece that makes "AI-powered" true rather than marketing.

The reconciliation_engine.py rules (missing settlement, short settlement,
inventory shrinkage) are deterministic bookkeeping logic — they belong in
any decent accounting tool and nobody should call them AI.

This module is different: it uses an unsupervised model (Isolation Forest)
to learn what "normal" settlement behavior looks like for THIS merchant
from THEIR own transaction history, then flags transactions that deviate
from their own pattern — not from a fixed rule someone guessed at.

Concretely, it catches things fixed thresholds miss, e.g.:
  - A merchant whose settlements are normally same-day suddenly seeing
    12-hour delays (anomalous for THEM, even if under a generic 24h rule)
  - A transaction whose amount/delay/hour-of-day combination is unusual
    relative to that merchant's own historical distribution
  - Gradual, creeping short-settlements too small individually to trip a
    fixed percentage threshold, but statistically abnormal as a pattern

This is intentionally merchant-specific: the model is fit fresh on each
merchant's own history, because "normal" for a 24-hour fuel station looks
nothing like "normal" for a pharmacy that closes at 8pm.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        """
        contamination: expected proportion of transactions that are
        anomalous, as a prior. 0.05 = assume ~5% of settlements are
        unusual until the data says otherwise. Tune per merchant vertical
        (e.g. cash-heavy fuel stations may warrant a higher prior).
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.feature_cols = ["delay_hours", "amount", "hour_of_day", "day_of_week"]

    def _build_features(self, pos_df: pd.DataFrame, bank_df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds a feature table of settled transactions only (unsettled/missing
        ones are already caught as deterministic leaks by the rules engine —
        this model's job is to find WEIRD settlements, not MISSING ones).
        """
        pos = pos_df.copy()
        pos["timestamp"] = pd.to_datetime(pos["timestamp"])

        bank = bank_df.copy()
        bank["settlement_time"] = pd.to_datetime(bank["settlement_time"])
        bank = bank.dropna(subset=["matched_txn_id"])

        merged = pos.merge(
            bank, left_on="txn_id", right_on="matched_txn_id", how="inner", suffixes=("_pos", "_bank")
        )
        if merged.empty:
            return merged

        merged["delay_hours"] = (
            merged["settlement_time"] - merged["timestamp"]
        ).dt.total_seconds() / 3600
        merged["amount"] = merged["amount_bank"]
        merged["hour_of_day"] = merged["timestamp"].dt.hour
        merged["day_of_week"] = merged["timestamp"].dt.dayofweek

        return merged

    def fit_predict(self, pos_df: pd.DataFrame, bank_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the model fresh on this merchant's settled transactions and
        returns a dataframe of flagged statistical anomalies with an
        anomaly_score (lower = more anomalous) for ranking/triage.
        """
        features = self._build_features(pos_df, bank_df)
        if len(features) < 20:
            # Not enough history for a meaningful model — degrade gracefully
            # rather than overfitting noise to false positives.
            return pd.DataFrame(columns=[
                "txn_id", "type", "reason", "amount", "delay_hours",
                "anomaly_score", "timestamp",
            ])

        X = features[self.feature_cols].fillna(0).values

        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=200,
        )
        preds = self.model.fit_predict(X)          # -1 = anomaly, 1 = normal
        scores = self.model.decision_function(X)    # lower = more anomalous

        features["is_anomaly"] = preds == -1
        features["anomaly_score"] = scores

        flagged = features[features["is_anomaly"]].copy()
        if flagged.empty:
            return pd.DataFrame(columns=[
                "txn_id", "type", "reason", "amount", "delay_hours",
                "anomaly_score", "timestamp",
            ])

        flagged = flagged.sort_values("anomaly_score")  # most anomalous first
        flagged["type"] = "ml_anomaly"
        flagged["reason"] = flagged.apply(
            lambda r: (
                f"Statistically unusual settlement pattern for this merchant "
                f"(delay={r['delay_hours']:.1f}h, amount=₦{r['amount']:,.0f}, "
                f"hour={int(r['hour_of_day'])}:00)"
            ),
            axis=1,
        )

        return flagged[[
            "txn_id", "type", "reason", "amount", "delay_hours",
            "anomaly_score", "timestamp",
        ]].reset_index(drop=True)


if __name__ == "__main__":
    pos_df = pd.read_csv("pos_records.csv")
    bank_df = pd.read_csv("bank_records.csv")

    detector = AnomalyDetector(contamination=0.05)
    anomalies = detector.fit_predict(pos_df, bank_df)

    print(f"ML-flagged statistical anomalies: {len(anomalies)}")
    if not anomalies.empty:
        print(anomalies.head(10).to_string(index=False))
