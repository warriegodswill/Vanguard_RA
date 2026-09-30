"""
Vanguard RA - Recovery Tracker
--------------------------------
This is the module the business model was missing.

Detecting a leak is not recovering it. A missing bank settlement still
requires the merchant (or Vanguard RA, on their behalf) to dispute it with
the bank/PSP. Inventory shrinkage by a staff member requires the owner to
actually act — confront, investigate, adjust process. None of that happens
automatically just because the reconciliation engine found a gap.

The 0.1%-of-recovered-leakage fee only makes sense, and is only honest,
if there's a real lifecycle behind "recovered": someone has to log that
the money actually came back before it counts. This module enforces that.

Lifecycle:
    detected -> disputed -> recovered
                          -> written_off  (merchant/Vanguard couldn't recover it)
                          -> partial      (some but not all of it came back)

Only amounts in `recovered` or the recovered portion of `partial` status
are eligible for the success fee. Everything else is informational —
detection value, not billable value.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import pandas as pd
import json


class RecoveryStatus(str, Enum):
    DETECTED = "detected"
    DISPUTED = "disputed"
    RECOVERED = "recovered"
    PARTIAL = "partial"
    WRITTEN_OFF = "written_off"


SUCCESS_FEE_RATE = 0.001  # 0.1%, per the pitch deck


@dataclass
class RecoveryRecord:
    leak_id: str
    txn_id: str
    detected_amount: float
    status: RecoveryStatus = RecoveryStatus.DETECTED
    recovered_amount: float = 0.0
    notes: str = ""
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def billable_amount(self) -> float:
        """The ONLY number the success fee should ever be calculated on."""
        if self.status in (RecoveryStatus.RECOVERED, RecoveryStatus.PARTIAL):
            return self.recovered_amount
        return 0.0

    @property
    def success_fee(self) -> float:
        return round(self.billable_amount * SUCCESS_FEE_RATE, 2)

    def to_dict(self):
        d = self.__dict__.copy()
        d["status"] = self.status.value
        d["billable_amount"] = self.billable_amount
        d["success_fee"] = self.success_fee
        return d


class RecoveryTracker:
    """
    In-memory tracker for a single merchant's leak recovery lifecycle.
    Swap the storage layer for a real DB in production — the state
    machine and fee logic below is what actually matters and should
    not change when you do.
    """

    VALID_TRANSITIONS = {
        RecoveryStatus.DETECTED: {RecoveryStatus.DISPUTED, RecoveryStatus.WRITTEN_OFF},
        RecoveryStatus.DISPUTED: {RecoveryStatus.RECOVERED, RecoveryStatus.PARTIAL, RecoveryStatus.WRITTEN_OFF},
        RecoveryStatus.PARTIAL: {RecoveryStatus.RECOVERED, RecoveryStatus.WRITTEN_OFF},
        RecoveryStatus.RECOVERED: set(),      # terminal
        RecoveryStatus.WRITTEN_OFF: set(),    # terminal
    }

    def __init__(self):
        self.records: dict[str, RecoveryRecord] = {}

    def register_leak(self, leak_id: str, txn_id: str, detected_amount: float) -> RecoveryRecord:
        record = RecoveryRecord(leak_id=leak_id, txn_id=txn_id, detected_amount=detected_amount)
        self.records[leak_id] = record
        return record

    def register_leaks_from_df(self, leaks_df: pd.DataFrame) -> None:
        """Bulk-register every leak the reconciliation engine found."""
        for i, row in leaks_df.reset_index(drop=True).iterrows():
            leak_id = f"LEAK-{row.get('txn_id', i)}-{i}"
            amount = row.get("leaked_amount", 0.0) or 0.0
            if amount > 0:
                self.register_leak(leak_id, str(row.get("txn_id", "")), float(amount))

    def update_status(
        self, leak_id: str, new_status: RecoveryStatus, recovered_amount: float = None, notes: str = ""
    ) -> RecoveryRecord:
        if leak_id not in self.records:
            raise KeyError(f"Unknown leak_id: {leak_id}")

        record = self.records[leak_id]
        current = record.status

        if new_status not in self.VALID_TRANSITIONS.get(current, set()):
            raise ValueError(
                f"Invalid transition: {current.value} -> {new_status.value}. "
                f"Allowed from {current.value}: "
                f"{[s.value for s in self.VALID_TRANSITIONS.get(current, set())]}"
            )

        record.status = new_status
        record.last_updated = datetime.now(timezone.utc).isoformat()
        if notes:
            record.notes = notes

        if new_status == RecoveryStatus.RECOVERED:
            record.recovered_amount = (
                recovered_amount if recovered_amount is not None else record.detected_amount
            )
        elif new_status == RecoveryStatus.PARTIAL:
            if recovered_amount is None:
                raise ValueError("PARTIAL status requires an explicit recovered_amount")
            record.recovered_amount = recovered_amount
        elif new_status == RecoveryStatus.WRITTEN_OFF:
            record.recovered_amount = 0.0

        return record

    def summary(self) -> dict:
        if not self.records:
            return {
                "total_detected": 0.0, "total_recovered": 0.0, "total_written_off": 0.0,
                "total_success_fees": 0.0, "recovery_rate_pct": 0.0, "n_leaks": 0,
                "by_status": {},
            }

        total_detected = sum(r.detected_amount for r in self.records.values())
        total_recovered = sum(r.billable_amount for r in self.records.values())
        total_written_off = sum(
            r.detected_amount for r in self.records.values() if r.status == RecoveryStatus.WRITTEN_OFF
        )
        total_fees = sum(r.success_fee for r in self.records.values())

        by_status = {}
        for status in RecoveryStatus:
            matching = [r for r in self.records.values() if r.status == status]
            by_status[status.value] = {
                "count": len(matching),
                "amount": round(sum(r.detected_amount for r in matching), 2),
            }

        return {
            "total_detected": round(total_detected, 2),
            "total_recovered": round(total_recovered, 2),
            "total_written_off": round(total_written_off, 2),
            "total_success_fees": round(total_fees, 2),
            "recovery_rate_pct": round(100 * total_recovered / total_detected, 2) if total_detected else 0.0,
            "n_leaks": len(self.records),
            "by_status": by_status,
        }

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=[
                "leak_id", "txn_id", "detected_amount", "status",
                "recovered_amount", "billable_amount", "success_fee", "notes",
                "detected_at", "last_updated",
            ])
        return pd.DataFrame([r.to_dict() for r in self.records.values()])

    def export_json(self) -> str:
        return json.dumps([r.to_dict() for r in self.records.values()], indent=2)


if __name__ == "__main__":
    # Demo: register leaks from the reconciliation engine, walk a few through
    # the lifecycle, and show that the success fee only ever reflects what's
    # actually been recovered — not what was merely detected.
    from reconciliation_engine import ReconciliationEngine

    pos_df = pd.read_csv("pos_records.csv")
    bank_df = pd.read_csv("bank_records.csv")
    inventory_df = pd.read_csv("inventory_records.csv")

    engine = ReconciliationEngine(pos_df, bank_df, inventory_df)
    result = engine.run()

    tracker = RecoveryTracker()
    tracker.register_leaks_from_df(result.leaks)

    leak_ids = list(tracker.records.keys())
    print(f"Registered {len(leak_ids)} leaks for recovery tracking.\n")

    # Simulate realistic outcomes: not everything gets recovered
    for i, leak_id in enumerate(leak_ids):
        record = tracker.records[leak_id]
        tracker.update_status(leak_id, RecoveryStatus.DISPUTED, notes="Merchant raised dispute with bank")
        if i % 4 == 0:
            tracker.update_status(leak_id, RecoveryStatus.WRITTEN_OFF, notes="Bank rejected dispute")
        elif i % 4 == 1:
            partial = round(record.detected_amount * 0.6, 2)
            tracker.update_status(leak_id, RecoveryStatus.PARTIAL, recovered_amount=partial)
        else:
            tracker.update_status(leak_id, RecoveryStatus.RECOVERED)

    print("=== Recovery Summary (this is what should actually be billed) ===")
    for k, v in tracker.summary().items():
        print(f"{k}: {v}")
