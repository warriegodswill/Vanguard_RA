"""
Vanguard RA - Column Mapper
------------------------------
Lets a merchant's CSV, with whatever headers THEY use, get mapped to the
exact schema ReconciliationEngine expects, instead of assuming their file
already matches.
"""

import pandas as pd
import streamlit as st

REQUIRED_SCHEMAS = {
    "pos": {
        "txn_id": "Transaction ID",
        "timestamp": "Date/Time",
        "amount": "Amount",
        "payment_method": "Payment Method",
    },
    "bank": {
        "bank_ref": "Bank Reference",
        "matched_txn_id": "Matching Transaction ID",
        "settlement_time": "Settlement Date/Time",
        "amount": "Settled Amount",
    },
    "inventory": {
        "date": "Date",
        "product": "Product Name",
        "units_sold_pos": "Units Sold (POS)",
        "units_depleted_inventory": "Units Depleted (Inventory)",
    },
}

ALIASES = {
    "txn_id": ["transaction id", "txn id", "order id", "reference"],
    "timestamp": ["date", "time", "datetime", "transaction date"],
    "amount": ["amount", "value", "total"],
    "payment_method": ["payment method", "method", "channel"],
    "bank_ref": ["bank reference", "ref", "reference", "bank_ref"],
    "matched_txn_id": ["transaction id", "txn id", "matching id", "matched_txn_id"],
    "settlement_time": ["settlement date", "settled on", "value date", "settlement_time"],
    "date": ["date", "day"],
    "product": ["product", "item", "sku", "name", "description"],
    "units_sold_pos": ["sold", "qty sold", "quantity sold", "pos qty"],
    "units_depleted_inventory": ["stock", "depleted", "inventory count", "on hand"],
}


def _best_guess(field_key: str, columns: list) -> str | None:
    candidates = ALIASES.get(field_key, [field_key])
    for col in columns:
        col_norm = col.lower().strip()
        if any(alias in col_norm for alias in candidates):
            return col
    return None


def map_columns_ui(uploaded_df: pd.DataFrame, file_type: str) -> pd.DataFrame | None:
    """
    file_type: one of "pos", "bank", "inventory" — looks up REQUIRED_SCHEMAS.
    Returns a renamed dataframe matching the engine's expected schema,
    or None if the merchant hasn't finished mapping yet.
    """
    required_fields = REQUIRED_SCHEMAS[file_type]
    their_columns = list(uploaded_df.columns)

    with st.expander(f"🔧 Map your {file_type.upper()} columns", expanded=True):
        st.caption("We couldn't assume your file's headers — confirm what each column means.")
        mapping = {}
        for field_key, field_label in required_fields.items():
            guess = _best_guess(field_key, their_columns)
            mapping[field_key] = st.selectbox(
                f"'{field_label}' is which column?",
                options=["-- none --"] + their_columns,
                index=(their_columns.index(guess) + 1) if guess else 0,
                key=f"{file_type}_{field_key}_mapper",
            )

        # 1. Check for unselected fields
        missing = [k for k, v in mapping.items() if v == "-- none --"]
        if missing:
            st.warning(f"Still need to map: {', '.join(missing)}")
            return None

        # 2. Check for duplicate column selections
        selected_cols = [v for v in mapping.values() if v != "-- none --"]
        if len(selected_cols) != len(set(selected_cols)):
            st.error("Duplicate mapping detected: Each required field must map to a unique column in your CSV.")
            return None

        # 3. Perform safe renaming and column duplication if necessary
        renamed_df = pd.DataFrame()
        for target_col, source_col in mapping.items():
            if source_col in uploaded_df.columns:
                renamed_df[target_col] = uploaded_df[source_col]

        # 4. Use reindex to guarantee no KeyError is thrown
        final_df = renamed_df.reindex(columns=list(required_fields.keys()))

        st.success(f"{file_type.upper()} columns mapped successfully.")
        return final_df


def validate_schema(df: pd.DataFrame, file_type: str) -> list[str]:
    errors = []
    numeric_fields = {"amount", "units_sold_pos", "units_depleted_inventory"}
    date_fields = {"timestamp", "settlement_time", "date"}
    nullable_fields = {"matched_txn_id"}

    for col in df.columns:
        if col in nullable_fields:
            continue
        if df[col].isnull().any():
            errors.append(f"'{col}' has missing values in some rows.")
        if col in numeric_fields and not pd.api.types.is_numeric_dtype(pd.to_numeric(df[col], errors="coerce")):
            errors.append(f"'{col}' should be numeric — check for text or currency symbols.")
        if col in date_fields:
            try:
                pd.to_datetime(df[col], errors="raise")
            except Exception:
                errors.append(f"'{col}' couldn't be parsed as a date — check the format.")
    return errors