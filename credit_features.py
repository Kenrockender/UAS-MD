"""
credit_features.py
==================
Shared feature definitions and the raw-data cleaning transformer for the
Credit Score Classification project.

This module is imported by:
  - pipeline.py   (training)
  - inference.py  (inferencing)
  - app.py        (Streamlit deployment)

IMPORTANT: the trained model pickle stores a reference to ``CreditDataCleaner``
defined here, so this file MUST be importable wherever the model is loaded
(local, Streamlit Cloud, or AWS).
"""

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

# Target column in the raw dataset
TARGET = "Credit_Score"
TARGET_CLASSES = ["Poor", "Standard", "Good"]  # ordered worst -> best

# Identifier / leakage columns that are dropped before modelling
DROP_COLS = ["Unnamed: 0", "ID", "Customer_ID", "Month", "Name", "SSN", "Type_of_Loan"]

# Numeric features that arrive as dirty strings (trailing "_", blanks, etc.)
STR_NUMERIC_COLS = [
    "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly",
    "Monthly_Balance",
]

# Numeric features that are already (mostly) numeric in the raw file
RAW_NUMERIC_COLS = [
    "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
    "Interest_Rate", "Delay_from_due_date", "Num_Credit_Inquiries",
    "Credit_Utilization_Ratio", "Total_EMI_per_month",
]

# Engineered numeric feature derived from "Credit_History_Age"
ENGINEERED_NUMERIC_COLS = ["Credit_History_Age_Months"]

# Final numeric / categorical feature lists consumed by the preprocessor
NUM_COLS = STR_NUMERIC_COLS + RAW_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS
CAT_COLS = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]
FEATURE_COLS = NUM_COLS + CAT_COLS

# Placeholder / garbage tokens that mean "missing"
GARBAGE_TOKENS = {"", "nan", "NaN", "_", "_______", "!@9#%8", "#F%$D@*&8", "__-333333333333333333333333333__"}

# Categorical placeholders to blank out per column
CAT_PLACEHOLDERS = {
    "Occupation": {"_______"},
    "Credit_Mix": {"_"},
    "Payment_Behaviour": {"!@9#%8"},
    "Payment_of_Min_Amount": set(),  # Yes / No / NM are all valid
}

# Plausible value ranges. Anything outside -> NaN (later imputed).
VALUE_BOUNDS = {
    "Age": (0, 100),
    "Annual_Income": (0, 300_000),
    "Monthly_Inhand_Salary": (0, 50_000),
    "Num_Bank_Accounts": (0, 20),
    "Num_Credit_Card": (0, 20),
    "Interest_Rate": (0, 50),
    "Num_of_Loan": (0, 20),
    "Delay_from_due_date": (0, 100),
    "Num_of_Delayed_Payment": (0, 100),
    "Changed_Credit_Limit": (-50, 50),
    "Num_Credit_Inquiries": (0, 50),
    "Outstanding_Debt": (0, 10_000),
    "Credit_Utilization_Ratio": (0, 100),
    "Total_EMI_per_month": (0, 100_000),
    "Amount_invested_monthly": (0, 10_000),
    "Monthly_Balance": (0, 100_000),
    "Credit_History_Age_Months": (0, 1_200),
}


def _to_number(series: pd.Series) -> pd.Series:
    """Strip stray characters from a dirty string column and coerce to float."""
    cleaned = (
        series.astype(str)
        .str.replace("_", "", regex=False)
        .str.strip()
    )
    cleaned = cleaned.mask(cleaned.isin(GARBAGE_TOKENS))
    return pd.to_numeric(cleaned, errors="coerce")


def _credit_history_to_months(series: pd.Series) -> pd.Series:
    """Convert 'X Years and Y Months' -> total months (float)."""
    def parse(val):
        if pd.isna(val):
            return np.nan
        m = re.search(r"(\d+)\s*Years?.*?(\d+)\s*Months?", str(val))
        if m:
            return int(m.group(1)) * 12 + int(m.group(2))
        return np.nan
    return series.apply(parse)


class CreditDataCleaner(BaseEstimator, TransformerMixin):
    """
    Stateless, row-independent cleaner that turns the raw credit dataset into a
    tidy numeric/categorical frame ready for imputation + encoding.

    Being row-independent is what makes the model deployable: a single customer
    record can be cleaned exactly the same way as the training set.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()

        # 1. Dirty string -> numeric
        for col in STR_NUMERIC_COLS:
            if col in df.columns:
                df[col] = _to_number(df[col])

        # 2. Already-numeric columns -> ensure float
        for col in RAW_NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 3. Engineered feature: credit history age in months
        if "Credit_History_Age" in df.columns:
            df["Credit_History_Age_Months"] = _credit_history_to_months(df["Credit_History_Age"])
        elif "Credit_History_Age_Months" in df.columns:
            df["Credit_History_Age_Months"] = pd.to_numeric(df["Credit_History_Age_Months"], errors="coerce")
        else:
            df["Credit_History_Age_Months"] = np.nan

        # 4. Range bounds -> out-of-range becomes NaN
        for col, (lo, hi) in VALUE_BOUNDS.items():
            if col in df.columns:
                df[col] = df[col].where((df[col] >= lo) & (df[col] <= hi), np.nan)

        # 5. Categorical placeholders -> NaN
        for col, bad in CAT_PLACEHOLDERS.items():
            if col in df.columns:
                df[col] = df[col].astype("string")
                if bad:
                    df[col] = df[col].where(~df[col].isin(bad), pd.NA)
                df[col] = df[col].astype(object).where(df[col].notna(), np.nan)

        # 6. Guarantee every expected feature column exists, in fixed order
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = np.nan

        return df[FEATURE_COLS]

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_COLS)
