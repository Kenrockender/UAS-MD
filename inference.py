"""
inference.py
============
Inferencing code for the deployed Credit Score classifier.

Loads the serialised pipeline bundle produced by ``pipeline.py`` and exposes a
single ``CreditScorePredictor`` class plus a small CLI demo that runs one test
case per class (Good / Standard / Poor).

Usage:
    python inference.py            # runs the 3 built-in test cases
    python inference.py --json '{...}'   # score a single custom record
"""

import os
import json
import argparse

import joblib
import pandas as pd

from credit_features import FEATURE_COLS  # noqa: F401 (ensures module is importable)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(BASE_DIR, "models", "credit_score_model.pkl")


class CreditScorePredictor:
    """Thin wrapper around the trained pipeline bundle."""

    def __init__(self, model_path: str = DEFAULT_MODEL):
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.labels = bundle["label_classes"]
        self.model_name = bundle.get("model_name", "model")
        self.metrics = bundle.get("metrics", {})

    def predict(self, record: dict) -> dict:
        """Score a single raw customer record (dict of raw column values)."""
        df = pd.DataFrame([record])
        idx = int(self.pipeline.predict(df)[0])
        proba = self.pipeline.predict_proba(df)[0]
        return {
            "credit_score": self.labels[idx],
            "confidence": round(float(proba[idx]), 4),
            "probabilities": {lbl: round(float(p), 4) for lbl, p in zip(self.labels, proba)},
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score many records at once; returns a copy with prediction columns."""
        out = df.copy()
        idx = self.pipeline.predict(df)
        proba = self.pipeline.predict_proba(df)
        out["predicted_credit_score"] = [self.labels[i] for i in idx]
        out["confidence"] = proba.max(axis=1).round(4)
        return out


# ---------------------------------------------------------------------------
# Built-in test cases: one representative record per class
# ---------------------------------------------------------------------------
TEST_CASES = {
    "Expected: Good": {
        "Age": 32, "Occupation": "Media_Manager", "Annual_Income": 15829.88,
        "Monthly_Inhand_Salary": 1242.16, "Num_Bank_Accounts": 4, "Num_Credit_Card": 2,
        "Interest_Rate": 4, "Num_of_Loan": 1, "Delay_from_due_date": 10,
        "Num_of_Delayed_Payment": 4, "Changed_Credit_Limit": 2.7,
        "Num_Credit_Inquiries": 3, "Credit_Mix": "Good", "Outstanding_Debt": 968.61,
        "Credit_Utilization_Ratio": 32.49, "Credit_History_Age": "19 Years and 10 Months",
        "Payment_of_Min_Amount": "No", "Total_EMI_per_month": 6.72,
        "Amount_invested_monthly": 58.25,
        "Payment_Behaviour": "High_spent_Medium_value_payments",
        "Monthly_Balance": 319.25,
    },
    "Expected: Standard": {
        "Age": 39, "Occupation": "Writer", "Annual_Income": 34763.33,
        "Monthly_Inhand_Salary": 2907.94, "Num_Bank_Accounts": 7, "Num_Credit_Card": 5,
        "Interest_Rate": 8, "Num_of_Loan": 2, "Delay_from_due_date": 17,
        "Num_of_Delayed_Payment": 17, "Changed_Credit_Limit": 7.5,
        "Num_Credit_Inquiries": 4, "Credit_Mix": "Standard", "Outstanding_Debt": 856.97,
        "Credit_Utilization_Ratio": 28.59, "Credit_History_Age": "18 Years and 3 Months",
        "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 44.58,
        "Amount_invested_monthly": 193.73,
        "Payment_Behaviour": "Low_spent_Small_value_payments",
        "Monthly_Balance": 342.48,
    },
    "Expected: Poor": {
        "Age": 30, "Occupation": "Manager", "Annual_Income": 20107.21,
        "Monthly_Inhand_Salary": 1631.60, "Num_Bank_Accounts": 6, "Num_Credit_Card": 6,
        "Interest_Rate": 32, "Num_of_Loan": 2, "Delay_from_due_date": 15,
        "Num_of_Delayed_Payment": 10, "Changed_Credit_Limit": 10.4,
        "Num_Credit_Inquiries": 11, "Credit_Mix": "Bad", "Outstanding_Debt": 2544.6,
        "Credit_Utilization_Ratio": 29.20, "Credit_History_Age": "19 Years and 7 Months",
        "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 27.11,
        "Amount_invested_monthly": 160.33,
        "Payment_Behaviour": "Low_spent_Small_value_payments",
        "Monthly_Balance": 265.73,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Credit score inferencing")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--json", default=None, help="JSON string of a single record")
    args = parser.parse_args()

    predictor = CreditScorePredictor(args.model)
    print(f"Loaded model: {predictor.model_name} | classes: {predictor.labels}\n")

    if args.json:
        record = json.loads(args.json)
        print(json.dumps(predictor.predict(record), indent=2))
        return

    for name, record in TEST_CASES.items():
        result = predictor.predict(record)
        print(f"[{name}] -> {result['credit_score']} "
              f"(confidence {result['confidence']:.2%})")
        print(f"    probabilities: {result['probabilities']}")


if __name__ == "__main__":
    main()
