"""
aws/inference_sagemaker.py
==========================
SageMaker inference handler for the credit score model (Path B in README_AWS.md).

SageMaker's SKLearn serving container calls these four functions. The model
bundle is the same pickle produced by pipeline.py / aws/train.py.
"""

import os
import json

import joblib
import pandas as pd


def model_fn(model_dir):
    """Load the model bundle from the SageMaker model directory."""
    for name in ("model.pkl", "credit_score_model.pkl"):
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return joblib.load(path)  # joblib.load also reads plain pickle files
    raise FileNotFoundError("No model pickle found in model_dir")


def input_fn(request_body, content_type="application/json"):
    """Parse a single JSON record (or list of records) into a DataFrame."""
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    data = json.loads(request_body)
    if isinstance(data, dict):
        data = [data]
    return pd.DataFrame(data)


def predict_fn(input_df, bundle):
    """Run the pipeline and return labels + probabilities."""
    pipe, labels = bundle["pipeline"], bundle["label_classes"]
    idx = pipe.predict(input_df)
    proba = pipe.predict_proba(input_df)
    return [
        {
            "credit_score": labels[i],
            "probabilities": {lbl: round(float(p), 4) for lbl, p in zip(labels, row)},
        }
        for i, row in zip(idx, proba)
    ]


def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), "application/json"
