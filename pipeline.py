"""
pipeline.py
===========
Object-oriented, locally-runnable training pipeline for the Credit Score
classification problem, with MLflow experiment tracking.

Design (OOP, one responsibility per class):
    DataLoader      -> load raw csv, split features/target, train/test split
    Preprocessor    -> cleaning + imputation + scaling + encoding (sklearn)
    ModelTrainer    -> trains a single candidate model inside an MLflow run
    ModelEvaluator  -> computes & logs evaluation metrics
    ExperimentRunner-> orchestrates: tries several models, keeps the best,
                       and serialises the winning pipeline to a pickle bundle.

Run:
    python pipeline.py
    # then inspect: mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import os
import argparse

import numpy as np
import pandas as pd
import joblib

import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

from credit_features import (
    CreditDataCleaner, NUM_COLS, CAT_COLS, TARGET, TARGET_CLASSES,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
class DataLoader:
    """Loads the raw dataset and produces train/test splits."""

    def __init__(self, csv_path, target=TARGET, test_size=0.2, random_state=42):
        self.csv_path = csv_path
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()

    def load(self):
        df = pd.read_csv(self.csv_path)
        df = df.dropna(subset=[self.target]).reset_index(drop=True)

        X = df.drop(columns=[self.target])
        # LabelEncoder assigns indices alphabetically: Good=0, Poor=1, Standard=2.
        # The bundle stores label_encoder.classes_ so inference maps index->name correctly.
        self.label_encoder.fit(TARGET_CLASSES)
        y = self.label_encoder.transform(df[self.target])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.random_state, stratify=y,
        )
        print(f"[DataLoader] train={X_train.shape} test={X_test.shape} "
              f"classes={list(self.label_encoder.classes_)}")
        return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
class Preprocessor:
    """Builds the cleaning + imputation + scaling + encoding sklearn step."""

    def __init__(self, num_cols=NUM_COLS, cat_cols=CAT_COLS):
        self.num_cols = num_cols
        self.cat_cols = cat_cols

    def build(self):
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        column_transformer = ColumnTransformer([
            ("num", num_pipe, self.num_cols),
            ("cat", cat_pipe, self.cat_cols),
        ])
        # Cleaner runs first so the ColumnTransformer always sees tidy columns.
        return Pipeline([
            ("cleaner", CreditDataCleaner()),
            ("transform", column_transformer),
        ])


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------
class ModelEvaluator:
    """Computes classification metrics and logs them to MLflow."""

    def __init__(self, class_names=TARGET_CLASSES):
        self.class_names = class_names

    def evaluate(self, model, X_test, y_test):
        preds = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1_weighted": f1_score(y_test, preds, average="weighted"),
            "f1_macro": f1_score(y_test, preds, average="macro"),
            "precision_weighted": precision_score(y_test, preds, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, preds, average="weighted", zero_division=0),
        }
        report = classification_report(y_test, preds, target_names=self.class_names, zero_division=0)
        cm = confusion_matrix(y_test, preds)
        return metrics, report, cm


# ---------------------------------------------------------------------------
# 4. Training (one model = one MLflow run)
# ---------------------------------------------------------------------------
class ModelTrainer:
    """Trains a single candidate model and records it in MLflow."""

    def __init__(self, name, estimator, preprocessor, evaluator):
        self.name = name
        self.estimator = estimator
        self.preprocessor = preprocessor
        self.evaluator = evaluator

    def train(self, X_train, y_train, X_test, y_test):
        pipe = Pipeline([
            ("preprocessor", self.preprocessor.build()),
            ("classifier", self.estimator),
        ])
        with mlflow.start_run(run_name=self.name):
            pipe.fit(X_train, y_train)
            metrics, report, cm = self.evaluator.evaluate(pipe, X_test, y_test)

            mlflow.log_param("model", self.name)
            for k, v in self.estimator.get_params().items():
                mlflow.log_param(f"hp_{k}", v)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("report", report)
            mlflow.sklearn.log_model(pipe, name=f"model_{self.name}")

            print(f"\n[{self.name}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            print(report)
            print("Confusion matrix (rows=true, cols=pred):")
            print(pd.DataFrame(cm, index=self.evaluator.class_names,
                               columns=self.evaluator.class_names))
        return pipe, metrics


# ---------------------------------------------------------------------------
# 5. Orchestration
# ---------------------------------------------------------------------------
class ExperimentRunner:
    """Runs all candidate models, keeps the best, and serialises it."""

    def __init__(self, csv_path, experiment_name="credit_score_classification",
                 tracking_uri="sqlite:///mlflow.db", selection_metric="f1_weighted"):
        self.csv_path = csv_path
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.selection_metric = selection_metric

    def candidate_models(self):
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
            "DecisionTree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=42),
            "RandomForest": RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_leaf=2,
                n_jobs=-1, random_state=42),
        }

    def run(self, model_out="models/credit_score_model.pkl"):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        loader = DataLoader(self.csv_path)
        X_train, X_test, y_train, y_test = loader.load()

        preprocessor = Preprocessor()
        # use the encoder's own class order so per-class names line up with labels
        evaluator = ModelEvaluator(class_names=list(loader.label_encoder.classes_))

        best_pipe, best_name, best_score, best_metrics = None, None, -1.0, None
        for name, estimator in self.candidate_models().items():
            trainer = ModelTrainer(name, estimator, preprocessor, evaluator)
            pipe, metrics = trainer.train(X_train, y_train, X_test, y_test)
            if metrics[self.selection_metric] > best_score:
                best_score = metrics[self.selection_metric]
                best_pipe, best_name, best_metrics = pipe, name, metrics

        print(f"\n==> BEST MODEL: {best_name} "
              f"({self.selection_metric}={best_score:.4f})")

        # Serialise a self-contained bundle for inference/deployment
        out_path = os.path.join(BASE_DIR, model_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        bundle = {
            "pipeline": best_pipe,
            "model_name": best_name,
            "label_classes": [str(c) for c in loader.label_encoder.classes_],
            "feature_cols": NUM_COLS + CAT_COLS,
            "metrics": best_metrics,
        }
        # joblib + compression keeps the Random Forest bundle small enough for
        # GitHub / Streamlit Cloud (full-accuracy model ~33 MB instead of ~154 MB).
        joblib.dump(bundle, out_path, compress=3)
        print(f"==> Saved bundle to {out_path}")
        return bundle


def main():
    parser = argparse.ArgumentParser(description="Train credit score classifier")
    parser.add_argument("--data", default=os.path.join(BASE_DIR, "data_A.csv"),
                        help="Path to training CSV")
    parser.add_argument("--out", default="models/credit_score_model.pkl")
    args = parser.parse_args()

    runner = ExperimentRunner(args.data)
    runner.run(model_out=args.out)


if __name__ == "__main__":
    main()
