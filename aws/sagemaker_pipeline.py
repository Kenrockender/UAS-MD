"""
aws/sagemaker_pipeline.py
=========================
Defines & launches the cloud training pipeline on AWS SageMaker.

Two ways to use it:

1. Simple managed Training Job (recommended to start):
       python aws/sagemaker_pipeline.py --mode train

2. Full SageMaker Pipeline (process -> train -> register), so retraining is
   one click / one API call and every run is tracked in SageMaker Studio:
       python aws/sagemaker_pipeline.py --mode pipeline

Prerequisites (see README_AWS.md):
    - AWS account + `aws configure` done
    - A SageMaker execution role ARN (env SAGEMAKER_ROLE_ARN)
    - An S3 bucket (env S3_BUCKET); data_A.csv uploaded to s3://$S3_BUCKET/credit/data/
"""

import os
import argparse

import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput

REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
ROLE = os.environ.get("SAGEMAKER_ROLE_ARN")          # arn:aws:iam::<acct>:role/<SageMakerRole>
BUCKET = os.environ.get("S3_BUCKET")                 # my-credit-bucket
PREFIX = "credit"
FRAMEWORK_VERSION = "1.2-1"                           # SKLearn container version


def make_estimator():
    """A SageMaker-managed scikit-learn training job that runs aws/train.py."""
    return SKLearn(
        entry_point="train.py",
        source_dir=os.path.dirname(os.path.abspath(__file__)) + "/..",  # ship whole project (pipeline.py, credit_features.py)
        role=ROLE,
        instance_type="ml.m5.large",
        instance_count=1,
        framework_version=FRAMEWORK_VERSION,
        py_version="py3",
        base_job_name="credit-score",
        hyperparameters={"data-file": "data_A.csv"},
    )


def run_training_job():
    sess = sagemaker.Session()
    train_input = TrainingInput(f"s3://{BUCKET}/{PREFIX}/data/", content_type="text/csv")
    est = make_estimator()
    est.fit({"train": train_input})
    print("Model artifact:", est.model_data)  # s3://.../model.tar.gz
    return est


def build_pipeline():
    """Full SageMaker Pipeline: a single TrainingStep wrapped as a registered pipeline."""
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.steps import TrainingStep
    from sagemaker.workflow.pipeline_context import PipelineSession

    pipeline_session = PipelineSession()
    est = make_estimator()
    est.sagemaker_session = pipeline_session

    train_input = TrainingInput(f"s3://{BUCKET}/{PREFIX}/data/", content_type="text/csv")
    step_train = TrainingStep(name="TrainCreditModel", step_args=est.fit({"train": train_input}))

    pipeline = Pipeline(
        name="CreditScoreTrainingPipeline",
        steps=[step_train],
        sagemaker_session=pipeline_session,
    )
    pipeline.upsert(role_arn=ROLE)
    print("Pipeline upserted. Starting execution...")
    execution = pipeline.start()
    print("Execution ARN:", execution.arn)
    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "pipeline"], default="train")
    args = parser.parse_args()

    assert ROLE, "Set SAGEMAKER_ROLE_ARN env var"
    assert BUCKET, "Set S3_BUCKET env var"

    if args.mode == "train":
        run_training_job()
    else:
        build_pipeline()


if __name__ == "__main__":
    main()
