"""
aws/train.py
============
SageMaker-compatible training entry point.

When run inside a SageMaker Training Job (using the SKLearn framework
container), SageMaker calls this script with the data channel mounted at
SM_CHANNEL_TRAIN and expects the model written to SM_MODEL_DIR.

It reuses the exact same OOP pipeline + cleaning logic as the local run, so the
local and cloud models are reproducible from one codebase.

Locally you can simulate it:
    python aws/train.py --train ./ --model-dir ./aws/model_out
"""

import os
import argparse
import shutil

from pipeline import ExperimentRunner  # reuses local OOP pipeline


def parse_args():
    p = argparse.ArgumentParser()
    # SageMaker injects these as environment variables
    p.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "."))
    p.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model_out"))
    p.add_argument("--data-file", type=str, default="data_A.csv")
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = os.path.join(args.train, args.data_file)
    os.makedirs(args.model_dir, exist_ok=True)

    # MLflow tracking is optional in the cloud; keep a local file store inside the job
    runner = ExperimentRunner(
        csv_path=csv_path,
        experiment_name="credit_score_sagemaker",
        tracking_uri=f"file://{os.path.join(args.model_dir, 'mlruns')}",
    )
    out_pkl = os.path.join(args.model_dir, "credit_score_model.pkl")
    bundle = runner.run(model_out=out_pkl)

    # Also write the canonical SageMaker artifact name (model.pkl)
    shutil.copyfile(out_pkl, os.path.join(args.model_dir, "model.pkl"))
    print("Training complete. Best model:", bundle["model_name"], bundle["metrics"])


if __name__ == "__main__":
    main()
