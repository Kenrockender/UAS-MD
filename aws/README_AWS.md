# AWS Cloud Pipeline & Deployment — Step-by-Step

This guide covers **Tugas 4** (cloud ML build pipeline on AWS) and **Tugas 5**
(cloud deployment on AWS + Streamlit with a public URL). You run these steps with
your own AWS account; every command is ready to copy.

The cloud pipeline reuses the **exact same code** as the local one
(`pipeline.py`, `credit_features.py`) — only the *execution environment* changes.
That reproducibility is the whole point of the local→cloud migration.

---

## 0. Prerequisites

```bash
pip install awscli boto3 sagemaker
aws configure          # enter Access Key, Secret, region (e.g. ap-southeast-1)
```

You need:
- An **S3 bucket**, e.g. `credit-score-<yourname>`.
- A **SageMaker execution role** (IAM role with `AmazonSageMakerFullAccess` +
  S3 access). Copy its ARN.

```bash
export AWS_REGION=ap-southeast-1
export S3_BUCKET=credit-score-yourname
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<ACCOUNT_ID>:role/<SageMakerExecutionRole>
```

---

## Tugas 4 — Cloud ML Build Pipeline (SageMaker)

### Step 1 — Upload the dataset to S3
```bash
aws s3 mb s3://$S3_BUCKET
aws s3 cp ../data_A.csv s3://$S3_BUCKET/credit/data/data_A.csv
```

### Step 2 — Launch a managed training job
```bash
python sagemaker_pipeline.py --mode train
```
What happens:
1. SageMaker spins up an `ml.m5.large` instance with a scikit-learn container.
2. It runs `aws/train.py`, which calls the **same `ExperimentRunner`** used
   locally — trying Logistic Regression, Decision Tree, Random Forest.
3. The best model (Random Forest) is written to `model.tar.gz` in S3.
4. The artifact S3 URI is printed at the end.

### Step 3 — (Optional) Full, repeatable SageMaker Pipeline
For **easy retraining + monitoring** (the assignment's emphasis), register a
SageMaker Pipeline so each retrain is one API call and is tracked in **SageMaker
Studio → Pipelines**:
```bash
python sagemaker_pipeline.py --mode pipeline
```
This upserts `CreditScoreTrainingPipeline` and starts an execution. Re-running it
(or triggering from EventBridge on a schedule) gives logged, versioned retraining
— the cloud equivalent of the local MLflow tracking.

### Step 4 — Download the trained model artifact
```bash
aws s3 cp <printed-model-s3-uri> model.tar.gz
tar -xzf model.tar.gz          # -> credit_score_model.pkl (+ model.pkl)
cp credit_score_model.pkl ../models/credit_score_model.pkl
```

---

## Tugas 5 — Cloud Deployment (AWS + Streamlit, public URL)

You have two deployment surfaces. Use **Path A** for the required public Streamlit
URL; **Path B** is the AWS-native real-time endpoint (good to show in the video).

### Path A — Public Streamlit (Streamlit Community Cloud) ✅ required
The model produced on AWS SageMaker is served by the Streamlit app.

1. Push the project to a **public GitHub repo** (include `app.py`,
   `inference.py`, `credit_features.py`, `requirements.txt`, and
   `models/credit_score_model.pkl`).
2. Go to **https://share.streamlit.io** → *New app* → pick the repo/branch →
   main file `app.py` → **Deploy**.
3. You get a public URL like `https://<app-name>.streamlit.app` — accessible by
   anyone (assignment requirement).
4. Open the app, click each **Quick test case** (Good / Standard / Poor), and
   screenshot the prediction for each class.

> The model was *built on AWS*; the public web front-end is Streamlit. This is a
> common, valid cloud architecture (managed training on AWS + hosted UI).

### Path B — AWS-native real-time endpoint (optional, for the video)
Deploy the same model as a SageMaker endpoint:
```python
from sagemaker.sklearn.model import SKLearnModel
model = SKLearnModel(model_data="s3://.../model.tar.gz",
                     role=os.environ["SAGEMAKER_ROLE_ARN"],
                     entry_point="inference_sagemaker.py",
                     framework_version="1.2-1")
predictor = model.deploy(instance_type="ml.t2.medium", initial_instance_count=1)
print(predictor.predict(test_record))
predictor.delete_endpoint()   # IMPORTANT: delete to stop billing
```

### Path C — Host Streamlit on AWS EC2 (alternative public URL)
```bash
# on an Ubuntu EC2 (open port 8501 in the security group)
sudo apt update && sudo apt install -y python3-pip
pip3 install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
# public URL: http://<EC2_PUBLIC_IP>:8501
```

---

## Cost hygiene
- **Delete SageMaker endpoints** when done (`predictor.delete_endpoint()`).
- **Stop/terminate EC2** instances after the demo.
- Training jobs stop billing automatically when finished.
