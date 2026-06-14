# Final Project — Model Deployment
## Credit Score Classification (Poor / Standard / Good)

**NIM:** 2802413871

Memprediksi performa kredit nasabah (`Credit_Score`) di institusi keuangan
menggunakan machine learning, lengkap dengan pipeline training, tracking
eksperimen (MLflow), deployment web (Streamlit), dan migrasi ke cloud (AWS).

---

## Struktur Berkas

```
.
├── notebook.ipynb            # (Tugas 1) Eksplorasi data + modelling
├── pipeline.py               # (Tugas 2) Pipeline training OOP + MLflow
├── credit_features.py        # Cleaning & definisi fitur (dipakai bersama)
├── inference.py              # (Tugas 3) Kode inferencing + 3 test case
├── app.py                    # (Tugas 3 & 5) Deployment Streamlit
├── requirements.txt          # Dependensi
├── data_A.csv                # Dataset (untuk reproduksi training)
├── models/
│   └── credit_score_model.pkl   # Bundel model terbaik (Random Forest)
├── aws/                      # (Tugas 4 & 5) Pipeline & deployment cloud
│   ├── train.py
│   ├── sagemaker_pipeline.py
│   ├── inference_sagemaker.py
│   └── README_AWS.md
└── docs/
    └── comparison_local_vs_cloud.md   # (Tugas 6) Perbandingan local vs cloud
```

---

## Cara Menjalankan (Local)

```bash
pip install -r requirements.txt

# 1. Training (mencoba 3 model, log ke MLflow, simpan model terbaik)
python pipeline.py
mlflow ui --backend-store-uri sqlite:///mlflow.db     # lihat eksperimen

# 2. Inferencing (3 test case: Good / Standard / Poor)
python inference.py

# 3. Deployment web
streamlit run app.py
```

## Model
- **Algoritma dicoba:** Logistic Regression, Decision Tree, Random Forest.
- **Terbaik:** Random Forest — accuracy ≈ **74%**, F1 (weighted) ≈ **0.74**.
- **Metrik pemilihan:** F1 weighted (karena target tidak seimbang).
- Bundel `credit_score_model.pkl` berisi pipeline lengkap (cleaning → impute →
  scale/encode → classifier) + daftar label, sehingga siap untuk inferensi 1 record.

## Cloud (AWS)
Lihat [`aws/README_AWS.md`](aws/README_AWS.md) untuk langkah lengkap SageMaker
(pipeline training) dan deployment (Streamlit Community Cloud public URL +
opsi SageMaker endpoint / EC2).

## Test Cases (untuk screenshot deployment)
`inference.py` dan tombol *Quick test case* di Streamlit menyediakan satu contoh
per kelas yang sudah diverifikasi:
| Test case | Prediksi | Confidence |
|---|---|---|
| Good      | Good     | ~0.93 |
| Standard  | Standard | ~0.99 |
| Poor      | Poor     | ~0.81 |
