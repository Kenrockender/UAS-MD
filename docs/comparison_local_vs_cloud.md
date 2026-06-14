# Perbandingan Pendekatan Local vs Cloud (AWS)

**Tugas 6.** Berdasarkan pengalaman membangun pipeline & deployment secara lokal
maupun di cloud (AWS), berikut perbandingan kelebihan dan kekurangannya.

## Ringkasan

| Aspek | Local (laptop + MLflow + Streamlit) | Cloud / AWS (SageMaker + S3 + Streamlit Cloud) |
|---|---|---|
| **Setup awal** | Cepat, cukup `pip install` | Perlu akun AWS, IAM role, S3, konfigurasi |
| **Biaya** | Gratis (pakai resource sendiri) | Bayar per pemakaian (instance, storage, endpoint) |
| **Skalabilitas** | Terbatas pada 1 mesin | Mudah scale up/out (instance lebih besar, paralel) |
| **Ketersediaan (uptime)** | Mati saat laptop mati | 24/7, managed, ada SLA |
| **Reproducibility** | MLflow lokal (1 user) | SageMaker Pipelines + Model Registry (tim) |
| **Kolaborasi** | Sulit dibagikan | Artifact di S3, endpoint/URL bisa diakses bersama |
| **Retraining terjadwal** | Manual / cron lokal | Otomatis (EventBridge → Pipeline) |
| **Akses publik** | Perlu tunneling | Native (Streamlit Cloud / endpoint / EC2) |
| **Keamanan & audit** | Bergantung mesin lokal | IAM, VPC, CloudTrail, enkripsi |
| **Maintenance** | Diurus sendiri | Sebagian besar managed oleh AWS |

## Local — Kelebihan
- **Cepat & murah** untuk eksperimen dan iterasi awal.
- **Kontrol penuh** atas environment; tidak perlu koneksi internet.
- **MLflow** memberi tracking eksperimen yang ringan dan mudah dibaca.
- Cocok untuk **prototyping** dan debugging sebelum naik ke produksi.

## Local — Kekurangan
- **Tidak scalable**: dibatasi CPU/RAM laptop; training lambat untuk data besar.
- **Tidak selalu tersedia**: layanan mati ketika mesin dimatikan.
- **Sulit kolaborasi** dan tidak ada jaminan uptime/SLA.
- Reproducibility terbatas pada satu lingkungan.

## Cloud / AWS — Kelebihan
- **Scalable & elastis**: pilih instance sesuai kebutuhan, bisa paralel.
- **Managed & andal**: SageMaker mengurus infrastruktur, endpoint 24/7.
- **Reproducible untuk tim**: SageMaker Pipelines + Model Registry + artifact di S3.
- **Retraining otomatis** & termonitor (EventBridge, CloudWatch).
- **Akses publik** mudah (Streamlit Cloud / SageMaker endpoint / EC2).
- **Keamanan enterprise**: IAM, VPC, enkripsi, audit (CloudTrail).

## Cloud / AWS — Kekurangan
- **Biaya** berjalan terus selama resource aktif (endpoint, EC2) — perlu disiplin
  mematikan resource.
- **Kurva belajar** lebih curam (IAM, role, S3, SDK SageMaker).
- **Vendor lock-in** terhadap layanan AWS tertentu.
- **Latency setup** lebih lama (cold start training job, provisioning endpoint).

## Kesimpulan
Pendekatan **lokal** ideal untuk **eksperimen, prototyping, dan iterasi cepat**
dengan biaya nol. Begitu model terbukti stabil dan butuh **skalabilitas,
ketersediaan, dan kolaborasi**, migrasi ke **cloud (AWS)** memberi nilai jauh
lebih besar meski menambah biaya dan kompleksitas. Karena kode pipeline kita
identik di kedua lingkungan (`pipeline.py` + `credit_features.py`), transisi
local→cloud menjadi mulus dan reproducible.
