# Project Setup & Context — Hotel Reservation MLOps

> **Last updated:** 2026-07-19  
> **Course:** [Mastering Advanced MLOps on GCP — CI/CD, Kubernetes, Kubeflow](https://www.udemy.com/course/mastering-advanced-mlops-on-gcp-cicd-kubernetes-kubeflow/) (Udemy)  
> **Instructor:** Sudhanshu Gusain / KRISHAI Technologies  
> **Course recorded:** March 2025  
> **Project:** Hotel Reservation Prediction with MLFlow, Jenkins and GCP Deployment

---

## 1. Environment Overview

### 1.1 System
| Item | Value |
|------|-------|
| **OS** | Ubuntu 26.04 (Linux) |
| **Shell** | bash |
| **Architecture** | x86_64 |

### 1.2 Python Environment
- **Python version:** 3.14.4
- **Virtual env:** `venv/` (created with `python3 -m venv venv`)
- **Activation:** `source venv/bin/activate`
- **Package manager:** pip

### 1.3 Key Distinction From Instructor
The instructor works on **Windows**. We work on **Ubuntu Linux**. This affects:
- Path conventions (`/` vs `\`) — handled by `os.path.join()` and `pathlib`
- Python venv activation: `source venv/bin/activate` (ours) vs `venv\Scripts\activate` (theirs)
- GCP authentication method (see Section 4)
- Shell scripts: `.sh` (ours) vs `.bat`/`.ps1` (theirs)
- Docker / Jenkins / CI-CD scripts (will need Ubuntu/Debian base images)

---

## 2. Project Structure

```
.
├── artifacts/                     # Generated outputs (gitignored)
│   └── raw/                       # Raw data from ingestion
│       ├── raw.csv                # Full CSV downloaded from GCS
│       ├── train.csv              # Training split (80%)
│       └── test.csv               # Test split (20%)
├── config/
│   ├── __init__.py
│   ├── config.yaml                # Hyperparameters, column names, bucket config
│   └── paths_config.py            # All file/directory paths as constants
├── logs/
│   └── log_2026-07-19.log         # Current day's log
├── src/
│   ├── __init__.py
│   ├── logger.py                  # Logging module (daily log files)
│   ├── custom_exception.py        # Custom exception with traceback
│   └── data_ingestion.py          # GCS download + train/test split
├── utils/
│   ├── __init__.py
│   └── common_functions.py        # read_yaml(), load_data()
├── static/                        # For Flask app (future)
├── templates/                     # For Flask app (future)
├── notebook/                      # For EDA notebooks (future)
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installer (pip install -e .)
├── testing.py                     # Initial logger/exception test script
└── setup.md                       # This file
```

---

## 3. Files & Components — Detailed

### 3.1 `src/logger.py`
- Creates a `logs/` directory if missing
- Generates a daily log file: `logs/log_YYYY-MM-DD.log`
- Uses Python's `logging` module with `basicConfig`
- Provides `get_logger(name)` function for all modules

```python
LOGS_DIR = "logs"
LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")
```

### 3.2 `src/custom_exception.py`
- Custom exception class that captures:
  - File name where error occurred
  - Line number of the error
  - Error message
- Usage: `raise CustomException("message", sys)`
- Note: Uses `traceback.sys.exc_info()` — this works but `sys.exc_info()` is cleaner

### 3.3 `src/data_ingestion.py`
**Data Ingestion class** with 3 methods:
1. `download_csv_from_gcp()` — Downloads CSV from GCS bucket using ADC
2. `split_data()` — Splits into train/test (80/20) using `sklearn.model_selection.train_test_split`
3. `run()` — Orchestrates the pipeline

**Key:** Uses `storage.Client()` with **no arguments** — relies on Application Default Credentials (ADC). See Section 4.

### 3.4 `config/paths_config.py`
Central path constants:
```python
RAW_DIR = "artifacts/raw"
RAW_FILE_PATH = "artifacts/raw/raw.csv"
TRAIN_FILE_PATH = "artifacts/raw/train.csv"
TEST_FILE_PATH = "artifacts/raw/test.csv"
CONFIG_PATH = "config/config.yaml"
PROCESSED_DIR = "artifacts/processed"
PROCESSED_TRAIN_DATA_PATH = "artifacts/processed/processed_train.csv"
PROCESSED_TEST_DATA_PATH = "artifacts/processed/processed_test.csv"
MODEL_OUTPUT_PATH = "artifacts/models/lgbm_model.pkl"
```

### 3.5 `config/config.yaml`
```yaml
data_ingestion:
  bucket_name: "my_bucket9787"
  bucket_file_name: "Hotel_Reservations.csv"
  train_ratio: 0.8

data_processing:
  categorical_columns: [type_of_meal_plan, required_car_parking_space, ...]
  numerical_columns: [no_of_adults, no_of_children, ...]
  skewness_threshold: 5
  no_of_features: 10
```

### 3.6 `utils/common_functions.py`
Two utility functions:
- `read_yaml(file_path)` — Reads and parses YAML config files
- `load_data(path)` — Loads CSV into pandas DataFrame

### 3.7 `setup.py`
```python
from setuptools import setup, find_packages
with open("requirements.txt") as f:
    requirements = f.read().splitlines()
setup(
    name="Hotel-Reservation1",
    version="0.2",
    author="Safwen",
    packages=find_packages(),
    install_requires=requirements,
)
```

### 3.8 `testing.py`
Initial smoke test for logging and exception system.

---

## 4. GCP Authentication — ADC (Application Default Credentials)

### 4.1 The Problem
Google's **Organization Policy** (`iam.disableServiceAccountKeyCreation`) **blocks** creating downloadable JSON service account keys on free-tier GCP accounts ($300 credit). This is enforced by Google's "Secure by Default" initiative.

Attempting to create a key in GCP Console → IAM → Service Accounts → Keys gives:
```
Service account key creation is disabled. An Organization Policy that blocks
service accounts key creation has been enforced on your organization.
```

Attempting to override via Organization Policies console fails because personal Gmail accounts lack sufficient `orgpolicy.policyAdmin` permissions.

### 4.2 The Solution: ADC (Application Default Credentials)
Instead of a service account JSON key file, we use **ADC** which stores credentials locally after OAuth login.

**Why this works:**
- `storage.Client()` (no args) auto-discovers ADC credentials
- The Python code is **identical** to the instructor's code
- Google's own client libraries prefer ADC over JSON keys

### 4.3 Setup Commands Executed

```bash
# 1. Install gcloud CLI
sudo apt-get update
sudo apt-get install ca-certificates gnupg curl
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update && sudo apt-get install google-cloud-cli -y

# 2. Verify
gcloud --version   # → Google Cloud SDK 576.0.0

# 3. Authenticate (opens browser for Google login)
gcloud init

# 4. Set up ADC (stores credentials at ~/.config/gcloud/application_default_credentials.json)
gcloud auth application-default login

# 5. Set quota project (required for client libraries to track API usage)
gcloud auth application-default set-quota-project project-150342d7-e272-4d0c-ab9
```

### 4.4 GCP Resources Created
| Resource | Value |
|----------|-------|
| **GCP Project ID** | `project-150342d7-e272-4d0c-ab9` |
| **GCP Project Name** | `Hotel-reservation-mlops` |
| **GCP Account** | `safwencherif4@gmail.com` |
| **GCS Bucket** | `gs://my_bucket9787/` |
| **CSV File** | `Hotel_Reservations.csv` |

### 4.5 ADC Credential Location
```
~/.config/gcloud/application_default_credentials.json
```

### 4.6 Impact on Future Work
- **Local development:** `storage.Client()` works transparently
- **Docker containers:** Will need to mount the ADC file or use Workload Identity / GCE default service account
- **Jenkins CI/CD:** Will need to configure GCP credentials via Jenkins plugins or environment variables
- **No `GOOGLE_APPLICATION_CREDENTIALS` env var needed locally** — ADC auto-detects the default file

---

## 5. Data Ingestion — Verified Working ✅

### 5.1 Log Output (2026-07-19)
```
15:44:50 - INFO - Successfully read the YAML file
15:44:50 - INFO - Data Ingestion started with my_bucket9787 and file is Hotel_Reservations.csv
15:44:50 - INFO - Starting data ingestion process
15:44:53 - INFO - CSV file is successfully downloaded to artifacts/raw/raw.csv
15:44:53 - INFO - Starting the splitting process
15:44:53 - INFO - Train data saved to artifacts/raw/train.csv
15:44:53 - INFO - Test data saved to artifacts/raw/test.csv
15:44:53 - INFO - Data ingestion completed successfully
15:44:53 - INFO - Data ingestion completed
```

### 5.2 Generated Artifacts
| File | Path | Size (approx) |
|------|------|---------------|
| Full dataset | `artifacts/raw/raw.csv` | ~350 KB |
| Training set (80%) | `artifacts/raw/train.csv` | ~280 KB |
| Test set (20%) | `artifacts/raw/test.csv` | ~70 KB |

---

## 6. Installed Dependencies

All installed in `venv/` via pip:

```
pandas
numpy
scikit-learn
google-cloud-storage
pyyaml
imbalanced-learn
lightgbm
mlflow
flask
joblib
scipy
```

> Many of the above (mlflow, lightgbm, imbalanced-learn, flask) are installed ahead even though they're needed in later stages.

---

## 7. Course Roadmap — What's Ahead

This is the **first project** (8h 46min) of a larger course (54h+). The full pipeline is:

```
1. ✅ Data Ingestion (GCS → raw/train/test CSV)
2. 📝 Data Preprocessing (Label Encoding, Skewness, SMOTE balancing)
3. 📝 Model Training (LightGBM + RandomizedSearchCV + MLflow tracking)
4. 📝 Flask App (Web interface for predictions)
5. 📝 Docker (Containerize the app)
6. 📝 Jenkins CI/CD (Automated pipeline)
7. 📝 GCP Deployment (Cloud Run / Compute Engine)

Later projects cover:
- Anime Recommender (Comet-ML, DVC, Jenkins, Kubernetes)
- Survival Prediction (Airflow, SQL, Redis, Grafana, Prometheus)
- Object Detection (TensorBoard, DVC, FastAPI)
- Cancer Prediction (MLflow+DagsHUB, Minikube, Kubeflow)
```

---

## 8. Known Issues & Notes

### 8.1 `custom_exception.py` — Minor Improvement Possible
Line 11 uses `traceback.sys.exc_info()` which is an implementation detail (accesses `sys` via the traceback module's import). The safer approach is `sys.exc_info()` since `sys` is already imported. Both work, but the latter is more readable.

### 8.2 Package Versions (Course: March 2025 vs Now: July 2026)
Some packages have had major version bumps:
- Python is now 3.14 (was likely 3.11-3.12 in the course)
- lightgbm, scikit-learn, etc. have newer versions
- If compatibility issues arise, specific versions can be pinned

### 8.3 GCP Console May Look Different
The GCP UI changes frequently. Console screenshots from the course (March 2025) will look dated.

---

## 9. How to Continue

### Activate environment & run:
```bash
cd ~/"Mlops Projects/Hotel Reservation 1"
source venv/bin/activate
python src/data_ingestion.py
```

### Next step expected:
Create `src/data_preprocessing.py` with `DataProcessor` class that performs:
1. Drop unnecessary columns (`Unnamed: 0`, `Booking_ID`)
2. Remove duplicates
3. Label encode categorical columns
4. Handle skewness with log transform
5. Balance data with SMOTE
